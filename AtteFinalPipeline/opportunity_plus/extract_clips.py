import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def find_sample_pngs(xai_dir):
    pattern = os.path.join(xai_dir, "*_class*_*_s*.png")
    return sorted(glob.glob(pattern))


def json_for_png(png_path):
    json_path = os.path.splitext(png_path)[0] + ".json"
    return json_path if os.path.exists(json_path) else None


def resolve_video_path(metadata, opp_plus_root):
    video_rel = metadata.get("video_file", "")
    candidates = []
    if opp_plus_root and video_rel:
        candidates.append(os.path.join(opp_plus_root, video_rel))
        candidates.append(os.path.join(opp_plus_root, os.path.basename(video_rel)))
    baked = metadata.get("video_abspath", "")
    if baked:
        candidates.append(baked)
    for c in candidates:
        if c and os.path.exists(c):
            return (c, candidates)
    return (None, candidates)


def load_sample_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def matching_png_for_json(json_path):
    png_path = os.path.splitext(json_path)[0] + ".png"
    return png_path if os.path.exists(png_path) else None


def _check_ffmpeg_available():
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            print(f"[FATAL] '{tool}' not found on PATH. Install ffmpeg and retry.")
            sys.exit(1)


def extract_clip(video_path, start_time_s, end_time_s, context_seconds, out_path):
    start_t = max(0.0, float(start_time_s) - context_seconds)
    end_t = float(end_time_s) + context_seconds
    duration = max(0.1, end_t - start_t)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_t:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-an",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60)
        return (True, None)
    except subprocess.CalledProcessError as e:
        return (False, f"ffmpeg returncode {e.returncode}")
    except subprocess.TimeoutExpired:
        return (False, "ffmpeg timeout")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def _norm_activity(name):
    return name.replace("_", " ").strip().lower()


def safe_folder_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")


def build_sample_folder_name(entry):
    idx = int(entry["sample_idx"])
    cls = entry["_filename_class"]
    name = entry["_filename_activity"]
    return f"sample_{idx:04d}_class{cls:02d}_{safe_folder_name(name)}"


def parse_filename_class_and_activity(json_path):
    base = os.path.basename(json_path)
    stem = os.path.splitext(base)[0]
    if "_class" not in stem:
        return (None, None)
    _, rest = stem.split("_class", 1)
    parts = rest.split("_", 1)
    if len(parts) < 2:
        return (None, None)
    try:
        cls = int(parts[0])
    except ValueError:
        return (None, None)
    name_part = parts[1]
    if "_s" in name_part:
        name_part = name_part.rsplit("_s", 1)[0]
    return (cls, name_part)


def process_sample(
    json_path,
    opp_plus_root,
    output_dir,
    context_seconds,
    include_no_png,
    classes_filter,
    runs_filter,
    null_filter,
    min_confidence,
    class_zero_id,
):
    try:
        entry = load_sample_json(json_path)
    except Exception as e:
        return {
            "status": "error",
            "json": json_path,
            "reason": f"could not read JSON: {e}",
        }
    cls, activity = parse_filename_class_and_activity(json_path)
    if cls is None:
        return {
            "status": "skip",
            "json": json_path,
            "reason": "filename pattern doesn't expose class id",
        }
    entry["_filename_class"] = cls
    entry["_filename_activity"] = activity
    if null_filter and cls == class_zero_id:
        return {"status": "skip", "json": json_path, "reason": "null class (filtered)"}
    if classes_filter is not None and cls not in classes_filter:
        return {
            "status": "skip",
            "json": json_path,
            "reason": f"class {cls} not in --classes",
        }
    metadata = entry.get("opportunity_plus_metadata", {})
    if not metadata:
        return {
            "status": "skip",
            "json": json_path,
            "reason": "no opportunity_plus_metadata block — wrong JSON source?",
        }
    run = metadata.get("run", "")
    if runs_filter is not None and run not in runs_filter:
        return {
            "status": "skip",
            "json": json_path,
            "reason": f"run {run} not in --runs",
        }
    confidence = float(entry.get("confidence", 0.0))
    if confidence < min_confidence:
        return {
            "status": "skip",
            "json": json_path,
            "reason": f"confidence {confidence:.3f} < {min_confidence:.3f}",
        }
    png_path = matching_png_for_json(json_path)
    if png_path is None and (not include_no_png):
        return {
            "status": "skip",
            "json": json_path,
            "reason": "no matching PNG (re-run with --include_no_png to keep)",
        }
    video_rel = metadata.get("video_file", "")
    video_abs, tried = resolve_video_path(metadata, opp_plus_root)
    if video_abs is None:
        return {
            "status": "skip",
            "json": json_path,
            "reason": f"video not found: tried {tried!r} (rel={video_rel!r}, root={opp_plus_root!r})",
        }
    fps = float(metadata.get("video_fps") or 30.0)
    start_frame = int(metadata["start_frame"])
    end_frame = int(metadata["end_frame"])
    start_time_s = metadata.get("start_time_s")
    end_time_s = metadata.get("end_time_s")
    if start_time_s is None or end_time_s is None:
        start_time_s = start_frame / fps
        end_time_s = end_frame / fps
    folder_name = build_sample_folder_name(entry)
    sample_dir = os.path.join(output_dir, folder_name)
    os.makedirs(sample_dir, exist_ok=True)
    png_copied = False
    if png_path is not None:
        try:
            shutil.copy2(png_path, os.path.join(sample_dir, "attribution.png"))
            png_copied = True
        except Exception as e:
            return {
                "status": "error",
                "json": json_path,
                "reason": f"PNG copy failed: {e}",
            }
    clip_path = os.path.join(sample_dir, "clip.mp4")
    ok, err = extract_clip(
        video_abs, start_time_s, end_time_s, context_seconds, clip_path
    )
    if not ok:
        return {"status": "error", "json": json_path, "reason": f"ffmpeg failed: {err}"}
    meta = {
        "sample_idx": int(entry["sample_idx"]),
        "true_class_id": cls,
        "true_class_name": activity,
        "predicted_class_name": entry.get("activity", ""),
        "confidence": confidence,
        "correct": _norm_activity(entry.get("activity", ""))
        == _norm_activity(activity),
        "context_seconds_added": context_seconds,
        "model_window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": metadata.get("start_time_s"),
            "end_time_s": metadata.get("end_time_s"),
            "video_fps": fps,
        },
        "run": run,
        "video_file": video_rel,
        "sensor_file": metadata.get("sensor_file", ""),
        "srt_file": metadata.get("srt_file", ""),
        "used_srt_anchor": metadata.get("used_srt_anchor"),
        "source_json": os.path.relpath(json_path),
        "source_png": os.path.relpath(png_path) if png_path else None,
        "png_in_folder": png_copied,
    }
    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {
        "status": "ok",
        "json": json_path,
        "folder": sample_dir,
        "had_png": png_copied,
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Pair XAI attribution PNGs with their Opportunity++ video clips into one folder per sample for side-by-side review.",
    )
    ap.add_argument(
        "--xai_dir",
        required=True,
        help="Output dir from run_xai_opportunity_plus.py (contains the per-sample JSONs).",
    )
    ap.add_argument(
        "--opp_plus_root",
        required=True,
        help="Opportunity++ root, for resolving video_file paths stored relative inside each JSON.",
    )
    ap.add_argument(
        "--output_dir", required=True, help="Where to write the paired sample folders."
    )
    ap.add_argument(
        "--context_seconds",
        type=float,
        default=1.5,
        help="Seconds of pre/post video padding around the model window. 0.0 = exactly the 24-sample window (~0.8s).",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of samples processed.",
    )
    ap.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these class IDs (true label).",
    )
    ap.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these Opportunity++ runs (e.g. S2-ADL4 S4-ADL1).",
    )
    ap.add_argument(
        "--skip_null",
        action="store_true",
        default=True,
        help="Skip class 0 (Null). ON by default.",
    )
    ap.add_argument(
        "--include_null",
        action="store_true",
        help="Override --skip_null and include Null samples.",
    )
    ap.add_argument(
        "--include_no_png",
        action="store_true",
        help="Also process samples that don't have a PNG yet (only the clip will be extracted).",
    )
    ap.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Drop samples with confidence below this.",
    )
    ap.add_argument("--workers", type=int, default=8, help="Parallel ffmpeg workers.")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without copying / extracting.",
    )
    args = ap.parse_args()
    if not args.dry_run:
        _check_ffmpeg_available()
    os.makedirs(args.output_dir, exist_ok=True)
    png_paths = find_sample_pngs(args.xai_dir)
    print(f"[*] Found {len(png_paths)} dashboard PNGs in {args.xai_dir}")
    if not png_paths:
        print(
            "[FATAL] No dashboard PNGs found. Is --xai_dir correct, and did run_xai_opportunity_plus.py run with --plot_every_n > 0?"
        )
        sys.exit(1)
    json_paths = []
    pngs_without_json = 0
    for pp in png_paths:
        jp = json_for_png(pp)
        if jp is None:
            pngs_without_json += 1
        else:
            json_paths.append(jp)
    if pngs_without_json:
        print(f"    [WARN] {pngs_without_json} PNGs had no sibling JSON — skipped.")
    print(f"[*] Paired {len(json_paths)} PNG+JSON samples to process")
    null_filter = args.skip_null and (not args.include_null)
    classes_filter = set(args.classes) if args.classes else None
    runs_filter = set(args.runs) if args.runs else None
    if args.max_samples is not None:
        json_paths = json_paths[: args.max_samples]
        print(f"    --max_samples cap applied: {len(json_paths)} samples")
    if args.dry_run:
        print("\n[DRY RUN] Walking JSONs and showing what would be paired ...")
        kept = 0
        for jp in json_paths:
            cls, activity = parse_filename_class_and_activity(jp)
            png = matching_png_for_json(jp)
            tag = []
            if cls is None:
                tag.append("BAD-NAME")
            else:
                if null_filter and cls == 0:
                    tag.append(f"null-skip(cls={cls})")
                if classes_filter is not None and cls not in classes_filter:
                    tag.append(f"class-filter(cls={cls})")
            if png is None and (not args.include_no_png):
                tag.append("no-png-skip")
            if not tag:
                tag.append(
                    f"OK cls={cls} act={activity} png={('yes' if png else 'no')}"
                )
                kept += 1
            print(f"  {os.path.basename(jp):60s}  →  {' | '.join(tag)}")
        print(f"\n[DRY RUN] {kept} of {len(json_paths)} samples would be paired.")
        return
    print(
        f"\n[*] Processing with {args.workers} workers (context_seconds={args.context_seconds}) ..."
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                process_sample,
                jp,
                args.opp_plus_root,
                args.output_dir,
                args.context_seconds,
                args.include_no_png,
                classes_filter,
                runs_filter,
                null_filter,
                args.min_confidence,
                class_zero_id=0,
            )
            for jp in json_paths
        ]
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            unit="sample",
            desc="Pairing",
            dynamic_ncols=True,
        )
        for fut in pbar:
            res = fut.result()
            results.append(res)
            pbar.set_postfix_str(
                f"ok={sum((1 for r in results if r['status'] == 'ok'))} skip={sum((1 for r in results if r['status'] == 'skip'))} err={sum((1 for r in results if r['status'] == 'error'))}"
            )
        pbar.close()
    ok = [r for r in results if r["status"] == "ok"]
    skip = [r for r in results if r["status"] == "skip"]
    error = [r for r in results if r["status"] == "error"]
    with_png = sum((1 for r in ok if r.get("had_png")))
    without_png = len(ok) - with_png
    print(f"\n{'=' * 60}")
    print(
        f"  Paired successfully:    {len(ok)} (with PNG: {with_png}, clip-only: {without_png})"
    )
    print(f"  Skipped (filtered):     {len(skip)}")
    print(f"  Errors:                 {len(error)}")
    print(f"  Output:                 {args.output_dir}")
    print(f"{'=' * 60}")
    if skip:
        from collections import Counter

        reason_counts = Counter((r["reason"].split("(")[0].strip() for r in skip))
        print("\n  Skip reasons:")
        for reason, n in reason_counts.most_common():
            print(f"    {n:6d}  {reason}")
    if error:
        print("\n  First few errors:")
        for r in error[:5]:
            print(f"    {os.path.basename(r['json'])}: {r['reason']}")
        if len(error) > 5:
            print(f"    ... and {len(error) - 5} more")
    if ok:
        import csv

        index_path = os.path.join(args.output_dir, "_pairing_index.csv")
        with open(index_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "folder",
                    "sample_idx",
                    "class_id",
                    "activity",
                    "had_png",
                    "source_json",
                ]
            )
            for r in ok:
                meta_path = os.path.join(r["folder"], "metadata.json")
                try:
                    with open(meta_path) as mf:
                        m = json.load(mf)
                    w.writerow(
                        [
                            os.path.basename(r["folder"]),
                            m["sample_idx"],
                            m["true_class_id"],
                            m["true_class_name"],
                            r["had_png"],
                            m["source_json"],
                        ]
                    )
                except Exception:
                    pass
        print(f"\n  Index CSV: {index_path}")


if __name__ == "__main__":
    main()
