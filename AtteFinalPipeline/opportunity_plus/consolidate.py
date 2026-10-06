import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

FOLDER_RX = re.compile("^reasonings_(\\d{6})_(.+)$")


def scan_reasoning_dir(root):
    if not os.path.isdir(root):
        sys.exit(f"[FATAL] Directory not found: {root}")
    found = {}
    for json_path in glob.glob(os.path.join(root, "reasonings_*", "reasoning.json")):
        folder_path = os.path.dirname(json_path)
        folder_name = os.path.basename(folder_path)
        m = FOLDER_RX.match(folder_name)
        if not m:
            print(f"  [warn] unexpected folder name, skipping: {folder_name}")
            continue
        wid = int(m.group(1))
        token = m.group(2)
        found[wid] = (os.path.abspath(folder_path), token)
    n_empty = 0
    for f in glob.glob(os.path.join(root, "reasonings_*")):
        if os.path.isdir(f) and (not os.path.isfile(os.path.join(f, "reasoning.json"))):
            n_empty += 1
    return (OrderedDict(sorted(found.items(), key=lambda kv: kv[0])), n_empty)


def load_predictions_csv_map(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        print(
            f"  [WARN] predictions.csv not found at {path}; on-demand extraction will be impossible."
        )
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            wid = int(row["window_idx"])
            try:
                out[wid] = {
                    "run": row.get("run", "") or "unknown_run",
                    "video_file": row.get("video_file", ""),
                    "video_fps": float(row.get("video_fps") or 0.0),
                    "start_time_s": float(row.get("start_time_s") or 0.0),
                    "end_time_s": float(row.get("end_time_s") or 0.0),
                    "start_frame": int(row.get("start_frame") or 0),
                    "end_frame": int(row.get("end_frame") or 0),
                    "pred_label": int(row["pred_label"]),
                    "true_label": int(row["true_label"]),
                }
            except (KeyError, ValueError) as e:
                print(f"  [warn] predictions row for wid {wid} unusable: {e!r}")
    return out


def _materialise(src, dst, mode, overwrite):
    if os.path.exists(dst) or os.path.islink(dst):
        if not overwrite:
            return "skipped_exists"
        if os.path.islink(dst) or os.path.isfile(dst):
            os.unlink(dst)
        else:
            shutil.rmtree(dst)
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copyfile(src, dst)
            return "copy_fallback"
    shutil.copyfile(src, dst)
    return "copy"


def build_ffmpeg_job(wid, pred_row, opp_plus_root, out_path, pad_seconds):
    video_rel = pred_row.get("video_file", "")
    if not video_rel:
        return (None, "no video_file in predictions.csv")
    video_abs = os.path.join(opp_plus_root, video_rel)
    if not os.path.isfile(video_abs):
        return (None, f"video not found: {video_abs}")
    fps = pred_row["video_fps"]
    if fps <= 0:
        return (None, "video_fps <= 0")
    t_start = max(0.0, pred_row["start_time_s"] - pad_seconds)
    t_end = pred_row["end_time_s"] + pad_seconds
    if t_end <= t_start:
        t_start = max(0.0, pred_row["start_frame"] / fps - pad_seconds)
        t_end = pred_row["end_frame"] / fps + pad_seconds
    duration = max(t_end - t_start, 1.0 / fps)
    return (
        {
            "window_idx": wid,
            "video": os.path.abspath(video_abs),
            "t_start": t_start,
            "duration": duration,
            "out_path": out_path,
        },
        None,
    )


def run_ffmpeg(job, ffmpeg_bin, copy_codec):
    os.makedirs(os.path.dirname(job["out_path"]), exist_ok=True)
    if copy_codec:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{job['t_start']:.3f}",
            "-i",
            job["video"],
            "-t",
            f"{job['duration']:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-loglevel",
            "error",
            job["out_path"],
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            job["video"],
            "-ss",
            f"{job['t_start']:.3f}",
            "-t",
            f"{job['duration']:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            "-loglevel",
            "error",
            job["out_path"],
        ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=60
        )
        if proc.returncode != 0:
            if os.path.exists(job["out_path"]):
                try:
                    os.remove(job["out_path"])
                except OSError:
                    pass
            return (False, (proc.stderr or proc.stdout or "ffmpeg failed").strip())
        if not os.path.isfile(job["out_path"]) or os.path.getsize(job["out_path"]) == 0:
            return (False, "ffmpeg produced empty output")
        return (True, "")
    except subprocess.TimeoutExpired:
        return (False, "ffmpeg timed out (60s)")
    except Exception as e:
        return (False, repr(e))


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument(
        "--trace_dir",
        required=True,
        help="Output root from run_trace_reasoning_opp_plus.py.",
    )
    p.add_argument(
        "--opentslm_dir",
        required=True,
        help="Output root from run_opentslm_opp_plus.py.",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Where to write consolidated reasonings_*/ folders.",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="predictions.csv from infer_opportunity_plus.py. Required for on-demand clip extraction.",
    )
    p.add_argument(
        "--opp_plus_root",
        default=None,
        help="Opportunity++ root (parent of data/run-name dirs). Required for on-demand clip extraction.",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--copy", action="store_const", const="copy", dest="mode")
    grp.add_argument("--symlink", action="store_const", const="symlink", dest="mode")
    p.set_defaults(mode="hardlink")
    p.add_argument(
        "--overwrite", action="store_true", help="Replace files already in out_dir."
    )
    p.add_argument(
        "--require_clip",
        action="store_true",
        help="Delete windows that end up with no clip from out_dir.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Process at most this many common windows (-1=all).",
    )
    p.add_argument(
        "--csv", default=None, help="Optional CSV listing every consolidated window."
    )
    p.add_argument(
        "--no_extract",
        action="store_true",
        help="Disable on-demand ffmpeg extraction entirely. Missing clips are just reported.",
    )
    p.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary.")
    p.add_argument(
        "--ffmpeg_workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Parallel ffmpeg processes for on-demand extraction.",
    )
    p.add_argument(
        "--pad_seconds",
        type=float,
        default=0.0,
        help="Widen extracted clips by this many seconds on each side. Useful since the raw 24-sample window is ~0.8s.",
    )
    p.add_argument(
        "--copy_codec",
        action="store_true",
        help="Use ffmpeg -c copy (stream copy). Faster but seeks snap to nearest keyframe — clips may start early.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Consolidating common reasoning windows")
    print("=" * 70)
    print(f"  TRACE dir       : {args.trace_dir}")
    print(f"  OpenTSLM dir    : {args.opentslm_dir}")
    print(f"  Out dir         : {args.out_dir}")
    print(f"  Predictions CSV : {args.predictions or '(none)'}")
    print(f"  OPP++ root      : {args.opp_plus_root or '(none)'}")
    print(f"  Link mode       : {args.mode}")
    print(f"  Extract missing : {not args.no_extract}")
    print(f"  pad_seconds     : {args.pad_seconds}")
    print(f"  ffmpeg_workers  : {args.ffmpeg_workers}")
    print(f"  Require clip    : {args.require_clip}")
    print(f"  Limit           : {args.limit}")
    print("=" * 70)
    extraction_enabled = not args.no_extract
    if extraction_enabled:
        if not args.predictions or not args.opp_plus_root:
            print(
                "\n[info] --predictions and --opp_plus_root not both given; on-demand extraction is disabled."
            )
            extraction_enabled = False
        elif shutil.which(args.ffmpeg) is None:
            print(
                f"\n[WARN] ffmpeg binary not found at {args.ffmpeg!r}; on-demand extraction disabled."
            )
            extraction_enabled = False
    os.makedirs(args.out_dir, exist_ok=True)
    print("\n[1/4] Scanning input directories")
    trace_map, trace_empty = scan_reasoning_dir(args.trace_dir)
    print(f"  TRACE   : {len(trace_map)} processed  ({trace_empty} empty folders)")
    ots_map, ots_empty = scan_reasoning_dir(args.opentslm_dir)
    print(f"  OpenTSLM: {len(ots_map)} processed  ({ots_empty} empty folders)")
    common = sorted(set(trace_map.keys()) & set(ots_map.keys()))
    print(f"  Common  : {len(common)}")
    if args.limit > 0:
        common = common[: args.limit]
        print(f"  --limit applied: processing {len(common)} windows.")
    if not common:
        sys.exit("[!] No common windows. Nothing to do.")
    pred_map = {}
    if extraction_enabled:
        print(f"\n[2/4] Loading {args.predictions}")
        pred_map = load_predictions_csv_map(args.predictions)
        print(f"  {len(pred_map)} prediction rows loaded")
    else:
        print("\n[2/4] Skipping predictions.csv (on-demand extraction disabled)")
    print(f"\n[3/4] Consolidating {len(common)} windows")
    t0 = time.time()
    n_token_mismatch = 0
    n_clip_from_trace = 0
    n_clip_from_ots = 0
    n_clip_already_in_out = 0
    extraction_jobs = []
    extraction_skips = []
    modes_used = {
        "hardlink": 0,
        "copy": 0,
        "copy_fallback": 0,
        "symlink": 0,
        "skipped_exists": 0,
    }
    pending_for_csv = {}
    progress_step = max(100, len(common) // 50)
    for i, wid in enumerate(common, 1):
        trace_folder, trace_token = trace_map[wid]
        ots_folder, ots_token = ots_map[wid]
        if trace_token != ots_token:
            n_token_mismatch += 1
            print(
                f"  [warn] wid {wid:06d}: token mismatch trace={trace_token!r} opentslm={ots_token!r}; using TRACE."
            )
        out_folder = os.path.join(args.out_dir, f"reasonings_{wid:06d}_{trace_token}")
        os.makedirs(out_folder, exist_ok=True)
        trace_json = os.path.join(trace_folder, "reasoning.json")
        ots_json = os.path.join(ots_folder, "reasoning.json")
        m1 = _materialise(
            trace_json,
            os.path.join(out_folder, "trace_reasoning.json"),
            args.mode,
            args.overwrite,
        )
        modes_used[m1] = modes_used.get(m1, 0) + 1
        m2 = _materialise(
            ots_json,
            os.path.join(out_folder, "opentslm_reasoning.json"),
            args.mode,
            args.overwrite,
        )
        modes_used[m2] = modes_used.get(m2, 0) + 1
        out_clip = os.path.join(out_folder, "clip.mp4")
        trace_clip = os.path.join(trace_folder, "clip.mp4")
        ots_clip = os.path.join(ots_folder, "clip.mp4")
        clip_origin = "missing"
        if os.path.isfile(out_clip) and (not args.overwrite):
            clip_origin = "preexisting_out"
            n_clip_already_in_out += 1
        elif os.path.isfile(trace_clip):
            m3 = _materialise(trace_clip, out_clip, args.mode, args.overwrite)
            modes_used[m3] = modes_used.get(m3, 0) + 1
            clip_origin = "trace"
            n_clip_from_trace += 1
        elif os.path.isfile(ots_clip):
            m3 = _materialise(ots_clip, out_clip, args.mode, args.overwrite)
            modes_used[m3] = modes_used.get(m3, 0) + 1
            clip_origin = "opentslm"
            n_clip_from_ots += 1
        elif extraction_enabled:
            pred_row = pred_map.get(wid)
            if pred_row is None:
                extraction_skips.append((wid, "no row in predictions.csv"))
                clip_origin = "missing"
            else:
                job, err = build_ffmpeg_job(
                    wid, pred_row, args.opp_plus_root, out_clip, args.pad_seconds
                )
                if job is None:
                    extraction_skips.append((wid, err))
                    clip_origin = "missing"
                else:
                    extraction_jobs.append((wid, job))
                    clip_origin = "pending_extract"
        true_act = har_act = trace_pred = ots_pred = ""
        try:
            with open(trace_json) as fh:
                td = json.load(fh)
            har_act = td.get("har_predicted_activity", "")
            true_act = td.get("true_activity", "")
            trace_pred = td.get("reasoning_predicted_activity", "")
        except Exception as e:
            print(f"  [warn] wid {wid:06d}: TRACE JSON unreadable: {e!r}")
        try:
            with open(ots_json) as fh:
                od = json.load(fh)
            ots_pred = od.get("opentslm_predicted_activity", "")
            if not har_act:
                har_act = od.get("har_predicted_activity", "")
            if not true_act:
                true_act = od.get("true_activity", "")
        except Exception as e:
            print(f"  [warn] wid {wid:06d}: OpenTSLM JSON unreadable: {e!r}")
        pending_for_csv[wid] = {
            "window_idx": f"{wid:06d}",
            "folder": os.path.basename(out_folder),
            "har_activity": har_act,
            "true_activity": true_act,
            "trace_predicted": trace_pred,
            "opentslm_predicted": ots_pred,
            "clip_origin": clip_origin,
            "token_mismatch": "yes" if trace_token != ots_token else "no",
        }
        if i % progress_step == 0 or i == len(common):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(common) - i) / rate if rate > 0 else 0.0
            print(
                f"  [{i}/{len(common)}]  {rate:.0f} win/s  ETA {eta:.1f}s  pending_extract={len(extraction_jobs)}",
                flush=True,
            )
    n_extracted = 0
    n_extract_failed = 0
    extract_errors = []
    if extraction_jobs:
        print(
            f"\n[4a/4] Extracting {len(extraction_jobs)} missing clips with ffmpeg ({args.ffmpeg_workers} workers)"
        )
        t1 = time.time()
        step = max(50, len(extraction_jobs) // 40)
        with ThreadPoolExecutor(max_workers=args.ffmpeg_workers) as ex:
            futs = {
                ex.submit(run_ffmpeg, j, args.ffmpeg, args.copy_codec): wid
                for wid, j in extraction_jobs
            }
            done = 0
            for fut in as_completed(futs):
                wid = futs[fut]
                ok, msg = fut.result()
                done += 1
                if ok:
                    n_extracted += 1
                    pending_for_csv[wid]["clip_origin"] = "extracted"
                else:
                    n_extract_failed += 1
                    extract_errors.append((wid, msg))
                    pending_for_csv[wid]["clip_origin"] = "extract_failed"
                if done % step == 0 or done == len(extraction_jobs):
                    elapsed = time.time() - t1
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(extraction_jobs) - done) / rate if rate > 0 else 0.0
                    print(
                        f"  [{done}/{len(extraction_jobs)}]  {rate:.1f} clips/s  ok={n_extracted}  err={n_extract_failed}  ETA {eta / 60:.1f} min",
                        flush=True,
                    )
    else:
        print("\n[4a/4] No on-demand extraction needed.")
    if args.require_clip:
        removed = []
        for wid in list(pending_for_csv.keys()):
            row = pending_for_csv[wid]
            if row["clip_origin"] in ("missing", "extract_failed", "pending_extract"):
                folder = os.path.join(args.out_dir, row["folder"])
                shutil.rmtree(folder, ignore_errors=True)
                pending_for_csv.pop(wid)
                removed.append(wid)
        if removed:
            print(
                f"\n  --require_clip: removed {len(removed)} windows that ended up with no clip."
            )
    csv_rows = [pending_for_csv[wid] for wid in sorted(pending_for_csv.keys())]
    print("\n[4b/4] Writing summary")
    n_clip_missing_final = sum(
        (
            1
            for r in csv_rows
            if r["clip_origin"] in ("missing", "extract_failed", "pending_extract")
        )
    )
    summary = {
        "timestamp": datetime.now().isoformat(),
        "trace_dir": os.path.abspath(args.trace_dir),
        "opentslm_dir": os.path.abspath(args.opentslm_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "predictions_csv": os.path.abspath(args.predictions)
        if args.predictions
        else None,
        "opp_plus_root": os.path.abspath(args.opp_plus_root)
        if args.opp_plus_root
        else None,
        "mode": args.mode,
        "extraction_enabled": extraction_enabled,
        "trace_processed": len(trace_map),
        "trace_empty_folders": trace_empty,
        "opentslm_processed": len(ots_map),
        "opentslm_empty_folders": ots_empty,
        "common_windows": len(common),
        "windows_written": len(csv_rows),
        "token_mismatches": n_token_mismatch,
        "clip_from_trace": n_clip_from_trace,
        "clip_from_opentslm": n_clip_from_ots,
        "clip_preexisting": n_clip_already_in_out,
        "clip_extracted": n_extracted,
        "clip_extract_failed": n_extract_failed,
        "clip_extract_skipped": len(extraction_skips),
        "clip_missing_final": n_clip_missing_final,
        "extraction_skips_sample": [
            {"wid": w, "reason": r} for w, r in extraction_skips[:50]
        ],
        "extraction_errors_sample": [
            {"wid": w, "msg": m.splitlines()[0] if m else ""}
            for w, m in extract_errors[:20]
        ],
        "operation_counts": modes_used,
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        cols = [
            "window_idx",
            "folder",
            "har_activity",
            "true_activity",
            "trace_predicted",
            "opentslm_predicted",
            "clip_origin",
            "token_mismatch",
        ]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  CSV: {args.csv}  ({len(csv_rows)} rows)")
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Out dir            : {args.out_dir}")
    print(f"  Windows written    : {len(csv_rows)}")
    print(f"  Token mismatches   : {n_token_mismatch}")
    print(f"  Clip from TRACE    : {n_clip_from_trace}")
    print(f"  Clip from OpenTSLM : {n_clip_from_ots}")
    print(f"  Clip preexisting   : {n_clip_already_in_out}")
    print(f"  Clip extracted     : {n_extracted}")
    print(f"  Clip extract fails : {n_extract_failed}")
    print(f"  Clip still missing : {n_clip_missing_final}")
    print(f"  Operation counts   : {modes_used}")
    print(f"  Summary            : {summary_path}")
    print(f"  Elapsed            : {summary['elapsed_s']}s")
    print("=" * 70)
    if n_extract_failed and extract_errors:
        print("\nFirst extraction errors:")
        for wid, msg in extract_errors[:10]:
            first_line = msg.splitlines()[0] if msg else ""
            print(f"  wid={wid:06d}: {first_line}")
        if len(extract_errors) > 10:
            print(f"  ... and {len(extract_errors) - 10} more.")


if __name__ == "__main__":
    main()
