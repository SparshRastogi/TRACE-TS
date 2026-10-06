import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime

FOLDER_RX = re.compile("^reasonings_(\\d{6})_(.+)$")


def scan_reasoning_dir(root):
    if not os.path.isdir(root):
        sys.exit(f"[FATAL] Directory not found: {root}")
    found = {}
    pattern = os.path.join(root, "reasonings_*", "reasoning.json")
    for json_path in glob.glob(pattern):
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
    ordered = OrderedDict(sorted(found.items(), key=lambda kv: kv[0]))
    return (ordered, n_empty)


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
        help="Where to write the consolidated reasonings_*/ folders.",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--copy",
        action="store_const",
        const="copy",
        dest="mode",
        help="Copy files (uses real disk space). Default is hardlink.",
    )
    grp.add_argument(
        "--symlink",
        action="store_const",
        const="symlink",
        dest="mode",
        help="Write symlinks (cheapest; breaks if sources move).",
    )
    p.set_defaults(mode="hardlink")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files already in out_dir. By default they're left alone (handy for resuming).",
    )
    p.add_argument(
        "--require_clip",
        action="store_true",
        help="Skip windows where neither side has a clip.mp4.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Process at most this many common windows (-1 = all). Applied to the sorted intersection, useful for a quick smoke test.",
    )
    p.add_argument(
        "--csv", default=None, help="Optional CSV listing every consolidated window."
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Consolidating common reasoning windows")
    print("=" * 70)
    print(f"  TRACE dir    : {args.trace_dir}")
    print(f"  OpenTSLM dir : {args.opentslm_dir}")
    print(f"  Out dir      : {args.out_dir}")
    print(f"  Mode         : {args.mode}")
    print(f"  Overwrite    : {args.overwrite}")
    print(f"  Require clip : {args.require_clip}")
    print(f"  Limit        : {args.limit}")
    print("=" * 70)
    os.makedirs(args.out_dir, exist_ok=True)
    print("\n[1/3] Scanning input directories")
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
    print(f"\n[2/3] Consolidating {len(common)} windows")
    t0 = time.time()
    n_written = 0
    n_skipped = 0
    n_token_mismatch = 0
    n_clip_missing = 0
    n_clip_from_trace = 0
    n_clip_from_ots = 0
    modes_used = {
        "hardlink": 0,
        "copy": 0,
        "copy_fallback": 0,
        "symlink": 0,
        "skipped_exists": 0,
    }
    csv_rows = []
    progress_step = max(100, len(common) // 50)
    for i, wid in enumerate(common, 1):
        trace_folder, trace_token = trace_map[wid]
        ots_folder, ots_token = ots_map[wid]
        if trace_token != ots_token:
            n_token_mismatch += 1
            print(
                f"  [warn] wid {wid:06d}: token mismatch trace={trace_token!r} opentslm={ots_token!r}; using TRACE."
            )
        out_token = trace_token
        out_folder = os.path.join(args.out_dir, f"reasonings_{wid:06d}_{out_token}")
        trace_json = os.path.join(trace_folder, "reasoning.json")
        ots_json = os.path.join(ots_folder, "reasoning.json")
        trace_clip = os.path.join(trace_folder, "clip.mp4")
        ots_clip = os.path.join(ots_folder, "clip.mp4")
        if os.path.isfile(trace_clip):
            clip_src, clip_origin = (trace_clip, "trace")
        elif os.path.isfile(ots_clip):
            clip_src, clip_origin = (ots_clip, "opentslm")
        else:
            clip_src, clip_origin = (None, "missing")
        if args.require_clip and clip_src is None:
            n_skipped += 1
            continue
        os.makedirs(out_folder, exist_ok=True)
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
        if clip_src:
            m3 = _materialise(
                clip_src,
                os.path.join(out_folder, "clip.mp4"),
                args.mode,
                args.overwrite,
            )
            modes_used[m3] = modes_used.get(m3, 0) + 1
            if clip_origin == "trace":
                n_clip_from_trace += 1
            else:
                n_clip_from_ots += 1
        else:
            n_clip_missing += 1
        n_written += 1
        true_act = har_act = trace_pred = ots_pred = ""
        try:
            with open(trace_json) as fh:
                td = json.load(fh)
            har_act = td.get("har_predicted_activity", "")
            true_act = td.get("true_activity", "")
            trace_pred = td.get("reasoning_predicted_activity", "")
        except Exception as e:
            print(f"  [warn] wid {wid:06d}: couldn't read TRACE JSON: {e!r}")
        try:
            with open(ots_json) as fh:
                od = json.load(fh)
            ots_pred = od.get("opentslm_predicted_activity", "")
            if not har_act:
                har_act = od.get("har_predicted_activity", "")
            if not true_act:
                true_act = od.get("true_activity", "")
        except Exception as e:
            print(f"  [warn] wid {wid:06d}: couldn't read OpenTSLM JSON: {e!r}")
        csv_rows.append(
            {
                "window_idx": f"{wid:06d}",
                "folder": os.path.basename(out_folder),
                "har_activity": har_act,
                "true_activity": true_act,
                "trace_predicted": trace_pred,
                "opentslm_predicted": ots_pred,
                "clip_origin": clip_origin,
                "token_mismatch": "yes" if trace_token != ots_token else "no",
            }
        )
        if i % progress_step == 0 or i == len(common):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(common) - i) / rate if rate > 0 else 0.0
            print(
                f"  [{i}/{len(common)}]  {rate:.0f} win/s  ETA {eta:.1f}s  written={n_written}  skipped={n_skipped}",
                flush=True,
            )
    print("\n[3/3] Writing summary")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "trace_dir": os.path.abspath(args.trace_dir),
        "opentslm_dir": os.path.abspath(args.opentslm_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "mode": args.mode,
        "overwrite": args.overwrite,
        "require_clip": args.require_clip,
        "trace_processed": len(trace_map),
        "trace_empty_folders": trace_empty,
        "opentslm_processed": len(ots_map),
        "opentslm_empty_folders": ots_empty,
        "common_windows": len(common),
        "windows_written": n_written,
        "windows_skipped": n_skipped,
        "token_mismatches": n_token_mismatch,
        "clip_from_trace": n_clip_from_trace,
        "clip_from_opentslm": n_clip_from_ots,
        "clip_missing": n_clip_missing,
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
    print(f"  Windows written    : {n_written}")
    print(f"  Windows skipped    : {n_skipped}")
    print(f"  Token mismatches   : {n_token_mismatch}")
    print(f"  Clip from TRACE    : {n_clip_from_trace}")
    print(f"  Clip from OpenTSLM : {n_clip_from_ots}")
    print(f"  Clip missing       : {n_clip_missing}")
    print(f"  Operation counts   : {modes_used}")
    print(f"  Summary            : {summary_path}")
    print(f"  Elapsed            : {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
