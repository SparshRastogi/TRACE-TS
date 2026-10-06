import os
import re
import csv
import json
import argparse
import subprocess
import numpy as np
import scipy.io as sio

MAT_CHANNEL_RAW_COLS = (
    list(range(37, 48))
    + list(range(50, 59))
    + list(range(63, 72))
    + list(range(76, 85))
    + list(range(89, 98))
    + list(range(102, 134))
)
assert len(MAT_CHANNEL_RAW_COLS) == 79, (
    f"Expected 79 channels, got {len(MAT_CHANNEL_RAW_COLS)}"
)
EXPECTED_CHANNELS = 79
WINDOW = 24
STRIDE_TEST = 1
SENSOR_HZ = 30
LABEL_COL = 249
OPP_GESTURE_VALUES = {
    406516,
    406517,
    404516,
    404517,
    406520,
    404520,
    406505,
    404505,
    406519,
    404519,
    406511,
    404511,
    406508,
    404508,
    408512,
    407521,
    405506,
}
OPP_GESTURE_MAP = {
    0: 0,
    404505: 1,
    404508: 2,
    404511: 3,
    404516: 4,
    404517: 5,
    404519: 6,
    404520: 7,
    405506: 8,
    406505: 9,
    406508: 10,
    406511: 11,
    406516: 12,
    406517: 13,
    406519: 14,
    406520: 15,
    407521: 16,
    408512: 17,
}
OPP_CLASS_NAMES = [
    "Null",
    "Close Dishwasher",
    "Close Drawer 3",
    "Close Drawer 2",
    "Close Door 1",
    "Close Door 2",
    "Close Drawer 1",
    "Close Fridge",
    "Toggle Switch",
    "Open Dishwasher",
    "Open Drawer 3",
    "Open Drawer 2",
    "Open Door 1",
    "Open Door 2",
    "Open Drawer 1",
    "Open Fridge",
    "Drink from Cup",
    "Clean Table",
]
ALL_RUNS = [
    f"S{s}-{r}"
    for s in (1, 2, 3, 4)
    for r in ("ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill")
]
DEFAULT_RUNS = ["S2-ADL4", "S2-ADL5", "S3-ADL4", "S3-ADL5"]
SRT_SUFFIX_ANCHOR = "_side.ml_both_arms.srt"


def impute_nans(X):
    for col in range(X.shape[1]):
        nans = np.isnan(X[:, col])
        if not nans.any():
            continue
        not_nan = np.where(~nans)[0]
        if len(not_nan) == 0:
            X[:, col] = 0.0
        else:
            X[:, col] = np.interp(
                np.arange(X.shape[0]), not_nan, X[not_nan, col]
            ).astype(np.float32)
    return X


def parse_sensor_file(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    n_rows = len(rows)
    n_cols_max = max((len(r) for r in rows))
    if n_cols_max != 250:
        raise RuntimeError(
            f"[FATAL] {filepath}: expected 250 columns, found {n_cols_max}. Opportunity++ sensor file layout doesn't match the original."
        )
    data = np.full((n_rows, n_cols_max), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try:
                data[i, j] = float(v)
            except ValueError:
                pass
    label_raw = data[:, LABEL_COL]
    y = np.array(
        [
            OPP_GESTURE_MAP.get(0 if np.isnan(v) else int(round(v)), 0)
            for v in label_raw
        ],
        dtype=np.int64,
    )
    X = data[:, MAT_CHANNEL_RAW_COLS].astype(np.float32)
    X = impute_nans(X)
    X = (X / 1000.0).astype(np.float32)
    return (X, y)


def compute_second_pass_stats(opp_mat_path):
    print(f"[*] Recovering second-pass z-score stats from {opp_mat_path}")
    m = sio.loadmat(opp_mat_path)
    x_train = m["trainingData"].astype(np.float32).T
    if x_train.shape[1] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"opportunity.mat has {x_train.shape[1]} channels, expected {EXPECTED_CHANNELS}."
        )
    mean = np.mean(x_train, axis=0).astype(np.float32)
    std = np.std(x_train, axis=0).astype(np.float32)
    std[std == 0] = 1.0
    print(f"  train shape: {x_train.shape}")
    print(f"  mean[:5] = {mean[:5].tolist()}")
    print(f"  std [:5] = {std[:5].tolist()}")
    return (mean, std)


def locate_run_files(opp_plus_root, runs):
    candidates_data_dir = [os.path.join(opp_plus_root, "data"), opp_plus_root]
    data_dir = None
    for cand in candidates_data_dir:
        if os.path.isdir(os.path.join(cand, runs[0])):
            data_dir = cand
            break
    if data_dir is None:
        raise FileNotFoundError(
            f"Could not find Opportunity++ run folders under {opp_plus_root}. Expected {opp_plus_root}/data/{runs[0]}/... or {opp_plus_root}/{runs[0]}/..."
        )
    found = []
    for run in runs:
        rdir = os.path.join(data_dir, run)
        if not os.path.isdir(rdir):
            raise FileNotFoundError(f"Run folder missing: {rdir}")
        sensor = os.path.join(rdir, f"{run}_sensors_data.txt")
        video = os.path.join(rdir, f"{run}_side.avi")
        srt = os.path.join(rdir, f"{run}{SRT_SUFFIX_ANCHOR}")
        if not os.path.isfile(sensor):
            raise FileNotFoundError(
                f"[FATAL] Missing sensor file: {sensor}\n        Expected name '<RUN>_sensors_data.txt'."
            )
        if not os.path.isfile(video):
            print(
                f"  [WARN] No video file at {video} — clips will be skipped for {run}."
            )
            video = None
        if not os.path.isfile(srt):
            print(
                f"  [WARN] No ml_both_arms.srt at {srt} — using naive timing for {run}."
            )
            srt = None
        found.append({"run": run, "sensor": sensor, "video": video, "srt": srt})
    return found


def probe_video_fps(video_path):
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                video_path,
            ],
            stderr=subprocess.DEVNULL,
        )
        info = json.loads(out)
        streams = info.get("streams", [])
        if not streams:
            raise RuntimeError("no video stream in ffprobe output")
        rate = streams[0].get("r_frame_rate") or streams[0].get("avg_frame_rate")
        if not rate or "/" not in rate:
            raise RuntimeError(f"unparseable rate: {rate!r}")
        num, den = rate.split("/")
        den = float(den)
        if den == 0:
            raise RuntimeError("zero denominator in frame rate")
        return float(num) / den
    except Exception as e:
        print(f"  [WARN] ffprobe failed on {video_path}: {e}; defaulting to 30 fps.")
        return 30.0


_SRT_TIME_RE = re.compile(
    "(\\d{2}):(\\d{2}):(\\d{2})[,.](\\d{3})\\s*-->\\s*(\\d{2}):(\\d{2}):(\\d{2})[,.](\\d{3})"
)


def _hms_to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path):
    if path is None or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        block_text = []
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.strip() == "":
                if block_text:
                    out.extend(_parse_srt_block(block_text))
                    block_text = []
            else:
                block_text.append(line)
        if block_text:
            out.extend(_parse_srt_block(block_text))
    return out


def _parse_srt_block(lines):
    time_line_idx = None
    for i, l in enumerate(lines):
        if _SRT_TIME_RE.search(l):
            time_line_idx = i
            break
    if time_line_idx is None:
        return []
    m = _SRT_TIME_RE.search(lines[time_line_idx])
    start_s = _hms_to_sec(*m.group(1, 2, 3, 4))
    end_s = _hms_to_sec(*m.group(5, 6, 7, 8))
    text = " ".join(lines[time_line_idx + 1 :]).strip()
    return [(start_s, end_s, text)]


def derive_video_anchor(srt_entries, dat_labels_30hz):
    if not srt_entries:
        return (0, 0.0, False)
    first_srt_nonnull = None
    for st, et, lbl in srt_entries:
        if lbl and lbl.strip().lower() not in ("null", "none", ""):
            first_srt_nonnull = (st, et, lbl)
            break
    if first_srt_nonnull is None:
        return (0, 0.0, False)
    srt_anchor_time = first_srt_nonnull[0]
    nz = np.where(dat_labels_30hz != 0)[0]
    if nz.size == 0:
        return (0, 0.0, False)
    dat_anchor_sample = int(nz[0])
    return (dat_anchor_sample, float(srt_anchor_time), True)


def sample_to_frame(sample_idx, anchor_sample, anchor_video_s, video_fps):
    video_t = (sample_idx - anchor_sample) / float(SENSOR_HZ) + anchor_video_s
    video_t = max(0.0, video_t)
    return int(round(video_t * video_fps))


def sample_to_time(sample_idx, anchor_sample, anchor_video_s):
    video_t = (sample_idx - anchor_sample) / float(SENSOR_HZ) + anchor_video_s
    return max(0.0, video_t)


def check_layout(opp_plus_root, runs):
    print("=" * 70)
    print(f"[CHECK MODE] Validating {opp_plus_root}")
    print(f"             Runs: {runs}")
    print("=" * 70)
    files = locate_run_files(opp_plus_root, runs)
    for f in files:
        print(f"\n── {f['run']} ──")
        print(
            f"  sensor: {f['sensor']}  ({os.path.getsize(f['sensor']) / 1000000.0:.1f} MB)"
        )
        print(f"  video : {f['video'] or '—'}")
        print(f"  srt   : {f['srt'] or '—'}")
        with open(f["sensor"]) as fh:
            first = fh.readline().strip().split()
        print(f"  sensor cols (first row): {len(first)}")
        if f["video"]:
            fps = probe_video_fps(f["video"])
            print(f"  video fps (ffprobe): {fps:.3f}")
        if f["srt"]:
            entries = parse_srt(f["srt"])
            non_null = [e for e in entries if e[2] and e[2].lower() != "null"]
            print(f"  SRT entries: {len(entries)} ({len(non_null)} non-null)")
            if non_null:
                print(
                    f"  first non-null: t={non_null[0][0]:.2f}s  label={non_null[0][2]!r}"
                )
    print("\n[CHECK MODE] Layout looks usable. Re-run without --check.")


def build(opp_plus_root, opp_mat, output_mat, output_csv, runs, skip_second_pass=False):
    print("=" * 70)
    print("[Opportunity++ inference prep — v4 (corrected gesture map)]")
    print(f"  opp_plus_root = {opp_plus_root}")
    print(f"  opp_mat       = {opp_mat}")
    print(f"  output_mat    = {output_mat}")
    print(f"  output_csv    = {output_csv}")
    print(f"  runs          = {runs}")
    if skip_second_pass:
        print(f"  [!] skip_second_pass=True — output will only be /1000 normalised.")
    print("=" * 70)
    files = locate_run_files(opp_plus_root, runs)
    print("\n[1/6] Files discovered:")
    for f in files:
        v = os.path.basename(f["video"]) if f["video"] else "—"
        s = os.path.basename(f["srt"]) if f["srt"] else "—"
        print(
            f"  {f['run']:10s}  sensor={os.path.basename(f['sensor']):<28s}  video={v:<22s}  srt={s}"
        )
    print("\n[2/6] Probing video frame rates with ffprobe")
    for f in files:
        if f["video"]:
            f["fps"] = probe_video_fps(f["video"])
            print(f"  {f['run']}: {f['fps']:.3f} fps")
        else:
            f["fps"] = 30.0
            print(f"  {f['run']}: no video; fps defaulted to 30")
    if not skip_second_pass:
        print("\n[3/6] Recovering second-pass z-score stats from opportunity.mat")
        second_mean, second_std = compute_second_pass_stats(opp_mat)
    else:
        print("\n[3/6] Skipping second-pass stats (--skip_second_pass set)")
        second_mean = None
        second_std = None
    print("\n[4/6] Parsing each run with the verified channel mapping")
    all_X = []
    all_y = []
    per_run_info = []
    for f in files:
        print(f"\n  ── {f['run']} ──")
        X, y = parse_sensor_file(f["sensor"])
        print(f"    parsed: X={X.shape}, y={y.shape}, non-null = {int((y != 0).sum())}")
        if X.shape[1] != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"[FATAL] {f['run']}: parsed {X.shape[1]} channels, expected {EXPECTED_CHANNELS}."
            )
        all_X.append(X)
        all_y.append(y)
        per_run_info.append(
            {
                "run": f["run"],
                "N": X.shape[0],
                "fps": f["fps"],
                "srt_path": f["srt"],
                "video_path": f["video"],
                "sensor_path": f["sensor"],
                "raw_labels": y,
            }
        )
    if second_mean is not None:
        print("\n[5/6] Applying second-pass z-score to OPP++ data")
        for i in range(len(all_X)):
            all_X[i] = ((all_X[i] - second_mean) / second_std).astype(np.float32)
        Xcat = np.concatenate(all_X, axis=0)
        print(
            f"  After both passes — global mean={Xcat.mean():.4f}, std={Xcat.std():.4f}  (should be roughly 0, 1)"
        )
        del Xcat
    else:
        print("\n[5/6] (skipping second pass; output is /1000 only)")
    print("\n[6/6] Enumerating windows and building index CSV")
    index_rows = []
    cumulative_sample_offset = 0
    cumulative_window_offset = 0
    for X, info in zip(all_X, per_run_info):
        run = info["run"]
        N = info["N"]
        fps = info["fps"]
        srt_p = info["srt_path"]
        video = info["video_path"]
        sensor = info["sensor_path"]
        y = info["raw_labels"]
        srt_entries = parse_srt(srt_p)
        anchor_sample, anchor_video_s, used_srt = derive_video_anchor(srt_entries, y)
        if used_srt:
            print(
                f"  {run}: SRT anchor dat_sample={anchor_sample} ↔ video_t={anchor_video_s:.3f}s"
            )
        else:
            print(f"  {run}: [WARN] no SRT anchor — naive timing (video t=0)")
        run_windows = 0
        start = 0
        while start + WINDOW < N:
            end = start + WINDOW
            label_window = int(y[end - 1])
            start_t = sample_to_time(start, anchor_sample, anchor_video_s)
            end_t = sample_to_time(end - 1, anchor_sample, anchor_video_s)
            start_f = sample_to_frame(start, anchor_sample, anchor_video_s, fps)
            end_f = sample_to_frame(end - 1, anchor_sample, anchor_video_s, fps)
            index_rows.append(
                {
                    "window_idx": cumulative_window_offset + run_windows,
                    "run": run,
                    "sensor_file": os.path.relpath(sensor, opp_plus_root),
                    "video_file": os.path.relpath(video, opp_plus_root)
                    if video
                    else "",
                    "srt_file": os.path.relpath(srt_p, opp_plus_root) if srt_p else "",
                    "video_fps": round(fps, 4),
                    "start_sample": cumulative_sample_offset + start,
                    "end_sample": cumulative_sample_offset + end - 1,
                    "run_local_start": start,
                    "run_local_end": end - 1,
                    "start_time_s": round(start_t, 4),
                    "end_time_s": round(end_t, 4),
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "true_label": label_window,
                    "true_label_name": OPP_CLASS_NAMES[label_window],
                    "used_srt_anchor": used_srt,
                }
            )
            run_windows += 1
            start += STRIDE_TEST
        print(f"    windows: {run_windows}")
        cumulative_sample_offset += N
        cumulative_window_offset += run_windows
    X_test = np.concatenate(all_X, axis=0).astype(np.float32)
    y_test = np.concatenate(all_y, axis=0).astype(np.int64)
    print(f"\n  Combined: X={X_test.shape}, y={y_test.shape}")
    print(f"  Class histogram (label : count):")
    for c in range(18):
        n = int((y_test == c).sum())
        if n > 0:
            print(f"    {c:2d} {OPP_CLASS_NAMES[c]:<22s} {n:>8d}")
    os.makedirs(os.path.dirname(os.path.abspath(output_mat)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    runs_str = ",".join(runs)
    provenance = (
        f"Opportunity++ runs={runs_str} — channel mapping verified vs opportunity.mat; first-pass /1000; "
        + (
            "second-pass z-score baked in"
            if second_mean is not None
            else "second pass SKIPPED"
        )
        + "; gesture-ID->class map = numerically-sorted (matches "
        + "opportunity.mat['testingLabels'])."
    )
    sio.savemat(
        output_mat,
        {
            "trainingData": X_test.T,
            "trainingLabels": (y_test + 1).reshape(1, -1),
            "valData": X_test.T,
            "valLabels": (y_test + 1).reshape(1, -1),
            "testingData": X_test.T,
            "testingLabels": (y_test + 1).reshape(1, -1),
            "_provenance": np.array(provenance, dtype=object),
            "_runs": np.array(runs_str, dtype=object),
        },
    )
    print(f"\n  Wrote {output_mat}  ({os.path.getsize(output_mat) / 1000000.0:.1f} MB)")
    fieldnames = list(index_rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in index_rows:
            w.writerow(row)
    print(f"  Wrote {output_csv}  ({len(index_rows)} window rows)")
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Runs included:             {runs}")
    print(f"  Sensor samples (total):    {X_test.shape[0]}")
    print(
        f"  Sensor channels:           {X_test.shape[1]} (expected {EXPECTED_CHANNELS} — {('OK' if X_test.shape[1] == EXPECTED_CHANNELS else 'MISMATCH')})"
    )
    print(f"  Windows:                   {len(index_rows)}")
    print(
        f"  Windows with SRT anchor:   {sum((1 for r in index_rows if r['used_srt_anchor']))}/{len(index_rows)}"
    )
    print("=" * 70)
    print(
        "\nNext: run eval_opportunity_plus_sanity.py with --mat / --index_csv pointing at the files above."
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build an inference-ready .mat from Opportunity++ runs.",
    )
    p.add_argument(
        "--opp_plus_root",
        required=True,
        help="Path to extracted Opportunity++ root (contains data/ or run folders directly).",
    )
    p.add_argument(
        "--opp_mat",
        default="./dataset/opportunity.mat",
        help="Path to the existing opportunity.mat used during training. Its `trainingData` is used to recover the second-pass z-score stats so the output .mat matches the training distribution.",
    )
    p.add_argument(
        "--output_mat",
        default="./dataset/opportunity_plus_S2S3_ADL45.mat",
        help="Output .mat path.",
    )
    p.add_argument(
        "--output_csv",
        default="./dataset/opportunity_plus_S2S3_ADL45_index.csv",
        help="Output index CSV path.",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        choices=ALL_RUNS,
        metavar="RUN",
        help="Which Opportunity++ runs to include. Default = original A&D test split (S2-ADL4 S2-ADL5 S3-ADL4 S3-ADL5). Choices: any of "
        + ", ".join(ALL_RUNS),
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate layout (files, fps, SRT) without writing outputs.",
    )
    p.add_argument(
        "--skip_second_pass",
        action="store_true",
        help="Skip the second z-score pass (output will be /1000 only). Use this only if you plan to push the .mat through preprocess.py:load_mat(), which will apply the second pass on load — same effect as baking it in here.",
    )
    args = p.parse_args()
    if args.check:
        check_layout(args.opp_plus_root, args.runs)
    else:
        build(
            args.opp_plus_root,
            args.opp_mat,
            args.output_mat,
            args.output_csv,
            args.runs,
            skip_second_pass=args.skip_second_pass,
        )


if __name__ == "__main__":
    main()
