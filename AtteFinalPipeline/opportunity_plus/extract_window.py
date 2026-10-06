import os
import csv
import argparse
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
assert len(MAT_CHANNEL_RAW_COLS) == 79
EXPECTED_CHANNELS = 79
LABEL_COL = 249
SENSOR_HZ = 30
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


def parse_sensor_file(filepath, do_first_pass=True):
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
            f"[FATAL] {filepath}: expected 250 columns, found {n_cols_max}."
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
    if do_first_pass:
        X = (X / 1000.0).astype(np.float32)
    return (X, y)


def compute_second_pass_stats(opp_mat_path):
    m = sio.loadmat(opp_mat_path)
    x_train = m["trainingData"].astype(np.float32).T
    if x_train.shape[1] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"opportunity.mat has {x_train.shape[1]} channels, expected {EXPECTED_CHANNELS}."
        )
    mean = np.mean(x_train, axis=0).astype(np.float32)
    std = np.std(x_train, axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return (mean, std)


def find_window_row(index_csv_path, window_idx):
    with open(index_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["window_idx"]) == window_idx:
                return row
    raise KeyError(
        f"window_idx={window_idx} not found in {index_csv_path}. Check that the CSV was generated from the same run set you expect."
    )


def resolve_sensor_path(opp_plus_root, row):
    rel = row["sensor_file"]
    abs_path = os.path.join(opp_plus_root, rel)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Sensor file not found: {abs_path}\n(reconstructed from --opp_plus_root + sensor_file column '{rel}')"
        )
    return abs_path


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n\n")[0],
    )
    p.add_argument(
        "--window_idx",
        type=int,
        required=True,
        help="The window_idx to extract (e.g. 21825).",
    )
    p.add_argument(
        "--index_csv",
        required=True,
        help="Per-window index CSV from prepare_opportunity_plus.py (e.g. opportunity_plus_S2S3_ADL45_index.csv).",
    )
    p.add_argument(
        "--opp_plus_root",
        required=True,
        help="Opportunity++ root used by Stage 1 — needed to resolve the relative sensor_file path in the index CSV.",
    )
    p.add_argument(
        "--opp_mat",
        default="./dataset/opportunity.mat",
        help="opportunity.mat used during training. Needed to recover the second-pass z-score stats. Ignored if --raw or --skip_second_pass is set.",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path. Defaults to window_{window_idx:06d}.csv in CWD.",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Dump unscaled sensor readings (skip BOTH /1000 and z-score). Useful for raw inspection.",
    )
    p.add_argument(
        "--skip_second_pass",
        action="store_true",
        help="Apply only the /1000 first pass; skip the z-score second pass. Output then matches what's stored in opportunity_plus_*.mat BEFORE preprocess.load_mat().",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"Extracting window_idx = {args.window_idx}")
    print("=" * 70)
    row = find_window_row(args.index_csv, args.window_idx)
    run = row["run"]
    local_start = int(row["run_local_start"])
    local_end = int(row["run_local_end"])
    global_start = int(row["start_sample"])
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    start_time_s = float(row["start_time_s"])
    end_time_s = float(row["end_time_s"])
    fps = float(row["video_fps"])
    label = int(row["true_label"])
    label_name = row["true_label_name"]
    used_srt = row["used_srt_anchor"]
    print(f"  run              : {run}")
    print(
        f"  local samples    : {local_start} .. {local_end} ({local_end - local_start + 1} rows)"
    )
    print(f"  video frames     : {start_frame} .. {end_frame}  @ {fps:.3f} fps")
    print(f"  time (s)         : {start_time_s:.3f} .. {end_time_s:.3f}")
    print(f"  label            : {label} ({label_name})")
    print(f"  used_srt_anchor  : {used_srt}")
    sensor_path = resolve_sensor_path(args.opp_plus_root, row)
    print(f"\n  sensor_file      : {sensor_path}")
    if args.raw:
        do_first_pass = False
        do_second_pass = False
        norm_desc = "raw (no normalisation)"
    elif args.skip_second_pass:
        do_first_pass = True
        do_second_pass = False
        norm_desc = "/1000 only (first pass)"
    else:
        do_first_pass = True
        do_second_pass = True
        norm_desc = "/1000 + z-score (both passes — matches model input)"
    print(f"  normalisation    : {norm_desc}")
    print(f"\n  parsing sensor file …")
    X, y = parse_sensor_file(sensor_path, do_first_pass=do_first_pass)
    print(f"  parsed X={X.shape}, y={y.shape}")
    if do_second_pass:
        if not os.path.isfile(args.opp_mat):
            raise FileNotFoundError(
                f"--opp_mat not found at {args.opp_mat}. Pass --skip_second_pass or --raw to skip z-score normalisation."
            )
        print(f"  recovering z-score stats from {args.opp_mat}")
        mean, std = compute_second_pass_stats(args.opp_mat)
        X = ((X - mean) / std).astype(np.float32)
    if local_end >= X.shape[0]:
        raise RuntimeError(
            f"Window range {local_start}..{local_end} exceeds sensor file length {X.shape[0]} — index CSV and sensor file disagree."
        )
    X_win = X[local_start : local_end + 1]
    y_win = y[local_start : local_end + 1]
    T = X_win.shape[0]
    print(f"\n  window slice     : X={X_win.shape}, y={y_win.shape}")
    times = start_time_s + np.arange(T) / SENSOR_HZ
    frames = np.round(times * fps).astype(np.int64)
    out_path = args.output_csv or f"window_{args.window_idx:06d}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    CHANNEL_NAMES = [
        "IMU_BACK_AccX",
        "IMU_BACK_AccY",
        "IMU_BACK_AccZ",
        "IMU_BACK_GyroX",
        "IMU_BACK_GyroY",
        "IMU_BACK_GyroZ",
        "IMU_BACK_MagX",
        "IMU_BACK_MagY",
        "IMU_BACK_MagZ",
        "IMU_BACK_Quat1",
        "IMU_BACK_Quat2",
        "IMU_RUA_AccX",
        "IMU_RUA_AccY",
        "IMU_RUA_AccZ",
        "IMU_RUA_GyroX",
        "IMU_RUA_GyroY",
        "IMU_RUA_GyroZ",
        "IMU_RUA_MagX",
        "IMU_RUA_MagY",
        "IMU_RUA_MagZ",
        "IMU_RLA_AccX",
        "IMU_RLA_AccY",
        "IMU_RLA_AccZ",
        "IMU_RLA_GyroX",
        "IMU_RLA_GyroY",
        "IMU_RLA_GyroZ",
        "IMU_RLA_MagX",
        "IMU_RLA_MagY",
        "IMU_RLA_MagZ",
        "IMU_LUA_AccX",
        "IMU_LUA_AccY",
        "IMU_LUA_AccZ",
        "IMU_LUA_GyroX",
        "IMU_LUA_GyroY",
        "IMU_LUA_GyroZ",
        "IMU_LUA_MagX",
        "IMU_LUA_MagY",
        "IMU_LUA_MagZ",
        "IMU_LLA_AccX",
        "IMU_LLA_AccY",
        "IMU_LLA_AccZ",
        "IMU_LLA_GyroX",
        "IMU_LLA_GyroY",
        "IMU_LLA_GyroZ",
        "IMU_LLA_MagX",
        "IMU_LLA_MagY",
        "IMU_LLA_MagZ",
        "LSHOE_EuX",
        "LSHOE_EuY",
        "LSHOE_EuZ",
        "LSHOE_Nav_AccX",
        "LSHOE_Nav_AccY",
        "LSHOE_Nav_AccZ",
        "LSHOE_Body_AccX",
        "LSHOE_Body_AccY",
        "LSHOE_Body_AccZ",
        "LSHOE_AngVelBodyX",
        "LSHOE_AngVelBodyY",
        "LSHOE_AngVelBodyZ",
        "LSHOE_AngVelNavX",
        "LSHOE_AngVelNavY",
        "LSHOE_AngVelNavZ",
        "LSHOE_Compass",
        "RSHOE_EuX",
        "RSHOE_EuY",
        "RSHOE_EuZ",
        "RSHOE_Nav_AccX",
        "RSHOE_Nav_AccY",
        "RSHOE_Nav_AccZ",
        "RSHOE_Body_AccX",
        "RSHOE_Body_AccY",
        "RSHOE_Body_AccZ",
        "RSHOE_AngVelBodyX",
        "RSHOE_AngVelBodyY",
        "RSHOE_AngVelBodyZ",
        "RSHOE_AngVelNavX",
        "RSHOE_AngVelNavY",
        "RSHOE_AngVelNavZ",
        "RSHOE_Compass",
    ]
    assert len(CHANNEL_NAMES) == EXPECTED_CHANNELS
    header = [
        "sample_idx_local",
        "sample_idx_global",
        "time_s",
        "frame",
        "label",
        "label_name",
    ] + CHANNEL_NAMES
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in range(T):
            row_out = [
                local_start + t,
                global_start + t,
                round(float(times[t]), 6),
                int(frames[t]),
                int(y_win[t]),
                OPP_CLASS_NAMES[int(y_win[t])],
            ] + [float(v) for v in X_win[t]]
            w.writerow(row_out)
    print(f"\n  wrote {out_path}  ({T} rows x {len(header)} cols)")
    print("=" * 70)
    print("[DONE]")


if __name__ == "__main__":
    main()
