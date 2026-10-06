import argparse
import sys
import glob
from pathlib import Path
import numpy as np
import scipy.io as sio

SUBJECT_SPLIT = {"train": list(range(1, 8)), "val": [8], "test": [9, 10]}
ACTIVITY_NAMES = [
    "null",
    "standing still",
    "sitting and relaxing",
    "lying down",
    "walking",
    "climbing stairs",
    "waist bends forward",
    "frontal elevation of arms",
    "knees bending",
    "cycling",
    "jogging",
    "running",
    "jump front and back",
]
N_EXPECTED_SENSOR_COLS = 23
GYRO_COLS = list(range(9, 12)) + list(range(18, 21))
DEG_TO_RAD = np.pi / 180.0
MHEALTH_JOINTS = [11, 3, 21]


def find_subject_file(data_dir, subject_id):
    pattern = str(data_dir / f"*ubject{subject_id}.log")
    matches = glob.glob(pattern)
    if not matches:
        matches = glob.glob(str(data_dir / f"*ubject{subject_id}.LOG"))
    if not matches:
        raise FileNotFoundError(f"Cannot find subject {subject_id} in {data_dir}")
    return Path(matches[0])


def load_subject(path, drop_null):
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_sensor = data.shape[1] - 1
    if n_sensor != N_EXPECTED_SENSOR_COLS:
        print(
            f"  [WARN] {path.name}: {n_sensor} sensor cols (expected {N_EXPECTED_SENSOR_COLS})"
        )
    x = data[:, :n_sensor].astype(np.float32)
    y = data[:, -1].astype(np.int64)
    nan_rows = np.any(~np.isfinite(x), axis=1)
    if nan_rows.any():
        x = x[~nan_rows]
        y = y[~nan_rows]
    if y.min() >= 1:
        y = y - 1
    if drop_null:
        mask = y > 0
        x = x[mask]
        y = y[mask] - 1
    if n_sensor >= 21:
        x[:, GYRO_COLS] = x[:, GYRO_COLS] * DEG_TO_RAD
    return (x, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./mhealth_dataset")
    p.add_argument("--out_dir", default="./dataset")
    p.add_argument("--drop_null", action="store_true")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        sys.exit(f"[ERROR] data_dir {data_dir} not found.")
    n_classes = 12 if args.drop_null else 13
    activity_names = ACTIVITY_NAMES[1:] if args.drop_null else ACTIVITY_NAMES
    out_name = "mhealth_nonull_unimts.mat" if args.drop_null else "mhealth_unimts.mat"
    print(f"MHEALTH UniMTS prep: drop_null={args.drop_null}, n_classes={n_classes}")
    print(f"Sampling rate: 50 Hz")
    print(f"Conversions: gyro × π/180 (deg/s → rad/s); acc already m/s²")
    splits = {}
    for split_name, sids in SUBJECT_SPLIT.items():
        xs, ys = ([], [])
        for sid in sids:
            fp = find_subject_file(data_dir, sid)
            print(f"  Loading S{sid} ({split_name}) from {fp.name}...")
            x, y = load_subject(fp, drop_null=args.drop_null)
            xs.append(x)
            ys.append(y)
        splits[split_name] = (np.concatenate(xs, 0), np.concatenate(ys, 0))
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"\n[MHEALTH] Final (physical units, no z-score):")
    print(f"  X_train {X_train.shape}  classes {np.unique(y_train).tolist()}")
    print(f"  X_val   {X_val.shape}    classes {np.unique(y_val).tolist()}")
    print(f"  X_test  {X_test.shape}   classes {np.unique(y_test).tolist()}")
    print(
        f"  Chest acc magnitude (col 0-2): {np.linalg.norm(X_train[:, :3], axis=1).mean():.2f} m/s²"
    )
    out_path = out_dir / out_name
    sio.savemat(
        str(out_path),
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_val,
            "y_valid": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "sampling_rate": 50,
            "joints": np.array(MHEALTH_JOINTS),
            "class_names": np.array(activity_names, dtype=object),
            "drop_null": np.array(args.drop_null),
        },
        do_compression=True,
    )
    print(f"\n[+] Saved -> {out_path}")


if __name__ == "__main__":
    main()
