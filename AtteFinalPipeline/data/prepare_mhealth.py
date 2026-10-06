import argparse
import sys
import glob
from pathlib import Path
import numpy as np
import scipy.io as sio

SUBJECT_SPLIT = {"train": list(range(1, 8)), "val": [8], "test": [9, 10]}
ACTIVITY_NAMES = [
    "Null (no activity)",
    "Standing still",
    "Sitting and relaxing",
    "Lying down",
    "Walking",
    "Climbing stairs",
    "Waist bends forward",
    "Frontal elevation of arms",
    "Knees bending (crouching)",
    "Cycling",
    "Jogging",
    "Running",
    "Jump front & back",
]
N_EXPECTED_SENSOR_COLS = 23
N_TOTAL_SUBJECTS = 10


def find_subject_file(data_dir: Path, subject_id: int) -> Path:
    pattern = str(data_dir / f"*ubject{subject_id}.log")
    matches = glob.glob(pattern, recursive=False)
    if not matches:
        pattern_lc = str(data_dir / f"*ubject{subject_id}.LOG")
        matches = glob.glob(pattern_lc)
    if not matches:
        raise FileNotFoundError(
            f"Cannot find subject {subject_id} file in {data_dir}. Expected pattern: mHealth_subject{subject_id}.log"
        )
    return Path(matches[0])


def load_subject(path: Path, drop_null: bool) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_cols = data.shape[1]
    n_sensor = n_cols - 1
    if n_sensor != N_EXPECTED_SENSOR_COLS:
        print(
            f"  [WARN] {path.name}: expected {N_EXPECTED_SENSOR_COLS} sensor cols, found {n_sensor}. Proceeding with {n_sensor} channels."
        )
    x = data[:, :n_sensor].astype(np.float32)
    y = data[:, -1].astype(np.int64)
    nan_rows = np.any(~np.isfinite(x), axis=1)
    if nan_rows.any():
        n_bad = int(nan_rows.sum())
        print(f"  [WARN] {path.name}: dropping {n_bad} rows with NaN/Inf values.")
        x = x[~nan_rows]
        y = y[~nan_rows]
    if y.min() >= 1:
        print(
            f"  [INFO] {path.name}: labels appear 1-based (min={y.min()}), shifting to 0-based."
        )
        y = y - 1
    if drop_null:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        y = y - 1
    return (x, y)


def z_score_normalize(x_train, x_val, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    return (x_train, x_val, x_test, mean, std)


def print_split_stats(name, x, y, activity_names):
    n_classes = len(np.unique(y))
    print(
        f"\n  [{name}]  shape={x.shape}  dtype={x.dtype}  label_range=[{y.min()}, {y.max()}]  n_classes={n_classes}"
    )
    counts = np.bincount(y, minlength=len(activity_names))
    for i, cnt in enumerate(counts):
        if cnt > 0:
            label = activity_names[i] if i < len(activity_names) else f"class_{i}"
            print(f"          class {i:2d}  {label:<30s}: {cnt:7d} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MHEALTH dataset for AttendDiscriminate pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="./mhealth_dataset",
        help="Directory containing mHealth_subject*.log files.",
    )
    parser.add_argument(
        "--out_dir",
        default="./dataset",
        help="Directory where mhealth.mat will be written.",
    )
    parser.add_argument(
        "--drop_null",
        action="store_true",
        default=False,
        help="Drop rows where label == 0 (null activity) and remap labels 1-12 -> 0-11, giving num_class=12.",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        default=False,
        help="Skip z-score normalisation (not recommended).",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        sys.exit(
            f"[ERROR] data_dir '{data_dir}' not found. Run the download script first."
        )
    n_classes = 12 if args.drop_null else 13
    activity_names = ACTIVITY_NAMES[1:] if args.drop_null else ACTIVITY_NAMES
    print("=" * 60)
    print("MHEALTH Preprocessing")
    print(f"  data_dir    : {data_dir}")
    print(f"  out_dir     : {out_dir}")
    print(f"  drop_null   : {args.drop_null}  (num_class = {n_classes})")
    print(f"  normalize   : {not args.no_normalize}")
    print(f"  split       : train=S1-7  val=S8  test=S9-10")
    print("=" * 60)
    splits_x, splits_y = ({}, {})
    for split_name, subject_ids in SUBJECT_SPLIT.items():
        xs, ys = ([], [])
        for sid in subject_ids:
            path = find_subject_file(data_dir, sid)
            print(f"  Loading subject {sid:2d}  ({split_name}) from {path.name} ...")
            x, y = load_subject(path, drop_null=args.drop_null)
            xs.append(x)
            ys.append(y)
            print(f"    → {x.shape[0]:7d} samples, labels: {np.unique(y).tolist()}")
        splits_x[split_name] = np.concatenate(xs, axis=0)
        splits_y[split_name] = np.concatenate(ys, axis=0)
    x_train = splits_x["train"]
    y_train = splits_y["train"]
    x_val = splits_x["val"]
    y_val = splits_y["val"]
    x_test = splits_x["test"]
    y_test = splits_y["test"]
    if not args.no_normalize:
        print("\n[*] Applying z-score normalisation (train statistics)...")
        x_train, x_val, x_test, mean, std = z_score_normalize(x_train, x_val, x_test)
        print(f"    per-channel mean (first 5): {mean[:5].round(4).tolist()}")
        print(f"    per-channel std  (first 5): {std[:5].round(4).tolist()}")
        post_mean = x_train.mean()
        post_std = x_train.std()
        print(
            f"    post-norm global: mean={post_mean:.4f}, std={post_std:.4f} (should be ~0, ~1)"
        )
    else:
        print("\n[WARN] Skipping normalisation — preprocess.py will apply it later.")
    print("\n── Dataset statistics ──────────────────────────────────────────")
    print_split_stats("train", x_train, y_train, activity_names)
    print_split_stats("val", x_val, y_val, activity_names)
    print_split_stats("test", x_test, y_test, activity_names)
    print("\n── Sanity checks ───────────────────────────────────────────────")
    n_sensor_cols = x_train.shape[1]
    print(f"  Sensor channels : {n_sensor_cols}  (expected {N_EXPECTED_SENSOR_COLS})")
    if n_sensor_cols != N_EXPECTED_SENSOR_COLS:
        print(
            f"  [WARN] Channel count mismatch! Update settings.py:  input_dim = {n_sensor_cols}"
        )
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        max_label = y.max()
        if max_label >= n_classes:
            print(f"  [WARN] {split_name}: label {max_label} >= n_classes {n_classes}!")
        else:
            print(f"  {split_name}: labels OK  [0, {max_label}] ⊆ [0, {n_classes - 1}]")
    out_path = out_dir / "mhealth.mat"
    print(f"\n[*] Saving to {out_path} ...")
    sio.savemat(
        str(out_path),
        {
            "X_train": x_train,
            "y_train": y_train,
            "X_valid": x_val,
            "y_valid": y_val,
            "X_test": x_test,
            "y_test": y_test,
        },
        do_compression=True,
    )
    print(f"[+] Saved: {out_path}")
    null_comment = (
        "# null class dropped; labels 1-12 remapped to 0-11"
        if args.drop_null
        else "# null class retained as label 0"
    )
    class_map_str = (
        "[\n"
        + "".join((f'            "{name}",\n' for name in activity_names))
        + "        ]"
    )
    print("\n" + "=" * 60)
    print("Add this entry to _DATASET_CONFIGS in settings.py:\n")
    print(
        f'    "mhealth": dict(\n        dataset        = "mhealth",\n        input_dim      = {n_sensor_cols},        {null_comment}\n        num_class      = {n_classes},\n        window         = 100,         # 50 Hz × 2.0 s\n        stride         = 50,          # 50 % overlap\n        stride_test    = 1,\n        path_data      = "./dataset/mhealth.mat",\n        path_raw       = "./data/mhealth/raw",\n        path_processed = "./data/mhealth/processed/100_50",\n        class_map      = {class_map_str},\n        results_dir    = "./results/mhealth",\n    ),\n'
    )
    print("Add this to _DATASET_TRAIN_OVERRIDES:\n")
    print(
        '    "mhealth": dict(\n        init_weights = "orthogonal",\n        beta         = 0.003,\n        dropout      = 0.5,\n        dropout_rnn  = 0.25,\n        dropout_cls  = 0.5,\n    ),\n'
    )
    print("=" * 60)
    print("\nDone. Now run:  python preprocess.py --dataset mhealth")


if __name__ == "__main__":
    main()
