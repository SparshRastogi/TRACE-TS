from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
import os
import argparse
import glob
import numpy as np
import scipy.io as sio

USC_HAD_ACTIVITIES = list(USCHAD_CLASS_MAP)
NUM_CLASSES = 12
INPUT_DIM = 6


def z_score_normalize(x_train, x_val, x_test):
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    return (x_train, x_val, x_test)


def sliding_window(x, y, window, stride):
    data, target = ([], [])
    start = 0
    while start + window <= len(x):
        data.append(x[start : start + window])
        target.append(y[start + window - 1])
        start += stride
    if not data:
        return (
            np.empty((0, window, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (np.array(data, dtype=np.float32), np.array(target, dtype=np.int64))


def load_subject(subject_dir):
    x_list, y_list = ([], [])
    mat_files = sorted(glob.glob(os.path.join(subject_dir, "*.mat")))
    if not mat_files:
        print(f"  [WARN] No .mat files found in {subject_dir}")
        return (x_list, y_list)
    for fpath in mat_files:
        try:
            data = sio.loadmat(fpath)
        except Exception as e:
            print(f"  [WARN] Could not load {fpath}: {e}")
            continue
        if "sensor_readings" in data:
            readings = data["sensor_readings"].astype(np.float32)
        elif "Sensor_readings" in data:
            readings = data["Sensor_readings"].astype(np.float32)
        else:
            print(f"  [WARN] No sensor_readings field in {fpath}, skipping.")
            continue
        if readings.shape[0] == 6 and readings.ndim == 2:
            readings = readings.T
        if "activity_number" in data:
            act_num = int(np.squeeze(data["activity_number"]))
        else:
            fname = os.path.basename(fpath)
            digits = "".join((c for c in fname.split("t")[0] if c.isdigit()))
            if not digits:
                print(f"  [WARN] Cannot determine activity for {fpath}, skipping.")
                continue
            act_num = int(digits)
        label = act_num - 1
        if label < 0 or label >= NUM_CLASSES:
            print(
                f"  [WARN] Label {label} out of range [0,{NUM_CLASSES - 1}] in {fpath}, skipping."
            )
            continue
        labels = np.full(len(readings), label, dtype=np.int64)
        x_list.append(readings)
        y_list.append(labels)
    return (x_list, y_list)


def load_subjects(data_dir, subject_ids, label=""):
    all_x, all_y = ([], [])
    for sid in subject_ids:
        subj_dir = os.path.join(data_dir, f"Subject{sid}")
        if not os.path.isdir(subj_dir):
            print(f"[WARN] Subject directory not found: {subj_dir}")
            continue
        print(f"  Loading Subject{sid} ...")
        x_list, y_list = load_subject(subj_dir)
        all_x.extend(x_list)
        all_y.extend(y_list)
    if not all_x:
        raise RuntimeError(f"No data loaded for {label} subjects: {subject_ids}")
    x = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    print(f"  [{label}] {x.shape[0]:,} samples, {len(np.unique(y))} classes")
    return (x, y)


def prepare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "uschad.mat")
    all_subject_ids = list(range(1, 15))
    val_set = set(args.val_subjects)
    test_set = set(args.test_subjects)
    train_set = set(all_subject_ids) - val_set - test_set
    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"val and test subjects overlap: {overlap}")
    if not train_set:
        raise ValueError("No subjects left for training after val/test split.")
    print(f"Train subjects : {sorted(train_set)}")
    print(f"Val subjects   : {sorted(val_set)}")
    print(f"Test subjects  : {sorted(test_set)}")
    print()
    print("=== Loading training subjects ===")
    x_train_raw, y_train_raw = load_subjects(args.data_dir, sorted(train_set), "train")
    print("\n=== Loading validation subjects ===")
    x_val_raw, y_val_raw = load_subjects(args.data_dir, sorted(val_set), "val")
    print("\n=== Loading test subjects ===")
    x_test_raw, y_test_raw = load_subjects(args.data_dir, sorted(test_set), "test")
    print("\n=== Normalising ===")
    x_train_raw, x_val_raw, x_test_raw = z_score_normalize(
        x_train_raw, x_val_raw, x_test_raw
    )
    print(f"  train mean={x_train_raw.mean():.4f}  std={x_train_raw.std():.4f}")
    print(f"\n=== Windowing (window={args.window}, stride={args.stride}) ===")
    X_train, y_train = (x_train_raw, y_train_raw)
    X_valid, y_valid = (x_val_raw, y_val_raw)
    X_test, y_test = (x_test_raw, y_test_raw)
    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_valid : {X_valid.shape}  y_valid : {y_valid.shape}")
    print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    for name, y in [("train", y_train), ("val", y_valid), ("test", y_test)]:
        unique = np.unique(y)
        print(f"  [{name}] unique labels: {unique.tolist()}")
        if unique.min() < 0 or unique.max() >= NUM_CLASSES:
            print(f"  [WARN] Labels out of expected range [0, {NUM_CLASSES - 1}]!")
    print(f"\n=== Split sizes ===")
    print(
        f"Train samples : {len(y_train_raw):,}  |  classes: {sorted(set(y_train_raw.tolist()))}"
    )
    print(
        f"Val samples   : {len(y_val_raw):,}  |  classes: {sorted(set(y_val_raw.tolist()))}"
    )
    print(
        f"Test samples  : {len(y_test_raw):,}  |  classes: {sorted(set(y_test_raw.tolist()))}"
    )
    from collections import Counter

    print(
        f"\nTrain class counts : {dict(sorted(Counter(y_train_raw.tolist()).items()))}"
    )
    print(f"Val class counts   : {dict(sorted(Counter(y_val_raw.tolist()).items()))}")
    print(f"\n=== Saving to {out_path} ===")
    sio.savemat(
        out_path,
        {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_valid": X_valid,
            "y_valid": y_valid.reshape(-1, 1),
            "X_test": X_test,
            "y_test": y_test.reshape(-1, 1),
        },
        do_compression=True,
    )
    print("[+] Done!")
    print(f"\nNext steps:")
    print(f"  python train.py --dataset uschad --train_mode")


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare USC-HAD dataset for the AttendDiscriminate HAR pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Root directory of USC-HAD (contains Subject1 .. Subject14 folders)",
    )
    p.add_argument(
        "--out_dir",
        default="./dataset",
        help="Output directory for the generated .mat file",
    )
    p.add_argument(
        "--window",
        default=100,
        type=int,
        help="Sliding window length in samples (default 100 ≈ 0.5s at 200Hz)",
    )
    p.add_argument(
        "--stride",
        default=50,
        type=int,
        help="Sliding window stride (default 50 → 50%% overlap)",
    )
    p.add_argument(
        "--val_subjects",
        nargs="+",
        type=int,
        default=[13, 14],
        help="Subject IDs to use as validation set",
    )
    p.add_argument(
        "--test_subjects",
        nargs="+",
        type=int,
        default=[11, 12],
        help="Subject IDs to use as test set",
    )
    return p.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
