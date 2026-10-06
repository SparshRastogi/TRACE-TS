import os
import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = "./shoaib_dataset/extracted/DataSet"
OUT_DIR = "./dataset"
OUT_PATH = os.path.join(OUT_DIR, "shoaib.mat")
TEST_SUBJECTS = [1, 9]
VAL_SUBJECTS = [10]
TRAIN_SUBJECTS = [2, 3, 4, 5, 6, 7, 8]
CLASS_MAP = [
    "Walking",
    "Standing",
    "Jogging",
    "Sitting",
    "Biking",
    "Walking Upstairs",
    "Walking Downstairs",
]
N_CLASSES = len(CLASS_MAP)
ACTIVITY_MAP = {
    "walking": 0,
    "standing": 1,
    "jogging": 2,
    "sitting": 3,
    "biking": 4,
    "upstairs": 5,
    "downstairs": 6,
}


def normalise_label(s):
    s = str(s).strip().lower()
    if "upstairs" in s:
        return "upstairs"
    if "downstairs" in s:
        return "downstairs"
    if "jog" in s:
        return "jogging"
    if "bik" in s or "cycl" in s:
        return "biking"
    if "sit" in s:
        return "sitting"
    if "stand" in s:
        return "standing"
    if "walk" in s:
        return "walking"
    return None


N_POSITIONS = 5
COLS_PER_POS_RAW = 14
KEEP_OFFSETS = [1, 2, 3, 7, 8, 9, 4, 5, 6]
N_CHANNELS_PER_POS = len(KEEP_OFFSETS)
N_CHANNELS = N_POSITIONS * N_CHANNELS_PER_POS


def load_participant(csv_path):
    df = pd.read_csv(csv_path, header=[0, 1], low_memory=False, dtype=str)
    df.columns = [f"{a}__{b}" for a, b in df.columns]
    n_total = df.shape[1]
    expected = N_POSITIONS * COLS_PER_POS_RAW
    if n_total != expected:
        raise ValueError(
            f"{csv_path}: expected {expected} columns ({N_POSITIONS} × {COLS_PER_POS_RAW}), got {n_total}. CSV layout differs from what this script assumes."
        )
    raw_labels = df.iloc[:, -1].astype(str).values
    numeric = df.apply(pd.to_numeric, errors="coerce").values
    selected = []
    for p in range(N_POSITIONS):
        block = numeric[:, p * COLS_PER_POS_RAW : (p + 1) * COLS_PER_POS_RAW]
        selected.append(block[:, KEEP_OFFSETS])
    X = np.concatenate(selected, axis=1).astype(np.float32)
    y_raw = [normalise_label(l) for l in raw_labels]
    keep_row = np.array([k is not None and k in ACTIVITY_MAP for k in y_raw])
    y = np.array([ACTIVITY_MAP[k] for k in y_raw if k in ACTIVITY_MAP], dtype=np.int64)
    X = X[keep_row]
    finite = ~np.isnan(X).any(axis=1)
    X, y = (X[finite], y[finite])
    return (X, y)


def load_subjects(subject_ids, label):
    print(f"\nLoading {label} subjects: {subject_ids}")
    Xs, ys = ([], [])
    for sid in subject_ids:
        path = os.path.join(DATA_DIR, f"Participant_{sid}.csv")
        if not os.path.exists(path):
            print(f"  [WARN] missing {path} — skipping")
            continue
        print(f"  Loading {path}")
        X, y = load_participant(path)
        hist = np.bincount(y, minlength=N_CLASSES).tolist()
        print(f"    -> X={X.shape}, y={y.shape}, hist={hist}")
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise RuntimeError(f"No data loaded for split '{label}'")
    return (np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, y_train = load_subjects(TRAIN_SUBJECTS, "TRAIN")
    X_val, y_val = load_subjects(VAL_SUBJECTS, "VAL")
    X_test, y_test = load_subjects(TEST_SUBJECTS, "TEST")
    mean_tr = X_train.mean(axis=0)
    std_tr = X_train.std(axis=0)
    std_tr[std_tr == 0] = 1.0
    X_train = (X_train - mean_tr) / std_tr
    X_val = (X_val - mean_tr) / std_tr
    X_test = (X_test - mean_tr) / std_tr
    print("\n" + "=" * 60)
    print(
        f"Post-norm train stats : mean={X_train.mean():.4f}, std={X_train.std():.4f}  (should be ~0, ~1)"
    )
    print("Final shapes:")
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_val   {X_val.shape},   y_val   {y_val.shape}")
    print(f"  X_test  {X_test.shape},  y_test  {y_test.shape}")
    print(f"  train hist: {np.bincount(y_train, minlength=N_CLASSES).tolist()}")
    print(f"  val   hist: {np.bincount(y_val, minlength=N_CLASSES).tolist()}")
    print(f"  test  hist: {np.bincount(y_test, minlength=N_CLASSES).tolist()}")
    print("=" * 60)
    sio.savemat(
        OUT_PATH,
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_val,
            "y_valid": y_val,
            "X_test": X_test,
            "y_test": y_test,
        },
        do_compression=True,
    )
    print(f"\n[+] Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
