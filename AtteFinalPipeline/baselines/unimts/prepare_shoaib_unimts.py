import os
import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = "./shoaib_dataset/extracted/DataSet"
OUT_PATH = "./dataset/shoaib_unimts.mat"
TEST_SUBJECTS = [1, 9]
VAL_SUBJECTS = [10]
TRAIN_SUBJECTS = [2, 3, 4, 5, 6, 7, 8]
CLASS_MAP = [
    "walking",
    "standing",
    "jogging",
    "sitting",
    "biking",
    "walking upstairs",
    "walking downstairs",
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
N_POSITIONS = 5
COLS_PER_POS_RAW = 14
KEEP_OFFSETS = [1, 2, 3, 7, 8, 9, 4, 5, 6]
N_CHANNELS = N_POSITIONS * len(KEEP_OFFSETS)
SHOAIB_JOINTS = [1, 5, 21, 20, 0]


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


def load_participant(csv_path):
    df = pd.read_csv(csv_path, header=[0, 1], low_memory=False, dtype=str)
    df.columns = [f"{a}__{b}" for a, b in df.columns]
    n_total = df.shape[1]
    expected = N_POSITIONS * COLS_PER_POS_RAW
    if n_total != expected:
        raise ValueError(f"{csv_path}: expected {expected} cols, got {n_total}")
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
    return (X[finite], y[finite])


def load_subjects(subject_ids, label):
    print(f"\nLoading {label}: {subject_ids}")
    Xs, ys = ([], [])
    for sid in subject_ids:
        path = os.path.join(DATA_DIR, f"Participant_{sid}.csv")
        if not os.path.exists(path):
            print(f"  [WARN] missing {path}")
            continue
        X, y = load_participant(path)
        Xs.append(X)
        ys.append(y)
        print(
            f"  P{sid}: X={X.shape}  hist={np.bincount(y, minlength=N_CLASSES).tolist()}"
        )
    if not Xs:
        raise RuntimeError(f"No data for {label}")
    return (np.concatenate(Xs, 0), np.concatenate(ys, 0))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    X_train, y_train = load_subjects(TRAIN_SUBJECTS, "TRAIN")
    X_val, y_val = load_subjects(VAL_SUBJECTS, "VAL")
    X_test, y_test = load_subjects(TEST_SUBJECTS, "TEST")
    print(f"\n[Shoaib] Final (physical units, no z-score):")
    print(f"  X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    print(f"  L-pocket acc x mean magnitude: {np.abs(X_train[:, 0]).mean():.2f} m/s²")
    print(f"  L-pocket gyro x mean magnitude: {np.abs(X_train[:, 3]).mean():.2f} rad/s")
    sio.savemat(
        OUT_PATH,
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_val,
            "y_valid": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "sampling_rate": 50,
            "joints": np.array(SHOAIB_JOINTS),
            "class_names": np.array(CLASS_MAP, dtype=object),
        },
        do_compression=True,
    )
    print(f"\n[+] Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
