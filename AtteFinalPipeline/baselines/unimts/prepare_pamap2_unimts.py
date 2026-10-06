import os
import zipfile
import subprocess
import numpy as np
import scipy.io as sio
from tqdm import tqdm

PAMAP2_FEAT_COLS = [2] + list(range(3, 20)) + list(range(20, 37)) + list(range(37, 54))
PAMAP2_TRAIN_SUBJECTS = [1, 2, 3, 4, 7, 8, 9]
PAMAP2_VAL_SUBJECTS = [5]
PAMAP2_TEST_SUBJECTS = [6]
PAMAP2_ACTIVITY_IDS = [24, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17]
PAMAP2_ID_TO_CLASS = {aid: i for i, aid in enumerate(PAMAP2_ACTIVITY_IDS)}
PAMAP2_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
    "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
]
PAMAP2_DEST = "./downloads/pamap2.zip"
PAMAP2_DIR = "./downloads/pamap2"
PAMAP2_CLASS_NAMES = [
    "rope jumping",
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "nordic walking",
    "ascending stairs",
    "descending stairs",
    "vacuum cleaning",
    "ironing",
]
PAMAP2_BLOCK_STARTS_IN_52COL = [1, 18, 35]


def download_file(url, dest_path, desc="Downloading"):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        print(f"  [skip] Already exists: {dest_path}")
        return
    print(f"  [download] {desc}")
    for cmd in [
        [
            "wget",
            "-q",
            "--show-progress",
            "--no-check-certificate",
            "-O",
            dest_path,
            url,
        ],
        ["curl", "-L", "-k", "--progress-bar", "-o", dest_path, url],
    ]:
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
                return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    import warnings, requests

    warnings.filterwarnings("ignore", message="Unverified HTTPS")
    r = requests.get(
        url,
        stream=True,
        timeout=300,
        headers={"User-Agent": "Mozilla/5.0"},
        verify=False,
    )
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with (
        open(dest_path, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True) as bar,
    ):
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def unzip(zip_path, dest_dir):
    print(f"  [unzip] {zip_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


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


def find_subject_file(protocol_dir, sid):
    candidates = [
        os.path.join(protocol_dir, f"subject10{sid}.dat"),
        os.path.join(protocol_dir, f"subject{sid:03d}.dat"),
        os.path.join(protocol_dir, f"subject{sid}.dat"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def load_pamap2_subject(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    if not rows:
        return (np.zeros((0, len(PAMAP2_FEAT_COLS)), np.float32), np.zeros(0, np.int64))
    n_cols = max((len(r) for r in rows))
    data = np.full((len(rows), n_cols), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try:
                data[i, j] = float(v)
            except ValueError:
                pass
    act_ids = np.round(data[:, 1]).astype(int)
    mask = np.isin(act_ids, PAMAP2_ACTIVITY_IDS)
    data = data[mask]
    act_ids = act_ids[mask]
    if data.shape[0] == 0:
        return (np.zeros((0, len(PAMAP2_FEAT_COLS)), np.float32), np.zeros(0, np.int64))
    max_col = max(PAMAP2_FEAT_COLS)
    if data.shape[1] <= max_col:
        pad = np.full((data.shape[0], max_col + 1 - data.shape[1]), np.nan)
        data = np.concatenate([data, pad], axis=1)
    X = data[:, PAMAP2_FEAT_COLS].astype(np.float32)
    X = impute_nans(X)
    y = np.array([PAMAP2_ID_TO_CLASS[int(a)] for a in act_ids], dtype=np.int64)
    X = X[::3]
    y = y[::3]
    return (X, y)


def build_pamap2_unimts(output_path="./dataset/pamap2_unimts.mat"):
    print("\n" + "=" * 60)
    print("[PAMAP2 / UniMTS] building physical-units .mat (no z-score)")
    print("  Split: train={1,2,3,4,7,8,9}  val={5}  test={6}")
    print("  Sampling rate after [::3] downsample : 33 Hz")
    print("  Units: acc6 in m/s², gyro in rad/s  (PAMAP2 native)")
    downloaded = False
    for url in PAMAP2_URLS:
        if downloaded:
            break
        try:
            download_file(url, PAMAP2_DEST, "PAMAP2 dataset (~650 MB)")
            if os.path.exists(PAMAP2_DEST) and os.path.getsize(PAMAP2_DEST) > 100000:
                downloaded = True
        except Exception as e:
            print(f"  [warn] {e}")
            if os.path.exists(PAMAP2_DEST):
                os.remove(PAMAP2_DEST)
    if not downloaded:
        print("\n[PAMAP2] Manual download required.")
        return
    if not zipfile.is_zipfile(PAMAP2_DEST):
        raise RuntimeError(f"{PAMAP2_DEST} is not a valid zip.")
    unzip(PAMAP2_DEST, PAMAP2_DIR)
    protocol_dir = None
    for root, dirs, files in os.walk(PAMAP2_DIR):
        if os.path.basename(root).lower() == "protocol" and any(
            (f.startswith("subject") and f.endswith(".dat") for f in files)
        ):
            protocol_dir = root
            break
    if protocol_dir is None:
        for root, dirs, files in os.walk(PAMAP2_DIR):
            if os.path.basename(root).lower() == "optional":
                continue
            if any((f.startswith("subject") and f.endswith(".dat") for f in files)):
                protocol_dir = root
                break
    if protocol_dir is None:
        raise FileNotFoundError("Protocol folder not found.")
    print(f"  Protocol dir: {protocol_dir}")

    def load_subjects(subject_ids, split_name):
        Xs, ys = ([], [])
        for sid in subject_ids:
            fpath = find_subject_file(protocol_dir, sid)
            if fpath is None:
                print(f"  [warn] Subject {sid} not found, skipping.")
                continue
            print(
                f"  Loading subject {sid} ({os.path.basename(fpath)}) ...",
                end=" ",
                flush=True,
            )
            X, y = load_pamap2_subject(fpath)
            print(f"shape={X.shape}  classes={np.unique(y).tolist()}")
            Xs.append(X)
            ys.append(y)
        if not Xs:
            raise RuntimeError(f"No subjects loaded for {split_name}!")
        return (np.concatenate(Xs, 0), np.concatenate(ys, 0))

    X_train, y_train = load_subjects(PAMAP2_TRAIN_SUBJECTS, "train")
    X_val, y_val = load_subjects(PAMAP2_VAL_SUBJECTS, "val")
    X_test, y_test = load_subjects(PAMAP2_TEST_SUBJECTS, "test")
    print("\n[PAMAP2] Final shapes (physical units, no z-score):")
    print(f"  train: {X_train.shape}, classes {np.unique(y_train).tolist()}")
    print(f"  val:   {X_val.shape},   classes {np.unique(y_val).tolist()}")
    print(f"  test:  {X_test.shape},  classes {np.unique(y_test).tolist()}")
    hand_acc6_z = X_train[:, 1 + 6]
    print(
        f"  hand acc6 z magnitude check: |mean|={abs(hand_acc6_z.mean()):.2f}, std={hand_acc6_z.std():.2f}  (expect a few m/s² when standing)"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sio.savemat(
        output_path,
        {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_valid": X_val,
            "y_valid": y_val.reshape(-1, 1),
            "X_test": X_test,
            "y_test": y_test.reshape(-1, 1),
            "sampling_rate": 33,
            "feat_cols": np.array(PAMAP2_FEAT_COLS),
            "block_starts_52": np.array(PAMAP2_BLOCK_STARTS_IN_52COL),
            "class_names": np.array(PAMAP2_CLASS_NAMES, dtype=object),
            "activity_ids": np.array(PAMAP2_ACTIVITY_IDS),
        },
    )
    print(f"\n[+] Saved -> {output_path}")


if __name__ == "__main__":
    build_pamap2_unimts()
