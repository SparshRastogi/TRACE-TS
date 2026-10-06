import os
import argparse
import zipfile
import subprocess
import numpy as np
import scipy.io as sio
from tqdm import tqdm


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
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
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


PAMAP2_FEAT_COLS = [2] + list(range(3, 20)) + list(range(20, 37)) + list(range(37, 54))
PAMAP2_TRAIN_SUBJECTS = [1, 2, 3, 4, 7, 8, 9]
PAMAP2_VAL_SUBJECTS = [5]
PAMAP2_TEST_SUBJECTS = [6]
PAMAP2_ACTIVITY_IDS = [24, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17]
PAMAP2_ID_TO_CLASS = {aid: i for i, aid in enumerate(PAMAP2_ACTIVITY_IDS)}
PAMAP2_CHANNEL_NAMES = (
    ["heart_rate"]
    + [
        f"hand_{s}"
        for s in [
            "temp",
            "acc16g_x",
            "acc16g_y",
            "acc16g_z",
            "acc6g_x",
            "acc6g_y",
            "acc6g_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "orient_0",
            "orient_1",
            "orient_2",
            "orient_3",
        ]
    ]
    + [
        f"chest_{s}"
        for s in [
            "temp",
            "acc16g_x",
            "acc16g_y",
            "acc16g_z",
            "acc6g_x",
            "acc6g_y",
            "acc6g_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "orient_0",
            "orient_1",
            "orient_2",
            "orient_3",
        ]
    ]
    + [
        f"ankle_{s}"
        for s in [
            "temp",
            "acc16g_x",
            "acc16g_y",
            "acc16g_z",
            "acc6g_x",
            "acc6g_y",
            "acc6g_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "orient_0",
            "orient_1",
            "orient_2",
            "orient_3",
        ]
    ]
)
PAMAP2_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
    "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
]
PAMAP2_DEST = "./downloads/pamap2.zip"
PAMAP2_DIR = "./downloads/pamap2"


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


def find_subject_file(protocol_dir, sid):
    candidates = [
        os.path.join(protocol_dir, f"subject10{sid}.dat"),
        os.path.join(protocol_dir, f"subject{sid:03d}.dat"),
        os.path.join(protocol_dir, f"subject{sid}.dat"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def build_pamap2(output_path="./dataset/pamap2.mat"):
    print("\n" + "=" * 60)
    print("[PAMAP2] Starting...")
    print("  Split: Train=subjects{1,2,3,4,7,8,9}  Val=subject{5}  Test=subject{6}")
    print("  (Replicates the A&D within-subject run-level split)")
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
        print("\n[PAMAP2] AUTO-DOWNLOAD FAILED — manual download required:")
        print(
            "  https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
        )
        print(f"  Save to: {os.path.abspath(PAMAP2_DEST)}")
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
        raise FileNotFoundError(
            f"No Protocol subject .dat files found under {PAMAP2_DIR}. Expected a 'Protocol' subdirectory inside the zip."
        )
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

    print("\n[PAMAP2] Loading train subjects (1,2,3,4,7,8,9)...")
    X_train, y_train = load_subjects(PAMAP2_TRAIN_SUBJECTS, "train")
    print("\n[PAMAP2] Loading val subject (5)...")
    X_val, y_val = load_subjects(PAMAP2_VAL_SUBJECTS, "val")
    print("\n[PAMAP2] Loading test subject (6)...")
    X_test, y_test = load_subjects(PAMAP2_TEST_SUBJECTS, "test")
    mean = X_train.mean(0)
    std = X_train.std(0)
    std[std == 0] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)
    print(f"\n[PAMAP2] Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    print(f"[PAMAP2] Classes in train: {np.unique(y_train).tolist()}")
    print(f"[PAMAP2] Classes in val:   {np.unique(y_val).tolist()}")
    print(f"[PAMAP2] Classes in test:  {np.unique(y_test).tolist()}")
    train_cls = set(np.unique(y_train).tolist())
    for split, y in [("val", y_val), ("test", y_test)]:
        missing = train_cls - set(np.unique(y).tolist())
        if missing:
            missing_names = [PAMAP2_ACTIVITY_IDS[c] for c in missing]
            print(
                f"  [WARN] {split} is missing classes {missing} (activity IDs {missing_names}) — subject may not have performed them"
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
            "channel_names": np.array(PAMAP2_CHANNEL_NAMES, dtype=object),
            "activity_ids": np.array(PAMAP2_ACTIVITY_IDS),
            "train_mean": mean,
            "train_std": std,
        },
    )
    print(f"[PAMAP2] Saved -> {output_path}")


OPP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
OPP_DEST = "./downloads/opportunity.zip"
OPP_DIR = "./downloads/opportunity"
OPP_COLS_TO_DELETE = (
    list(range(46, 50))
    + list(range(59, 63))
    + list(range(72, 76))
    + list(range(85, 89))
    + list(range(98, 102))
    + list(range(134, 243))
    + list(range(244, 250))
)
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
    406516: 1,
    406517: 2,
    404516: 3,
    404517: 4,
    406520: 5,
    404520: 6,
    406505: 7,
    404505: 8,
    406519: 9,
    404519: 10,
    406511: 11,
    404511: 12,
    406508: 13,
    404508: 14,
    408512: 15,
    407521: 16,
    405506: 17,
}
OPP_FEATURE_SLICE = slice(1, 80)
OPP_TRAIN_FILES = [
    "S1-ADL1.dat",
    "S1-ADL2.dat",
    "S1-ADL3.dat",
    "S1-ADL4.dat",
    "S1-ADL5.dat",
    "S1-Drill.dat",
    "S2-ADL1.dat",
    "S2-ADL2.dat",
    "S2-ADL3.dat",
    "S3-ADL1.dat",
    "S3-ADL2.dat",
    "S3-ADL3.dat",
]
OPP_VAL_FILES = ["S2-ADL4.dat", "S2-ADL5.dat"]
OPP_TEST_FILES = [
    "S4-ADL1.dat",
    "S4-ADL2.dat",
    "S4-ADL3.dat",
    "S4-ADL4.dat",
    "S4-ADL5.dat",
    "S4-Drill.dat",
]


def find_gesture_col(filepath):
    with open(filepath) as f:
        for line in f:
            for j, v in enumerate(line.strip().split()):
                try:
                    if int(float(v)) in OPP_GESTURE_VALUES:
                        return j
                except ValueError:
                    pass
    return None


def load_opp_file(filepath, gesture_col):
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    n_cols = max((len(r) for r in rows))
    data = np.full((len(rows), n_cols), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try:
                data[i, j] = float(v)
            except ValueError:
                pass
    label_raw = data[:, gesture_col]
    y = np.array(
        [
            OPP_GESTURE_MAP.get(0 if np.isnan(v) else int(round(v)), 0)
            for v in label_raw
        ],
        dtype=np.int64,
    )
    cols_del = [c for c in OPP_COLS_TO_DELETE if c < n_cols]
    X = np.delete(data, cols_del, axis=1)[:, OPP_FEATURE_SLICE].astype(np.float32)
    return (impute_nans(X), y)


def build_opportunity(output_path="./dataset/opportunity.mat"):
    print("\n" + "=" * 60 + "\n[Opportunity] Starting...")
    download_file(OPP_URL, OPP_DEST, "Opportunity UCI dataset (~270 MB)")
    if not zipfile.is_zipfile(OPP_DEST):
        raise RuntimeError(f"{OPP_DEST} is not a valid zip.")
    unzip(OPP_DEST, OPP_DIR)
    opp_data_dir = None
    for root, dirs, files in os.walk(OPP_DIR):
        if any((f.endswith(".dat") for f in files)):
            opp_data_dir = root
            break
    probe = os.path.join(opp_data_dir, "S1-Drill.dat")
    gesture_col = find_gesture_col(probe)
    if gesture_col is None:
        raise RuntimeError("Could not find gesture label column.")
    print(f"  Gesture label at raw column {gesture_col}")

    def load_split(flist, name):
        Xs, ys = ([], [])
        for fname in flist:
            fp = os.path.join(opp_data_dir, fname)
            if not os.path.exists(fp):
                continue
            X, y = load_opp_file(fp, gesture_col)
            Xs.append(X)
            ys.append(y)
        return (np.concatenate(Xs, 0), np.concatenate(ys, 0))

    X_train, y_train = load_split(OPP_TRAIN_FILES, "train")
    X_val, y_val = load_split(OPP_VAL_FILES, "val")
    X_test, y_test = load_split(OPP_TEST_FILES, "test")
    mean = X_train.mean(0)
    std = X_train.std(0)
    std[std == 0] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sio.savemat(
        output_path,
        {
            "trainingData": X_train.T,
            "trainingLabels": (y_train + 1).reshape(1, -1),
            "valData": X_val.T,
            "valLabels": (y_val + 1).reshape(1, -1),
            "testingData": X_test.T,
            "testingLabels": (y_test + 1).reshape(1, -1),
        },
    )
    print(f"[Opportunity] Saved -> {output_path}")


SKODA_URLS = [
    "http://www.ife.ee.ethz.ch/uploads/tx_ethpublications/skoda.zip",
    "https://zenodo.org/records/3932973/files/skoda.zip?download=1",
]
SKODA_DEST = "./downloads/skoda.zip"
SKODA_DIR = "./downloads/skoda"
SKODA_LABEL_MAP = {
    32: 0,
    48: 1,
    49: 2,
    50: 3,
    51: 4,
    52: 5,
    53: 6,
    54: 7,
    55: 8,
    56: 9,
    57: 10,
}


def build_skoda(output_path="./dataset/skoda.mat"):
    print("\n" + "=" * 60 + "\n[Skoda] Starting...")
    if not (os.path.exists(SKODA_DEST) and os.path.getsize(SKODA_DEST) > 10000):
        for url in SKODA_URLS:
            try:
                download_file(url, SKODA_DEST, "Skoda dataset")
                if os.path.exists(SKODA_DEST) and zipfile.is_zipfile(SKODA_DEST):
                    break
                if os.path.exists(SKODA_DEST):
                    os.remove(SKODA_DEST)
            except:
                pass
    unzip(SKODA_DEST, SKODA_DIR)
    skoda_file = None
    for root, _, files in os.walk(SKODA_DIR):
        for fname in sorted(files):
            if fname.endswith(".dat") or fname.endswith(".csv"):
                skoda_file = os.path.join(root, fname)
                break
        if skoda_file:
            break
    rows = []
    with open(skoda_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    rows.append([float(v) for v in parts])
                except ValueError:
                    pass
    data = np.array(rows, dtype=np.float32)
    n_feat = min(60, data.shape[1] - 1)
    X = impute_nans(data[:, :n_feat].copy())
    y = np.array([SKODA_LABEL_MAP.get(int(v), 0) for v in data[:, -1]], dtype=np.int64)
    n = len(X)
    t1, t2 = (int(n * 0.6), int(n * 0.8))
    mean = X[:t1].mean(0)
    std = X[:t1].std(0)
    std[std == 0] = 1.0
    X = ((X - mean) / std).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sio.savemat(
        output_path,
        {
            "X_train": X[:t1],
            "y_train": y[:t1].reshape(-1, 1),
            "X_valid": X[t1:t2],
            "y_valid": y[t1:t2].reshape(-1, 1),
            "X_test": X[t2:],
            "y_test": y[t2:].reshape(-1, 1),
        },
    )
    print(f"[Skoda] Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="all", choices=["all", "opportunity", "pamap2", "skoda"]
    )
    parser.add_argument("--output_dir", default="./dataset")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset in ("all", "opportunity"):
        build_opportunity(os.path.join(args.output_dir, "opportunity.mat"))
    if args.dataset in ("all", "pamap2"):
        build_pamap2(os.path.join(args.output_dir, "pamap2.mat"))
    if args.dataset in ("all", "skoda"):
        build_skoda(os.path.join(args.output_dir, "skoda.mat"))
    print("\n" + "=" * 60 + "\nSummary:")
    for name in ["opportunity", "pamap2", "skoda"]:
        path = os.path.join(args.output_dir, f"{name}.mat")
        if os.path.exists(path):
            print(f"  ✓ {path}  ({os.path.getsize(path) / 1000000.0:.1f} MB)")
        else:
            print(f"  ✗ {path}  (not created)")


if __name__ == "__main__":
    main()
