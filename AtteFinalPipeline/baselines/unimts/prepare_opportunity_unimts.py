import argparse
import os
import zipfile
import subprocess
import numpy as np
import scipy.io as sio
from tqdm import tqdm

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
UNIMTS_TRAIN_FILES = [
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
UNIMTS_VAL_FILES = ["S2-ADL4.dat", "S2-ADL5.dat"]
UNIMTS_TEST_FILES = [
    "S4-ADL1.dat",
    "S4-ADL2.dat",
    "S4-ADL3.dat",
    "S4-ADL4.dat",
    "S4-ADL5.dat",
    "S4-Drill.dat",
]
AD_TRAIN_FILES = [
    "S1-ADL1.dat",
    "S1-ADL2.dat",
    "S1-ADL3.dat",
    "S1-ADL4.dat",
    "S1-ADL5.dat",
    "S1-Drill.dat",
    "S2-ADL1.dat",
    "S2-ADL2.dat",
    "S2-Drill.dat",
    "S3-ADL1.dat",
    "S3-ADL2.dat",
    "S3-Drill.dat",
    "S4-ADL1.dat",
    "S4-ADL2.dat",
    "S4-ADL3.dat",
    "S4-ADL4.dat",
    "S4-ADL5.dat",
    "S4-Drill.dat",
]
AD_VAL_FILES = ["S2-ADL3.dat", "S3-ADL3.dat"]
AD_TEST_FILES = ["S2-ADL4.dat", "S2-ADL5.dat", "S3-ADL4.dat", "S3-ADL5.dat"]
OPP_CLASS_NAMES = [
    "null",
    "open door 1",
    "open door 2",
    "close door 1",
    "close door 2",
    "open fridge",
    "close fridge",
    "open dishwasher",
    "close dishwasher",
    "open drawer 1",
    "close drawer 1",
    "open drawer 2",
    "close drawer 2",
    "open drawer 3",
    "close drawer 3",
    "clean table",
    "drink from cup",
    "toggle switch",
]
OPP_IMU_BLOCKS = [
    (0, 3, 3, 6),
    (9, 12, 12, 15),
    (18, 21, 21, 24),
    (27, 30, 30, 33),
    (36, 39, 39, 42),
]
OPP_JOINTS = [16, 20, 15, 10, 19]
OPP_ACC_TO_MS2 = 9.8 / 1000.0
OPP_GYRO_TO_RADS = 1.0 / 1000.0
OPP_NULL_LABEL = 0


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

    warnings.filterwarnings("ignore")
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


def _drop_null_rows(X, y):
    keep = y != OPP_NULL_LABEL
    n_dropped = int((~keep).sum())
    X_f = X[keep]
    y_f = (y[keep] - 1).astype(np.int64)
    return (X_f, y_f, n_dropped, int(keep.sum()))


def get_file_lists(split_choice):
    if split_choice == "unimts":
        return (UNIMTS_TRAIN_FILES, UNIMTS_VAL_FILES, UNIMTS_TEST_FILES)
    elif split_choice == "attend_disc":
        return (AD_TRAIN_FILES, AD_VAL_FILES, AD_TEST_FILES)
    else:
        raise ValueError(f"Unknown split: {split_choice}")


def get_output_path(split_choice, drop_null):
    if split_choice == "unimts":
        base = "opportunity"
    elif split_choice == "attend_disc":
        base = "opportunity_attenddisc"
    else:
        raise ValueError(f"Unknown split: {split_choice}")
    if drop_null:
        return f"./dataset/{base}_nonull_unimts.mat"
    return f"./dataset/{base}_unimts.mat"


def build_opportunity_unimts(split_choice="unimts", output_path=None, drop_null=False):
    train_files, val_files, test_files = get_file_lists(split_choice)
    if output_path is None:
        output_path = get_output_path(split_choice, drop_null)
    tag_split = f"split={split_choice}"
    tag_null = " (NULL DROPPED, 17-class)" if drop_null else " (18-class with null)"
    print("\n" + "=" * 60)
    print(f"[Opportunity / UniMTS] {tag_split}{tag_null}")
    print(f"  train files: {len(train_files)}")
    print(f"  val   files: {len(val_files)}")
    print(f"  test  files: {len(test_files)}")
    print(f"  output:      {output_path}")
    print("=" * 60)
    download_file(OPP_URL, OPP_DEST, "Opportunity (~270 MB)")
    if not zipfile.is_zipfile(OPP_DEST):
        raise RuntimeError(f"{OPP_DEST} not a valid zip.")
    unzip(OPP_DEST, OPP_DIR)
    opp_data_dir = None
    for root, dirs, files in os.walk(OPP_DIR):
        if any((f.endswith(".dat") for f in files)):
            opp_data_dir = root
            break
    if opp_data_dir is None:
        raise RuntimeError(f"No .dat files found under {OPP_DIR}")
    probe = os.path.join(opp_data_dir, "S1-Drill.dat")
    gesture_col = find_gesture_col(probe)
    if gesture_col is None:
        raise RuntimeError("Could not find gesture label column.")
    print(f"  Gesture label at raw column {gesture_col}")

    def load_split(flist, name):
        Xs, ys = ([], [])
        missing = []
        for fname in flist:
            fp = os.path.join(opp_data_dir, fname)
            if not os.path.exists(fp):
                missing.append(fname)
                continue
            X, y = load_opp_file(fp, gesture_col)
            print(f"    {name:5s}  {fname:14s}  N={X.shape[0]}")
            Xs.append(X)
            ys.append(y)
        if missing:
            print(f"  [!] {name}: missing files {missing}")
        if not Xs:
            raise RuntimeError(f"No data loaded for split '{name}'.")
        return (np.concatenate(Xs, 0), np.concatenate(ys, 0))

    print("\n  Loading files:")
    X_train, y_train = load_split(train_files, "train")
    X_val, y_val = load_split(val_files, "val")
    X_test, y_test = load_split(test_files, "test")
    for X in (X_train, X_val, X_test):
        for acc_lo, acc_hi, gy_lo, gy_hi in OPP_IMU_BLOCKS:
            X[:, acc_lo:acc_hi] *= OPP_ACC_TO_MS2
            X[:, gy_lo:gy_hi] *= OPP_GYRO_TO_RADS
    if drop_null:
        print("\n  Dropping null rows (label == 0) and remapping 1..17 → 0..16:")
        X_train, y_train, drT, kpT = _drop_null_rows(X_train, y_train)
        X_val, y_val, drV, kpV = _drop_null_rows(X_val, y_val)
        X_test, y_test, drTe, kpTe = _drop_null_rows(X_test, y_test)
        print(f"    train: dropped {drT}, kept {kpT}")
        print(f"    val:   dropped {drV}, kept {kpV}")
        print(f"    test:  dropped {drTe}, kept {kpTe}")
        class_names = OPP_CLASS_NAMES[1:]
        assert len(class_names) == 17
    else:
        class_names = OPP_CLASS_NAMES
    print(f"\n  Final shapes (physical units, no z-score):")
    print(f"    train: {X_train.shape}")
    print(f"    val:   {X_val.shape}")
    print(f"    test:  {X_test.shape}")
    if X_train.size:
        print(
            f"    RUA acc magnitude (col 0): |mean|={abs(X_train[:, 0].mean()):.2f} m/s²  (expect a few)"
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    provenance = f"opportunity_unimts split={split_choice} drop_null={drop_null}; train={len(train_files)} val={len(val_files)} test={len(test_files)} files"
    sio.savemat(
        output_path,
        {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_valid": X_val,
            "y_valid": y_val.reshape(-1, 1),
            "X_test": X_test,
            "y_test": y_test.reshape(-1, 1),
            "sampling_rate": 30,
            "imu_blocks": np.array(OPP_IMU_BLOCKS),
            "joints": np.array(OPP_JOINTS),
            "class_names": np.array(class_names, dtype=object),
            "null_dropped": np.array([1 if drop_null else 0]),
            "_split": np.array(split_choice, dtype=object),
            "_provenance": np.array(provenance, dtype=object),
        },
    )
    print(f"\n[+] Saved -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Prepare Opportunity for UniMTS fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--split",
        choices=["unimts", "attend_disc"],
        default="unimts",
        help="Which train/val/test split to use. 'unimts' keeps the original UniMTS cross-subject S4-test split. 'attend_disc' uses the Opportunity challenge partition that opportunity.mat (and the A&D checkpoint) was built with — test = S2/S3-ADL4/5. Use this for direct ablation against the A&D classifier on the same data partition.",
    )
    p.add_argument(
        "--drop_null",
        action="store_true",
        help="Filter out null-class rows and remap labels 1..17 → 0..16. Writes to a *_nonull_*.mat path so the default 18-class artefact is preserved. The raw downloaded data is never modified.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override the default output path. Default is derived from --split and --drop_null.",
    )
    args = p.parse_args()
    build_opportunity_unimts(
        split_choice=args.split, output_path=args.output, drop_null=args.drop_null
    )
