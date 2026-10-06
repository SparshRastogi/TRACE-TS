import argparse
import os
import zipfile
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import warnings

SAMPLE_RATE = 100
DATAFILES_PAT = "capture24/P[0-9][0-9][0-9].csv.gz"
ANNOFILE = "capture24/annotation-label-dictionary.csv"
WALMSLEY_MAP = {"sleep": 0, "sedentary": 1, "light": 2, "moderate-vigorous": 3}
WALMSLEY_MAP_INV = {v: k for k, v in WALMSLEY_MAP.items()}
WILLETTS_MAP = {
    "sleep": 0,
    "sit-stand": 1,
    "walking": 2,
    "bicycling": 3,
    "mixed": 4,
    "vehicle": 5,
}
WILLETTS_MAP_INV = {v: k for k, v in WILLETTS_MAP.items()}
LABEL_SCHEMAS = {
    "Walmsley2020": ("Walmsley2020", WALMSLEY_MAP, WALMSLEY_MAP_INV),
    "Willetts2018": ("Willetts2018", WILLETTS_MAP, WILLETTS_MAP_INV),
}
DEFAULT_LABEL_SCHEMA = "Willetts2018"
DEFAULT_N_SUBJECTS = 100
TRAIN_FRAC = 0.7
VAL_FRAC = 0.2


def download_capture24(datadir: str, overwrite: bool = False) -> None:
    import requests

    zip_path = os.path.join(datadir, "capture24.zip")
    capture24dir = os.path.join(datadir, "capture24")
    if overwrite or not os.path.exists(zip_path):
        os.makedirs(datadir, exist_ok=True)
        url = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"
        print(f"[*] Downloading Capture-24 (~6.9 GB) → {zip_path}")
        print("    This will take a while on a slow connection.")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = 100.0 * downloaded / total_size
                        mb = downloaded / 1000000.0
                        print(f"\r    {pct:5.1f}%  ({mb:.0f} MB)", end="", flush=True)
        print()
    else:
        print(f"[*] Found existing archive: {zip_path}")
    existing = glob(os.path.join(datadir, DATAFILES_PAT))
    if overwrite or len(existing) < 151:
        print(f"[*] Unzipping to {capture24dir} …")
        os.makedirs(capture24dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in tqdm(zf.namelist(), desc="Unzipping"):
                try:
                    zf.extract(member, datadir)
                except zipfile.BadZipFile:
                    pass
    else:
        print(f"[*] Capture-24 already extracted ({len(existing)} participants found).")


def _load_participant(datafile: str) -> pd.DataFrame:
    df = pd.read_csv(
        datafile,
        index_col="time",
        parse_dates=["time"],
        dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": "string"},
    )
    return df


def _is_good_window(x: np.ndarray, win_len: int) -> bool:
    return len(x) == win_len and (not np.isnan(x).any())


def _process_participant(
    datafile: str,
    anno_df: pd.DataFrame,
    winsec: int,
    sample_rate: int,
    anno_col: str = "Walmsley2020",
    label_map: dict = None,
) -> tuple:
    if label_map is None:
        label_map = WALMSLEY_MAP
    win_len = winsec * sample_rate
    step = win_len // 2
    try:
        data = _load_participant(datafile)
    except Exception as e:
        print(f"[WARN] Could not load {datafile}: {e}")
        return (
            np.empty((0, win_len, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            "",
        )
    label_lookup: dict = {}
    for raw_annot, row in anno_df.iterrows():
        label_str = str(row[f"label:{anno_col}"]).strip().lower()
        if label_str in label_map:
            label_lookup[raw_annot] = label_map[label_str]
    xyz = data[["x", "y", "z"]].to_numpy(dtype=np.float32)
    annots = data["annotation"].to_numpy()
    n_samples = len(xyz)
    X_list, y_list = ([], [])
    start = 0
    while start + win_len <= n_samples:
        end = start + win_len
        x_seg = xyz[start:end]
        a_seg = annots[start:end]
        if not _is_good_window(x_seg, win_len):
            start += step
            continue
        valid_mask = ~pd.isna(a_seg)
        if not valid_mask.any():
            start += step
            continue
        valid_annots = a_seg[valid_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            majority_raw = pd.Series(valid_annots).mode().iloc[0]
        if majority_raw not in label_lookup:
            start += step
            continue
        X_list.append(x_seg)
        y_list.append(label_lookup[majority_raw])
        start += step
    pid_str = Path(datafile).name.split(".")[0]
    if len(X_list) == 0:
        return (
            np.empty((0, win_len, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            pid_str,
        )
    X_out = np.stack(X_list, axis=0).astype(np.float32)
    y_out = np.array(y_list, dtype=np.int64)
    return (X_out, y_out, pid_str)


def build_npz(
    datadir: str,
    outdir: str,
    winsec: int = 2,
    n_jobs: int = 8,
    overwrite: bool = False,
    n_subjects: int = DEFAULT_N_SUBJECTS,
    label_schema: str = DEFAULT_LABEL_SCHEMA,
) -> None:
    if label_schema not in LABEL_SCHEMAS:
        raise ValueError(
            f"Unknown label_schema '{label_schema}'. Choose from: {list(LABEL_SCHEMAS.keys())}"
        )
    anno_col, label_map, label_map_inv = LABEL_SCHEMAS[label_schema]
    n_classes = len(label_map)
    npz_path = os.path.join(outdir, f"capture24_{label_schema}.npz")
    if not overwrite and os.path.exists(npz_path):
        print(f"[*] {npz_path} already exists. Use --overwrite to rebuild.")
        return
    print(
        f"[*] Using label schema: {label_schema} ({n_classes} classes: {list(label_map.keys())})"
    )
    all_datafiles = sorted(glob(os.path.join(datadir, DATAFILES_PAT)))
    if len(all_datafiles) == 0:
        raise FileNotFoundError(
            f"No participant files found matching {os.path.join(datadir, DATAFILES_PAT)}.\nCheck that the dataset was downloaded and extracted correctly."
        )
    if n_subjects > len(all_datafiles):
        print(
            f"[WARN] Requested {n_subjects} subjects but only {len(all_datafiles)} found. Using all {len(all_datafiles)}."
        )
        n_subjects = len(all_datafiles)
    datafiles = all_datafiles[:n_subjects]
    first_pid = Path(datafiles[0]).name.split(".")[0]
    last_pid = Path(datafiles[-1]).name.split(".")[0]
    print(f"[*] Found {len(all_datafiles)} participant files total.")
    print(f"[*] Using first {len(datafiles)} subjects ({first_pid} – {last_pid}).")
    anno_path = os.path.join(datadir, ANNOFILE)
    anno_df = pd.read_csv(anno_path, index_col="annotation", dtype=str)
    print(f"[*] Loaded annotation dictionary: {anno_path}")
    print(f"[*] Windowing (winsec={winsec}s, 50% overlap, {n_jobs} workers) …")
    results = Parallel(n_jobs=n_jobs)(
        (
            delayed(_process_participant)(
                f, anno_df, winsec, SAMPLE_RATE, anno_col=anno_col, label_map=label_map
            )
            for f in tqdm(datafiles, desc="Participants")
        )
    )
    pid_to_X: dict = {}
    pid_to_y: dict = {}
    for X_p, y_p, pid in results:
        if len(X_p) == 0:
            print(f"[WARN] No windows for {pid} — skipping.")
            continue
        pid_to_X[pid] = X_p
        pid_to_y[pid] = y_p
        print(f"  {pid}: {len(X_p):6d} windows, labels {np.unique(y_p).tolist()}")
    all_pids = sorted(pid_to_X.keys())
    n_valid = len(all_pids)
    n_train = int(round(n_valid * TRAIN_FRAC))
    n_val = int(round(n_valid * VAL_FRAC))
    n_test = n_valid - n_train - n_val
    train_pids = all_pids[:n_train]
    val_pids = all_pids[n_train : n_train + n_val]
    test_pids = all_pids[n_train + n_val :]
    print(f"\n[*] Participant-level split ({n_valid} subjects total):")
    print(
        f"    Train : {len(train_pids):3d} subjects  ({train_pids[0]} – {train_pids[-1]})"
    )
    print(f"    Val   : {len(val_pids):3d} subjects  ({val_pids[0]} – {val_pids[-1]})")
    print(
        f"    Test  : {len(test_pids):3d} subjects  ({test_pids[0]} – {test_pids[-1]})"
    )

    def _concat(pids):
        X = np.concatenate([pid_to_X[p] for p in pids if p in pid_to_X], axis=0)
        y = np.concatenate([pid_to_y[p] for p in pids if p in pid_to_y], axis=0)
        return (X, y)

    X_train, y_train = _concat(train_pids)
    X_valid, y_valid = _concat(val_pids)
    X_test, y_test = _concat(test_pids)
    N_tr, T, C = X_train.shape
    X_flat = X_train.reshape(-1, C)
    mean_train = X_flat.mean(axis=0)
    std_train = X_flat.std(axis=0)
    std_train[std_train == 0] = 1.0
    X_train = ((X_train.reshape(-1, C) - mean_train) / std_train).reshape(N_tr, T, C)
    X_valid = ((X_valid.reshape(-1, C) - mean_train) / std_train).reshape(X_valid.shape)
    X_test = ((X_test.reshape(-1, C) - mean_train) / std_train).reshape(X_test.shape)
    print(
        f"\n[*] Per-channel means after normalisation: {X_train.reshape(-1, C).mean(axis=0).round(4).tolist()}"
    )
    print(
        f"[*] Per-channel stds  after normalisation: {X_train.reshape(-1, C).std(axis=0).round(4).tolist()}"
    )
    print()
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        counts = {
            label_map_inv[i]: int((y == i).sum())
            for i in range(n_classes)
            if int((y == i).sum()) > 0
        }
        print(f"  {split:5s}: X={X.shape}, y={y.shape}  labels={counts}")
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        n_subjects=np.array(n_subjects),
        n_train_subjects=np.array(len(train_pids)),
        n_val_subjects=np.array(len(val_pids)),
        n_test_subjects=np.array(len(test_pids)),
        mean_train=mean_train,
        std_train=std_train,
        label_schema=np.array(label_schema),
        n_classes=np.array(n_classes),
    )
    size_mb = os.path.getsize(npz_path) / 1000000.0
    print(f"\n[+] Saved → {npz_path}  ({size_mb:.1f} MB)")
    stats_path = os.path.join(outdir, f"capture24_{label_schema}_norm_stats.npz")
    np.savez(stats_path, mean=mean_train, std=std_train)
    print(f"[+] Normalisation stats → {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Capture-24 and build capture24.npz for the HAR pipeline.\nOnly the first --n_subjects participants are used (P001 upward),\nsplit 70 / 20 / 10 by participant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datadir", default="data", help="where to store the raw download"
    )
    parser.add_argument(
        "--outdir", default="dataset", help="where to write capture24.npz"
    )
    parser.add_argument(
        "--winsec", default=2, type=int, help="window length in seconds"
    )
    parser.add_argument("--n_jobs", default=8, type=int, help="parallel workers")
    parser.add_argument(
        "--n_subjects",
        default=100,
        type=int,
        help="how many subjects to include (taken from P001 upward)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-process and overwrite existing capture24.npz",
    )
    parser.add_argument(
        "--label_schema",
        default=DEFAULT_LABEL_SCHEMA,
        choices=list(LABEL_SCHEMAS.keys()),
        help="Which activity label schema to use. 'Walmsley2020' = 4-class (sleep/sedentary/light/moderate-vigorous). 'Willetts2018' = 6-class (sleep/sitting/standing/walking/bicycling/mixed). Default: Willetts2018.",
    )
    args = parser.parse_args()
    download_capture24(args.datadir, args.overwrite)
    build_npz(
        args.datadir,
        args.outdir,
        winsec=args.winsec,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
        n_subjects=args.n_subjects,
        label_schema=args.label_schema,
    )


if __name__ == "__main__":
    main()
