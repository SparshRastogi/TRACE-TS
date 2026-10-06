import argparse
import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from AtteFinalPipeline.data.prepare_capture24 import (
    _process_participant,
    SAMPLE_RATE,
    DATAFILES_PAT,
    ANNOFILE,
    LABEL_SCHEMAS,
    DEFAULT_LABEL_SCHEMA,
    TRAIN_FRAC,
    VAL_FRAC,
    DEFAULT_N_SUBJECTS,
)


def _stratified_subsample_indices(y, fraction, floor, cap, rng):
    chosen = []
    for cls in np.unique(y):
        cls_idxs = np.where(y == cls)[0]
        n_cell = len(cls_idxs)
        if n_cell == 0:
            continue
        n_take = int(round(fraction * n_cell))
        n_take = max(n_take, floor)
        n_take = min(n_take, cap)
        n_take = min(n_take, n_cell)
        picked = rng.choice(cls_idxs, size=n_take, replace=False)
        chosen.append(picked)
    if not chosen:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(chosen))


def _load_full_norm_stats(full_npz_path):
    if not full_npz_path or not os.path.exists(full_npz_path):
        return (None, None)
    try:
        with np.load(full_npz_path) as f:
            keys = f.files
            if "mean_train" in keys and "std_train" in keys:
                mean = f["mean_train"].astype(np.float64)
                std = f["std_train"].astype(np.float64)
                if mean.shape == (3,) and std.shape == (3,):
                    return (mean, std)
                print(
                    f"[WARN] {full_npz_path} has mean_train/std_train but with unexpected shape mean={mean.shape}, std={std.shape} — falling back to recomputing."
                )
            else:
                print(
                    f"[WARN] {full_npz_path} does not contain mean_train/std_train keys — falling back to recomputing."
                )
    except Exception as e:
        print(f"[WARN] Could not read {full_npz_path}: {e} — recomputing stats.")
    return (None, None)


def _compute_norm_stats_from_scratch(pid_to_X, train_pids):
    print(
        "[*] Recomputing train mean/std from full train participants (this reproduces prepare_capture24.py's stats bit-for-bit)."
    )
    X_full_train = np.concatenate(
        [pid_to_X[p] for p in train_pids if p in pid_to_X], axis=0
    )
    C = X_full_train.shape[-1]
    flat = X_full_train.reshape(-1, C).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return (mean, std)


def build_subset_npz(
    datadir,
    outdir,
    winsec,
    n_jobs,
    n_subjects,
    fraction,
    floor,
    cap,
    seed,
    overwrite,
    full_npz_path,
    label_schema=DEFAULT_LABEL_SCHEMA,
):
    if label_schema not in LABEL_SCHEMAS:
        raise ValueError(
            f"Unknown label_schema '{label_schema}'. Choose from: {list(LABEL_SCHEMAS.keys())}"
        )
    anno_col, label_map, label_map_inv = LABEL_SCHEMAS[label_schema]
    n_classes = len(label_map)
    npz_path = os.path.join(outdir, f"capture24_{label_schema}_subset.npz")
    if not overwrite and os.path.exists(npz_path):
        print(f"[*] {npz_path} already exists. Use --overwrite to rebuild.")
        return
    print(f"[*] Building {label_schema} ({n_classes}-class) 10%% subset → {npz_path}")
    all_datafiles = sorted(glob(os.path.join(datadir, DATAFILES_PAT)))
    if len(all_datafiles) == 0:
        raise FileNotFoundError(
            f"No participant files found matching {os.path.join(datadir, DATAFILES_PAT)}. Run prepare_capture24.py --datadir {datadir} first (downloads + extracts the archive)."
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
    pid_to_X, pid_to_y = ({}, {})
    for X_p, y_p, pid in results:
        if len(X_p) == 0:
            print(f"[WARN] No windows for {pid} — skipping.")
            continue
        pid_to_X[pid] = X_p
        pid_to_y[pid] = y_p
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
    print(
        f"\n[*] Subsampling: fraction={fraction}, floor={floor}, cap={cap}, seed={seed}"
    )
    master_rng = np.random.default_rng(seed)

    def _sample_split(split_name, pids):
        X_parts, y_parts, sid_parts = ([], [], [])
        total_before = 0
        total_after = 0
        per_cell_log = []
        for pid in pids:
            if pid not in pid_to_X:
                continue
            Xp = pid_to_X[pid]
            yp = pid_to_y[pid]
            total_before += len(yp)
            pid_seed = int(master_rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(pid_seed)
            idx = _stratified_subsample_indices(
                yp, fraction=fraction, floor=floor, cap=cap, rng=sub_rng
            )
            if len(idx) == 0:
                continue
            X_parts.append(Xp[idx])
            y_parts.append(yp[idx])
            sid_parts.append(np.array([pid] * len(idx), dtype="<U4"))
            total_after += len(idx)
            for cls in np.unique(yp):
                n_before = int(np.sum(yp == cls))
                n_after = int(np.sum(yp[idx] == cls))
                per_cell_log.append((pid, label_map_inv[int(cls)], n_before, n_after))
        if len(X_parts) == 0:
            print(f"[WARN] {split_name}: no windows sampled!")
            return (
                np.empty((0, winsec * SAMPLE_RATE, 3), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype="<U4"),
                total_before,
                0,
                per_cell_log,
            )
        X_sub = np.concatenate(X_parts, axis=0).astype(np.float32)
        y_sub = np.concatenate(y_parts, axis=0).astype(np.int64)
        sid_sub = np.concatenate(sid_parts, axis=0)
        return (X_sub, y_sub, sid_sub, total_before, total_after, per_cell_log)

    X_train, y_train, sid_train, tb_tr, ta_tr, log_tr = _sample_split(
        "train", train_pids
    )
    X_valid, y_valid, sid_valid, tb_va, ta_va, log_va = _sample_split("val", val_pids)
    X_test, y_test, sid_test, tb_te, ta_te, log_te = _sample_split("test", test_pids)
    mean_train, std_train = _load_full_norm_stats(full_npz_path)
    if mean_train is None:
        mean_train, std_train = _compute_norm_stats_from_scratch(pid_to_X, train_pids)
    else:
        print(f"[*] Reusing mean_train/std_train from {full_npz_path}")
    print(f"    mean_train = {mean_train.round(4).tolist()}")
    print(f"    std_train  = {std_train.round(4).tolist()}")

    def _zscore(X):
        if X.size == 0:
            return X
        N, T, C = X.shape
        flat = X.reshape(-1, C).astype(np.float32)
        flat = (flat - mean_train.astype(np.float32)) / std_train.astype(np.float32)
        return flat.reshape(N, T, C)

    X_train = _zscore(X_train)
    X_valid = _zscore(X_valid)
    X_test = _zscore(X_test)
    if X_train.size > 0:
        chk_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        chk_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
        print(
            f"    post-norm TRAIN subset per-channel mean: {chk_mean.round(4).tolist()}  (should be near 0)"
        )
        print(
            f"    post-norm TRAIN subset per-channel std:  {chk_std.round(4).tolist()}   (should be near 1)"
        )
    print(
        f"\n[*] Subset sizes (after {fraction * 100:.1f}% stratified sampling, floor={floor}, cap={cap}):"
    )
    for name, tb, ta, X, y in [
        ("train", tb_tr, ta_tr, X_train, y_train),
        ("valid", tb_va, ta_va, X_valid, y_valid),
        ("test", tb_te, ta_te, X_test, y_test),
    ]:
        if tb > 0:
            pct = 100.0 * ta / tb
        else:
            pct = 0.0
        counts = {
            label_map_inv[i]: int((y == i).sum())
            for i in range(n_classes)
            if int((y == i).sum()) > 0
        }
        print(
            f"    {name:5s}: {ta:6d} / {tb:7d} windows  ({pct:5.2f}%)   X={X.shape}  labels={counts}"
        )
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        subject_ids_train=sid_train,
        X_valid=X_valid,
        y_valid=y_valid,
        subject_ids_valid=sid_valid,
        X_test=X_test,
        y_test=y_test,
        subject_ids_test=sid_test,
        mean_train=mean_train,
        std_train=std_train,
        n_subjects=np.array(n_valid),
        n_train_subjects=np.array(len(train_pids)),
        n_val_subjects=np.array(len(val_pids)),
        n_test_subjects=np.array(len(test_pids)),
        label_schema=np.array(label_schema),
        n_classes=np.array(n_classes),
        subset_fraction=np.array(fraction),
        subset_cap_per_cell=np.array(cap),
        subset_floor=np.array(floor),
        subset_seed=np.array(seed),
    )
    size_mb = os.path.getsize(npz_path) / 1000000.0
    print(f"\n[+] Saved → {npz_path}  ({size_mb:.1f} MB)")
    log_csv = os.path.join(outdir, f"capture24_{label_schema}_subset_sampling_log.csv")
    rows = []
    for split_name, log in [("train", log_tr), ("valid", log_va), ("test", log_te)]:
        for pid, cls_name, nb, na in log:
            rows.append(
                {
                    "split": split_name,
                    "subject": pid,
                    "class": cls_name,
                    "n_before": nb,
                    "n_after": na,
                    "kept_pct": round(100.0 * na / nb, 2) if nb > 0 else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(log_csv, index=False)
    print(f"[+] Per-cell sampling log → {log_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a (subject, class)-stratified 10% subset of Capture-24 for attribution/embedding use. Reuses the classifier's training normalisation stats so XAI results stay faithful.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datadir",
        default="data",
        help="folder containing the unzipped capture24/ data (same as prepare_capture24.py --datadir)",
    )
    parser.add_argument(
        "--outdir",
        default="dataset",
        help="where to write capture24_{schema}_subset.npz",
    )
    parser.add_argument(
        "--label_schema",
        default=DEFAULT_LABEL_SCHEMA,
        choices=list(LABEL_SCHEMAS.keys()),
        help="Label schema to use. 'Walmsley2020' = 4-class. 'Willetts2018' = 6-class (default). Must match the schema used when training the model.",
    )
    parser.add_argument(
        "--full_npz",
        default=None,
        help="Path to the full capture24_{schema}.npz whose mean_train/std_train will be reused for normalisation. If omitted, defaults to dataset/capture24_{label_schema}.npz. If missing, stats are recomputed from scratch.",
    )
    parser.add_argument("--winsec", type=int, default=2)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=DEFAULT_N_SUBJECTS,
        help="first N subjects to include (P001 upward)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="fraction of each (subject, class) cell to keep",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=1,
        help="minimum samples per non-empty (subject, class) cell",
    )
    parser.add_argument(
        "--cap", type=int, default=500, help="maximum samples per (subject, class) cell"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducible sampling"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    full_npz = args.full_npz
    if full_npz is None:
        full_npz = os.path.join(args.outdir, f"capture24_{args.label_schema}.npz")
    build_subset_npz(
        datadir=args.datadir,
        outdir=args.outdir,
        winsec=args.winsec,
        n_jobs=args.n_jobs,
        n_subjects=args.n_subjects,
        fraction=args.fraction,
        floor=args.floor,
        cap=args.cap,
        seed=args.seed,
        overwrite=args.overwrite,
        full_npz_path=full_npz,
        label_schema=args.label_schema,
    )


if __name__ == "__main__":
    main()
