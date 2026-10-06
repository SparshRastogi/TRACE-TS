import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from AtteFinalPipeline.data.prepare_capture24 import (
    download_capture24,
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

G_TO_MS2 = 9.80665
CAPTURE24_JOINTS = [21]


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


def _sample_split(
    split_name,
    pids,
    pid_to_X,
    pid_to_y,
    label_map_inv,
    n_classes,
    winsec,
    fraction,
    floor,
    cap,
    master_rng,
):
    X_parts, y_parts, sid_parts = ([], [], [])
    total_before, total_after = (0, 0)
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
    if not X_parts:
        print(f"[WARN] {split_name}: no windows sampled!")
        win_len = winsec * SAMPLE_RATE
        return (
            np.empty((0, win_len, 3), dtype=np.float32),
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


def build_unimts_npz(
    datadir,
    outdir,
    label_schema,
    winsec,
    n_jobs,
    n_subjects,
    fraction,
    floor,
    cap,
    seed,
    overwrite,
):
    if label_schema not in LABEL_SCHEMAS:
        raise ValueError(f"Unknown schema {label_schema}")
    anno_col, label_map, label_map_inv = LABEL_SCHEMAS[label_schema]
    n_classes = len(label_map)
    out_path = os.path.join(outdir, f"capture24_{label_schema}_unimts.npz")
    if not overwrite and os.path.exists(out_path):
        print(f"[*] {out_path} already exists. Use --overwrite to rebuild.")
        return
    download_capture24(datadir, overwrite=False)
    all_files = sorted(glob(os.path.join(datadir, DATAFILES_PAT)))
    if not all_files:
        raise FileNotFoundError(f"No participant files in {datadir}")
    if n_subjects > len(all_files):
        n_subjects = len(all_files)
    files = all_files[:n_subjects]
    anno_path = os.path.join(datadir, ANNOFILE)
    anno_df = pd.read_csv(anno_path, index_col="annotation", dtype=str)
    print(f"[*] Windowing {len(files)} subjects with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs)(
        (
            delayed(_process_participant)(
                f, anno_df, winsec, SAMPLE_RATE, anno_col=anno_col, label_map=label_map
            )
            for f in tqdm(files, desc="participants")
        )
    )
    pid_to_X, pid_to_y = ({}, {})
    for X_p, y_p, pid in results:
        if len(X_p) > 0:
            pid_to_X[pid] = X_p
            pid_to_y[pid] = y_p
    all_pids = sorted(pid_to_X.keys())
    n_valid = len(all_pids)
    n_train = int(round(n_valid * TRAIN_FRAC))
    n_val = int(round(n_valid * VAL_FRAC))
    train_pids = all_pids[:n_train]
    val_pids = all_pids[n_train : n_train + n_val]
    test_pids = all_pids[n_train + n_val :]
    print(f"\n[*] Participant split:")
    print(f"    Train: {len(train_pids)} subjects ({train_pids[0]}-{train_pids[-1]})")
    print(f"    Val:   {len(val_pids)} subjects ({val_pids[0]}-{val_pids[-1]})")
    print(f"    Test:  {len(test_pids)} subjects ({test_pids[0]}-{test_pids[-1]})")
    print(
        f"\n[*] Subsampling: fraction={fraction}, floor={floor}, cap={cap}, seed={seed}"
    )
    master_rng = np.random.default_rng(seed)
    X_train, y_train, sid_train, tb_tr, ta_tr, log_tr = _sample_split(
        "train",
        train_pids,
        pid_to_X,
        pid_to_y,
        label_map_inv,
        n_classes,
        winsec,
        fraction,
        floor,
        cap,
        master_rng,
    )
    X_val, y_val, sid_val, tb_va, ta_va, log_va = _sample_split(
        "val",
        val_pids,
        pid_to_X,
        pid_to_y,
        label_map_inv,
        n_classes,
        winsec,
        fraction,
        floor,
        cap,
        master_rng,
    )
    X_test, y_test, sid_test, tb_te, ta_te, log_te = _sample_split(
        "test",
        test_pids,
        pid_to_X,
        pid_to_y,
        label_map_inv,
        n_classes,
        winsec,
        fraction,
        floor,
        cap,
        master_rng,
    )
    X_train = (X_train * G_TO_MS2).astype(np.float32)
    X_val = (X_val * G_TO_MS2).astype(np.float32)
    X_test = (X_test * G_TO_MS2).astype(np.float32)
    print(f"\n[*] Final (physical units, no z-score, after stratified sampling):")
    for name, tb, ta, X, y in [
        ("train", tb_tr, ta_tr, X_train, y_train),
        ("valid", tb_va, ta_va, X_val, y_val),
        ("test", tb_te, ta_te, X_test, y_test),
    ]:
        pct = 100.0 * ta / tb if tb > 0 else 0.0
        counts = {
            label_map_inv[i]: int((y == i).sum())
            for i in range(n_classes)
            if int((y == i).sum()) > 0
        }
        print(
            f"    {name:5s}: {ta:6d} / {tb:7d} windows ({pct:5.2f}%)  X={X.shape}  labels={counts}"
        )
    print(
        f"    train acc-norm |mean|={np.linalg.norm(X_train, axis=-1).mean():.2f} m/s²  (expect ~10 baseline gravity)"
    )
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        subject_ids_train=sid_train,
        X_valid=X_val,
        y_valid=y_val,
        subject_ids_valid=sid_val,
        X_test=X_test,
        y_test=y_test,
        subject_ids_test=sid_test,
        sampling_rate=np.array(SAMPLE_RATE),
        joints=np.array(CAPTURE24_JOINTS),
        label_schema=np.array(label_schema),
        class_names=np.array(
            [label_map_inv[i] for i in range(n_classes)], dtype=object
        ),
        subset_fraction=np.array(fraction),
        subset_cap_per_cell=np.array(cap),
        subset_floor=np.array(floor),
        subset_seed=np.array(seed),
    )
    print(f"\n[+] Saved -> {out_path}")
    log_csv = os.path.join(outdir, f"capture24_{label_schema}_unimts_sampling_log.csv")
    rows = []
    for split_name, log in [("train", log_tr), ("val", log_va), ("test", log_te)]:
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
    print(f"[+] Per-cell sampling log -> {log_csv}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datadir", default="data")
    p.add_argument("--outdir", default="dataset")
    p.add_argument(
        "--label_schema",
        default=DEFAULT_LABEL_SCHEMA,
        choices=list(LABEL_SCHEMAS.keys()),
    )
    p.add_argument("--winsec", type=int, default=2)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--n_subjects", type=int, default=DEFAULT_N_SUBJECTS)
    p.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="fraction of each (subject, class) cell to keep",
    )
    p.add_argument(
        "--floor",
        type=int,
        default=1,
        help="minimum samples per non-empty (subject, class) cell",
    )
    p.add_argument(
        "--cap", type=int, default=500, help="maximum samples per (subject, class) cell"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed — MUST match prepare_capture24_subset.py",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    build_unimts_npz(
        args.datadir,
        args.outdir,
        args.label_schema,
        args.winsec,
        args.n_jobs,
        args.n_subjects,
        args.fraction,
        args.floor,
        args.cap,
        args.seed,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
