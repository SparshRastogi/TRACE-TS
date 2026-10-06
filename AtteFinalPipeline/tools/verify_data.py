import os
import argparse
import numpy as np
import scipy.io as sio

MAT_CHANNEL_RAW_COLS = (
    list(range(37, 48))
    + list(range(50, 59))
    + list(range(63, 72))
    + list(range(76, 85))
    + list(range(89, 98))
    + list(range(102, 134))
)
assert len(MAT_CHANNEL_RAW_COLS) == 79
ALL_RUNS = [
    f"S{s}-{r}"
    for s in (1, 2, 3, 4)
    for r in ("ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill")
]


def load_raw_sensors(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    if not rows:
        return None
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows))
    data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            try:
                data[i, j] = float(v)
            except ValueError:
                pass
    return data


def impute_nans(X):
    X = X.copy()
    for col in range(X.shape[1]):
        nans = np.isnan(X[:, col])
        if not nans.any():
            continue
        not_nan = np.where(~nans)[0]
        if len(not_nan) == 0:
            X[:, col] = 0.0
        else:
            X[:, col] = np.interp(np.arange(X.shape[0]), not_nan, X[not_nan, col])
    return X


def get_raw_signature(opp_plus_root, run, ref_channel_idx=0, n_probe=200):
    candidates = [
        os.path.join(opp_plus_root, "data", run, f"{run}_sensors_data.txt"),
        os.path.join(opp_plus_root, run, f"{run}_sensors_data.txt"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return (None, None, None)
    raw = load_raw_sensors(path)
    if raw is None:
        return (None, None, None)
    raw = impute_nans(raw).astype(np.float32)
    raw_79 = raw[:, MAT_CHANNEL_RAW_COLS] / 1000.0
    N = raw_79.shape[0]
    ref = raw_79[: min(n_probe, N), ref_channel_idx].astype(np.float32)
    return (N, ref, raw_79)


def find_run_at_offset(
    mat_segment,
    offset,
    raw_signatures,
    ref_channel_idx=0,
    n_probe=200,
    corr_thresh=0.9999,
):
    if offset >= mat_segment.shape[0]:
        return (None, 0, 0.0)
    best_run = None
    best_N = 0
    best_corr = -1.0
    for run, sig in raw_signatures.items():
        if sig is None:
            continue
        N_run, ref, _full = sig
        if offset + n_probe > mat_segment.shape[0]:
            continue
        if offset + N_run > mat_segment.shape[0]:
            continue
        seg = mat_segment[offset : offset + n_probe, ref_channel_idx]
        ref_short = ref[:n_probe]
        if seg.std() == 0 or ref_short.std() == 0:
            continue
        seg_c = seg - seg.mean()
        ref_c = ref_short - ref_short.mean()
        corr = float(np.mean(seg_c * ref_c) / (seg.std() * ref_short.std()))
        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_run = run
            best_N = N_run
    if best_corr < corr_thresh:
        return (None, 0, best_corr)
    return (best_run, best_N, best_corr)


def verify_match(mat_segment, offset, raw_full, ref_channel_idx=0):
    N = raw_full.shape[0]
    if offset + N > mat_segment.shape[0]:
        return (False, float("inf"))
    seg = mat_segment[offset : offset + N, ref_channel_idx]
    raw = raw_full[:, ref_channel_idx]
    return (True, float(np.max(np.abs(seg - raw))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp_plus_root", required=True)
    ap.add_argument("--opp_mat", default="./dataset/opportunity.mat")
    ap.add_argument(
        "--ref_channel_idx",
        type=int,
        default=0,
        help="Which of the 79 mat channels to use as the correlation reference. 0 = BACK accX, fine in almost every case.",
    )
    args = ap.parse_args()
    print(f"[*] Loading {args.opp_mat}")
    m = sio.loadmat(args.opp_mat)
    x_train_z = m["trainingData"].astype(np.float32).T
    train_mean = x_train_z.mean(axis=0)
    train_std = x_train_z.std(axis=0)
    train_std[train_std == 0] = 1.0
    splits = {}
    for split_name, key_X, key_y in [
        ("train", "trainingData", "trainingLabels"),
        ("val", "valData", "valLabels"),
        ("test", "testingData", "testingLabels"),
    ]:
        if key_X not in m:
            print(f"  [!] {key_X} missing from .mat — skipping {split_name}")
            continue
        X_z = m[key_X].astype(np.float32).T
        X_unz = X_z * train_std + train_mean
        splits[split_name] = X_unz
        print(f"    {split_name:5s}  shape={X_unz.shape}")
    print(f"\n[*] Computing raw signatures for all runs in {args.opp_plus_root}")
    raw_sigs = {}
    for run in ALL_RUNS:
        N, ref, full = get_raw_signature(
            args.opp_plus_root, run, ref_channel_idx=args.ref_channel_idx
        )
        if N is None:
            print(f"    {run:10s}  not found")
        else:
            print(f"    {run:10s}  N={N}")
            raw_sigs[run] = (N, ref, full)
    print("\n" + "=" * 70)
    print("Recovering file order per split")
    print("=" * 70)
    found = {}
    for split_name, mat_seg in splits.items():
        print(f"\n  ── {split_name} (N_total = {mat_seg.shape[0]}) ──")
        offset = 0
        order = []
        while offset < mat_seg.shape[0]:
            best_run, N_run, corr = find_run_at_offset(
                mat_seg, offset, raw_sigs, ref_channel_idx=args.ref_channel_idx
            )
            if best_run is None:
                remaining = mat_seg.shape[0] - offset
                print(
                    f"    [!] No matching run starts at offset {offset} (remaining={remaining}). Best corr seen: {corr:.4f}"
                )
                break
            _ok, max_diff = verify_match(
                mat_seg,
                offset,
                raw_sigs[best_run][2],
                ref_channel_idx=args.ref_channel_idx,
            )
            print(
                f"    offset={offset:>7d}  {best_run:10s}  N={N_run:>6d}  corr={corr:.4f}  max_diff_full={max_diff:.6f}"
            )
            order.append((best_run, N_run))
            offset += N_run
        if offset == mat_seg.shape[0]:
            print(
                f"    ✓ {split_name}: fully recovered ({sum((n for _, n in order))} samples)"
            )
        found[split_name] = order
    print("\n" + "=" * 70)
    print("SUMMARY: file lists per split")
    print("=" * 70)
    for split, order in found.items():
        print(f"\n  {split.upper()}_FILES = [")
        for run, N in order:
            print(f'      "{run}.dat",   # N={N}')
        print(f"  ]   # total N={sum((n for _, n in order))}")


if __name__ == "__main__":
    main()
