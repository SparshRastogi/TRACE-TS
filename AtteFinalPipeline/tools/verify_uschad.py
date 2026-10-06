import argparse
import os
import sys
from collections import Counter
import numpy as np


def _load_mat(path):
    try:
        from scipy.io import loadmat

        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        return ({k: v for k, v in d.items() if not k.startswith("__")}, "scipy")
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"[scipy.io.loadmat failed: {e}] — trying h5py …")
    try:
        import h5py
    except ImportError:
        print("ERROR: file appears to be MATLAB v7.3 (HDF5) but h5py is not installed.")
        print("       pip install h5py --break-system-packages")
        sys.exit(1)
    out = {}
    with h5py.File(path, "r") as f:

        def _read(name, obj):
            if isinstance(obj, h5py.Dataset):
                arr = np.array(obj)
                if arr.ndim >= 2:
                    arr = arr.T
                out[name] = arr

        f.visititems(_read)
    return (out, "h5py")


def _fmt_shape(arr):
    if not hasattr(arr, "shape"):
        return f"<{type(arr).__name__}>"
    return f"shape={arr.shape}, dtype={arr.dtype}"


def _print_keys(d):
    print("\n── Top-level keys ────────────────────────────────────────────")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, np.ndarray):
            print(f"  {k:24s}  {_fmt_shape(v)}")
        else:
            print(f"  {k:24s}  type={type(v).__name__}, value={v!r}")


def _print_split(name, X, y):
    print(f"\n── Split: {name} ─────────────────────────────────────────────")
    if X is None:
        print("  (no X found)")
        return
    print(f"  X: {_fmt_shape(X)}")
    if y is not None:
        print(f"  y: {_fmt_shape(y)}")
        y_flat = np.asarray(y).reshape(-1).astype(np.int64)
        counts = Counter(y_flat.tolist())
        total = sum(counts.values())
        print(f"  Label distribution (n={total}):")
        for lbl in sorted(counts.keys()):
            c = counts[lbl]
            pct = 100.0 * c / total
            print(f"    class {lbl:>3d}:  {c:>10d}  ({pct:5.2f}%)")


def _channel_stats(X, n_show=6):
    if X is None or X.ndim != 2:
        return
    print("\n── Per-channel statistics (train) ────────────────────────────")
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    n_ch = X.shape[1]
    print(f"  {'ch':>3s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    for c in range(n_ch):
        print(
            f"  {c:>3d}  {means[c]:>10.4f}  {stds[c]:>10.4f}  {mins[c]:>10.4f}  {maxs[c]:>10.4f}"
        )


def _run_length_stats(y):
    if y is None:
        return None
    y = np.asarray(y).reshape(-1).astype(np.int64)
    if y.size == 0:
        return None
    change = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [y.size]))
    labels = y[starts]
    lengths = ends - starts
    per_label = {}
    for lbl, L in zip(labels.tolist(), lengths.tolist()):
        per_label.setdefault(lbl, []).append(L)
    print("\n── Run-length stats per class (samples per contiguous block) ─")
    print(
        f"  {'class':>5s}  {'runs':>6s}  {'min':>8s}  {'median':>10s}  {'mean':>10s}  {'max':>10s}  {'total':>12s}"
    )
    for lbl in sorted(per_label.keys()):
        L = np.asarray(per_label[lbl])
        print(
            f"  {lbl:>5d}  {L.size:>6d}  {L.min():>8d}  {int(np.median(L)):>10d}  {L.mean():>10.1f}  {L.max():>10d}  {L.sum():>12d}"
        )
    return per_label


def _infer_fs_from_runs(per_label, candidate_rates=(50, 100, 200)):
    if per_label is None or len(per_label) == 0:
        return
    all_runs = np.concatenate([np.asarray(v) for v in per_label.values()])
    med = float(np.median(all_runs))
    print("\n── Sampling-rate inference (heuristic) ───────────────────────")
    print(f"  Median run length across all classes: {med:.0f} samples")
    print(f"  Implied recording duration at candidate fs:")
    for fs in candidate_rates:
        secs = med / fs
        print(f"    fs = {fs:>4d} Hz  →  {secs:>8.2f} s per typical run")
    print("\n  USC-HAD per-trial recording length is documented as roughly")
    print("  5–30 s for short activities and up to ~100 s for walking trials.")
    print("  Whichever candidate fs gives durations in that range is the")
    print("  most likely actual sampling rate of your .mat file.")


def _check_explicit_fs_keys(d):
    candidates = [
        "fs",
        "Fs",
        "FS",
        "sample_rate",
        "sampling_rate",
        "srate",
        "sr",
        "hz",
        "Hz",
    ]
    found = {}
    for k in candidates:
        if k in d:
            found[k] = d[k]
    if found:
        print("\n── Explicit sampling-rate keys found ─────────────────────────")
        for k, v in found.items():
            print(f"  {k}: {v}")
    else:
        print("\n── No explicit 'fs' / 'sample_rate' key in the .mat ──────────")


def _maybe_plot(X_train, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not available — skipping plot]")
        return
    if X_train is None or X_train.ndim != 2:
        return
    n_show = min(2000, X_train.shape[0])
    n_ch = X_train.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 1.2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for c in range(n_ch):
        axes[c].plot(X_train[:n_show, c], linewidth=0.6)
        axes[c].set_ylabel(f"ch {c}", fontsize=8)
    axes[-1].set_xlabel("sample index")
    fig.suptitle(f"USC-HAD: first {n_show} samples of X_train, per channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot saved to {out_path}]")


def main():
    p = argparse.ArgumentParser(description="Inspect a USC-HAD .mat file.")
    p.add_argument("--path", default="./dataset/uschad.mat", help="Path to uschad.mat")
    p.add_argument(
        "--expected_fs",
        type=int,
        default=None,
        help="If set, also report implied duration at this fs.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a per-channel snippet plot of X_train.",
    )
    p.add_argument("--plot_path", default="uschad_snippet.png")
    args = p.parse_args()
    if not os.path.exists(args.path):
        print(f"ERROR: file not found: {args.path}")
        sys.exit(1)
    print(f"Loading: {args.path}")
    d, backend = _load_mat(args.path)
    print(f"Loader:  {backend}")
    _print_keys(d)
    _check_explicit_fs_keys(d)
    splits = {
        "train": (d.get("X_train"), d.get("y_train")),
        "valid": (d.get("X_valid"), d.get("y_valid")),
        "test": (d.get("X_test"), d.get("y_test")),
    }
    if splits["valid"][0] is None and "X_val" in d:
        splits["valid"] = (d.get("X_val"), d.get("y_val"))
    for name, (X, y) in splits.items():
        _print_split(name, X, y)
    X_train = splits["train"][0]
    y_train = splits["train"][1]
    if X_train is not None and X_train.ndim == 2:
        _channel_stats(X_train)
    per_label = _run_length_stats(y_train)
    candidate_rates = [50, 100, 200]
    if args.expected_fs and args.expected_fs not in candidate_rates:
        candidate_rates.append(args.expected_fs)
    _infer_fs_from_runs(per_label, candidate_rates=tuple(sorted(candidate_rates)))
    if args.expected_fs:
        print(f"\n  (Expected fs from settings.py: {args.expected_fs} Hz)")
    if args.plot:
        _maybe_plot(X_train, args.plot_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
