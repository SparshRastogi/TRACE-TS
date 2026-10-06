from __future__ import annotations
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

UCI_HAR_CLASSES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


def stratified_val_split(X, y, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_frac)))
        val_idx.extend(cls_idx[:n_val].tolist())
    val_idx = np.array(sorted(val_idx))
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    return (X[train_mask], y[train_mask], X[val_idx], y[val_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        default=str(UCIHAR_DATA_DIR),
        help="Directory containing X_train.npy, y_train.npy, X_test.npy, y_test.npy as produced by your UCI-HAR loader (channels: total_acc xyz in m/s^2, body_gyro xyz in rad/s, body_acc xyz).",
    )
    ap.add_argument(
        "--out_path",
        default="./dataset/ucihar_unimts.mat",
        help="Where to write the UniMTS-ready .mat file (matches the convention used by the other prepare_*_unimts.py scripts).",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of train held out as validation (stratified).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[uci_har_unimts] loading from {src}")
    X_train_full = np.load(src / "X_train.npy")
    y_train_full = np.load(src / "y_train.npy")
    X_test = np.load(src / "X_test.npy")
    y_test = np.load(src / "y_test.npy")
    for name, X in (("X_train", X_train_full), ("X_test", X_test)):
        if X.ndim != 3 or X.shape[1] != 128 or X.shape[2] != 9:
            raise ValueError(
                f"{name} has shape {X.shape}; expected (N, 128, 9). Did your loader run with the 9-file stack?"
            )
    print(f"[uci_har_unimts] X_train_full {X_train_full.shape} | X_test {X_test.shape}")
    X_train_full = X_train_full[:, :, :6].astype(np.float32, copy=False)
    X_test = X_test[:, :, :6].astype(np.float32, copy=False)
    uniq = np.unique(y_train_full)
    if uniq.min() == 1 and uniq.max() == 6:
        y_train_full = (y_train_full - 1).astype(np.int64)
        y_test = (y_test - 1).astype(np.int64)
    elif uniq.min() == 0 and uniq.max() == 5:
        y_train_full = y_train_full.astype(np.int64)
        y_test = y_test.astype(np.int64)
    else:
        raise ValueError(
            f"Unexpected label range in y_train: min={uniq.min()}, max={uniq.max()}. Expected 1..6 (raw UCI-HAR) or 0..5 (already remapped)."
        )
    acc_max = float(np.abs(X_train_full[:, :, :3]).max())
    if acc_max < 3.0:
        print(
            f"[uci_har_unimts] WARNING: max |acc| = {acc_max:.3f}. That looks like g, not m/s^2. Did your upstream loader skip the *9.81 step?"
        )
    else:
        print(f"[uci_har_unimts] acc range OK (max |acc| = {acc_max:.2f} m/s^2)")
    gyro_max = float(np.abs(X_train_full[:, :, 3:]).max())
    print(f"[uci_har_unimts] gyro max |omega| = {gyro_max:.2f} rad/s")
    X_train, y_train, X_valid, y_valid = stratified_val_split(
        X_train_full, y_train_full, val_frac=args.val_frac, seed=args.seed
    )

    def hist(y):
        return {int(c): int((y == c).sum()) for c in np.unique(y)}

    print(f"[uci_har_unimts] train n={len(y_train)}  dist={hist(y_train)}")
    print(f"[uci_har_unimts] valid n={len(y_valid)}  dist={hist(y_valid)}")
    print(f"[uci_har_unimts] test  n={len(y_test)}   dist={hist(y_test)}")
    sio.savemat(
        str(out),
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
        },
        do_compression=True,
    )
    print(f"[uci_har_unimts] wrote -> {out}")
    print(f"  X_train {X_train.shape}  X_valid {X_valid.shape}  X_test {X_test.shape}")
    print(f"  classes: {UCI_HAR_CLASSES}")


if __name__ == "__main__":
    main()
