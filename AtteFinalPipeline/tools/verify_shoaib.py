import argparse
import os
import sys
import numpy as np
import scipy.io as sio

DATASET_MAT = "./dataset/shoaib.mat"
RAW_DIR = "./data/shoaib/raw"
PROCESSED_DIR = "./data/shoaib/processed/100_50"
NUM_CLASSES = 7
INPUT_DIM = 45
WINDOW = 100
STRIDE = 50
CLASS_MAP = [
    "Walking",
    "Standing",
    "Jogging",
    "Sitting",
    "Biking",
    "Walking Upstairs",
    "Walking Downstairs",
]
EXPECTED_SAMPLE_SHAPES = {
    "train": (432000, INPUT_DIM),
    "val": (63000, INPUT_DIM),
    "test": (126000, INPUT_DIM),
}


def hr(c="─", n=70):
    print(c * n)


def section(title):
    print()
    hr("═")
    print(f"  {title}")
    hr("═")


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


class Report:
    def __init__(self, strict=False):
        self.strict = strict
        self.n_ok = 0
        self.n_warn = 0
        self.n_fail = 0

    def ok(self, msg):
        ok(msg)
        self.n_ok += 1

    def warn(self, msg):
        warn(msg)
        self.n_warn += 1

    def fail(self, msg):
        fail(msg)
        self.n_fail += 1

    def exit_code(self):
        if self.n_fail > 0:
            return 1
        if self.strict and self.n_warn > 0:
            return 1
        return 0


def check_no_nan_inf(rep, name, arr):
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan == 0 and n_inf == 0:
        rep.ok(f"{name}: no NaN/Inf")
    else:
        rep.fail(f"{name}: contains {n_nan} NaN and {n_inf} Inf values")


def check_label_range(rep, name, y, num_classes=NUM_CLASSES):
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        rep.fail(f"{name}: empty label array")
        return
    lo, hi = (int(y.min()), int(y.max()))
    if lo >= 0 and hi < num_classes:
        rep.ok(f"{name}: labels in [{lo}, {hi}] ⊂ [0, {num_classes - 1}]")
    else:
        rep.fail(f"{name}: labels in [{lo}, {hi}] — expected [0, {num_classes - 1}]")


def check_histogram(rep, name, y, num_classes=NUM_CLASSES):
    hist = np.bincount(np.asarray(y).reshape(-1), minlength=num_classes).tolist()
    classes_present = sum((1 for h in hist if h > 0))
    classes_missing = [i for i, h in enumerate(hist) if h == 0]
    print(f"         hist: {hist}")
    print(f"         classes present: {classes_present}/{num_classes}")
    if classes_missing:
        print(
            f"         classes missing: {classes_missing} ({[CLASS_MAP[i] for i in classes_missing]})"
        )


def check_per_channel_stats(rep, name, X, expect_normed=True):
    if X.ndim == 3:
        flat = X.reshape(-1, X.shape[-1])
    else:
        flat = X
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    print(f"         per-channel mean range: [{mean.min():+.4f}, {mean.max():+.4f}]")
    print(f"         per-channel std  range: [{std.min():.4f}, {std.max():.4f}]")
    if not expect_normed:
        return
    max_abs_mean = float(np.abs(mean).max())
    max_std_dev = float(np.abs(std - 1.0).max())
    if max_abs_mean < 0.5 and max_std_dev < 0.5:
        rep.ok(
            f"{name}: per-channel stats look normalised (|mean|≤{max_abs_mean:.3f}, |std-1|≤{max_std_dev:.3f})"
        )
    else:
        rep.warn(
            f"{name}: per-channel stats drift (|mean|≤{max_abs_mean:.3f}, |std-1|≤{max_std_dev:.3f}). Some drift on val/test is expected from leave-subjects-out splits."
        )


def check_shape(rep, name, arr, expected, exact=True):
    if exact:
        if arr.shape == expected:
            rep.ok(f"{name}: shape {arr.shape} matches expected {expected}")
        else:
            rep.fail(f"{name}: shape {arr.shape} ≠ expected {expected}")
    elif arr.shape[1:] == expected[1:]:
        rep.ok(
            f"{name}: trailing shape {arr.shape[1:]} matches expected {expected[1:]} (N={arr.shape[0]})"
        )
    else:
        rep.fail(f"{name}: trailing shape {arr.shape[1:]} ≠ expected {expected[1:]}")


def stage_mat(rep):
    section("Stage 1 — raw .mat at ./dataset/shoaib.mat")
    if not os.path.exists(DATASET_MAT):
        rep.fail(f"missing file: {DATASET_MAT} — run prepare_shoaib.py first")
        return None
    contents = sio.loadmat(DATASET_MAT)
    keys_needed = ["X_train", "y_train", "X_valid", "y_valid", "X_test", "y_test"]
    missing = [k for k in keys_needed if k not in contents]
    if missing:
        rep.fail(f"missing keys in .mat: {missing}")
        return None
    rep.ok(f"all required keys present: {keys_needed}")
    splits = {}
    for split, x_key, y_key, exp_shape in [
        ("train", "X_train", "y_train", EXPECTED_SAMPLE_SHAPES["train"]),
        ("val", "X_valid", "y_valid", EXPECTED_SAMPLE_SHAPES["val"]),
        ("test", "X_test", "y_test", EXPECTED_SAMPLE_SHAPES["test"]),
    ]:
        print(f"\n  -- {split} --")
        X = contents[x_key].astype(np.float32)
        y = np.asarray(contents[y_key]).reshape(-1).astype(np.int64)
        print(f"         X dtype={X.dtype}, y dtype={y.dtype}")
        check_shape(rep, f"{split} X", X, exp_shape, exact=True)
        if y.shape[0] != X.shape[0]:
            rep.fail(f"{split}: X has {X.shape[0]} rows but y has {y.shape[0]}")
        else:
            rep.ok(f"{split}: X and y row counts match ({X.shape[0]})")
        check_no_nan_inf(rep, f"{split} X", X)
        check_label_range(rep, f"{split} y", y)
        check_histogram(rep, f"{split} y", y)
        check_per_channel_stats(rep, f"{split} X", X, expect_normed=True)
        splits[split] = (X, y)
    return splits


def stage_raw_npz(rep):
    section("Stage 2 — sample-level .npz at ./data/shoaib/raw/")
    if not os.path.isdir(RAW_DIR):
        rep.warn(
            f"directory does not exist yet: {RAW_DIR} (will be created on first preprocess.py run)"
        )
        return None
    splits = {}
    for split, exp_shape in [
        ("train", EXPECTED_SAMPLE_SHAPES["train"]),
        ("val", EXPECTED_SAMPLE_SHAPES["val"]),
        ("test", EXPECTED_SAMPLE_SHAPES["test"]),
    ]:
        path = os.path.join(RAW_DIR, f"{split}.npz")
        print(f"\n  -- {split} ({path}) --")
        if not os.path.exists(path):
            rep.warn(f"missing: {path} (run preprocess.py to create)")
            continue
        z = np.load(path)
        if "x" not in z.files or "y" not in z.files:
            rep.fail(f"{split}: expected keys 'x' and 'y', got {z.files}")
            continue
        X, y = (z["x"], z["y"])
        print(f"         X dtype={X.dtype}, y dtype={y.dtype}")
        check_shape(rep, f"{split} X", X, exp_shape, exact=True)
        if y.shape[0] != X.shape[0]:
            rep.fail(f"{split}: X/y row mismatch ({X.shape[0]} vs {y.shape[0]})")
        else:
            rep.ok(f"{split}: X/y row counts match ({X.shape[0]})")
        check_no_nan_inf(rep, f"{split} X", X)
        check_label_range(rep, f"{split} y", y)
        check_histogram(rep, f"{split} y", y)
        check_per_channel_stats(rep, f"{split} X", X, expect_normed=True)
        splits[split] = (X, y)
    return splits


def expected_segment_count(n_samples, window, stride):
    if n_samples <= window:
        return 0
    return (n_samples - window - 1) // stride + 1


def stage_processed_npz(rep, sample_splits, skip_test_sw=False):
    section("Stage 3 — windowed .npz at ./data/shoaib/processed/100_50/")
    if not os.path.isdir(PROCESSED_DIR):
        rep.warn(
            f"directory does not exist yet: {PROCESSED_DIR} (will be created on first preprocess.py run)"
        )
        return
    files = [
        ("train", "train.npz", STRIDE),
        ("val", "val.npz", STRIDE),
        ("test", "test.npz", STRIDE),
    ]
    if not skip_test_sw:
        files.append(("test_sample_wise", "test_sample_wise.npz", 1))
    for split, fname, stride in files:
        path = os.path.join(PROCESSED_DIR, fname)
        print(f"\n  -- {split} ({path}) --")
        if not os.path.exists(path):
            rep.warn(f"missing: {path} (run preprocess.py to create)")
            continue
        z = np.load(path)
        if "data" not in z.files or "target" not in z.files:
            rep.fail(f"{split}: expected keys 'data' and 'target', got {z.files}")
            continue
        data, target = (z["data"], z["target"])
        print(f"         data dtype={data.dtype}, target dtype={target.dtype}")
        check_shape(
            rep, f"{split} data", data, (data.shape[0], WINDOW, INPUT_DIM), exact=False
        )
        ref_split = "test" if split == "test_sample_wise" else split
        if sample_splits is not None and ref_split in sample_splits:
            n_samples = sample_splits[ref_split][0].shape[0]
            n_expected = expected_segment_count(n_samples, WINDOW, stride)
            if data.shape[0] == n_expected:
                rep.ok(
                    f"{split}: segment count {data.shape[0]} matches expected {n_expected} (N={n_samples}, w={WINDOW}, s={stride})"
                )
            else:
                rep.fail(
                    f"{split}: segment count {data.shape[0]} ≠ expected {n_expected} (N={n_samples}, w={WINDOW}, s={stride})"
                )
        if target.shape[0] != data.shape[0]:
            rep.fail(
                f"{split}: data/target row mismatch ({data.shape[0]} vs {target.shape[0]})"
            )
        else:
            rep.ok(f"{split}: data/target row counts match ({data.shape[0]})")
        check_no_nan_inf(rep, f"{split} data", data)
        check_label_range(rep, f"{split} target", target)
        check_histogram(rep, f"{split} target", target)
        check_per_channel_stats(rep, f"{split} data", data, expect_normed=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures."
    )
    parser.add_argument(
        "--skip-test-sw",
        action="store_true",
        help="Skip the 126k-segment test_sample_wise check (faster).",
    )
    args = parser.parse_args()
    rep = Report(strict=args.strict)
    print()
    hr("═")
    print("  Shoaib dataset verification")
    print(
        f"  num_classes = {NUM_CLASSES} | input_dim = {INPUT_DIM} | window = {WINDOW} | stride = {STRIDE}"
    )
    hr("═")
    sample_splits = stage_mat(rep)
    raw_splits = stage_raw_npz(rep)
    splits_for_processed = raw_splits if raw_splits else sample_splits
    stage_processed_npz(rep, splits_for_processed, skip_test_sw=args.skip_test_sw)
    print()
    hr("═")
    print(f"  SUMMARY:  {rep.n_ok} OK   {rep.n_warn} WARN   {rep.n_fail} FAIL")
    hr("═")
    code = rep.exit_code()
    if code == 0:
        print("  ✓ verification passed")
    else:
        print(
            "  ✗ verification failed (see [FAIL]"
            + (" / [WARN] in --strict mode" if args.strict else "")
            + " above)"
        )
    sys.exit(code)


if __name__ == "__main__":
    main()
