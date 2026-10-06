from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
import argparse
import os
import sys
import numpy as np

try:
    import scipy.io as sio
except ImportError:
    sio = None
_OFFICIAL_CLASS_MAPS = {
    "uschad": list(USCHAD_CLASS_MAP),
    "opportunity": [
        "Null",
        "Close Dishwasher",
        "Close Drawer 3",
        "Close Drawer 2",
        "Close Door 1",
        "Close Door 2",
        "Close Drawer 1",
        "Close Fridge",
        "Toggle Switch",
        "Open Dishwasher",
        "Open Drawer 3",
        "Open Drawer 2",
        "Open Door 1",
        "Open Door 2",
        "Open Drawer 1",
        "Open Fridge",
        "Drink from Cup",
        "Clean Table",
    ],
    "opportunity_nonull": [
        "Close Dishwasher",
        "Close Drawer 3",
        "Close Drawer 2",
        "Close Door 1",
        "Close Door 2",
        "Close Drawer 1",
        "Close Fridge",
        "Toggle Switch",
        "Open Dishwasher",
        "Open Drawer 3",
        "Open Drawer 2",
        "Open Door 1",
        "Open Door 2",
        "Open Drawer 1",
        "Open Fridge",
        "Drink from Cup",
        "Clean Table",
    ],
}


def _get_dataset_config(dataset_name):
    if "." not in sys.path and "" not in sys.path:
        sys.path.insert(0, ".")
    from AtteFinalPipeline.expert.settings import get_args as _settings_get_args

    saved_argv = sys.argv
    sys.argv = ["settings", "--dataset", dataset_name]
    try:
        args, _, _ = _settings_get_args()
    finally:
        sys.argv = saved_argv
    class_map = _OFFICIAL_CLASS_MAPS.get(dataset_name, args.class_map)
    return {
        "dataset": dataset_name,
        "class_map": class_map,
        "class_map_src": "override"
        if dataset_name in _OFFICIAL_CLASS_MAPS
        else "settings.py",
        "input_dim": args.input_dim,
        "num_class": args.num_class,
        "window": args.window,
        "stride": args.stride,
        "path_data": args.path_data,
        "path_raw": args.path_raw,
        "path_processed": args.path_processed,
    }


_EXCLUDED_DATASETS = {"skoda", "capture24_walmsley", "capture24_full"}


def _all_dataset_names():
    if "." not in sys.path and "" not in sys.path:
        sys.path.insert(0, ".")
    from AtteFinalPipeline.expert.settings import _DATASET_CONFIGS

    return [n for n in _DATASET_CONFIGS.keys() if n not in _EXCLUDED_DATASETS]


def _selectable_dataset_names():
    if "." not in sys.path and "" not in sys.path:
        sys.path.insert(0, ".")
    from AtteFinalPipeline.expert.settings import _DATASET_CONFIGS

    return [n for n in _DATASET_CONFIGS.keys() if n not in _EXCLUDED_DATASETS]


def _load_npz_target(npz_path):
    if not os.path.exists(npz_path):
        return None
    ds = np.load(npz_path)
    if "target" in ds:
        return ds["target"].astype(np.int64)
    return None


def _load_ucihar_split(base, split):
    y_path = os.path.join(base, f"y_{split}.npy")
    if os.path.exists(y_path):
        y = np.load(y_path).astype(np.int64)
        if y.size and y.min() > 0:
            y = y - 1
        return y
    npz_path = os.path.join(base, f"{split}.npz")
    if os.path.exists(npz_path):
        ds = np.load(npz_path)
        if "target" in ds:
            y = ds["target"].astype(np.int64)
            if y.size and y.min() > 0:
                y = y - 1
            return y
    return None


def load_window_labels(cfg, splits=("train", "val", "test")):
    out = {}
    dataset = cfg["dataset"]
    base = cfg["path_processed"]
    if dataset == "ucihar":
        bases = [cfg["path_data"], cfg["path_processed"]]
        bases = [b for b in bases if b]
        for split in splits:
            y = None
            for b in bases:
                y = _load_ucihar_split(b, split)
                if y is not None:
                    break
            if y is not None:
                out[split] = y
    else:
        for split in splits:
            split_file = "val" if split in ("validation", "valid") else split
            y = _load_npz_target(os.path.join(base, f"{split_file}.npz"))
            if y is not None:
                out[split] = y
    if not out:
        return None
    out["all"] = np.concatenate([out[s] for s in splits if s in out])
    return out


def _load_uschad_samples(mat_path):
    contents = sio.loadmat(mat_path)
    return {
        "train": contents["y_train"].reshape(-1).astype(np.int64),
        "val": contents["y_valid"].reshape(-1).astype(np.int64),
        "test": contents["y_test"].reshape(-1).astype(np.int64),
    }


def _load_keyed_mat_samples(mat_path):
    contents = sio.loadmat(mat_path)
    return {
        "train": contents["y_train"].reshape(-1).astype(np.int64),
        "val": contents["y_valid"].reshape(-1).astype(np.int64),
        "test": contents["y_test"].reshape(-1).astype(np.int64),
    }


def _load_opportunity_samples(mat_path, drop_null=False):
    contents = sio.loadmat(mat_path)
    y_tr = contents["trainingLabels"].reshape(-1).astype(np.int64) - 1
    y_v = contents["valLabels"].reshape(-1).astype(np.int64) - 1
    y_te = contents["testingLabels"].reshape(-1).astype(np.int64) - 1
    if drop_null:

        def _drop(y):
            y = y[y != 0]
            return y - 1

        y_tr = _drop(y_tr)
        y_v = _drop(y_v)
        y_te = _drop(y_te)
    return {"train": y_tr, "val": y_v, "test": y_te}


def _load_capture24_samples(npz_path):
    contents = np.load(npz_path)
    return {
        "train": contents["y_train"].reshape(-1).astype(np.int64),
        "val": contents["y_valid"].reshape(-1).astype(np.int64),
        "test": contents["y_test"].reshape(-1).astype(np.int64),
    }


def _load_ucihar_samples(cfg):
    return load_window_labels(cfg)


def load_sample_labels(cfg):
    if sio is None:
        sys.exit("[FATAL] scipy is required for sample-level reads.")
    dataset = cfg["dataset"]
    path = cfg["path_data"]
    if not path or not os.path.exists(path):
        return None
    try:
        if dataset == "uschad":
            out = _load_uschad_samples(path)
        elif dataset in ("opportunity",):
            out = _load_opportunity_samples(path, drop_null=False)
        elif dataset == "opportunity_nonull":
            out = _load_opportunity_samples(path, drop_null=True)
        elif dataset in (
            "pamap2",
            "skoda",
            "shoaib",
            "mhealth",
            "mhealth_nonull",
            "hospital",
        ):
            out = _load_keyed_mat_samples(path)
        elif dataset.startswith("capture24"):
            out = _load_capture24_samples(path)
        elif dataset == "ucihar":
            out = _load_ucihar_samples(cfg)
            if out is None:
                return None
            out = {k: v for k, v in out.items() if k != "all"}
        else:
            return None
    except (KeyError, FileNotFoundError) as e:
        print(f"  [WARN] could not parse {path}: {e}")
        return None
    out["all"] = np.concatenate([out[s] for s in ("train", "val", "test") if s in out])
    return out


def _counts(labels, n_classes):
    return np.bincount(labels, minlength=n_classes)


def _fmt_proportions(pct, decimals=2):
    fmt = "{:." + str(decimals) + "f}"
    return ", ".join((fmt.format(p) for p in pct))


def _print_class_table(class_names, counts, pct, n_classes, label_col_width=28):
    print(f"  {'idx':>3s}  {'class':<{label_col_width}s}  {'count':>10s}  {'%':>7s}")
    for c in range(n_classes):
        name = class_names[c] if c < len(class_names) else f"Class_{c}"
        if len(name) > label_col_width:
            name = name[: label_col_width - 1] + "…"
        print(
            f"  {c:>3d}  {name:<{label_col_width}s}  {int(counts[c]):>10d}  {pct[c]:>6.2f}%"
        )
    print(
        f"  {'':>3s}  {'TOTAL':<{label_col_width}s}  {int(counts.sum()):>10d}  100.00%"
    )


def _print_pastable_row(name, n_classes, class_names, pct, decimals):
    classes_field = ", ".join(class_names[:n_classes])
    proportions_str = _fmt_proportions(pct, decimals)
    print()
    print(f"  Dataset     : {name}")
    print(f"  # Classes   : {n_classes}")
    print(f"  Classes     : {classes_field}")
    print(f"  Proportions : {proportions_str}")
    print()
    print("  ── Pastable Markdown row ──")
    print(f"  | {name} | {n_classes} | {classes_field} | {proportions_str} |")


def _report_one_dataset(cfg, labels_by_split, level, decimals, per_split, splits_arg):
    name = cfg["dataset"]
    class_names = cfg["class_map"]
    n_classes = cfg["num_class"]
    print()
    print("=" * 90)
    print(f" Dataset: {name}   (level={level}, class_map={cfg['class_map_src']})")
    print("=" * 90)
    print(
        f"  splits available : {', '.join((s for s in splits_arg if s in labels_by_split))}"
    )
    for split in splits_arg:
        if split in labels_by_split:
            print(f"    {split:>6s} : {int(labels_by_split[split].size):>10d}")
    print(f"    {'total':>6s} : {int(labels_by_split['all'].size):>10d}")
    if per_split:
        for split in splits_arg:
            if split not in labels_by_split:
                continue
            y = labels_by_split[split]
            if y.size == 0:
                continue
            counts = _counts(y, n_classes)
            total = counts.sum()
            pct = 100.0 * counts / max(total, 1)
            print()
            print(f"  ── split = {split.upper()} ─────────────────────────────")
            _print_class_table(class_names, counts, pct, n_classes)
    y_all = labels_by_split["all"]
    if y_all.size and (y_all.max() >= n_classes or y_all.min() < 0):
        print(
            f"  [WARN] label range [{y_all.min()}, {y_all.max()}] does not fit declared num_class={n_classes}. Counts may be misleading."
        )
    counts = _counts(y_all, n_classes)
    total = counts.sum()
    pct = 100.0 * counts / max(total, 1)
    print()
    print(
        f"  ── POOLED ({' + '.join((s for s in splits_arg if s in labels_by_split))}) ──"
    )
    _print_class_table(class_names, counts, pct, n_classes)
    _print_pastable_row(name, n_classes, class_names, pct, decimals)


def main():
    all_names = _all_dataset_names()
    selectable = _selectable_dataset_names()
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--dataset",
        nargs="+",
        default=all_names,
        choices=selectable,
        help="One or more dataset names. Default: all datasets in settings.py except "
        + ", ".join(sorted(_EXCLUDED_DATASETS))
        + ".",
    )
    ap.add_argument(
        "--level",
        choices=("window", "sample"),
        default="window",
        help="window = post-windowing .npz counts (what the model trains on); sample = pre-windowing counts from path_data.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to include when pooling.",
    )
    ap.add_argument(
        "--per_split",
        action="store_true",
        help="Also print per-class tables for each split (not just the pool).",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places in the printed percentages.",
    )
    args = ap.parse_args()
    print(f"[*] Datasets : {', '.join(args.dataset)}")
    print(f"[*] Level    : {args.level}")
    print(f"[*] Splits   : {', '.join(args.splits)}")
    summary_rows = []
    skipped = []
    for ds_name in args.dataset:
        try:
            cfg = _get_dataset_config(ds_name)
        except Exception as e:
            print(f"\n[!] Could not resolve settings for '{ds_name}': {e}")
            skipped.append((ds_name, "settings error"))
            continue
        if args.level == "window":
            labels = load_window_labels(cfg, splits=args.splits)
            source_hint = f"path_processed = {cfg['path_processed']}"
        else:
            labels = load_sample_labels(cfg)
            source_hint = f"path_data = {cfg['path_data']}"
        if labels is None or labels["all"].size == 0:
            print(f"\n[!] No data found for '{ds_name}'.")
            print(f"    Expected: {source_hint}")
            if args.level == "window":
                print(f"    Hint: run preprocess.py for '{ds_name}' first.")
            skipped.append((ds_name, "no data on disk"))
            continue
        _report_one_dataset(
            cfg, labels, args.level, args.decimals, args.per_split, args.splits
        )
        summary_rows.append((ds_name, cfg["num_class"], int(labels["all"].size)))
    print()
    print("=" * 90)
    print(" SUMMARY")
    print("=" * 90)
    if summary_rows:
        print(f"  {'dataset':<24s}  {'# classes':>10s}  {'total':>12s}")
        print(f"  {'-' * 24}  {'-' * 10}  {'-' * 12}")
        for n, k, t in summary_rows:
            print(f"  {n:<24s}  {k:>10d}  {t:>12d}")
    else:
        print("  (no datasets produced output)")
    if skipped:
        print()
        print("  Skipped:")
        for n, reason in skipped:
            print(f"    - {n}: {reason}")
    print()
    print(f"  Counts are at the {args.level} level.")
    if args.level == "window":
        print("  These are the rows the model actually sees per epoch.")
    else:
        print("  Sample-level counts; multiply by ~window/stride to estimate windows.")


if __name__ == "__main__":
    main()
