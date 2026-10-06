from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.data.sensors import get_null_class_indices
from AtteFinalPipeline.data.loading import _is_capture24_variant
from AtteFinalPipeline.embeddings.encoders import (
    MANTIS_OUTPUT_TOKEN,
    MANTIS_RETURN_TRANSF_LAYER,
    _resize_batch_for_mantis,
    load_mantis_model,
)
import os
import sys
import time
import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from AtteFinalPipeline.expert.settings import get_args as _settings_get_args

_OFFICIAL_CLASS_MAPS = {"uschad": list(USCHAD_CLASS_MAP)}


def _get_dataset_config(dataset_name):
    saved_argv = sys.argv
    sys.argv = ["settings", "--dataset", dataset_name]
    try:
        args, config_dataset, config_model = _settings_get_args()
        class_map = _OFFICIAL_CLASS_MAPS.get(dataset_name, args.class_map)
        if dataset_name in _OFFICIAL_CLASS_MAPS:
            print(
                f"  [class_map] Using official override for '{dataset_name}' ({len(class_map)} classes) — settings.py value ignored."
            )
    finally:
        sys.argv = saved_argv
    return {
        "class_map": class_map,
        "input_dim": args.input_dim,
        "num_class": args.num_class,
        "window": args.window,
        "stride": args.stride,
        "path_data": args.path_data,
        "path_raw": args.path_raw,
        "path_processed": args.path_processed,
    }


def load_data(dataset_name, data_path=None, split="test", ds_config=None):
    split = split.lower()
    if dataset_name == "uschad":
        processed_base = (
            data_path
            if data_path
            else ds_config["path_processed"]
            if ds_config
            else None
        )
        if processed_base:
            split_file = "val" if split in ("validation", "valid") else split
            npz_path = os.path.join(processed_base, f"{split_file}.npz")
            if os.path.exists(npz_path):
                ds = np.load(npz_path)
                return (ds["data"].astype(np.float32), ds["target"].astype(np.int64))
        import scipy.io as sio

        mat_path = (
            ds_config["path_data"] if ds_config else None
        ) or "./dataset/uschad.mat"
        if not os.path.exists(mat_path):
            print(f"ERROR: USC-HAD .mat not found at {mat_path}.")
            sys.exit(1)
        contents = sio.loadmat(mat_path)
        _key_map = {
            "train": ("X_train", "y_train"),
            "val": ("X_valid", "y_valid"),
            "valid": ("X_valid", "y_valid"),
            "validation": ("X_valid", "y_valid"),
            "test": ("X_test", "y_test"),
        }
        x_key, y_key = _key_map[split]
        return (
            contents[x_key].astype(np.float32),
            contents[y_key].reshape(-1).astype(np.int64),
        )
    if _is_capture24_variant(dataset_name):
        processed_base = (
            data_path
            if data_path
            else ds_config["path_processed"]
            if ds_config
            else None
        )
        if processed_base:
            split_file = "val" if split in ("validation", "valid") else split
            npz_path = os.path.join(processed_base, f"{split_file}.npz")
            if os.path.exists(npz_path):
                ds = np.load(npz_path)
                return (ds["data"].astype(np.float32), ds["target"].astype(np.int64))
        raw_npz_path = ds_config["path_data"] if ds_config else None
        if not raw_npz_path or not os.path.exists(raw_npz_path):
            print(f"ERROR: Capture-24 .npz not found at {raw_npz_path}.")
            sys.exit(1)
        contents = np.load(raw_npz_path)
        _key_map = {
            "train": ("X_train", "y_train"),
            "val": ("X_valid", "y_valid"),
            "valid": ("X_valid", "y_valid"),
            "validation": ("X_valid", "y_valid"),
            "test": ("X_test", "y_test"),
        }
        x_key, y_key = _key_map[split]
        return (
            contents[x_key].astype(np.float32),
            contents[y_key].reshape(-1).astype(np.int64),
        )
    if dataset_name == "ucihar":
        candidates = []
        if data_path:
            candidates.append(data_path)
        if ds_config:
            if ds_config.get("path_data"):
                candidates.append(ds_config["path_data"])
            if ds_config.get("path_processed"):
                candidates.append(ds_config["path_processed"])
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]
        for base in candidates:
            x_path = os.path.join(base, f"X_{split}.npy")
            y_path = os.path.join(base, f"y_{split}.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                data = np.load(x_path).astype(np.float32)
                target = np.load(y_path).astype(np.longlong)
                if np.min(target) > 0:
                    target = target - 1
                return (data, target)
            npz_path = os.path.join(base, f"{split}.npz")
            if os.path.exists(npz_path):
                ds = np.load(npz_path)
                data = ds["data"].astype(np.float32)
                target = ds["target"].astype(np.longlong)
                if np.min(target) > 0:
                    target = target - 1
                return (data, target)
        print(f"ERROR: No data files found for ucihar split='{split}'.")
        print(f"       Searched in: {candidates}")
        sys.exit(1)
    if dataset_name == "shoaib":
        processed_base = (
            data_path
            if data_path
            else ds_config["path_processed"]
            if ds_config
            else None
        )
        if processed_base:
            split_file = "val" if split in ("validation", "valid") else split
            npz_path = os.path.join(processed_base, f"{split_file}.npz")
            if os.path.exists(npz_path):
                ds = np.load(npz_path)
                return (ds["data"].astype(np.float32), ds["target"].astype(np.int64))
        mat_path = (
            ds_config["path_data"] if ds_config else None
        ) or "./dataset/shoaib.mat"
        if not os.path.exists(mat_path):
            print(
                f"ERROR: Shoaib processed .npz not found in '{processed_base}' and .mat fallback not found at '{mat_path}'.\n       Run prepare_shoaib.py then preprocess.py first."
            )
            sys.exit(1)
        print(
            f"  WARNING: Falling back to sample-level .mat at {mat_path} — Mantis expects (window={ds_config['window']}, D=45) windows. Run preprocess.py to generate the processed .npz files."
        )
        import scipy.io as sio

        contents = sio.loadmat(mat_path)
        _key_map = {
            "train": ("X_train", "y_train"),
            "val": ("X_valid", "y_valid"),
            "valid": ("X_valid", "y_valid"),
            "validation": ("X_valid", "y_valid"),
            "test": ("X_test", "y_test"),
        }
        x_key, y_key = _key_map[split]
        return (
            contents[x_key].astype(np.float32),
            contents[y_key].reshape(-1).astype(np.int64),
        )
    if dataset_name in ("mhealth", "mhealth_nonull"):
        processed_base = (
            data_path
            if data_path
            else ds_config["path_processed"]
            if ds_config
            else None
        )
        if processed_base:
            split_file = "val" if split in ("validation", "valid") else split
            npz_path = os.path.join(processed_base, f"{split_file}.npz")
            if os.path.exists(npz_path):
                ds = np.load(npz_path)
                return (ds["data"].astype(np.float32), ds["target"].astype(np.int64))
        default_mat = (
            "./dataset/mhealth_nonull.mat"
            if dataset_name == "mhealth_nonull"
            else "./dataset/mhealth.mat"
        )
        mat_path = (ds_config["path_data"] if ds_config else None) or default_mat
        if not os.path.exists(mat_path):
            print(
                f"ERROR: MHEALTH processed .npz not found in '{processed_base}' and .mat fallback not found at '{mat_path}'.\n       Run prepare_mhealth.py then preprocess.py first."
            )
            sys.exit(1)
        print(
            f"  WARNING: Falling back to sample-level .mat at {mat_path} — Mantis expects (window={ds_config['window']}, D=23) windows. Run preprocess.py to generate the processed .npz files."
        )
        import scipy.io as sio

        contents = sio.loadmat(mat_path)
        _key_map = {
            "train": ("X_train", "y_train"),
            "val": ("X_valid", "y_valid"),
            "valid": ("X_valid", "y_valid"),
            "validation": ("X_valid", "y_valid"),
            "test": ("X_test", "y_test"),
        }
        x_key, y_key = _key_map[split]
        return (
            contents[x_key].astype(np.float32),
            contents[y_key].reshape(-1).astype(np.int64),
        )
    base = (
        data_path if data_path else ds_config["path_processed"] if ds_config else None
    )
    split_file = "val" if split == "validation" else split
    npz_path = os.path.join(base, f"{split_file}.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: Data file not found: {npz_path}")
        sys.exit(1)
    ds = np.load(npz_path)
    return (ds["data"].astype(np.float32), ds["target"].astype(np.longlong))


def _save_embedding_json(args_tuple):
    (
        output_dir,
        split,
        sample_idx,
        true_label,
        label_name,
        embedding,
        embedding_dim,
        class_map,
    ) = args_tuple
    safe_label = label_name.replace(" ", "_").replace("/", "_")
    filename = f"{split}_class{true_label}_{safe_label}_s{sample_idx:04d}.json"
    path = os.path.join(output_dir, filename)
    entry = {
        "sample_idx": int(sample_idx),
        "split": split,
        "true_label": int(true_label),
        "activity": label_name,
        "class_names": class_map,
        "mantis_embedding": {
            "model": f"MantisV2 (paris-noah/MantisV2, layer={MANTIS_RETURN_TRANSF_LAYER}, token={MANTIS_OUTPUT_TOKEN})",
            "embedding_dim": embedding_dim,
            "values": np.round(embedding, 6).tolist(),
        },
    }
    with open(path, "w") as f:
        f.write(_json_dumps(entry))
    return path


def save_combined_npz(
    split, data, labels, embeddings, valid_indices, output_dir, class_map
):
    path = os.path.join(output_dir, f"{split}_mantis_embeddings.npz")
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32),
        sample_indices=valid_indices.astype(np.int64),
        labels=labels[valid_indices].astype(np.int64),
    )
    print(f"  Combined NPZ → {path}")
    return path


def _resolve_data_path(dataset_name, args, ds_config):
    if args.data_path:
        return args.data_path
    if dataset_name == "ucihar":
        return ds_config.get("path_data") or ds_config.get("path_processed")
    return ds_config["path_processed"]


def process_dataset(dataset_name, args, mantis_trainer, device):
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 70}")
    ds_config = _get_dataset_config(dataset_name)
    class_map = ds_config["class_map"]
    data_path = _resolve_data_path(dataset_name, args, ds_config)
    output_dir = args.output_dir or f"./mantis_embeddings/{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Classes:    {len(class_map)} — {class_map}")
    print(f"  Data path:  {data_path}")
    print(f"  Output:     {output_dir}")
    null_class_indices = get_null_class_indices(class_map, dataset_name)
    if null_class_indices:
        null_names = [
            class_map[i] for i in sorted(null_class_indices) if i < len(class_map)
        ]
        print(f"  Null classes to skip: {null_names}")
    total_saved = 0
    total_null = 0
    emb_dim = None
    for split_name in args.splits:
        print(f"\n  ── Split: {split_name} ──")
        try:
            data, labels = load_data(
                dataset_name, data_path, split=split_name, ds_config=ds_config
            )
        except SystemExit:
            print(
                f"  WARNING: Could not load split '{split_name}' for {dataset_name}, skipping."
            )
            continue
        print(f"  Loaded: {data.shape[0]} samples, shape={data.shape}")
        labels_int = labels.astype(int)
        null_mask = (
            np.isin(labels_int, list(null_class_indices))
            if null_class_indices
            else np.zeros(len(labels_int), dtype=bool)
        )
        valid_indices = np.where(~null_mask)[0]
        n_null = int(null_mask.sum())
        n_valid = len(valid_indices)
        if n_null:
            print(f"  Null-class filter: skipping {n_null} / {data.shape[0]} samples")
        print(f"  Processing {n_valid} samples ...")
        valid_data = data[valid_indices]
        all_embeddings = []
        for start in tqdm(
            range(0, n_valid, args.mantis_batch_size),
            desc=f"  {split_name.upper()} Mantis",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + args.mantis_batch_size, n_valid)
            x_resized = _resize_batch_for_mantis(valid_data[start:end])
            emb = mantis_trainer.transform(x_resized)
            all_embeddings.append(emb)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        emb_dim = int(all_embeddings.shape[1])
        print(f"  Embeddings done — dim={emb_dim}, n={n_valid}")
        save_args = []
        for local_i, gi in enumerate(valid_indices):
            true_label = int(labels_int[gi])
            label_name = (
                class_map[true_label]
                if true_label < len(class_map)
                else f"Class_{true_label}"
            )
            save_args.append(
                (
                    output_dir,
                    split_name,
                    int(gi),
                    true_label,
                    label_name,
                    all_embeddings[local_i],
                    emb_dim,
                    class_map,
                )
            )
        print(f"  Saving {n_valid} JSON files ...")
        with ThreadPoolExecutor(max_workers=args.save_workers) as ex:
            list(
                tqdm(
                    ex.map(_save_embedding_json, save_args),
                    total=n_valid,
                    desc=f"  {split_name.upper()} saving",
                    unit="file",
                    dynamic_ncols=True,
                )
            )
        if args.save_npz:
            save_combined_npz(
                split_name,
                data,
                labels,
                all_embeddings,
                valid_indices,
                output_dir,
                class_map,
            )
        total_saved += n_valid
        total_null += n_null
        print(
            f"  Split done — saved: {n_valid}"
            + (f", null-skipped: {n_null}" if n_null else "")
        )
    summary = {
        "dataset": dataset_name,
        "splits": args.splits,
        "mantis_checkpoint": args.mantis_checkpoint,
        "mantis_layer": MANTIS_RETURN_TRANSF_LAYER,
        "mantis_token": MANTIS_OUTPUT_TOKEN,
        "embedding_dim": emb_dim,
        "total_saved": total_saved,
        "total_null_skipped": total_null,
        "null_class_indices": sorted(null_class_indices),
        "class_map": class_map,
        "class_map_source": "official_override"
        if dataset_name in _OFFICIAL_CLASS_MAPS
        else "settings.py",
    }
    summary_path = os.path.join(output_dir, "embedding_summary.json")
    with open(summary_path, "w") as f:
        f.write(_json_dumps(summary))
    print(f"  Summary → {summary_path}")
    return (total_saved, total_null)


ALL_DATASETS = [
    "opportunity",
    "skoda",
    "pamap2",
    "hospital",
    "ucihar",
    "uschad",
    "shoaib",
    "mhealth",
    "mhealth_nonull",
    "capture24",
    "capture24_walmsley",
    "capture24_full",
]
_DEFAULT_DATASETS = ["uschad"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MantisV2 embeddings for one or more datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=_DEFAULT_DATASETS,
        choices=ALL_DATASETS,
        help="Dataset(s) to process. Choose one or more from: "
        + ", ".join(ALL_DATASETS)
        + ".  Default: "
        + ", ".join(_DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data path (applies to all datasets).",
    )
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output dir (per-dataset subdirs are created inside).",
    )
    parser.add_argument("--mantis_checkpoint", type=str, default="paris-noah/MantisV2")
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(4, min(16, (os.cpu_count() or 8) // 2))
    parser.add_argument("--save_workers", type=int, default=_cpu_default)
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save a combined .npz per split for fast loading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Datasets:   {args.dataset}")
    print(f"Splits:     {args.splits}")
    print(f"Device:     {device}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print()
    print("[1/2] Loading MantisV2 ...")
    mantis_trainer = load_mantis_model(args.mantis_checkpoint, device)
    print()
    print("[2/2] Processing datasets ...")
    t0 = time.time()
    grand_total_saved = 0
    grand_total_null = 0
    for dataset_name in args.dataset:
        try:
            saved, null = process_dataset(dataset_name, args, mantis_trainer, device)
            grand_total_saved += saved
            grand_total_null += null
        except Exception as e:
            print(f"\n  ERROR processing {dataset_name}: {e}")
            print(f"  Skipping and continuing with remaining datasets.\n")
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All done.")
    print(f"  Datasets processed     : {', '.join(args.dataset)}")
    print(f"  Total embeddings saved : {grand_total_saved}")
    if grand_total_null:
        print(f"  Total null-class skipped: {grand_total_null}")
    print(f"  Total time             : {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
