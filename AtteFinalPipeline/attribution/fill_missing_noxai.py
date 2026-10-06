from AtteFinalPipeline.paths import PIPELINE_ROOT
from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.data.loading import _get_dataset_config, _is_capture24_variant
from AtteFinalPipeline.data.sensors import GROUPS_CAPTURE24, SENSORS_CAPTURE24
from AtteFinalPipeline.embeddings.encoders import (
    MANTIS_RETURN_TRANSF_LAYER,
    compute_mantis_embeddings_batch,
    load_mantis_model,
)
from AtteFinalPipeline.attribution.engine import batched_predict
import os
import sys
import re
import glob
import json
import time
import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from AtteFinalPipeline.expert.model import create


def get_sensor_config(n_channels, dataset_name="capture24", **kwargs):
    if dataset_name.startswith("capture24"):
        if n_channels == 3:
            return (list(SENSORS_CAPTURE24), list(GROUPS_CAPTURE24))
        names = [f"Ch_{i}" for i in range(n_channels)]
        return (names, [("All Channels", list(range(n_channels)), names)])
    names = [f"Ch_{i}" for i in range(n_channels)]
    return (names, [("All Channels", list(range(n_channels)), names)])


def load_data(dataset_name, data_path=None, split="test", ds_config=None):
    split = split.lower()
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


def build_result_dict_noxai(
    data_np, pred_class, probs, sample_idx, true_label, sensor_names, split, class_map
):
    T, D = data_np.shape
    confidence = float(probs[pred_class])
    sensor_importance = np.ones(D, dtype=np.float32) / D
    n_phases = 5
    phase_size = T // n_phases
    phase_importance = []
    for p in range(n_phases):
        phase_importance.append(1.0 / n_phases)
    label_name = (
        class_map[true_label] if true_label < len(class_map) else f"Class_{true_label}"
    )
    return {
        "sample_idx": int(sample_idx),
        "true_label": int(true_label),
        "predicted_label": pred_class,
        "label_name": label_name,
        "confidence": confidence,
        "correct": bool(int(pred_class) == int(true_label)),
        "split": split,
        "class_probabilities": {
            class_map[i]: float(probs[i]) for i in range(len(probs))
        },
        "sensor_importance_ranking": {
            sensor_names[i]: float(sensor_importance[i])
            for i in np.argsort(sensor_importance)[::-1]
        },
        "temporal_phase_importance": [
            {
                "phase": f"Phase {p} ({p * phase_size}-{min((p + 1) * phase_size, T)})",
                "importance": phase_importance[p],
            }
            for p in range(n_phases)
        ],
        "attribution_threshold_p90": 0.0,
        "high_attribution_regions": [],
        "_data": data_np,
        "_sensor_importance": sensor_importance,
        "_phase_importance": phase_importance,
    }


def build_sample_json_noxai(
    result,
    sensor_names,
    mantis_embedding=None,
    dataset_name="capture24",
    class_map=None,
):
    if class_map is None:
        class_map = ["Unknown"]
    data_np = result["_data"]
    T, D = data_np.shape
    split = result["split"]
    raw_sensor_data = {
        ch_name: np.round(data_np[:, ch_idx], 6).tolist()
        for ch_idx, ch_name in enumerate(sensor_names)
    }
    xai_methods = ["None — no-XAI ablation (full time-series, no attributed regions)"]
    if mantis_embedding is not None:
        xai_methods.append("MantisV2 Pre-trained Embedding")
    entry = {
        "analysis_metadata": {
            "model": "AttendDiscriminate",
            "dataset": dataset_name,
            "split": split,
            "num_classes": len(class_map),
            "class_names": class_map,
            "sensor_channels": sensor_names,
            "num_timesteps": T,
            "xai_methods": xai_methods,
            "attribution_combination": {
                "method": "none",
                "formula": "No attribution — ablation baseline",
                "eps": 0.0,
            },
            "region_extraction": {
                "method": "none",
                "percentile": 0,
                "description": "No-XAI ablation: no high-attribution regions extracted. The full time-series is provided as-is for downstream tasks.",
            },
        },
        "sample_idx": result["sample_idx"],
        "split": split,
        "activity": result["label_name"],
        "confidence": round(result["confidence"], 4),
        "class_probabilities": {
            k: round(v, 4) for k, v in result["class_probabilities"].items()
        },
        "sensor_importance_ranking": {
            k: round(v, 4) for k, v in result["sensor_importance_ranking"].items()
        },
        "temporal_phase_importance": [
            {"phase": p["phase"], "importance": round(p["importance"], 4)}
            for p in result["temporal_phase_importance"]
        ],
        "attribution_threshold_p90": 0.0,
        "num_high_attribution_regions": 0,
        "high_attribution_regions": [],
        "raw_sensor_data": raw_sensor_data,
    }
    if mantis_embedding is not None:
        entry["mantis_embedding"] = {
            "model": f"MantisV2 (paris-noah/MantisV2, layer={MANTIS_RETURN_TRANSF_LAYER})",
            "embedding_dim": int(mantis_embedding.shape[0]),
            "values": np.round(mantis_embedding, 6).tolist(),
        }
    return entry


def save_sample_json_noxai(
    result,
    output_dir,
    sensor_names,
    mantis_embedding=None,
    dataset_name="capture24",
    class_map=None,
):
    entry = build_sample_json_noxai(
        result,
        sensor_names,
        mantis_embedding,
        dataset_name=dataset_name,
        class_map=class_map,
    )
    label_name = result["label_name"]
    split = result["split"]
    safe_label = label_name.replace(" ", "_").replace("/", "_")
    filename = f"{split}_class{result['true_label']}_{safe_label}_s{result['sample_idx']:04d}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(_json_dumps(entry))
    return path


def find_checkpoint(dataset_name, explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    capture24_experiment_map = {
        "capture24": "capture24_Willetts2018_subset",
        "capture24_walmsley": "capture24_Walmsley2020_subset",
        "capture24_full": "capture24_Willetts2018_full",
    }
    search_patterns = []
    if dataset_name in capture24_experiment_map:
        exp = capture24_experiment_map[dataset_name]
        search_patterns.extend(
            [
                f"./models/capture24/train_{exp}/checkpoints/checkpoint_best.pth",
                f"./models/capture24/train_{exp}/checkpoints/checkpoint_*.pth",
                f"./models/{exp}/train_{exp}/checkpoints/checkpoint_best.pth",
                f"./models/{exp}/checkpoints/checkpoint_best.pth",
                f"./results/{exp}/checkpoints/checkpoint_best.pth",
            ]
        )
    search_patterns.extend(
        [
            f"./models/{dataset_name}/train_*/checkpoints/checkpoint_best.pth",
            f"./weights/checkpoint_{dataset_name}.pth",
        ]
    )
    best_matches = []
    numbered_matches = []
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for m in matches:
            if os.path.basename(m) == "checkpoint_best.pth":
                best_matches.append(m)
            else:
                numbered_matches.append(m)
    if best_matches:
        best_matches.sort(key=lambda p: os.path.getmtime(p))
        return best_matches[-1]
    if numbered_matches:
        numbered_matches.sort(key=lambda p: os.path.getmtime(p))
        return numbered_matches[-1]
    return None


def discover_missing(subset_dir, source_dir):
    METADATA = {
        "sample_manifest.csv",
        "sample_summary.json",
        "perf_log.jsonl",
        "prediction_summary.json",
        "progress.json",
    }
    subset_jsons = {
        f for f in os.listdir(subset_dir) if f.endswith(".json") and f not in METADATA
    }
    existing_jsons = set()
    if os.path.isdir(source_dir):
        existing_jsons = {
            f
            for f in os.listdir(source_dir)
            if f.endswith(".json") and f not in METADATA
        }
    missing = subset_jsons - existing_jsons
    parsed = {}
    for f in missing:
        m = re.match("(train|test)_class(\\d+)_(.+)_s(\\d+)\\.json", f)
        if m:
            split = m.group(1)
            idx = int(m.group(4))
            if split not in parsed:
                parsed[split] = []
            parsed[split].append(idx)
    for split in parsed:
        parsed[split].sort()
    return (missing, parsed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate missing no-XAI ablation JSONs for capture24 subset samples"
    )
    parser.add_argument("--dataset", type=str, default="capture24")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mantis_checkpoint", type=str, default="paris-noah/MantisV2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(8, min(24, (os.cpu_count() or 16) // 2))
    parser.add_argument("--save_workers", type=int, default=_cpu_default)
    return parser.parse_args()


def main():
    args = parse_args()
    BASE = PIPELINE_ROOT
    SUBSET_DIR = os.path.join(
        BASE, "xai_output", "capture24_willetts_subset_sampled_20pct"
    )
    NOXAI_SRC = os.path.join(BASE, "xai_output", "capture24_noxai")
    NOXAI_DST = os.path.join(
        BASE, "xai_output_noxai", "capture24_willetts_subset_sampled_20pct"
    )
    missing_files, missing_parsed = discover_missing(SUBSET_DIR, NOXAI_SRC)
    if os.path.isdir(NOXAI_DST):
        for f in list(missing_files):
            if os.path.exists(os.path.join(NOXAI_DST, f)):
                missing_files.discard(f)
    missing_parsed = {}
    for f in missing_files:
        m = re.match("(train|test)_class(\\d+)_(.+)_s(\\d+)\\.json", f)
        if m:
            split = m.group(1)
            idx = int(m.group(4))
            if split not in missing_parsed:
                missing_parsed[split] = []
            missing_parsed[split].append(idx)
    for s in missing_parsed:
        missing_parsed[s].sort()
    all_splits = sorted(missing_parsed.keys())
    total_missing = sum((len(v) for v in missing_parsed.values()))
    print(f"Missing noxai JSONs: {total_missing}")
    print(f"Splits: {all_splits}")
    print()
    if total_missing == 0:
        print("Nothing to do — all subset samples already have noxai JSONs.")
        return
    ds_config = _get_dataset_config(args.dataset)
    class_map = ds_config["class_map"]
    input_dim = ds_config["input_dim"]
    num_class = ds_config["num_class"]
    config_model_from_settings = ds_config["config_model"]
    os.makedirs(NOXAI_DST, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset:       {args.dataset}")
    print(f"Classes:       {num_class} — {class_map}")
    print(f"Input dim:     {input_dim}")
    print(f"Device:        {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()
    print("[1/4] Loading data ...")
    data_path = args.data_path or ds_config["path_processed"]
    split_data = {}
    for split_name in all_splits:
        d, t = load_data(args.dataset, data_path, split=split_name, ds_config=ds_config)
        split_data[split_name] = (d, t)
        print(f"  {split_name}: {d.shape[0]} samples, shape={d.shape}")
    first_split = all_splits[0]
    n_channels = split_data[first_split][0].shape[2]
    if n_channels != input_dim:
        print(
            f"  WARNING: Data has {n_channels} channels but settings expects {input_dim}. Using {n_channels}."
        )
        input_dim = n_channels
    sensor_names, sensor_groups = get_sensor_config(n_channels, args.dataset)
    print(f"  Sensor names ({len(sensor_names)}): {sensor_names}")
    print("\n[2/4] Building model + loading checkpoint ...")
    config_model = dict(config_model_from_settings)
    config_model["input_dim"] = input_dim
    config_model["num_class"] = num_class
    config_model["train_mode"] = False
    config_model["experiment"] = "noxai_ablation"
    model = create("AttendDiscriminate", config_model).to(device)
    ckpt_path = find_checkpoint(args.dataset, args.checkpoint)
    if ckpt_path:
        print(f"  Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("  Checkpoint loaded.")
    else:
        print("  ERROR: No checkpoint found. Pass --checkpoint <path> explicitly.")
        sys.exit(1)
    model.eval()
    print("\n[3/4] Loading MantisV2 ...")
    mantis_trainer = None
    if args.mantis_checkpoint and args.mantis_checkpoint.lower() != "none":
        mantis_trainer = load_mantis_model(args.mantis_checkpoint, device)
    else:
        print("  MantisV2 disabled.")
    print(f"\n[4/4] Processing missing samples (no-XAI ablation) ...")
    total_saved = 0
    t0 = time.time()
    for split_name in all_splits:
        data, labels = split_data[split_name]
        N = data.shape[0]
        target_indices = np.array(missing_parsed[split_name])
        valid = target_indices < N
        if not valid.all():
            n_invalid = (~valid).sum()
            print(
                f"  WARNING: {n_invalid} indices exceed data size {N} for {split_name}, skipping them"
            )
            target_indices = target_indices[valid]
        if len(target_indices) == 0:
            print(f"  [{split_name.upper()}] No missing samples.")
            continue
        print(
            f"\n  [{split_name.upper()}] Processing {len(target_indices)} missing samples ..."
        )
        print(
            f"  [{split_name.upper()}] Running batched prediction on full {split_name} set ({N} samples) ..."
        )
        pred_classes, all_probs = batched_predict(
            model, data, device, batch_size=args.batch_size
        )
        labels_int = labels.astype(int)
        target_correct = []
        target_skipped = 0
        for gi in target_indices:
            if pred_classes[gi] == labels_int[gi]:
                target_correct.append(gi)
            else:
                target_skipped += 1
        target_correct = np.array(target_correct)
        print(
            f"  [{split_name.upper()}] {len(target_correct)} correct, {target_skipped} incorrect (skipped)"
        )
        if len(target_correct) == 0:
            continue
        save_executor = ThreadPoolExecutor(max_workers=args.save_workers)
        pending_futures = []
        for gi in tqdm(
            target_correct,
            desc=f"  {split_name.upper()} samples",
            unit="sample",
            dynamic_ncols=True,
        ):
            gi = int(gi)
            pred_class = int(pred_classes[gi])
            result = build_result_dict_noxai(
                data[gi],
                pred_class,
                all_probs[gi],
                gi,
                int(labels_int[gi]),
                sensor_names,
                split_name,
                class_map=class_map,
            )
            future = save_executor.submit(
                save_sample_json_noxai,
                result,
                NOXAI_DST,
                sensor_names,
                None,
                args.dataset,
                class_map,
            )
            pending_futures.append((future, gi))
            total_saved += 1
            still_pending = []
            for f, idx in pending_futures:
                if f.done():
                    try:
                        f.result()
                    except Exception as e:
                        print(f"\n  WARNING: Save failed for sample {idx}: {e}")
                else:
                    still_pending.append((f, idx))
            pending_futures = still_pending
        if pending_futures:
            print(
                f"  [{split_name.upper()}] Draining {len(pending_futures)} remaining saves ..."
            )
            for future, gi in pending_futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  WARNING: Save failed for sample {gi}: {e}")
        save_executor.shutdown(wait=True)
        if mantis_trainer is not None and len(target_correct) > 0:
            print(f"  [{split_name.upper()}] Computing Mantis embeddings ...")
            mantis_data = data[target_correct]
            all_emb = compute_mantis_embeddings_batch(
                mantis_trainer, mantis_data, device, batch_size=args.mantis_batch_size
            )
            print(
                f"  [{split_name.upper()}] Mantis done (dim={all_emb.shape[1]}). Patching JSONs ..."
            )

            def _patch_json(patch_args):
                gi, emb = patch_args
                true_label = int(labels_int[gi])
                label_name = (
                    class_map[true_label]
                    if true_label < len(class_map)
                    else f"Class_{true_label}"
                )
                safe_label = label_name.replace(" ", "_").replace("/", "_")
                filename = f"{split_name}_class{true_label}_{safe_label}_s{gi:04d}.json"
                path = os.path.join(NOXAI_DST, filename)
                if not os.path.exists(path):
                    return
                try:
                    with open(path, "r") as f:
                        entry = json.loads(f.read())
                    entry["mantis_embedding"] = {
                        "model": f"MantisV2 (paris-noah/MantisV2, layer={MANTIS_RETURN_TRANSF_LAYER})",
                        "embedding_dim": int(emb.shape[0]),
                        "values": np.round(emb, 6).tolist(),
                    }
                    with open(path, "w") as f:
                        f.write(_json_dumps(entry))
                except Exception as e:
                    print(f"\n  WARNING: JSON patch failed for sample {gi}: {e}")

            with ThreadPoolExecutor(max_workers=args.save_workers) as patch_executor:
                list(
                    patch_executor.map(
                        _patch_json,
                        [(int(gi), all_emb[i]) for i, gi in enumerate(target_correct)],
                    )
                )
            print(f"  [{split_name.upper()}] JSON patch complete.")
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done!")
    print(f"  Noxai JSONs saved: {total_saved} → {NOXAI_DST}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
