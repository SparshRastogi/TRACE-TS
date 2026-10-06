from AtteFinalPipeline.paths import PIPELINE_ROOT
from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.data.sensors import (
    GROUPS_CAPTURE24,
    SENSORS_CAPTURE24,
    get_null_class_indices,
)
from AtteFinalPipeline.data.loading import _is_capture24_variant
from AtteFinalPipeline.embeddings.encoders import (
    MANTIS_RETURN_TRANSF_LAYER,
    compute_mantis_embeddings_batch,
    load_mantis_model,
)
from AtteFinalPipeline.attribution.engine import _LogitWrapper, batched_predict
from AtteFinalPipeline.attribution.regions import build_result_dict
import os
import sys
import re
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from AtteFinalPipeline.baselines.deepconvlstm.model import DeepConvLSTM
from AtteFinalPipeline.baselines.deepconvlstm.settings import (
    _DATASET_CONFIGS,
    _MODEL_DEFAULTS,
)
from captum.attr import IntegratedGradients

MODEL_NAME = "DeepConvLSTM"


class _DeepConvLSTMAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return (None, logits, None)


def _get_dataset_config(dataset_name):
    if dataset_name not in _DATASET_CONFIGS:
        print(
            f"ERROR: Unknown dataset '{dataset_name}'. Available: {list(_DATASET_CONFIGS.keys())}"
        )
        sys.exit(1)
    ds = _DATASET_CONFIGS[dataset_name]
    config_model = dict(_MODEL_DEFAULTS)
    config_model["window_size"] = ds["window"]
    config_model["nb_channels"] = ds["input_dim"]
    config_model["nb_classes"] = ds["num_class"]
    config_model["seed"] = 42
    return {
        "class_map": list(ds["class_map"]),
        "input_dim": ds["input_dim"],
        "num_class": ds["num_class"],
        "window": ds["window"],
        "stride": ds["stride"],
        "path_data": ds["path_data"],
        "path_raw": ds["path_raw"],
        "path_processed": ds["path_processed"],
        "config_model": config_model,
    }


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


class XAIEngine:
    def __init__(
        self,
        model,
        ig_baseline,
        shap_background,
        device,
        use_compile=True,
        use_amp=False,
    ):
        self.device = device
        device_type = device.type
        self.wrapper = _LogitWrapper(model, use_amp=use_amp, device_type=device_type)
        self.wrapper.train()
        for m in self.wrapper.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()
        if use_compile and torch.cuda.is_available():
            print("  Compiling wrapper with torch.compile(mode='default') ...")
            self.compiled_wrapper = torch.compile(
                self.wrapper, mode="default", fullgraph=False
            )
            print("  Compilation registered (JIT will fire on first call).")
        else:
            self.compiled_wrapper = self.wrapper
        self.ig = IntegratedGradients(self.compiled_wrapper)
        self.ig_baseline = ig_baseline
        self._shap_background = shap_background
        self._n_bg = shap_background.shape[0]
        self._shap_rng = torch.Generator(device=device)
        self._shap_rng.manual_seed(42)
        print("  Caching num_classes (may trigger first JIT compile, ~30s) ...")
        with torch.no_grad():
            self._num_classes = int(self.compiled_wrapper(self.ig_baseline).shape[-1])
        print(
            f"  XAIEngine ready | num_classes={self._num_classes} | compile={('ON' if use_compile else 'OFF')} | AMP={('ON' if use_amp else 'OFF')} | cuDNN=ON"
        )

    def warm_up(self, sample_shape, target_class=0, n_steps=5, production_n_steps=None):
        print(f"  Warm-up pass 1: fast kernel (n_steps={n_steps}) ...")
        dummy = torch.zeros((1, *sample_shape), device=self.device)
        _ = self.ig.attribute(
            dummy,
            baselines=self.ig_baseline,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=n_steps,
        )
        _ = self.compute_shap_batch(
            dummy, torch.tensor([target_class], device=self.device)
        )
        print(f"  Warm-up pass 1 done.")
        if production_n_steps is not None and production_n_steps != n_steps:
            print(
                f"  Warm-up pass 2: production kernel (n_steps={production_n_steps}) ..."
            )
            _ = self.ig.attribute(
                dummy,
                baselines=self.ig_baseline,
                target=target_class,
                n_steps=production_n_steps,
                internal_batch_size=production_n_steps,
            )
            print(f"  Warm-up pass 2 done.")
        print("  Warm-up complete — all Triton kernels cached.")

    def compute_ig_batch(self, batch_tensor, target_classes, n_steps=25):
        B = batch_tensor.shape[0]
        internal_bs = max(1, min(n_steps * B, 512))
        baseline = self.ig_baseline.expand(B, -1, -1)
        attr = self.ig.attribute(
            batch_tensor,
            baselines=baseline,
            target=target_classes,
            n_steps=n_steps,
            internal_batch_size=internal_bs,
        )
        attr_np = attr.detach().cpu().numpy()
        np.abs(attr_np, out=attr_np)
        maxvals = attr_np.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        attr_np[nonzero] /= maxvals[nonzero, None, None]
        return attr_np

    def compute_shap_batch(self, batch_tensor, target_classes):
        B = batch_tensor.shape[0]
        bg = self._shap_background.detach()
        n_bg = self._n_bg
        alphas = torch.rand(
            B,
            n_bg,
            1,
            1,
            device=self.device,
            dtype=batch_tensor.dtype,
            generator=self._shap_rng,
        )
        x_exp = batch_tensor.unsqueeze(1)
        bg_exp = bg.unsqueeze(0)
        diffs = x_exp - bg_exp
        interp = (bg_exp + alphas * diffs).reshape(B * n_bg, *batch_tensor.shape[1:])
        interp = interp.requires_grad_(True)
        self.compiled_wrapper.zero_grad(set_to_none=True)
        logits = self.compiled_wrapper(interp)
        targets_expanded = target_classes.repeat_interleave(n_bg)
        target_logits = logits.gather(1, targets_expanded.unsqueeze(1)).squeeze(1)
        target_logits.sum().backward()
        grads = interp.grad.reshape(B, n_bg, *batch_tensor.shape[1:])
        diffs_r = diffs
        attr = (grads * diffs_r).mean(dim=1)
        attr_np = attr.detach().cpu().numpy()
        np.abs(attr_np, out=attr_np)
        maxvals = attr_np.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        attr_np[nonzero] /= maxvals[nonzero, None, None]
        return attr_np

    def compute_combined_batch(
        self, batch_tensor, target_classes, n_steps=25, eps=1e-08
    ):
        ig_attrs = self.compute_ig_batch(batch_tensor, target_classes, n_steps)
        shap_attrs = self.compute_shap_batch(batch_tensor, target_classes)
        return self._combine_batch(ig_attrs, shap_attrs, eps=eps)

    def _combine_batch(self, ig_attrs, shap_attrs, eps=1e-08):
        combined = np.sqrt((ig_attrs + eps) * (shap_attrs + eps))
        B = combined.shape[0]
        maxvals = combined.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        combined[nonzero] /= maxvals[nonzero, None, None]
        return combined


def build_sample_json(
    result,
    sensor_names,
    mantis_embedding=None,
    dataset_name="capture24",
    class_map=None,
):
    if class_map is None:
        class_map = result.get("_class_map", ["Unknown"])
    data_np = result["_data"]
    T, D = data_np.shape
    split = result["split"]
    raw_sensor_data = {
        ch_name: np.round(data_np[:, ch_idx], 6).tolist()
        for ch_idx, ch_name in enumerate(sensor_names)
    }
    xai_methods = [
        "Captum Integrated Gradients (training-mean baseline)",
        "Batched Gradient SHAP (expected gradient, vectorised over background)",
        "Combination: Geometric Mean — continuous AND",
    ]
    if mantis_embedding is not None:
        xai_methods.append("MantisV2 Pre-trained Embedding")
    regions = result["high_attribution_regions"]
    entry = {
        "analysis_metadata": {
            "model": MODEL_NAME,
            "dataset": dataset_name,
            "split": split,
            "num_classes": len(class_map),
            "class_names": class_map,
            "sensor_channels": sensor_names,
            "num_timesteps": T,
            "xai_methods": xai_methods,
            "attribution_combination": {
                "method": "geometric_mean",
                "formula": "combined = sqrt( (IG + eps) * (SHAP + eps) )",
                "eps": 1e-08,
            },
            "region_extraction": {
                "method": "global_percentile_threshold",
                "percentile": 90,
                "description": "Contiguous runs of timesteps per sensor where the combined attribution >= global 90th percentile.",
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
        "attribution_threshold_p90": round(result["attribution_threshold_p90"], 6),
        "num_high_attribution_regions": len(regions),
        "high_attribution_regions": [
            {
                "sensor": reg["sensor"],
                "sensor_idx": reg["sensor_idx"],
                "start_t": reg["start_t"],
                "end_t": reg["end_t"],
                "length": reg["length"],
                "mean_importance": round(reg["mean_importance"], 6),
                "max_importance": round(reg["max_importance"], 6),
                "peak_timestep": reg["peak_timestep"],
                "importance_values": reg["importance_values"],
            }
            for reg in regions
        ],
        "raw_sensor_data": raw_sensor_data,
    }
    if mantis_embedding is not None:
        entry["mantis_embedding"] = {
            "model": f"MantisV2 (paris-noah/MantisV2, layer={MANTIS_RETURN_TRANSF_LAYER})",
            "embedding_dim": int(mantis_embedding.shape[0]),
            "values": np.round(mantis_embedding, 6).tolist(),
        }
    return entry


def save_sample_json(
    result,
    output_dir,
    sensor_names,
    mantis_embedding=None,
    dataset_name="capture24",
    class_map=None,
):
    entry = build_sample_json(
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
    search_patterns = [
        f"./results_deepconvlstm/{dataset_name}/checkpoints/checkpoint_best.pth",
        f"./results_deepconvlstm/{dataset_name}/checkpoints/checkpoint_*.pth",
    ]
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
        description="Generate missing deepconvlstm XAI JSONs for capture24 subset samples"
    )
    parser.add_argument("--dataset", type=str, default="capture24")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mantis_checkpoint", type=str, default="paris-noah/MantisV2")
    parser.add_argument("--shap_bg_per_class", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--xai_batch_size", type=int, default=32)
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    _cpu_default = max(8, min(24, (os.cpu_count() or 16) // 2))
    parser.add_argument("--plot_workers", type=int, default=_cpu_default)
    return parser.parse_args()


def main():
    args = parse_args()
    BASE = PIPELINE_ROOT
    SUBSET_DIR = os.path.join(
        BASE, "xai_output", "capture24_willetts_subset_sampled_20pct"
    )
    DCL_SRC = os.path.join(BASE, "xai_output_deepconvlstm", "capture24")
    DCL_DST = os.path.join(
        BASE, "xai_output_deepconvlstm", "capture24_willetts_subset_sampled_20pct"
    )
    missing_files, missing_parsed = discover_missing(SUBSET_DIR, DCL_SRC)
    if os.path.isdir(DCL_DST):
        for f in list(missing_files):
            if os.path.exists(os.path.join(DCL_DST, f)):
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
    print(f"Missing deepconvlstm JSONs: {total_missing}")
    print(f"Splits: {all_splits}")
    print()
    if total_missing == 0:
        print("Nothing to do — all subset samples already have deepconvlstm JSONs.")
        return
    ds_config = _get_dataset_config(args.dataset)
    class_map = ds_config["class_map"]
    input_dim = ds_config["input_dim"]
    num_class = ds_config["num_class"]
    config_model_settings = ds_config["config_model"]
    os.makedirs(DCL_DST, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model:         {MODEL_NAME}")
    print(f"Dataset:       {args.dataset}")
    print(f"Classes:       {num_class} — {class_map}")
    print(f"Input dim:     {input_dim}")
    print(f"Device:        {device}")
    print(f"XAI batch:     {args.xai_batch_size}")
    print(f"IG n_steps:    {args.n_steps}")
    print(f"SHAP bg/class: {args.shap_bg_per_class}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()
    print("[1/6] Loading data ...")
    data_path = args.data_path or ds_config["path_processed"]
    split_data = {}
    for split_name in all_splits:
        d, t = load_data(args.dataset, data_path, split=split_name, ds_config=ds_config)
        split_data[split_name] = (d, t)
        print(f"  {split_name}: {d.shape[0]} samples, shape={d.shape}")
    first_split = all_splits[0]
    n_channels = split_data[first_split][0].shape[2]
    window_size = split_data[first_split][0].shape[1]
    if n_channels != input_dim:
        print(
            f"  WARNING: Data has {n_channels} channels but settings expects {input_dim}. Using {n_channels}."
        )
        input_dim = n_channels
    sensor_names, sensor_groups = get_sensor_config(n_channels, args.dataset)
    print(f"  Sensor names ({len(sensor_names)}): {sensor_names}")
    null_class_indices = get_null_class_indices(class_map, args.dataset)
    print("\n[2/6] Building IG baseline and SHAP background ...")
    if "train" in split_data:
        train_data, train_labels = split_data["train"]
    else:
        train_data, train_labels = load_data(
            args.dataset, data_path, split="train", ds_config=ds_config
        )
    if null_class_indices:
        null_mask_train = np.isin(train_labels.astype(int), list(null_class_indices))
        train_data_bg = train_data[~null_mask_train]
        train_labels_bg = train_labels[~null_mask_train]
    else:
        train_data_bg, train_labels_bg = (train_data, train_labels)
    baseline_np = train_data_bg.mean(axis=0).astype(np.float32)
    ig_baseline = torch.tensor(baseline_np, dtype=torch.float32).unsqueeze(0).to(device)
    rng = np.random.RandomState(args.seed)
    bg_indices = []
    for cls in sorted(np.unique(train_labels_bg)):
        cls_idxs = np.where(train_labels_bg == cls)[0]
        n = min(args.shap_bg_per_class, len(cls_idxs))
        bg_indices.extend(rng.choice(cls_idxs, size=n, replace=False).tolist())
    shap_background = torch.tensor(train_data_bg[bg_indices], dtype=torch.float32).to(
        device
    )
    print(f"  IG baseline shape: {tuple(ig_baseline.shape)}")
    print(f"  SHAP background: {shap_background.shape[0]} samples")
    print("\n[3/6] Loading MantisV2 ...")
    mantis_trainer = None
    if args.mantis_checkpoint and args.mantis_checkpoint.lower() != "none":
        mantis_trainer = load_mantis_model(args.mantis_checkpoint, device)
    else:
        print("  MantisV2 disabled.")
    print(f"\n[4/6] Building {MODEL_NAME} model ...")
    config_model_settings["window_size"] = window_size
    config_model_settings["nb_channels"] = n_channels
    config_model_settings["nb_classes"] = num_class
    raw_model = DeepConvLSTM(config_model_settings).to(device)
    print(f"  Parameters: {raw_model.number_of_parameters():,}")
    print("\n[5/6] Loading checkpoint ...")
    ckpt_path = find_checkpoint(args.dataset, args.checkpoint)
    if ckpt_path:
        print(f"  Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        raw_model.load_state_dict(state_dict, strict=False)
        print("  Checkpoint loaded.")
    else:
        print("  ERROR: No checkpoint found. Pass --checkpoint <path> explicitly.")
        sys.exit(1)
    raw_model.eval()
    model = _DeepConvLSTMAdapter(raw_model)
    xai_engine = XAIEngine(
        model,
        ig_baseline,
        shap_background,
        device,
        use_compile=not args.no_compile,
        use_amp=args.use_amp,
    )
    if not args.no_compile and torch.cuda.is_available():
        xai_engine.warm_up(
            sample_shape=(window_size, n_channels),
            target_class=0,
            n_steps=5,
            production_n_steps=args.n_steps,
        )
    print(f"\n[6/6] Processing missing samples ...")
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
        save_executor = ThreadPoolExecutor(max_workers=args.plot_workers)
        pending_futures = []
        n_correct = len(target_correct)
        pbar = tqdm(
            range(0, n_correct, args.xai_batch_size),
            desc=f"  {split_name.upper()} XAI batches",
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for batch_start in pbar:
            batch_end = min(batch_start + args.xai_batch_size, n_correct)
            batch_global_indices = target_correct[batch_start:batch_end]
            B = len(batch_global_indices)
            batch_np = data[batch_global_indices]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)
            batch_preds = torch.tensor(
                [int(pred_classes[gi]) for gi in batch_global_indices],
                dtype=torch.long,
                device=device,
            )
            combined_batch = xai_engine.compute_combined_batch(
                batch_tensor, batch_preds, n_steps=args.n_steps
            )
            for local_i, gi in enumerate(batch_global_indices):
                gi = int(gi)
                pred_class = int(pred_classes[gi])
                combined = combined_batch[local_i]
                result = build_result_dict(
                    data[gi],
                    combined,
                    pred_class,
                    all_probs[gi],
                    gi,
                    int(labels_int[gi]),
                    sensor_names,
                    split_name,
                    class_map=class_map,
                )
                future = save_executor.submit(
                    save_sample_json,
                    result,
                    DCL_DST,
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
        pbar.close()
        del batch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
                path = os.path.join(DCL_DST, filename)
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

            with ThreadPoolExecutor(max_workers=args.plot_workers) as patch_executor:
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
    print(f"  DeepConvLSTM JSONs saved: {total_saved} → {DCL_DST}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
