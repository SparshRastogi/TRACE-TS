from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.attribution.engine import batched_predict
from AtteFinalPipeline.attribution.progress import ProgressTracker
import os
import sys
import json
import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from AtteFinalPipeline.expert.model import create

try:
    import wandb as _wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
from AtteFinalPipeline.data.loading import _get_dataset_config
from AtteFinalPipeline.data.sensors import get_sensor_config
from AtteFinalPipeline.data.sensors import get_null_class_indices
from AtteFinalPipeline.data.loading import load_data
from AtteFinalPipeline.embeddings.encoders import MANTIS_RETURN_TRANSF_LAYER
from AtteFinalPipeline.embeddings.encoders import load_mantis_model
from AtteFinalPipeline.embeddings.encoders import compute_mantis_embeddings_batch
from AtteFinalPipeline.attribution.checkpoints import find_checkpoint


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
    result, sensor_names, mantis_embedding=None, dataset_name="ucihar", class_map=None
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
    dataset_name="ucihar",
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


from AtteFinalPipeline.attribution.combined import build_and_save_summary


def process_split(
    split,
    data,
    labels,
    model,
    mantis_trainer,
    output_dir,
    sensor_names,
    device,
    progress_tracker,
    class_map,
    dataset_name,
    predict_batch_size=256,
    mantis_batch_size=32,
    save_workers=8,
    min_confidence=0.0,
    worker_id=0,
    num_workers=1,
    null_class_indices=None,
):
    if null_class_indices is None:
        null_class_indices = set()
    N = data.shape[0]
    print(f"\n  [{split.upper()}] Batched prediction for {N} samples ...")
    pred_classes, all_probs = batched_predict(
        model, data, device, batch_size=predict_batch_size
    )
    labels_int = labels.astype(int)
    if null_class_indices:
        null_mask = np.isin(labels_int, list(null_class_indices))
        n_null = int(null_mask.sum())
        valid_mask = ~null_mask
        print(
            f"  [{split.upper()}] Null-class filter: skipping {n_null} samples ({N - n_null} remaining)"
        )
    else:
        null_mask = np.zeros(N, dtype=bool)
        valid_mask = np.ones(N, dtype=bool)
        n_null = 0
    valid_indices = np.where(valid_mask)[0]
    labels_valid = labels_int[valid_indices]
    pred_valid = pred_classes[valid_indices]
    correct_mask_valid = pred_valid == labels_valid
    correct_indices_local = np.where(correct_mask_valid)[0]
    incorrect_indices_local = np.where(~correct_mask_valid)[0]
    correct_indices = valid_indices[correct_indices_local]
    incorrect_indices = valid_indices[incorrect_indices_local]
    n_correct_total = len(correct_indices)
    n_incorrect = len(incorrect_indices)
    if num_workers > 1:
        chunk_size = int(np.ceil(n_correct_total / num_workers))
        slice_start = worker_id * chunk_size
        slice_end = min(slice_start + chunk_size, n_correct_total)
        correct_indices = correct_indices[slice_start:slice_end]
        print(
            f"  [{split.upper()}] Worker {worker_id}/{num_workers}: handling correct samples {slice_start}-{slice_end - 1} of {n_correct_total} total"
        )
    n_correct = len(correct_indices)
    print(
        f"  [{split.upper()}] Correct (this worker): {n_correct}, Incorrect (total): {n_incorrect}"
        + (f", Null-skipped: {n_null}" if n_null else "")
    )
    all_tracking = []
    for i in range(N):
        all_tracking.append(
            {
                "sample_idx": int(i),
                "true_label": int(labels_int[i]),
                "predicted_label": int(pred_classes[i]),
                "correct": False
                if null_mask[i]
                else bool(pred_classes[i] == labels_int[i]),
                "split": split,
                "null_skipped": bool(null_mask[i]),
            }
        )
    if worker_id == 0:
        for _ in np.where(null_mask)[0]:
            progress_tracker.update(split, correct=False)
        for _ in incorrect_indices:
            progress_tracker.update(split, correct=False)
    progress_tracker.write_periodic(every_n=1)
    if min_confidence > 0.0:
        conf_mask = np.array(
            [
                float(all_probs[gi][int(pred_classes[gi])]) >= min_confidence
                for gi in correct_indices
            ]
        )
        correct_indices = correct_indices[conf_mask]
        n_correct = len(correct_indices)
        print(
            f"  [{split.upper()}] After confidence filter (>={min_confidence:.2f}): {n_correct} samples remain"
        )
    print(f"  [{split.upper()}] Saving {n_correct} sample JSONs (no-XAI ablation) ...")
    saved_indices = []
    n_saved = 0

    def _save_one(args_tuple):
        result, out_dir, s_names, m_emb, ds_name, c_map = args_tuple
        save_sample_json_noxai(
            result, out_dir, s_names, m_emb, dataset_name=ds_name, class_map=c_map
        )

    save_executor = ThreadPoolExecutor(max_workers=save_workers)
    pending_futures = []
    try:
        for gi in tqdm(
            correct_indices,
            desc=f"  {split.upper()} samples",
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
                split,
                class_map=class_map,
            )
            save_arg = (result, output_dir, sensor_names, None, dataset_name, class_map)
            future = save_executor.submit(_save_one, save_arg)
            pending_futures.append((future, gi))
            saved_indices.append(gi)
            n_saved += 1
            progress_tracker.update(split, correct=True)
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
            progress_tracker.write_periodic(every_n=50)
        if pending_futures:
            print(
                f"  [{split.upper()}] Draining {len(pending_futures)} remaining saves ..."
            )
            for future, gi in pending_futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  WARNING: Save failed for sample {gi}: {e}")
    finally:
        save_executor.shutdown(wait=True)
    print(f"  [{split.upper()}] All {n_saved} JSONs saved.")
    if mantis_trainer is not None and n_saved > 0:
        print(
            f"  [{split.upper()}] Computing Mantis embeddings (batch={mantis_batch_size}) ..."
        )
        saved_indices_arr = np.array(saved_indices)
        correct_data_for_mantis = data[saved_indices_arr]
        all_emb = compute_mantis_embeddings_batch(
            mantis_trainer,
            correct_data_for_mantis,
            device,
            batch_size=mantis_batch_size,
        )
        print(
            f"  [{split.upper()}] Mantis done (dim={all_emb.shape[1]}). Patching JSONs ..."
        )

        def _patch_json(args):
            gi, emb = args
            true_label = int(labels_int[gi])
            label_name = (
                class_map[true_label]
                if true_label < len(class_map)
                else f"Class_{true_label}"
            )
            safe_label = label_name.replace(" ", "_").replace("/", "_")
            filename = f"{split}_class{true_label}_{safe_label}_s{gi:04d}.json"
            path = os.path.join(output_dir, filename)
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

        with ThreadPoolExecutor(max_workers=save_workers) as patch_executor:
            list(
                patch_executor.map(
                    _patch_json,
                    [(gi, all_emb[i]) for i, gi in enumerate(saved_indices)],
                )
            )
        print(f"  [{split.upper()}] JSON patch complete.")
    progress_tracker.finish_split(split)
    print(
        f"  [{split.upper()}] Done — saved: {n_saved}, skipped: {n_incorrect + (n_correct - n_saved)}"
        + (f", null-skipped: {n_null}" if n_null else "")
    )
    return all_tracking


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-XAI Ablation: same JSON format as attribution pipeline, but with no attributed regions — full time-series only."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="capture24",
        choices=[
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
        ],
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--mat_file", type=str, default=None)
    parser.add_argument("--opportunity_col_names", type=str, default=None)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mantis_checkpoint", type=str, default="paris-noah/MantisV2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(8, min(24, (os.cpu_count() or 16) // 2))
    parser.add_argument("--save_workers", type=int, default=_cpu_default)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ds_config = _get_dataset_config(args.dataset)
    class_map = ds_config["class_map"]
    input_dim = ds_config["input_dim"]
    num_class = ds_config["num_class"]
    config_model_from_settings = ds_config["config_model"]
    if args.output_dir is None:
        args.output_dir = f"./xai_output/{args.dataset}_noxai/"
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.worker_id >= args.num_workers:
        print(f"ERROR: worker_id={args.worker_id} >= num_workers={args.num_workers}")
        sys.exit(1)
    if _WANDB_OK and args.worker_id == 0:
        try:
            _wandb.init(
                project="sensorllm",
                name=f"noxai_ablation_{args.dataset}",
                tags=[args.dataset, "noxai_ablation"],
                config=vars(args),
                dir=args.output_dir,
                resume="allow",
            )
            print(f"W&B run initialised: {_wandb.run.url}")
        except Exception as e:
            print(f"W&B init failed (continuing without it): {e}")
    print(f"{'=' * 60}")
    print(f"  NO-XAI ABLATION — full time-series, no attribution regions")
    print(f"{'=' * 60}")
    print(f"Dataset:       {args.dataset}")
    print(f"Classes:       {num_class} — {class_map}")
    print(f"Input dim:     {input_dim}")
    print(f"Device:        {device}")
    print(f"Worker:        {args.worker_id} of {args.num_workers}")
    print(f"Splits:        {args.splits}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()
    print("[1/5] Loading data ...")
    data_path = args.data_path or ds_config["path_processed"]
    split_data = {}
    for split_name in args.splits:
        d, t = load_data(args.dataset, data_path, split=split_name, ds_config=ds_config)
        split_data[split_name] = (d, t)
        print(f"  {split_name}: {d.shape[0]} samples, shape={d.shape}")
    first_split = args.splits[0]
    n_channels = split_data[first_split][0].shape[2]
    if n_channels != input_dim:
        print(
            f"  WARNING: Data has {n_channels} channels but settings.py expects input_dim={input_dim}. Using actual={n_channels}."
        )
        input_dim = n_channels
    sensor_names, sensor_groups = get_sensor_config(
        n_channels,
        args.dataset,
        mat_path=args.mat_file,
        opportunity_col_names_path=args.opportunity_col_names,
    )
    print(f"  Sensor names ({len(sensor_names)}): {sensor_names}")
    null_class_indices = get_null_class_indices(class_map, args.dataset)
    print("\n[2/5] Loading MantisV2 ...")
    mantis_trainer = None
    if args.mantis_checkpoint and args.mantis_checkpoint.lower() != "none":
        mantis_trainer = load_mantis_model(args.mantis_checkpoint, device)
    else:
        print("  MantisV2 disabled.")
    print("\n[3/5] Building AttendDiscriminate model ...")
    config_model = dict(config_model_from_settings)
    config_model["input_dim"] = input_dim
    config_model["num_class"] = num_class
    config_model["train_mode"] = False
    config_model["experiment"] = "noxai_ablation"
    model = create("AttendDiscriminate", config_model).to(device)
    print("\n[4/5] Loading checkpoint ...")
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
    print(f"\n[5/5] Processing ({', '.join(args.splits)}) — no XAI ...")
    total_by_split = {s: split_data[s][0].shape[0] for s in args.splits}
    progress_tracker = ProgressTracker(
        args.output_dir, total_by_split, worker_id=args.worker_id
    )
    all_tracking = []
    for split_name in args.splits:
        s_data, s_labels = split_data[split_name]
        tracking = process_split(
            split_name,
            s_data,
            s_labels,
            model,
            mantis_trainer,
            args.output_dir,
            sensor_names,
            device,
            progress_tracker,
            class_map=class_map,
            dataset_name=args.dataset,
            predict_batch_size=args.batch_size,
            mantis_batch_size=args.mantis_batch_size,
            save_workers=args.save_workers,
            min_confidence=args.min_confidence,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            null_class_indices=null_class_indices,
        )
        all_tracking.extend(tracking)
    progress_tracker.finish()
    tracking_path = os.path.join(
        args.output_dir, f"tracking_worker{args.worker_id}.json"
    )
    with open(tracking_path, "w") as f:
        f.write(_json_dumps(all_tracking))
    print(f"  Tracking written: {tracking_path}")
    if args.worker_id == 0 and args.num_workers > 1:
        print(f"\nWorker 0: waiting for all workers to finish tracking ...")
        import time as _time

        all_tracking_merged = []
        for wid in range(args.num_workers):
            wpath = os.path.join(args.output_dir, f"tracking_worker{wid}.json")
            waited = 0
            while not os.path.exists(wpath) and waited < 3600:
                _time.sleep(5)
                waited += 5
            if os.path.exists(wpath):
                with open(wpath) as f:
                    all_tracking_merged.extend(json.loads(f.read()))
            else:
                print(f"  WARNING: tracking for worker {wid} never appeared, skipping.")
        all_tracking = all_tracking_merged
    print(f"\nGenerating summary ...")
    if args.worker_id == 0:
        summary_tracking = [t for t in all_tracking if not t.get("null_skipped", False)]
        summary_path, summary = build_and_save_summary(
            summary_tracking, args.output_dir, class_map
        )
    else:
        print(f"  Worker {args.worker_id}: skipping summary.")
        return

    def _print_split_block(label, block):
        print(f"\n  {label}")
        print(f"  {'─' * 68}")
        print(
            f"  Total: {block['total_samples']}  |  Correct: {block['correct_predictions']} ({block['overall_accuracy_percent']:.2f}%)  |  Wrong: {block['incorrect_predictions']} ({block['overall_error_rate_percent']:.2f}%)"
        )
        print(
            f"  {'Class':<25s} {'Total':>6s} {'Wrong':>6s} {'Error%':>8s}  Confused as"
        )
        print(f"  {'─' * 65}")
        for pc in block["per_class"]:
            confused_str = ""
            if pc["confused_as"]:
                parts = [f"{k}({v['count']})" for k, v in pc["confused_as"].items()]
                confused_str = ", ".join(parts)
            print(
                f"  {pc['class_name']:<25s} {pc['total_samples']:>6d} {pc['incorrect_predictions']:>6d} {pc['error_rate_percent']:>7.2f}%  {confused_str}"
            )

    print(f"\n{'=' * 70}")
    _print_split_block("OVERALL", summary["overall"])
    for sp in summary["summary_metadata"]["splits_processed"]:
        _print_split_block(f"{sp.upper()} SPLIT", summary[f"{sp}_split"])
    elapsed = progress_tracker.state["overall"]["elapsed_seconds"]
    total_saved = sum(
        (1 for t in all_tracking if t["correct"] and (not t.get("null_skipped", False)))
    )
    total_skipped = sum(
        (
            1
            for t in all_tracking
            if not t["correct"] and (not t.get("null_skipped", False))
        )
    )
    total_null = sum((1 for t in all_tracking if t.get("null_skipped", False)))
    print(f"\nOutputs:")
    print(f"  {total_saved} sample JSONs (no-XAI ablation, full time-series)")
    print(f"  {total_skipped} skipped (incorrect predictions)")
    if total_null:
        print(f"  {total_null} skipped (null-class ground truth)")
    print(f"  {summary_path}")
    print(f"  Output dir: {args.output_dir}/")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}")
    if _WANDB_OK and args.worker_id == 0:
        try:
            if _wandb.run is not None:
                ov = summary["overall"]
                _wandb.log(
                    {
                        "run/total_samples": ov["total_samples"],
                        "run/correct_saved": ov["correct_predictions"],
                        "run/accuracy_pct": ov["overall_accuracy_percent"],
                        "run/total_time_s": elapsed,
                    }
                )
                _wandb.finish()
        except Exception as e:
            print(f"  W&B finish failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
