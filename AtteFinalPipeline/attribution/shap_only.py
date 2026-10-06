from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.data.loading import _get_dataset_config, load_data
from AtteFinalPipeline.data.sensors import (
    SENSORS_MHEALTH,
    _PAMAP2_FALLBACK_NAMES,
    get_null_class_indices,
    get_sensor_config,
)
from AtteFinalPipeline.embeddings.encoders import (
    MANTIS_RETURN_TRANSF_LAYER,
    compute_mantis_embeddings_batch,
    load_mantis_model,
)
from AtteFinalPipeline.attribution.performance import PerformanceMonitor
from AtteFinalPipeline.attribution.engine import XAIEngine, batched_predict
from AtteFinalPipeline.attribution.regions import build_result_dict
from AtteFinalPipeline.attribution.progress import ProgressTracker
from AtteFinalPipeline.attribution.checkpoints import find_checkpoint
import os
import sys
import json
import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from AtteFinalPipeline.expert.model import create

try:
    import wandb as _wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
assert len(SENSORS_MHEALTH) == 23
assert len(_PAMAP2_FALLBACK_NAMES) == 52
SENSOR_COLORS = {
    "Acc_X": "#1565C0",
    "Acc_Y": "#E65100",
    "Acc_Z": "#2E7D32",
    "Gyro_X": "#00838F",
    "Gyro_Y": "#F57F17",
    "Gyro_Z": "#6A1B9A",
    "Total_Acc_X": "#C62828",
    "Total_Acc_Y": "#283593",
    "Total_Acc_Z": "#1B5E20",
    "Chest_ECG_I": "#AD1457",
    "Chest_ECG_II": "#6A1B9A",
    "Chest_ECG_III": "#283593",
    "Mag_X": "#00695C",
    "Mag_Y": "#558B2F",
    "Mag_Z": "#BF360C",
    "LinAcc_X": "#0277BD",
    "LinAcc_Y": "#D84315",
    "LinAcc_Z": "#4527A0",
}
_CMAP_HOT = LinearSegmentedColormap.from_list(
    "hot_custom", ["#FFFDE7", "#FFD54F", "#FF6F00", "#B71C1C", "#4A0000"]
)
_DYNAMIC_PALETTE = [
    "#1565C0",
    "#E65100",
    "#2E7D32",
    "#00838F",
    "#F57F17",
    "#6A1B9A",
    "#C62828",
    "#283593",
    "#1B5E20",
    "#D84315",
    "#4527A0",
    "#00695C",
    "#AD1457",
    "#F9A825",
    "#558B2F",
    "#0277BD",
    "#BF360C",
    "#311B92",
    "#004D40",
    "#FF6F00",
]
MAX_PLOT_CHANNELS = 12


def _select_top_channels(
    sensor_importance, sensor_names, max_channels=MAX_PLOT_CHANNELS
):
    D = len(sensor_names)
    if D <= max_channels:
        return (None, None, None)
    ranked = np.argsort(sensor_importance)[::-1][:max_channels]
    top_indices = sorted(ranked.tolist())
    top_names = [sensor_names[i] for i in top_indices]
    groups = []
    group_size = 3
    for g_start in range(0, len(top_indices), group_size):
        g_end = min(g_start + group_size, len(top_indices))
        local_indices = list(range(g_start, g_end))
        local_names = top_names[g_start:g_end]
        orig_range = f"{top_indices[g_start]}-{top_indices[g_end - 1]}"
        groups.append((f"Top Channels (orig {orig_range})", local_indices, local_names))
    return (top_indices, top_names, groups)


def _get_channel_color(ch_name, ch_local_idx):
    if ch_name in SENSOR_COLORS:
        return SENSOR_COLORS[ch_name]
    if "_" in ch_name:
        parts = ch_name.split("_")
        for start in range(1, len(parts)):
            suffix = "_".join(parts[start:])
            if suffix in SENSOR_COLORS:
                return SENSOR_COLORS[suffix]
    return _DYNAMIC_PALETTE[ch_local_idx % len(_DYNAMIC_PALETTE)]


def _get_top_markers_for_group(regions, channel_indices):
    return [r for r in regions if r["sensor_idx"] in channel_indices]


def plot_dashboard(
    result, output_dir, sensor_names, sensor_groups, save_dpi=100, dataset_name="ucihar"
):
    data = result["_data"]
    combined = result["_combined"]
    sensor_imp = result["_sensor_importance"]
    phase_imp = result["_phase_importance"]
    regions = result["high_attribution_regions"]
    T, D = data.shape
    time_axis = np.arange(T)
    label_name = result["label_name"]
    conf = result["confidence"]
    split = result["split"]
    top_indices, top_names, top_groups = _select_top_channels(
        sensor_imp, sensor_names, MAX_PLOT_CHANNELS
    )
    if top_indices is not None:
        plot_data = data[:, top_indices]
        plot_combined = combined[:, top_indices]
        plot_sensor_names = top_names
        plot_groups = top_groups
        plot_D = len(top_indices)
        idx_map = {orig: new for new, orig in enumerate(top_indices)}
        plot_regions = []
        for r in regions:
            if r["sensor_idx"] in idx_map:
                rr = dict(r)
                rr["sensor_idx"] = idx_map[r["sensor_idx"]]
                plot_regions.append(rr)
        plot_sensor_imp = sensor_imp[top_indices]
    else:
        plot_data = data
        plot_combined = combined
        plot_sensor_names = sensor_names
        plot_groups = sensor_groups
        plot_D = D
        plot_regions = regions
        plot_sensor_imp = sensor_imp
    n_sensor_rows = len(plot_groups)
    height_ratios = [1] * n_sensor_rows + [3.5, 1.2]
    total_rows = n_sensor_rows + 2
    fig_height = sum(height_ratios) * 3.2
    fig = plt.figure(figsize=(20, fig_height))
    gs = gridspec.GridSpec(
        total_rows,
        3,
        figure=fig,
        hspace=0.4,
        wspace=0.3,
        height_ratios=height_ratios,
        top=0.97,
    )
    title_suffix = f"  (top {plot_D}/{D} channels)" if top_indices is not None else ""
    fig.suptitle(
        f"Time-Sensor Attention Analysis: {label_name}  [{split.upper()} SET]{title_suffix}",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    for row_idx, (group_title, ch_indices, ch_names) in enumerate(plot_groups):
        ax = fig.add_subplot(gs[row_idx, :])
        for local_i, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_names)):
            color = _get_channel_color(ch_name, local_i)
            ax.plot(
                time_axis,
                plot_data[:, ch_idx],
                color=color,
                linewidth=1.5,
                label=ch_name,
                alpha=0.9,
            )
        group_regions = _get_top_markers_for_group(plot_regions, ch_indices)
        for rank, reg in enumerate(group_regions[:10], start=1):
            s_idx = reg["sensor_idx"]
            ch_name = plot_sensor_names[s_idx]
            color = _get_channel_color(ch_name, s_idx)
            ax.axvspan(reg["start_t"], reg["end_t"], alpha=0.12, color=color, zorder=1)
            t_peak = reg["peak_timestep"]
            val = plot_data[t_peak, s_idx]
            size = 80 + 200 * reg["max_importance"]
            ax.scatter(
                t_peak,
                val,
                marker="*",
                s=size,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                f"R{rank}",
                (t_peak, val),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=7,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8),
            )
        ax.set_title(group_title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Sensor Value", fontsize=10)
        if row_idx == n_sensor_rows - 1:
            ax.set_xlabel("Time Step", fontsize=10)
        ax.legend(fontsize=9, loc="upper right", ncol=min(len(ch_names), 4))
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, T - 1)
    heatmap_row = n_sensor_rows
    ax_combined = fig.add_subplot(gs[heatmap_row, :])
    im = ax_combined.imshow(
        plot_combined.T,
        aspect="auto",
        cmap=_CMAP_HOT,
        interpolation="bilinear",
        origin="upper",
        extent=[0, T - 1, plot_D - 0.5, -0.5],
        alpha=0.85,
    )
    lane_height = 0.4
    for ch_idx in range(plot_D):
        ch_name = plot_sensor_names[ch_idx]
        color = _get_channel_color(ch_name, ch_idx)
        signal = plot_data[:, ch_idx]
        sig_min, sig_max = (signal.min(), signal.max())
        sig_range = sig_max - sig_min if sig_max > sig_min else 1.0
        sig_norm = (signal - sig_min) / sig_range
        sig_scaled = ch_idx - lane_height + sig_norm * 2 * lane_height
        ax_combined.plot(
            time_axis, sig_scaled, color=color, linewidth=1.0, alpha=0.9, zorder=3
        )
    for rank, reg in enumerate(plot_regions[:10], start=1):
        ch_idx = reg["sensor_idx"]
        t_peak = reg["peak_timestep"]
        signal = plot_data[:, ch_idx]
        sig_min, sig_max = (signal.min(), signal.max())
        sig_range = sig_max - sig_min if sig_max > sig_min else 1.0
        sig_norm = (signal[t_peak] - sig_min) / sig_range
        y_pos = ch_idx - lane_height + sig_norm * 2 * lane_height
        ax_combined.plot(
            t_peak,
            y_pos,
            marker="*",
            markersize=16,
            color="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=6,
        )
        ax_combined.annotate(
            f"R{rank}",
            (t_peak, y_pos),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="white", alpha=0.7),
            zorder=7,
        )
    ax_combined.set_title(
        "Importance Map (Gradient SHAP ONLY)", fontsize=14, fontweight="bold"
    )
    ax_combined.set_xlabel("Time Step", fontsize=11)
    ax_combined.set_ylabel("Sensor Channel", fontsize=11)
    ax_combined.set_yticks(range(plot_D))
    ax_combined.set_yticklabels(plot_sensor_names, fontsize=9)
    ax_combined.set_xlim(0, T - 1)
    cbar = plt.colorbar(
        im, ax=ax_combined, orientation="horizontal", fraction=0.04, pad=0.1, shrink=0.5
    )
    cbar.set_label("Importance Score", fontsize=10)
    bottom_row = heatmap_row + 1
    ax_rank = fig.add_subplot(gs[bottom_row, 0])
    sorted_idx = np.argsort(plot_sensor_imp)
    sorted_names = [plot_sensor_names[i] for i in sorted_idx]
    sorted_vals = plot_sensor_imp[sorted_idx]
    p70 = np.percentile(plot_sensor_imp, 70)
    bar_colors = ["#FF6F00" if v >= p70 else "#42A5F5" for v in sorted_vals]
    ax_rank.barh(
        sorted_names, sorted_vals, color=bar_colors, edgecolor="black", linewidth=0.4
    )
    ax_rank.set_title("Sensor Importance Ranking", fontsize=12, fontweight="bold")
    ax_rank.set_xlabel("Average Importance", fontsize=9)
    ax_phase = fig.add_subplot(gs[bottom_row, 1])
    n_phases = len(phase_imp)
    phase_size = T // n_phases
    phase_labels = [
        f"Phase {p}\n({p * phase_size}-{min((p + 1) * phase_size, T)})"
        for p in range(n_phases)
    ]
    p70_ph = np.percentile(phase_imp, 70)
    phase_colors = ["#FF6F00" if v >= p70_ph else "#66BB6A" for v in phase_imp]
    ax_phase.bar(
        range(n_phases), phase_imp, color=phase_colors, edgecolor="black", linewidth=0.4
    )
    ax_phase.set_xticks(range(n_phases))
    ax_phase.set_xticklabels(phase_labels, fontsize=8)
    ax_phase.set_title("Temporal Phase Importance", fontsize=12, fontweight="bold")
    ax_phase.set_ylabel("Avg Importance", fontsize=9)
    ax_summary = fig.add_subplot(gs[bottom_row, 2])
    ax_summary.axis("off")
    sample_id = f"{split}_class{result['true_label']}_{label_name}_sample{result['sample_idx']:04d}"
    summary = f"Sample ID: {sample_id}\n\nDataset: {dataset_name}\nSplit: {split.upper()}\nActivity: {label_name}\nConfidence: {conf:.4f}\nThreshold (p90): {result['attribution_threshold_p90']:.4f}\n\nAttribution: Gradient SHAP (ONLY)\n"
    if top_indices is not None:
        summary += f"Plotted: top {plot_D} of {D} channels\n"
    summary += (
        f"\nTop High-Attribution Regions:\n  ({len(regions)} total regions found)\n"
    )
    for reg in regions[:7]:
        summary += f"  {reg['sensor']} t={reg['start_t']}-{reg['end_t']} (len {reg['length']})\n     mean={reg['mean_importance']:.3f} max={reg['max_importance']:.3f}\n"
    ax_summary.text(
        0.05,
        0.95,
        summary,
        transform=ax_summary.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            linewidth=1.5,
        ),
    )
    safe_label = label_name.replace(" ", "_").replace("/", "_")
    filename = f"{split}_class{result['true_label']}_{safe_label}_s{result['sample_idx']:04d}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _save_dashboard_and_json(args_tuple):
    (
        result,
        output_dir,
        sensor_names,
        sensor_groups,
        mantis_embedding,
        save_dpi,
        save_png,
        dataset_name,
        class_map,
    ) = args_tuple
    if save_png:
        plot_dashboard(
            result,
            output_dir,
            sensor_names,
            sensor_groups,
            save_dpi=save_dpi,
            dataset_name=dataset_name,
        )
    save_sample_json(
        result,
        output_dir,
        sensor_names,
        mantis_embedding,
        dataset_name=dataset_name,
        class_map=class_map,
    )


def build_sample_json(
    result, sensor_names, mantis_embedding=None, dataset_name="ucihar", class_map=None
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
        "Batched Gradient SHAP (expected gradient, vectorised over background) — SHAP-ONLY ablation"
    ]
    if mantis_embedding is not None:
        xai_methods.append("MantisV2 Pre-trained Embedding")
    regions = result["high_attribution_regions"]
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
                "method": "shap_only",
                "formula": "attribution = abs(SHAP) / max(abs(SHAP))  (no IG, no combination)",
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
    dataset_name="ucihar",
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


def build_and_save_summary(all_tracking, output_dir, class_map):
    num_classes = len(class_map)
    splits_seen = sorted({t["split"] for t in all_tracking})

    def _aggregate(tracking_subset):
        total = len(tracking_subset)
        class_total = {c: 0 for c in range(num_classes)}
        class_incorrect_count = {c: 0 for c in range(num_classes)}
        class_confused_as = {c: {} for c in range(num_classes)}
        for t in tracking_subset:
            tc = t["true_label"]
            pc = t["predicted_label"]
            class_total[tc] += 1
            if not t["correct"]:
                class_incorrect_count[tc] += 1
                pred_name = class_map[pc] if pc < len(class_map) else f"Class_{pc}"
                class_confused_as[tc][pred_name] = (
                    class_confused_as[tc].get(pred_name, 0) + 1
                )
        total_correct = sum((1 for t in tracking_subset if t["correct"]))
        total_incorrect = total - total_correct
        per_class = []
        for c in range(num_classes):
            n_total = class_total[c]
            n_wrong = class_incorrect_count[c]
            n_right = n_total - n_wrong
            error_pct = n_wrong / n_total * 100 if n_total > 0 else 0.0
            confused_sorted = dict(
                sorted(class_confused_as[c].items(), key=lambda x: x[1], reverse=True)
            )
            per_class.append(
                {
                    "class_id": c,
                    "class_name": class_map[c],
                    "total_samples": n_total,
                    "correct_predictions": n_right,
                    "incorrect_predictions": n_wrong,
                    "error_rate_percent": round(error_pct, 2),
                    "confused_as": {
                        k: {"count": v, "percent_of_class": round(v / n_total * 100, 2)}
                        for k, v in confused_sorted.items()
                    }
                    if n_wrong > 0
                    else {},
                }
            )
        return {
            "total_samples": total,
            "correct_predictions": total_correct,
            "incorrect_predictions": total_incorrect,
            "overall_accuracy_percent": round(total_correct / total * 100, 2)
            if total > 0
            else 0.0,
            "overall_error_rate_percent": round(total_incorrect / total * 100, 2)
            if total > 0
            else 0.0,
            "per_class": per_class,
        }

    summary = {
        "summary_metadata": {
            "model": "AttendDiscriminate",
            "dataset": class_map,
            "splits_processed": splits_seen,
            "num_classes": num_classes,
            "class_names": class_map,
        },
        "overall": _aggregate(all_tracking),
    }
    for sp in splits_seen:
        subset = [t for t in all_tracking if t["split"] == sp]
        summary[f"{sp}_split"] = _aggregate(subset)
    path = os.path.join(output_dir, "prediction_summary.json")
    with open(path, "w") as f:
        f.write(_json_dumps(summary))
    return (path, summary)


def process_split(
    split,
    data,
    labels,
    model,
    mantis_trainer,
    xai_engine,
    output_dir,
    sensor_names,
    sensor_groups,
    device,
    progress_tracker,
    class_map,
    dataset_name,
    n_steps=25,
    predict_batch_size=256,
    mantis_batch_size=32,
    plot_workers=8,
    save_dpi=100,
    min_confidence=0.0,
    worker_id=0,
    num_workers=1,
    perf_monitor=None,
    plot_every_n=50,
    null_class_indices=None,
    xai_batch_size=32,
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
    else:
        slice_start = 0
    n_correct = len(correct_indices)
    print(
        f"  [{split.upper()}] Correct (this worker): {n_correct}, Incorrect (total): {n_incorrect}"
        + (f", Null-skipped: {n_null}" if n_null else "")
    )
    print(f"  [{split.upper()}] XAI batch size: {xai_batch_size} samples/GPU-batch")
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
    print(f"  [{split.upper()}] Starting batched XAI attribution ...")
    saved_indices = []
    pending_futures = []
    n_saved = 0
    save_executor = ThreadPoolExecutor(max_workers=plot_workers)
    try:
        pbar = tqdm(
            range(0, n_correct, xai_batch_size),
            desc=f"  {split.upper()} XAI batches",
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for batch_start in pbar:
            batch_end = min(batch_start + xai_batch_size, n_correct)
            batch_global_indices = correct_indices[batch_start:batch_end]
            B = len(batch_global_indices)
            batch_np = data[batch_global_indices]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)
            batch_preds = torch.tensor(
                [int(pred_classes[gi]) for gi in batch_global_indices],
                dtype=torch.long,
                device=device,
            )
            combined_batch = xai_engine.compute_shap_batch(batch_tensor, batch_preds)
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
                    split,
                    class_map=class_map,
                )
                save_png = plot_every_n > 0 and n_saved % plot_every_n == 0
                save_arg = (
                    result,
                    output_dir,
                    sensor_names,
                    sensor_groups,
                    None,
                    save_dpi,
                    save_png,
                    dataset_name,
                    class_map,
                )
                future = save_executor.submit(_save_dashboard_and_json, save_arg)
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
            if perf_monitor is not None:
                perf_monitor.record_sample(save_queue_depth=len(pending_futures))
            progress_tracker.write_periodic(every_n=50)
        pbar.close()
        del batch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

        with ThreadPoolExecutor(max_workers=plot_workers) as patch_executor:
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
        description="XAI Ablation — Gradient SHAP ONLY (Multi-Dataset, Batched SHAP)"
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
    parser.add_argument("--shap_bg_per_class", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for initial batched_predict sweep.",
    )
    parser.add_argument(
        "--xai_batch_size",
        type=int,
        default=32,
        help="Number of samples processed simultaneously through IG and SHAP. Higher = better GPU utilisation. A100 80GB: try 64-256. Reduce if OOM. Default: 64.",
    )
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(8, min(24, (os.cpu_count() or 16) // 2))
    parser.add_argument("--plot_workers", type=int, default=_cpu_default)
    parser.add_argument("--save_dpi", type=int, default=100)
    parser.add_argument("--plot_every_n", type=int, default=50)
    parser.add_argument("--max_plot_channels", type=int, default=MAX_PLOT_CHANNELS)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
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
        args.output_dir = f"./xai_output_shap_only/{args.dataset}/"
    os.makedirs(args.output_dir, exist_ok=True)
    global MAX_PLOT_CHANNELS
    MAX_PLOT_CHANNELS = args.max_plot_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.worker_id >= args.num_workers:
        print(f"ERROR: worker_id={args.worker_id} >= num_workers={args.num_workers}")
        sys.exit(1)
    if _WANDB_OK and args.worker_id == 0:
        try:
            _wandb.init(
                project="sensorllm",
                name=f"attribution_{args.dataset}",
                tags=[args.dataset],
                config=vars(args),
                dir=args.output_dir,
                resume="allow",
            )
            print(f"W&B run initialised: {_wandb.run.url}")
        except Exception as e:
            print(f"W&B init failed (continuing without it): {e}")
    print(f"Dataset:       {args.dataset}")
    print(f"Classes:       {num_class} — {class_map}")
    print(f"Input dim:     {input_dim}")
    print(f"Device:        {device}")
    print(f"Worker:        {args.worker_id} of {args.num_workers}")
    print(f"Splits:        {args.splits}")
    print(f"XAI batch:     {args.xai_batch_size} samples/GPU-batch  ← key for GPU util")
    print(f"IG n_steps:    {args.n_steps}")
    print(f"SHAP bg/class: {args.shap_bg_per_class}")
    print(f"AMP float16:   {('ON' if args.use_amp else 'OFF')}")
    print(f"torch.compile: {('OFF' if args.no_compile else 'ON')}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()
    print("[1/6] Loading data ...")
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
    print("\n[4/6] Building AttendDiscriminate model ...")
    config_model = dict(config_model_from_settings)
    config_model["input_dim"] = input_dim
    config_model["num_class"] = num_class
    config_model["train_mode"] = False
    config_model["experiment"] = "xai_analysis"
    model = create("AttendDiscriminate", config_model).to(device)
    print("\n[5/6] Loading checkpoint ...")
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
    xai_engine = XAIEngine(
        model,
        ig_baseline,
        shap_background,
        device,
        use_compile=not args.no_compile,
        use_amp=args.use_amp,
    )
    if not args.no_compile and torch.cuda.is_available():
        T_sample = split_data[first_split][0].shape[1]
        xai_engine.warm_up(
            sample_shape=(T_sample, n_channels),
            target_class=0,
            n_steps=5,
            production_n_steps=args.n_steps,
        )
    print(f"\n[6/6] Analyzing ({', '.join(args.splits)}) ...")
    total_by_split = {s: split_data[s][0].shape[0] for s in args.splits}
    progress_tracker = ProgressTracker(
        args.output_dir, total_by_split, worker_id=args.worker_id
    )
    perf_monitor = PerformanceMonitor(args.output_dir, poll_interval=5.0)
    perf_monitor.start()
    all_tracking = []
    try:
        for split_name in args.splits:
            s_data, s_labels = split_data[split_name]
            tracking = process_split(
                split_name,
                s_data,
                s_labels,
                model,
                mantis_trainer,
                xai_engine,
                args.output_dir,
                sensor_names,
                sensor_groups,
                device,
                progress_tracker,
                class_map=class_map,
                dataset_name=args.dataset,
                n_steps=args.n_steps,
                predict_batch_size=args.batch_size,
                mantis_batch_size=args.mantis_batch_size,
                plot_workers=args.plot_workers,
                save_dpi=args.save_dpi,
                min_confidence=args.min_confidence,
                worker_id=args.worker_id,
                num_workers=args.num_workers,
                perf_monitor=perf_monitor,
                plot_every_n=args.plot_every_n,
                null_class_indices=null_class_indices,
                xai_batch_size=args.xai_batch_size,
            )
            all_tracking.extend(tracking)
    finally:
        perf_monitor.stop()
    progress_tracker.finish()
    perf_monitor.print_summary()
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
    print(f"  {total_saved} dashboard JSONs (+ PNGs every {args.plot_every_n} samples)")
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
