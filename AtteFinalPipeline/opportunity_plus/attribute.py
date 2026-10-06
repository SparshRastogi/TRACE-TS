from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.embeddings.encoders import (
    MANTIS_RETURN_TRANSF_LAYER,
    compute_mantis_embeddings_batch,
    load_mantis_model,
)
from AtteFinalPipeline.attribution.performance import _query_nvidia_smi
from AtteFinalPipeline.attribution.engine import _LogitWrapper, batched_predict
from AtteFinalPipeline.attribution.regions import extract_high_attribution_regions
from AtteFinalPipeline.attribution.progress import ProgressTracker
import os
import sys
import csv
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from AtteFinalPipeline.expert.model import create
from captum.attr import IntegratedGradients

try:
    import wandb as _wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
OPP_CLASS_NAMES = [
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
]
assert len(OPP_CLASS_NAMES) == 18
OPP_SENSOR_NAMES = [
    "IMU_BACK_AccX",
    "IMU_BACK_AccY",
    "IMU_BACK_AccZ",
    "IMU_BACK_GyroX",
    "IMU_BACK_GyroY",
    "IMU_BACK_GyroZ",
    "IMU_BACK_MagX",
    "IMU_BACK_MagY",
    "IMU_BACK_MagZ",
    "IMU_BACK_Quat1",
    "IMU_BACK_Quat2",
    "IMU_RUA_AccX",
    "IMU_RUA_AccY",
    "IMU_RUA_AccZ",
    "IMU_RUA_GyroX",
    "IMU_RUA_GyroY",
    "IMU_RUA_GyroZ",
    "IMU_RUA_MagX",
    "IMU_RUA_MagY",
    "IMU_RUA_MagZ",
    "IMU_RLA_AccX",
    "IMU_RLA_AccY",
    "IMU_RLA_AccZ",
    "IMU_RLA_GyroX",
    "IMU_RLA_GyroY",
    "IMU_RLA_GyroZ",
    "IMU_RLA_MagX",
    "IMU_RLA_MagY",
    "IMU_RLA_MagZ",
    "IMU_LUA_AccX",
    "IMU_LUA_AccY",
    "IMU_LUA_AccZ",
    "IMU_LUA_GyroX",
    "IMU_LUA_GyroY",
    "IMU_LUA_GyroZ",
    "IMU_LUA_MagX",
    "IMU_LUA_MagY",
    "IMU_LUA_MagZ",
    "IMU_LLA_AccX",
    "IMU_LLA_AccY",
    "IMU_LLA_AccZ",
    "IMU_LLA_GyroX",
    "IMU_LLA_GyroY",
    "IMU_LLA_GyroZ",
    "IMU_LLA_MagX",
    "IMU_LLA_MagY",
    "IMU_LLA_MagZ",
    "LSHOE_EuX",
    "LSHOE_EuY",
    "LSHOE_EuZ",
    "LSHOE_Nav_AccX",
    "LSHOE_Nav_AccY",
    "LSHOE_Nav_AccZ",
    "LSHOE_Body_AccX",
    "LSHOE_Body_AccY",
    "LSHOE_Body_AccZ",
    "LSHOE_AngVelBodyX",
    "LSHOE_AngVelBodyY",
    "LSHOE_AngVelBodyZ",
    "LSHOE_AngVelNavX",
    "LSHOE_AngVelNavY",
    "LSHOE_AngVelNavZ",
    "LSHOE_Compass",
    "RSHOE_EuX",
    "RSHOE_EuY",
    "RSHOE_EuZ",
    "RSHOE_Nav_AccX",
    "RSHOE_Nav_AccY",
    "RSHOE_Nav_AccZ",
    "RSHOE_Body_AccX",
    "RSHOE_Body_AccY",
    "RSHOE_Body_AccZ",
    "RSHOE_AngVelBodyX",
    "RSHOE_AngVelBodyY",
    "RSHOE_AngVelBodyZ",
    "RSHOE_AngVelNavX",
    "RSHOE_AngVelNavY",
    "RSHOE_AngVelNavZ",
    "RSHOE_Compass",
]
assert len(OPP_SENSOR_NAMES) == 79
OPP_SENSOR_GROUPS = [
    (
        "IMU BACK (acc / gyro / mag / Quat1-2)",
        list(range(0, 11)),
        OPP_SENSOR_NAMES[0:11],
    ),
    ("IMU RUA (acc / gyro / mag)", list(range(11, 20)), OPP_SENSOR_NAMES[11:20]),
    ("IMU RLA (acc / gyro / mag)", list(range(20, 29)), OPP_SENSOR_NAMES[20:29]),
    ("IMU LUA (acc / gyro / mag)", list(range(29, 38)), OPP_SENSOR_NAMES[29:38]),
    ("IMU LLA (acc / gyro / mag)", list(range(38, 47)), OPP_SENSOR_NAMES[38:47]),
    (
        "L-Shoe (Euler / NavAcc / BodyAcc / AngVel / Compass)",
        list(range(47, 63)),
        OPP_SENSOR_NAMES[47:63],
    ),
    (
        "R-Shoe (Euler / NavAcc / BodyAcc / AngVel / Compass)",
        list(range(63, 79)),
        OPP_SENSOR_NAMES[63:79],
    ),
]
OPP_INPUT_DIM = 79
OPP_NUM_CLASS = 18
OPP_WINDOW = 24
OPP_HIDDEN_DIM = 128
OPP_FILTER_NUM = 64
OPP_FILTER_SIZE = 5
OPP_ENC_LAYERS = 2
OPP_ENC_BIDIR = False
OPP_DROPOUT = 0.5
OPP_DROPOUT_RNN = 0.25
OPP_DROPOUT_CLS = 0.5
OPP_ACTIVATION = "ReLU"
OPP_SA_DIV = 1
OPP_NULL_CLASS_INDICES = {0}
try:
    import psutil as _psutil

    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
import threading


class PerformanceMonitor:
    def __init__(self, output_dir, poll_interval=5):
        self.poll_interval = poll_interval
        self._log_path = os.path.join(output_dir, "perf_log.jsonl")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._samples_per_sec = 0.0
        self._sample_ig_ms = []
        self._sample_shap_ms = []
        self._last_ig_ms = 0.0
        self._last_shap_ms = 0.0
        self._fh = open(self._log_path, "a", buffering=1)
        self._write(
            {
                "type": "run_start",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "psutil_available": _PSUTIL_OK,
            }
        )
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _write(self, record):
        line = json.dumps(record) + "\n"
        with self._lock:
            self._fh.write(line)

    def _poll_loop(self):
        while not self._stop.wait(self.poll_interval):
            gpu_util, gpu_mem_used, gpu_mem_total = _query_nvidia_smi()
            cpu_pct = _psutil.cpu_percent(interval=None) if _PSUTIL_OK else None
            if _PSUTIL_OK:
                vm = _psutil.virtual_memory()
                ram_used = round(vm.used / 1000000000.0, 2)
                ram_total = round(vm.total / 1000000000.0, 2)
            else:
                ram_used = ram_total = None
            self._write(
                {
                    "type": "poll",
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "gpu_util_pct": gpu_util,
                    "gpu_mem_used_mb": gpu_mem_used,
                    "gpu_mem_total_mb": gpu_mem_total,
                    "cpu_util_pct": cpu_pct,
                    "ram_used_gb": ram_used,
                    "ram_total_gb": ram_total,
                    "samples_per_sec": round(self._samples_per_sec, 3),
                }
            )

    def update_rate(self, samples_per_sec):
        with self._lock:
            self._samples_per_sec = samples_per_sec

    class _Timer:
        __slots__ = ("_mon", "_attr", "_t0")

        def __init__(self, mon, attr):
            self._mon = mon
            self._attr = attr

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            setattr(self._mon, self._attr, (time.perf_counter() - self._t0) * 1000.0)

    def time_ig(self):
        self._last_ig_ms = 0.0
        return self._Timer(self, "_last_ig_ms")

    def time_shap(self):
        self._last_shap_ms = 0.0
        return self._Timer(self, "_last_shap_ms")

    def record_sample(self, save_queue_depth=0):
        with self._lock:
            self._sample_ig_ms.append(self._last_ig_ms)
            self._sample_shap_ms.append(self._last_shap_ms)
            window = self._sample_ig_ms[-50:]
            shap_w = self._sample_shap_ms[-50:]
            if len(window) > 1:
                total_sec = sum((a + b for a, b in zip(window, shap_w))) / 1000.0
                self._samples_per_sec = len(window) / max(total_sec, 1e-09)
        self._write(
            {
                "type": "sample",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ig_ms": round(self._last_ig_ms, 2),
                "shap_ms": round(self._last_shap_ms, 2),
                "save_queue_depth": int(save_queue_depth),
            }
        )

    def start(self):
        pass

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.poll_interval + 2)
        self._write({"type": "run_end", "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        with self._lock:
            self._fh.close()

    def print_summary(self):
        with self._lock:
            ig_ms = sorted(self._sample_ig_ms)
            shap_ms = sorted(self._sample_shap_ms)
        if not ig_ms:
            print("  PerformanceMonitor: no samples timed.")
            return
        n = len(ig_ms)
        p = lambda lst, pct: lst[max(0, int(len(lst) * pct) - 1)]
        total_ms = sorted((a + b for a, b in zip(ig_ms, shap_ms)))
        print(f"\n  {'─' * 58}")
        print(f"  Performance summary  ({n} samples instrumented)")
        print(f"  {'─' * 58}")
        for label, lst in [
            ("IG (ms)", ig_ms),
            ("SHAP (ms)", shap_ms),
            ("IG+SHAP (ms)", total_ms),
        ]:
            mean = sum(lst) / len(lst)
            print(
                f"  {label:<30s} {mean:>7.1f} {p(lst, 0.5):>7.1f} {p(lst, 0.95):>7.1f}"
            )
        print(f"  {'─' * 58}")
        print(f"  Full log → {self._log_path}")

    @property
    def log_path(self):
        return self._log_path


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
        attr = (grads * diffs).mean(dim=1)
        attr_np = attr.detach().cpu().numpy()
        np.abs(attr_np, out=attr_np)
        maxvals = attr_np.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        attr_np[nonzero] /= maxvals[nonzero, None, None]
        return attr_np

    def _combine_batch(self, ig_attrs, shap_attrs, eps=1e-08):
        combined = np.sqrt((ig_attrs + eps) * (shap_attrs + eps))
        B = combined.shape[0]
        maxvals = combined.reshape(B, -1).max(axis=1)
        nonzero = maxvals > 0
        combined[nonzero] /= maxvals[nonzero, None, None]
        return combined


def build_result_dict(
    data_np,
    combined,
    pred_class,
    probs,
    sample_idx,
    true_label,
    sensor_names,
    split,
    class_map,
):
    T, D = data_np.shape
    confidence = float(probs[pred_class])
    sensor_importance = combined.mean(axis=0)
    n_phases = 5
    phase_size = T // n_phases
    phase_importance = []
    for p in range(n_phases):
        s = p * phase_size
        e = s + phase_size if p < n_phases - 1 else T
        phase_importance.append(float(combined[s:e].mean()))
    regions, attr_threshold = extract_high_attribution_regions(combined, sensor_names)
    label_name = (
        class_map[true_label] if true_label < len(class_map) else f"Class_{true_label}"
    )
    return {
        "sample_idx": int(sample_idx),
        "true_label": int(true_label),
        "predicted_label": int(pred_class),
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
        "attribution_threshold_p90": attr_threshold,
        "high_attribution_regions": regions,
        "_data": data_np,
        "_combined": combined,
        "_sensor_importance": sensor_importance,
        "_phase_importance": phase_importance,
    }


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
    "AccX": "#1565C0",
    "AccY": "#E65100",
    "AccZ": "#2E7D32",
    "GyroX": "#00838F",
    "GyroY": "#F57F17",
    "GyroZ": "#6A1B9A",
    "MagX": "#00695C",
    "MagY": "#558B2F",
    "MagZ": "#BF360C",
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
    result,
    output_dir,
    sensor_names,
    sensor_groups,
    save_dpi=100,
    dataset_name="opportunity_plus",
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
        "Combined Importance Map (Geometric Mean of IG & SHAP)",
        fontsize=14,
        fontweight="bold",
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
    summary = f"Sample ID: {sample_id}\n\nDataset: {dataset_name}\nSplit: {split.upper()}\nActivity: {label_name}\nConfidence: {conf:.4f}\nThreshold (p90): {result['attribution_threshold_p90']:.4f}\n\nCombination: Geometric Mean (AND)\n"
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
    result,
    sensor_names,
    mantis_embedding=None,
    dataset_name="opportunity_plus",
    class_map=None,
    extra_metadata=None,
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
            "model": "AttendDiscriminate",
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
    if extra_metadata is not None:
        entry["opportunity_plus_metadata"] = extra_metadata
    return entry


def save_sample_json(
    result,
    output_dir,
    sensor_names,
    mantis_embedding=None,
    dataset_name="opportunity_plus",
    class_map=None,
    extra_metadata=None,
):
    entry = build_sample_json(
        result,
        sensor_names,
        mantis_embedding,
        dataset_name=dataset_name,
        class_map=class_map,
        extra_metadata=extra_metadata,
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


def load_index_csv(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            row["window_idx"] = int(row["window_idx"])
            row["start_sample"] = int(row["start_sample"])
            row["end_sample"] = int(row["end_sample"])
            row["run_local_start"] = int(row["run_local_start"])
            row["run_local_end"] = int(row["run_local_end"])
            row["start_time_s"] = float(row["start_time_s"])
            row["end_time_s"] = float(row["end_time_s"])
            row["start_frame"] = int(row["start_frame"])
            row["end_frame"] = int(row["end_frame"])
            row["true_label"] = int(row["true_label"])
            row["used_srt_anchor"] = row["used_srt_anchor"] == "True"
            row["video_fps"] = float(row.get("video_fps") or 0.0)
            if "sensor_file" not in row and "dat_file" in row:
                row["sensor_file"] = row["dat_file"]
            rows.append(row)
    return rows


def build_windows_from_index(X, index_rows, window=OPP_WINDOW):
    N = X.shape[0]
    C = X.shape[1]
    n_win = len(index_rows)
    out = np.empty((n_win, window, C), dtype=np.float32)
    labels = np.empty(n_win, dtype=np.int64)
    bad_idx = []
    for i, row in enumerate(index_rows):
        s = row["start_sample"]
        e = s + window
        if e > N:
            bad_idx.append((i, s, e, N))
            slab = np.zeros((window, C), dtype=np.float32)
            avail = max(0, N - s)
            if avail > 0:
                slab[:avail] = X[s : s + avail]
            out[i] = slab
        else:
            out[i] = X[s:e]
        labels[i] = row["true_label"]
    if bad_idx:
        first = bad_idx[0]
        raise RuntimeError(
            f"[FATAL] {len(bad_idx)} CSV rows reference samples past the end of the .mat sensor array (e.g. row {first[0]}: start_sample={first[1]}, end={first[2]}, but N={first[3]}). The .mat and CSV are out of sync — re-run prepare_opportunity_plus.py."
        )
    return (out, labels)


def load_opportunity_plus_mat(mat_path):
    import scipy.io as sio

    mat = sio.loadmat(mat_path)
    X = mat["testingData"].astype(np.float32).T
    y = mat["testingLabels"].reshape(-1).astype(np.int64) - 1
    return (X, y)


def load_shap_background_from_opportunity_mat(
    opp_mat_path, shap_bg_per_class, rng, num_classes=18
):
    import scipy.io as sio

    print(f"  Loading SHAP background pool from {opp_mat_path}")
    m = sio.loadmat(opp_mat_path)
    X_train = m["trainingData"].astype(np.float32).T
    y_train = m["trainingLabels"].reshape(-1).astype(np.int64) - 1
    N = X_train.shape[0]
    window = OPP_WINDOW
    valid_end_positions = np.arange(window - 1, N)
    window_labels = y_train[valid_end_positions]
    bg_windows = []
    for cls in range(num_classes):
        cls_mask = window_labels == cls
        cls_end_positions = valid_end_positions[cls_mask]
        if len(cls_end_positions) == 0:
            continue
        n_pick = min(shap_bg_per_class, len(cls_end_positions))
        picked = rng.choice(cls_end_positions, size=n_pick, replace=False)
        for end_pos in picked:
            start_pos = end_pos - window + 1
            bg_windows.append(X_train[start_pos : end_pos + 1].copy())
    if not bg_windows:
        raise RuntimeError(
            "Could not build SHAP background — no valid windows in training data."
        )
    bg_array = np.stack(bg_windows, axis=0).astype(np.float32)
    return bg_array


def build_ig_baseline_from_opportunity_mat(opp_mat_path):
    import scipy.io as sio

    m = sio.loadmat(opp_mat_path)
    X_train = m["trainingData"].astype(np.float32).T
    mean_per_channel = X_train.mean(axis=0)
    baseline = np.tile(mean_per_channel[None, :], (OPP_WINDOW, 1)).astype(np.float32)
    return baseline


def process_opportunity_plus(
    split_name,
    X_windows,
    y_windows,
    index_rows,
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
    perf_monitor=None,
    plot_every_n=10,
    null_class_indices=None,
    xai_batch_size=32,
    opp_plus_root=None,
):
    if null_class_indices is None:
        null_class_indices = set()
    N = X_windows.shape[0]
    print(f"\n  [{split_name.upper()}] Batched prediction for {N} windows ...")
    pred_classes, all_probs = batched_predict(
        model, X_windows, device, batch_size=predict_batch_size
    )
    labels_int = y_windows.astype(int)
    if null_class_indices:
        null_mask = np.isin(labels_int, list(null_class_indices))
        n_null = int(null_mask.sum())
        valid_mask = ~null_mask
        print(
            f"  [{split_name.upper()}] Null-class filter: skipping {n_null} windows ({N - n_null} remaining)"
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
    n_correct = len(correct_indices)
    n_incorrect = len(incorrect_indices)
    print(
        f"  [{split_name.upper()}] Correct: {n_correct}, Incorrect: {n_incorrect}"
        + (f", Null-skipped: {n_null}" if n_null else "")
    )
    print(f"  [{split_name.upper()}] XAI batch size: {xai_batch_size}")
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
                "split": split_name,
                "null_skipped": bool(null_mask[i]),
            }
        )
    for _ in np.where(null_mask)[0]:
        progress_tracker.update(split_name, correct=False)
    for _ in incorrect_indices:
        progress_tracker.update(split_name, correct=False)
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
            f"  [{split_name.upper()}] After confidence filter (>={min_confidence:.2f}): {n_correct} samples remain"
        )
    print(f"  [{split_name.upper()}] Starting batched XAI attribution ...")
    saved_indices = []
    pending_futures = []
    n_saved = 0
    save_executor = ThreadPoolExecutor(max_workers=plot_workers)
    try:
        pbar = tqdm(
            range(0, n_correct, xai_batch_size),
            desc=f"  {split_name.upper()} XAI batches",
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for batch_start in pbar:
            batch_end = min(batch_start + xai_batch_size, n_correct)
            batch_global_indices = correct_indices[batch_start:batch_end]
            B = len(batch_global_indices)
            batch_np = X_windows[batch_global_indices]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)
            batch_preds = torch.tensor(
                [int(pred_classes[gi]) for gi in batch_global_indices],
                dtype=torch.long,
                device=device,
            )
            ig_attrs = xai_engine.compute_ig_batch(
                batch_tensor, batch_preds, n_steps=n_steps
            )
            shap_attrs = xai_engine.compute_shap_batch(batch_tensor, batch_preds)
            combined_batch = xai_engine._combine_batch(ig_attrs, shap_attrs)
            for local_i, gi in enumerate(batch_global_indices):
                gi = int(gi)
                pred_class = int(pred_classes[gi])
                combined = combined_batch[local_i]
                result = build_result_dict(
                    X_windows[gi],
                    combined,
                    pred_class,
                    all_probs[gi],
                    gi,
                    int(labels_int[gi]),
                    sensor_names,
                    split_name,
                    class_map=class_map,
                )
                csv_row = index_rows[gi]
                extra_metadata = {
                    "window_idx": csv_row["window_idx"],
                    "run": csv_row["run"],
                    "sensor_file": csv_row.get("sensor_file", ""),
                    "video_file": csv_row.get("video_file", ""),
                    "srt_file": csv_row.get("srt_file", ""),
                    "video_fps": csv_row["video_fps"],
                    "start_sample": csv_row["start_sample"],
                    "end_sample": csv_row["end_sample"],
                    "run_local_start": csv_row["run_local_start"],
                    "run_local_end": csv_row["run_local_end"],
                    "start_time_s": csv_row["start_time_s"],
                    "end_time_s": csv_row["end_time_s"],
                    "start_frame": csv_row["start_frame"],
                    "end_frame": csv_row["end_frame"],
                    "used_srt_anchor": csv_row["used_srt_anchor"],
                }
                if opp_plus_root and csv_row.get("video_file"):
                    extra_metadata["video_abspath"] = os.path.join(
                        opp_plus_root, csv_row["video_file"]
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
                    extra_metadata,
                )
                future = save_executor.submit(
                    _save_dashboard_and_json_oppplus, save_arg
                )
                pending_futures.append((future, gi))
                saved_indices.append(gi)
                n_saved += 1
                progress_tracker.update(split_name, correct=True)
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
                f"  [{split_name.upper()}] Draining {len(pending_futures)} remaining saves ..."
            )
            for future, gi in pending_futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  WARNING: Save failed for sample {gi}: {e}")
    finally:
        save_executor.shutdown(wait=True)
    print(f"  [{split_name.upper()}] All {n_saved} JSONs saved.")
    if mantis_trainer is not None and n_saved > 0:
        print(
            f"  [{split_name.upper()}] Computing Mantis embeddings (batch={mantis_batch_size}) ..."
        )
        saved_indices_arr = np.array(saved_indices)
        correct_data_for_mantis = X_windows[saved_indices_arr]
        all_emb = compute_mantis_embeddings_batch(
            mantis_trainer,
            correct_data_for_mantis,
            device,
            batch_size=mantis_batch_size,
        )
        print(
            f"  [{split_name.upper()}] Mantis done (dim={all_emb.shape[1]}). Patching JSONs ..."
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
            filename = f"{split_name}_class{true_label}_{safe_label}_s{gi:04d}.json"
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
        print(f"  [{split_name.upper()}] JSON patch complete.")
    progress_tracker.finish_split(split_name)
    print(
        f"  [{split_name.upper()}] Done — saved: {n_saved}, skipped: {n_incorrect + (len(correct_indices) - n_saved)}"
        + (f", null-skipped: {n_null}" if n_null else "")
    )
    return all_tracking


def _save_dashboard_and_json_oppplus(args_tuple):
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
        extra_metadata,
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
        extra_metadata=extra_metadata,
    )


DEFAULT_OPPORTUNITY_ROOTS = ["./models/opportunity", "./models/opportunity"]


def auto_find_checkpoint(explicit):
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = []
    if os.path.exists("./weights/checkpoint_opportunity.pth"):
        candidates.append("./weights/checkpoint_opportunity.pth")
    for root in DEFAULT_OPPORTUNITY_ROOTS:
        if os.path.isdir(root):
            hits = glob.glob(
                os.path.join(root, "*", "checkpoints", "checkpoint_best.pth")
            )
            hits.sort(key=os.path.getmtime, reverse=True)
            candidates.extend(hits)
            short = os.path.join(root, "checkpoints", "checkpoint_best.pth")
            if os.path.exists(short):
                candidates.append(short)
    for c in candidates:
        if os.path.exists(c):
            print(f"[*] Using checkpoint: {c}")
            return c
    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint /path/to/checkpoint_best.pth, or place one at ./weights/checkpoint_opportunity.pth, or train a model so a checkpoint appears under "
        + " | ".join(DEFAULT_OPPORTUNITY_ROOTS)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI Full Analysis Dashboard for Opportunity++ (IG + SHAP + MantisV2, same output format as run_xai_analysis.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mat",
        required=True,
        help="Path to the Opportunity++ .mat from prepare_opportunity_plus.py.",
    )
    parser.add_argument(
        "--index_csv",
        required=True,
        help="Path to the per-window index CSV from prepare_opportunity_plus.py.",
    )
    parser.add_argument(
        "--opp_mat",
        default="./dataset/opportunity.mat",
        help="Original opportunity.mat — used to recover the IG baseline and SHAP background (same training distribution as the checkpoint).",
    )
    parser.add_argument(
        "--opp_plus_root",
        default=None,
        help="Opportunity++ root, for resolving video paths stored in the per-sample JSONs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint_best.pth. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./xai_output/opportunity_plus/"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Logical split name used in output filenames and tracking. Opportunity++ duplicates the data into train/val/test in the .mat, so we just use one.",
    )
    parser.add_argument(
        "--mantis_checkpoint",
        type=str,
        default="paris-noah/MantisV2",
        help="Mantis checkpoint name, or 'none' to disable.",
    )
    parser.add_argument("--shap_bg_per_class", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for initial batched_predict.",
    )
    parser.add_argument(
        "--xai_batch_size",
        type=int,
        default=32,
        help="Samples per GPU-batch through IG and SHAP.",
    )
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(8, min(24, (os.cpu_count() or 16) // 2))
    parser.add_argument("--plot_workers", type=int, default=_cpu_default)
    parser.add_argument("--save_dpi", type=int, default=100)
    parser.add_argument(
        "--plot_every_n",
        type=int,
        default=10,
        help="Save a dashboard PNG every N saved samples (JSON is always saved). Default 10 for Opportunity++.",
    )
    parser.add_argument("--max_plot_channels", type=int, default=MAX_PLOT_CHANNELS)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument(
        "--skip_null",
        action="store_true",
        default=True,
        help="Skip windows whose true label is Null. ON by default.",
    )
    parser.add_argument(
        "--include_null",
        action="store_true",
        help="Override --skip_null and include Null windows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = "opportunity_plus"
    class_map = OPP_CLASS_NAMES
    sensor_names = OPP_SENSOR_NAMES
    sensor_groups = OPP_SENSOR_GROUPS
    os.makedirs(args.output_dir, exist_ok=True)
    global MAX_PLOT_CHANNELS
    MAX_PLOT_CHANNELS = args.max_plot_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _WANDB_OK:
        try:
            _wandb.init(
                project="sensorllm",
                name=f"attribution_{dataset_name}",
                tags=[dataset_name],
                config=vars(args),
                dir=args.output_dir,
                resume="allow",
            )
            print(f"W&B run initialised: {_wandb.run.url}")
        except Exception as e:
            print(f"W&B init failed (continuing without it): {e}")
    print(f"Dataset:       {dataset_name}")
    print(f"Classes:       {OPP_NUM_CLASS} — {class_map}")
    print(f"Input dim:     {OPP_INPUT_DIM}")
    print(f"Device:        {device}")
    print(f"Split:         {args.split_name}")
    print(f"XAI batch:     {args.xai_batch_size} samples/GPU-batch")
    print(f"IG n_steps:    {args.n_steps}")
    print(f"SHAP bg/class: {args.shap_bg_per_class}")
    print(f"PNG every:     {args.plot_every_n} samples")
    print(f"AMP float16:   {('ON' if args.use_amp else 'OFF')}")
    print(f"torch.compile: {('OFF' if args.no_compile else 'ON')}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()
    print("[1/6] Loading Opportunity++ .mat + index CSV ...")
    X_full, y_full = load_opportunity_plus_mat(args.mat)
    print(f"  .mat:  X={X_full.shape}, y={y_full.shape}")
    if X_full.shape[1] != OPP_INPUT_DIM:
        print(
            f"  ERROR: .mat has {X_full.shape[1]} channels, expected {OPP_INPUT_DIM}."
        )
        sys.exit(1)
    index_rows = load_index_csv(args.index_csv)
    print(f"  CSV:   {len(index_rows)} window rows")
    print("  Slicing windows from CSV start positions ...")
    X_windows, y_windows = build_windows_from_index(
        X_full, index_rows, window=OPP_WINDOW
    )
    print(f"  Windows: X={X_windows.shape}, y={y_windows.shape}")
    print(
        "\n[2/6] Building IG baseline and SHAP background from opportunity.mat training data ..."
    )
    if not os.path.exists(args.opp_mat):
        print(
            f"  ERROR: --opp_mat not found at {args.opp_mat}. The IG baseline and SHAP background must come from the same distribution the checkpoint was trained on."
        )
        sys.exit(1)
    baseline_np = build_ig_baseline_from_opportunity_mat(args.opp_mat)
    ig_baseline = torch.tensor(baseline_np, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"  IG baseline shape: {tuple(ig_baseline.shape)}")
    rng = np.random.RandomState(args.seed)
    shap_bg_np = load_shap_background_from_opportunity_mat(
        args.opp_mat, args.shap_bg_per_class, rng, num_classes=OPP_NUM_CLASS
    )
    shap_background = torch.tensor(shap_bg_np, dtype=torch.float32).to(device)
    print(
        f"  SHAP background: {shap_background.shape[0]} samples, shape={tuple(shap_background.shape)}"
    )
    print("\n[3/6] Loading MantisV2 ...")
    mantis_trainer = None
    if args.mantis_checkpoint and args.mantis_checkpoint.lower() != "none":
        mantis_trainer = load_mantis_model(args.mantis_checkpoint, device)
    else:
        print("  MantisV2 disabled.")
    print("\n[4/6] Building AttendDiscriminate model ...")
    config_model = dict(
        model="AttendDiscriminate",
        dataset="opportunity",
        input_dim=OPP_INPUT_DIM,
        hidden_dim=OPP_HIDDEN_DIM,
        filter_num=OPP_FILTER_NUM,
        filter_size=OPP_FILTER_SIZE,
        enc_num_layers=OPP_ENC_LAYERS,
        enc_is_bidirectional=OPP_ENC_BIDIR,
        dropout=OPP_DROPOUT,
        dropout_rnn=OPP_DROPOUT_RNN,
        dropout_cls=OPP_DROPOUT_CLS,
        activation=OPP_ACTIVATION,
        sa_div=OPP_SA_DIV,
        num_class=OPP_NUM_CLASS,
        train_mode=False,
        experiment="xai_analysis_opportunity_plus",
    )
    model = create("AttendDiscriminate", config_model).to(device)
    print("\n[5/6] Loading checkpoint ...")
    ckpt_path = auto_find_checkpoint(args.checkpoint)
    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("  Checkpoint loaded.")
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
        xai_engine.warm_up(
            sample_shape=(OPP_WINDOW, OPP_INPUT_DIM),
            target_class=0,
            n_steps=5,
            production_n_steps=args.n_steps,
        )
    print(f"\n[6/6] Analysing split '{args.split_name}' ...")
    total_by_split = {args.split_name: X_windows.shape[0]}
    progress_tracker = ProgressTracker(args.output_dir, total_by_split, worker_id=0)
    perf_monitor = PerformanceMonitor(args.output_dir, poll_interval=5.0)
    perf_monitor.start()
    null_class_indices = (
        OPP_NULL_CLASS_INDICES if args.skip_null and (not args.include_null) else set()
    )
    all_tracking = []
    try:
        tracking = process_opportunity_plus(
            args.split_name,
            X_windows,
            y_windows,
            index_rows,
            model,
            mantis_trainer,
            xai_engine,
            args.output_dir,
            sensor_names,
            sensor_groups,
            device,
            progress_tracker,
            class_map=class_map,
            dataset_name=dataset_name,
            n_steps=args.n_steps,
            predict_batch_size=args.batch_size,
            mantis_batch_size=args.mantis_batch_size,
            plot_workers=args.plot_workers,
            save_dpi=args.save_dpi,
            min_confidence=args.min_confidence,
            perf_monitor=perf_monitor,
            plot_every_n=args.plot_every_n,
            null_class_indices=null_class_indices,
            xai_batch_size=args.xai_batch_size,
            opp_plus_root=args.opp_plus_root,
        )
        all_tracking.extend(tracking)
    finally:
        perf_monitor.stop()
    progress_tracker.finish()
    perf_monitor.print_summary()
    tracking_path = os.path.join(args.output_dir, "tracking_worker0.json")
    with open(tracking_path, "w") as f:
        f.write(_json_dumps(all_tracking))
    print(f"  Tracking written: {tracking_path}")
    print(f"\nGenerating summary ...")
    summary_tracking = [t for t in all_tracking if not t.get("null_skipped", False)]
    summary_path, summary = build_and_save_summary(
        summary_tracking, args.output_dir, class_map
    )

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
    if _WANDB_OK:
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
