from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.data.sensors import (
    GROUPS_6CH,
    GROUPS_9CH,
    GROUPS_CAPTURE24,
    GROUPS_HOSPITAL,
    GROUPS_USCHAD,
    SENSORS_6CH,
    SENSORS_9CH,
    SENSORS_CAPTURE24,
    SENSORS_HOSPITAL,
    SENSORS_USCHAD,
    _OPP_RAW_COL_NAMES,
    _PAMAP2_FALLBACK_NAMES,
    _build_opportunity_79_groups,
    _channel_names_from_mat,
)
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
from AtteFinalPipeline.expert.settings import get_args as _settings_get_args

try:
    import wandb as _wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


def _get_dataset_config(dataset_name):
    saved_argv = sys.argv
    sys.argv = ["settings", "--dataset", dataset_name]
    try:
        args, config_dataset, config_model = _settings_get_args()
    finally:
        sys.argv = saved_argv
    return {
        "class_map": args.class_map,
        "input_dim": args.input_dim,
        "num_class": args.num_class,
        "window": args.window,
        "stride": args.stride,
        "path_data": args.path_data,
        "path_raw": args.path_raw,
        "path_processed": args.path_processed,
        "config_model": config_model,
    }


assert len(_PAMAP2_FALLBACK_NAMES) == 52


def _build_opportunity_79_names():
    cols_to_delete = set(
        list(range(46, 50))
        + list(range(59, 63))
        + list(range(72, 76))
        + list(range(85, 89))
        + list(range(98, 102))
        + list(range(134, 243))
        + list(range(244, 250))
    )
    surviving = [c for c in range(250) if c not in cols_to_delete]
    feature_raw_cols = surviving[1:80]
    names = []
    for raw_col in feature_raw_cols:
        names.append(_OPP_RAW_COL_NAMES.get(raw_col, f"Col_{raw_col}"))
    assert len(names) == 79, f"Expected 79 Opportunity channel names, got {len(names)}"
    return (names, feature_raw_cols)


_OPPORTUNITY_79_NAMES, _OPPORTUNITY_79_RAW_COLS = _build_opportunity_79_names()
_OPPORTUNITY_79_GROUPS = _build_opportunity_79_groups(_OPPORTUNITY_79_NAMES)


def _build_opportunity_113_names():
    names = []
    accel_locations = [
        "BACK",
        "RUA",
        "RLA",
        "LUA",
        "LLA",
        "HIP",
        "LKN",
        "RKN",
        "RANKE",
        "LANKE",
        "LWRI",
        "RWRI",
    ]
    for loc in accel_locations:
        for ax in ("x", "y", "z"):
            names.append(f"Acc_{loc}_{ax}")
    imu_locations_9ch = ["Back", "RUA", "RLA", "LUA", "LLA"]
    for loc in imu_locations_9ch:
        for mod in ("acc", "gyro", "mag"):
            for ax in ("x", "y", "z"):
                names.append(f"IMU_{loc}_{mod}_{ax}")
    shoe_locs = ["Lshoe", "Rshoe"]
    for loc in shoe_locs:
        for mod in ("acc", "gyro", "mag"):
            for ax in ("x", "y", "z"):
                names.append(f"IMU_{loc}_{mod}_{ax}")
        for nav in ("nav_mag", "nav_head", "nav_roll"):
            names.append(f"IMU_{loc}_{nav}")
    for i in range(8):
        names.append(f"LocomotionCtx_{i}")
    assert len(names) == 113, f"Expected 113, got {len(names)}"
    return names


_OPPORTUNITY_113_NAMES = _build_opportunity_113_names()
_OPPORTUNITY_113_GROUPS = [
    (
        "Triaxial Accelerometers — Back/Arms",
        list(range(0, 15)),
        _OPPORTUNITY_113_NAMES[0:15],
    ),
    (
        "Triaxial Accelerometers — Legs/Feet",
        list(range(15, 36)),
        _OPPORTUNITY_113_NAMES[15:36],
    ),
    (
        "IMU — Back + Right Arm (acc/gyro/mag)",
        list(range(36, 63)),
        _OPPORTUNITY_113_NAMES[36:63],
    ),
    (
        "IMU — Left Arm (acc/gyro/mag)",
        list(range(63, 81)),
        _OPPORTUNITY_113_NAMES[63:81],
    ),
    (
        "IMU — Shoes (acc/gyro/mag/nav)",
        list(range(81, 105)),
        _OPPORTUNITY_113_NAMES[81:105],
    ),
    ("Locomotion Context", list(range(105, 113)), _OPPORTUNITY_113_NAMES[105:113]),
]


def _channel_names_from_opportunity_file(col_names_path):
    if not col_names_path or not os.path.exists(col_names_path):
        return None
    try:
        names = []
        with open(col_names_path, "r", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if not parts:
                    continue
                if parts[0].lower() in ("index", "col", "column", "id", "name"):
                    continue
                if parts[0].lstrip("-").isdigit() and len(parts) >= 2:
                    names.append(" ".join(parts[1:]))
                else:
                    names.append(" ".join(parts))
        if not names:
            print(
                f"  [opportunity] No channel names parsed from {col_names_path} — using fallback names"
            )
            return None
        print(f"  [opportunity] Read {len(names)} channel names from {col_names_path}")
        return names
    except Exception as e:
        print(
            f"  [opportunity] Could not read {col_names_path}: {e} — using fallback names"
        )
        return None


def get_sensor_config(
    n_channels, dataset_name="ucihar", mat_path=None, opportunity_col_names_path=None
):
    if dataset_name == "ucihar":
        if n_channels == 9:
            return (SENSORS_9CH, GROUPS_9CH)
        elif n_channels == 6:
            return (SENSORS_6CH, GROUPS_6CH)
    elif dataset_name == "hospital":
        if n_channels == 6:
            return (SENSORS_HOSPITAL, GROUPS_HOSPITAL)
    elif dataset_name == "uschad":
        if n_channels == 6:
            print(
                f"  [uschad] Using hardcoded 6-channel sensor names (Acc_X/Y/Z + Gyro_X/Y/Z, 100 Hz)."
            )
            return (list(SENSORS_USCHAD), list(GROUPS_USCHAD))
        print(
            f"  [uschad] WARNING: expected 6 channels, got {n_channels} — using generic names."
        )
        names = [f"Ch_{i}" for i in range(n_channels)]
        groups = [("All Channels", list(range(n_channels)), names)]
        return (names, groups)
    elif dataset_name == "capture24":
        if n_channels == 3:
            print(
                f"  [capture24] Using hardcoded 3-channel sensor names (Acc_X/Y/Z, 100 Hz, 2 s windows)."
            )
            return (list(SENSORS_CAPTURE24), list(GROUPS_CAPTURE24))
        print(
            f"  [capture24] WARNING: expected 3 channels, got {n_channels} — using generic names."
        )
        names = [f"Ch_{i}" for i in range(n_channels)]
        groups = [("All Channels", list(range(n_channels)), names)]
        return (names, groups)
    elif dataset_name == "pamap2":
        mat_names = _channel_names_from_mat(mat_path)
        if mat_names and len(mat_names) == 52:
            sensor_names = mat_names
            print(
                f"  [pamap2] Using channel names from .mat file ({len(sensor_names)} channels)"
            )
        else:
            sensor_names = list(_PAMAP2_FALLBACK_NAMES)
            print(f"  [pamap2] Using fallback channel names (official readme ordering)")
        if n_channels == 52:
            groups = [
                ("Heart rate + hand IMU", list(range(0, 18)), sensor_names[0:18]),
                ("Chest IMU", list(range(18, 35)), sensor_names[18:35]),
                ("Ankle IMU", list(range(35, 52)), sensor_names[35:52]),
            ]
            return (sensor_names, groups)
        print(
            f"  [pamap2] WARNING: expected 52 channels, got {n_channels} — using generic names"
        )
    elif dataset_name == "opportunity":
        file_names = _channel_names_from_opportunity_file(opportunity_col_names_path)
        if file_names and len(file_names) == n_channels:
            sensor_names = file_names
            print(
                f"  [opportunity] Using channel names from file ({len(sensor_names)} channels)"
            )
            groups = []
            for g_start in range(0, n_channels, 10):
                g_end = min(g_start + 10, n_channels)
                indices = list(range(g_start, g_end))
                groups.append(
                    (
                        f"Channels {g_start}–{g_end - 1}",
                        indices,
                        sensor_names[g_start:g_end],
                    )
                )
            return (sensor_names, groups)
        if file_names and len(file_names) != n_channels:
            print(
                f"  [opportunity] WARNING: column-names file has {len(file_names)} names but data has {n_channels} channels — ignoring file."
            )
        if n_channels == 79:
            print(
                f"  [opportunity] Using 79-channel pipeline names (traced from OPP_COLS_TO_DELETE + OPP_FEATURE_SLICE)."
            )
            return (list(_OPPORTUNITY_79_NAMES), _OPPORTUNITY_79_GROUPS)
        if n_channels == 113:
            print(f"  [opportunity] Using hardcoded 113-channel challenge names.")
            return (_OPPORTUNITY_113_NAMES, _OPPORTUNITY_113_GROUPS)
        sensor_names = [f"Ch_{i}" for i in range(n_channels)]
        print(
            f"  [opportunity] {n_channels} channels — no hardcoded names for this count; using generic Ch_N names. Pass --opportunity_col_names to supply real names."
        )
        groups = []
        for g_start in range(0, n_channels, 10):
            g_end = min(g_start + 10, n_channels)
            indices = list(range(g_start, g_end))
            groups.append(
                (
                    f"Channels {g_start}–{g_end - 1}",
                    indices,
                    sensor_names[g_start:g_end],
                )
            )
        return (sensor_names, groups)
    elif dataset_name == "skoda":
        names = [f"Ch_{i}" for i in range(n_channels)]
        groups = []
        for g_start in range(0, n_channels, 10):
            g_end = min(g_start + 10, n_channels)
            indices = list(range(g_start, g_end))
            groups.append(
                (f"Channels {g_start}-{g_end - 1}", indices, names[g_start:g_end])
            )
        return (names, groups)
    names = [f"Ch_{i}" for i in range(n_channels)]
    groups = [("All Channels", list(range(n_channels)), names)]
    return (names, groups)


def get_null_class_indices(class_map, dataset_name):
    if dataset_name not in ("opportunity", "skoda"):
        return set()
    NULL_NAMES = {"null", "none", "background", "other", "unknown", "0"}
    null_indices = set()
    for idx, name in enumerate(class_map):
        if name.strip().lower() in NULL_NAMES or name.strip() == "0":
            null_indices.add(idx)
    null_indices.add(0)
    if null_indices:
        null_names = [
            class_map[i] if i < len(class_map) else str(i) for i in sorted(null_indices)
        ]
        print(
            f"  [{dataset_name}] Null-class filter active — skipping class indices {sorted(null_indices)} ({null_names})"
        )
    return null_indices


_CAPTURE24_SUBJECT_IDS = {}


def _load_capture24_from_bundle(npz_path, split):
    split = split.lower()
    _key_map = {
        "train": ("X_train", "y_train", "subject_ids_train"),
        "val": ("X_valid", "y_valid", "subject_ids_valid"),
        "valid": ("X_valid", "y_valid", "subject_ids_valid"),
        "validation": ("X_valid", "y_valid", "subject_ids_valid"),
        "test": ("X_test", "y_test", "subject_ids_test"),
    }
    if split not in _key_map:
        print(
            f"ERROR: Unknown split '{split}' for capture24. Choose from: train, val, test."
        )
        sys.exit(1)
    sids = None
    with np.load(npz_path, allow_pickle=False) as contents:
        keys = set(contents.files)
        x_key, y_key, sid_key = _key_map[split]
        if x_key not in keys:
            print(
                f"ERROR: Key '{x_key}' not found in {npz_path}. Re-run prepare_capture24*.py."
            )
            sys.exit(1)
        data = contents[x_key].astype(np.float32)
        target = contents[y_key].reshape(-1).astype(np.int64)
        if sid_key in keys:
            sids = contents[sid_key]
            sids = np.asarray(sids).astype(str).reshape(-1)
            if len(sids) != len(target):
                print(
                    f"  [capture24] WARNING: subject_ids length ({len(sids)}) != targets length ({len(target)}) in {npz_path} — ignoring subject IDs."
                )
                sids = None
            else:
                _CAPTURE24_SUBJECT_IDS[npz_path, split] = sids
                unique_subjects = sorted(set(sids.tolist()))
                print(
                    f"  [capture24] Bundle includes subject IDs ({len(unique_subjects)} unique subjects in {split})."
                )
        if "subset_fraction" in keys:
            frac = float(contents["subset_fraction"])
            cap = (
                int(contents["subset_cap_per_cell"])
                if "subset_cap_per_cell" in keys
                else None
            )
            floor_ = int(contents["subset_floor"]) if "subset_floor" in keys else None
            seed = int(contents["subset_seed"]) if "subset_seed" in keys else None
            print(
                f"  [capture24] Subset bundle detected — fraction={frac:.2f}"
                + (f", cap={cap}" if cap is not None else "")
                + (f", floor={floor_}" if floor_ is not None else "")
                + (f", seed={seed}" if seed is not None else "")
            )
    print(
        f"  [capture24] Loaded {split} from .npz: {npz_path} — shape {data.shape}, labels {np.unique(target).tolist()}"
    )
    return (data, target, sids)


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
                data = ds["data"].astype(np.float32)
                target = ds["target"].astype(np.int64)
                print(
                    f"  [uschad] Loaded {split} from processed NPZ: {npz_path} — shape {data.shape}"
                )
                return (data, target, None)
        mat_path = (
            ds_config["path_data"] if ds_config else None
        ) or "./dataset/uschad.mat"
        if not os.path.exists(mat_path):
            print(
                f"ERROR: USC-HAD .mat not found at {mat_path}. Run prepare_uschad.py first."
            )
            sys.exit(1)
        import scipy.io as sio

        contents = sio.loadmat(mat_path)
        _key_map = {
            "train": ("X_train", "y_train"),
            "val": ("X_valid", "y_valid"),
            "valid": ("X_valid", "y_valid"),
            "validation": ("X_valid", "y_valid"),
            "test": ("X_test", "y_test"),
        }
        if split not in _key_map:
            print(
                f"ERROR: Unknown split '{split}' for uschad. Choose from: train, val, test."
            )
            sys.exit(1)
        x_key, y_key = _key_map[split]
        if x_key not in contents:
            print(
                f"ERROR: Key '{x_key}' not found in {mat_path}. Re-run prepare_uschad.py."
            )
            sys.exit(1)
        data = contents[x_key].astype(np.float32)
        target = contents[y_key].reshape(-1).astype(np.int64)
        print(
            f"  [uschad] Loaded {split} from .mat: {mat_path} — shape {data.shape}, labels {np.unique(target).tolist()}"
        )
        return (data, target, None)
    if dataset_name == "capture24":
        if data_path and os.path.isfile(data_path) and data_path.endswith(".npz"):
            return _load_capture24_from_bundle(data_path, split)
        processed_base = (
            data_path
            if data_path and os.path.isdir(data_path)
            else ds_config["path_processed"]
            if ds_config
            else None
        )
        if processed_base and os.path.isdir(processed_base):
            split_file = "val" if split in ("validation", "valid") else split
            npz_path = os.path.join(processed_base, f"{split_file}.npz")
            if os.path.exists(npz_path):
                ds = np.load(npz_path)
                data = ds["data"].astype(np.float32)
                target = ds["target"].astype(np.int64)
                print(
                    f"  [capture24] Loaded {split} from processed NPZ: {npz_path} — shape {data.shape}"
                )
                return (data, target, None)
        subset_candidates = [
            "./dataset/capture24_Willetts2018_subset.npz",
            "dataset/capture24_Willetts2018_subset.npz",
            "./dataset/capture24_subset.npz",
            "dataset/capture24_subset.npz",
        ]
        for sub_path in subset_candidates:
            if os.path.exists(sub_path):
                print(
                    f"  [capture24] Found subset bundle at {sub_path} — using it (prepared by prepare_capture24_subset.py)."
                )
                return _load_capture24_from_bundle(sub_path, split)
        raw_npz_path = (
            ds_config["path_data"] if ds_config else None
        ) or "./dataset/capture24_Willetts2018.npz"
        if not os.path.exists(raw_npz_path):
            legacy = "./dataset/capture24.npz"
            if os.path.exists(legacy):
                print(f"  [capture24] Falling back to legacy bundle {legacy}.")
                raw_npz_path = legacy
            else:
                print(
                    f"ERROR: Capture-24 .npz not found at {raw_npz_path}. Run prepare_capture24.py --label_schema Willetts2018 (or prepare_Capture24_subset.py for the stratified subset) first."
                )
                sys.exit(1)
        return _load_capture24_from_bundle(raw_npz_path, split)
    if dataset_name == "ucihar":
        base = data_path or (ds_config["path_processed"] if ds_config else None)
        if base is None:
            print("ERROR: No data_path provided for ucihar")
            sys.exit(1)
        x_path = os.path.join(base, f"X_{split}.npy")
        y_path = os.path.join(base, f"y_{split}.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            data = np.load(x_path).astype(np.float32)
            target = np.load(y_path).astype(np.longlong)
            if np.min(target) > 0:
                target = target - 1
            return (data, target, None)
        npz_path = os.path.join(base, f"{split}.npz")
        if os.path.exists(npz_path):
            ds = np.load(npz_path)
            data = ds["data"].astype(np.float32)
            target = ds["target"].astype(np.longlong)
            if np.min(target) > 0:
                target = target - 1
            return (data, target, None)
        print(f"ERROR: No data files found for ucihar split='{split}' in {base}")
        sys.exit(1)
    base = (
        data_path if data_path else ds_config["path_processed"] if ds_config else None
    )
    if base is None:
        print(f"ERROR: No data_path or path_processed for dataset={dataset_name}")
        sys.exit(1)
    split_file = split
    if split == "validation":
        split_file = "val"
    npz_path = os.path.join(base, f"{split_file}.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: Data file not found: {npz_path}")
        sys.exit(1)
    ds = np.load(npz_path)
    data = ds["data"].astype(np.float32)
    target = ds["target"].astype(np.longlong)
    return (data, target, None)


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
            if _WANDB_OK:
                try:
                    if _wandb.run is not None:
                        wb_row = {}
                        if gpu_util is not None:
                            wb_row["sys/gpu_util_pct"] = gpu_util
                        if gpu_mem_used is not None:
                            wb_row["sys/gpu_mem_used_mb"] = gpu_mem_used
                        if cpu_pct is not None:
                            wb_row["sys/cpu_util_pct"] = cpu_pct
                        if ram_used is not None:
                            wb_row["sys/ram_used_gb"] = ram_used
                        with self._lock:
                            wb_row["throughput/samples_per_sec"] = round(
                                self._samples_per_sec, 3
                            )
                        if wb_row:
                            _wandb.log(wb_row, commit=False)
                except Exception:
                    pass

    def log_sample(
        self,
        split,
        sample_idx,
        pred_class,
        confidence,
        ig_sec,
        shap_sec,
        save_queue_depth,
    ):
        pass

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
            n = len(self._sample_ig_ms)
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
        if _WANDB_OK:
            try:
                if _wandb.run is not None:
                    _wandb.log(
                        {
                            "ig_ms": round(self._last_ig_ms, 2),
                            "shap_ms": round(self._last_shap_ms, 2),
                            "save_queue_depth": int(save_queue_depth),
                            "samples_processed": n,
                        }
                    )
            except Exception:
                pass

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
        print(f"  {'Metric':<30s} {'mean':>7s} {'p50':>7s} {'p95':>7s}")
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
        print(f"  {'─' * 58}")

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
            if use_compile:
                print("  torch.compile skipped (CUDA not available).")
            else:
                print("  torch.compile disabled via --no_compile.")
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
        _ = self.compute_shap(dummy[0], target_class)
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

    def compute_ig(self, data_tensor_on_gpu, target_class, n_steps=25):
        inp = data_tensor_on_gpu.unsqueeze(0)
        attr = self.ig.attribute(
            inp,
            baselines=self.ig_baseline,
            target=int(target_class),
            n_steps=n_steps,
            internal_batch_size=n_steps,
        )
        attr = attr[0].detach().cpu().numpy()
        np.abs(attr, out=attr)
        mx = attr.max()
        if mx > 0:
            attr *= 1.0 / mx
        return attr

    def compute_shap(self, data_tensor_on_gpu, target_class):
        x = data_tensor_on_gpu.unsqueeze(0)
        bg = self._shap_background.detach()
        n_bg = self._n_bg
        alphas = torch.rand(
            n_bg, 1, 1, device=self.device, dtype=x.dtype, generator=self._shap_rng
        )
        diffs = x - bg
        interp = bg + alphas * diffs
        interp = interp.requires_grad_(True)
        self.compiled_wrapper.zero_grad(set_to_none=True)
        logits = self.compiled_wrapper(interp)
        target_logits = logits[:, int(target_class)]
        target_logits.sum().backward()
        grads = interp.grad
        attr = (grads * diffs).mean(dim=0)
        attr = attr.detach().cpu().numpy()
        np.abs(attr, out=attr)
        mx = attr.max()
        if mx > 0:
            attr *= 1.0 / mx
        return attr

    def compute_combined(self, data_tensor_on_gpu, target_class, n_steps=25, eps=1e-08):
        ig_attr = self.compute_ig(data_tensor_on_gpu, target_class, n_steps)
        shap_attr = self.compute_shap(data_tensor_on_gpu, target_class)
        return self._combine(ig_attr, shap_attr, eps=eps)

    def _combine(self, ig_attr, shap_attr, eps=1e-08):
        combined = np.sqrt((ig_attr + eps) * (shap_attr + eps))
        mx = combined.max()
        if mx > 0:
            combined *= 1.0 / mx
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
    subject_id=None,
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
    regions, attr_threshold = extract_high_attribution_regions(
        combined, sensor_names, percentile=90
    )
    label_name = (
        class_map[true_label] if true_label < len(class_map) else f"Class_{true_label}"
    )
    result = {
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
        "attribution_threshold_p90": attr_threshold,
        "high_attribution_regions": regions,
        "_data": data_np,
        "_combined": combined,
        "_sensor_importance": sensor_importance,
        "_phase_importance": phase_importance,
    }
    if subject_id is not None:
        result["subject_id"] = str(subject_id)
    return result


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
    result, sensor_names, mantis_embedding=None, dataset_name="ucihar", class_map=None
):
    if class_map is None:
        class_map = result.get("_class_map", ["Unknown"])
    data_np = result["_data"]
    T, D = data_np.shape
    split = result["split"]
    raw_sensor_data = {}
    for ch_idx, ch_name in enumerate(sensor_names):
        raw_sensor_data[ch_name] = np.round(data_np[:, ch_idx], 6).tolist()
    xai_methods = [
        "Captum Integrated Gradients (training-mean baseline)",
        "Manual Batched Gradient SHAP (expected gradient, single fwd+bwd over all bg)",
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
                "description": "Contiguous runs of timesteps per sensor where the combined attribution >= global 90th percentile. Sensor-agnostic — sensors with no cells above threshold produce zero regions.",
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
    if "subject_id" in result:
        entry["subject_id"] = result["subject_id"]
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
    subject_ids=None,
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
            f"  [{split.upper()}] Null-class filter: skipping {n_null} samples with ground-truth in {sorted(null_class_indices)} ({N - n_null} remaining)"
        )
    else:
        null_mask = np.zeros(N, dtype=bool)
        valid_mask = np.ones(N, dtype=bool)
        n_null = 0
    valid_indices = np.where(valid_mask)[0]
    labels_valid = labels_int[valid_indices]
    pred_valid = pred_classes[valid_indices]
    probs_valid = all_probs[valid_indices]
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
    all_tracking = []
    for i in range(N):
        if null_mask[i]:
            all_tracking.append(
                {
                    "sample_idx": int(i),
                    "true_label": int(labels_int[i]),
                    "predicted_label": int(pred_classes[i]),
                    "correct": False,
                    "split": split,
                    "null_skipped": True,
                }
            )
        else:
            all_tracking.append(
                {
                    "sample_idx": int(i),
                    "true_label": int(labels_int[i]),
                    "predicted_label": int(pred_classes[i]),
                    "correct": bool(pred_classes[i] == labels_int[i]),
                    "split": split,
                    "null_skipped": False,
                }
            )
    if worker_id == 0:
        for i in np.where(null_mask)[0]:
            progress_tracker.update(split, correct=False)
        for _ in incorrect_indices:
            progress_tracker.update(split, correct=False)
    progress_tracker.write_periodic(every_n=1)
    print(
        f"  [{split.upper()}] XAI attribution + live saving (n_steps={n_steps}, workers={plot_workers}, dpi={save_dpi}) ..."
    )
    correct_data_tensor = torch.tensor(data[correct_indices], dtype=torch.float32).to(
        device
    )
    saved_indices = []
    pending_futures = []
    n_saved = 0
    save_executor = ThreadPoolExecutor(max_workers=plot_workers)
    try:
        pbar = tqdm(
            range(n_correct),
            desc=f"  {split.upper()} XAI+Save",
            unit="sample",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for local_idx in pbar:
            gi = int(correct_indices[local_idx])
            pred_class = int(pred_classes[gi])
            confidence = float(all_probs[gi][pred_class])
            if confidence < min_confidence:
                progress_tracker.update(split, correct=True)
                continue
            sample_on_gpu = correct_data_tensor[local_idx]
            subject_id = subject_ids[gi] if subject_ids is not None else None
            if perf_monitor is not None:
                with perf_monitor.time_ig():
                    ig_attr = xai_engine.compute_ig(
                        sample_on_gpu, pred_class, n_steps=n_steps
                    )
                with perf_monitor.time_shap():
                    shap_attr = xai_engine.compute_shap(sample_on_gpu, pred_class)
                combined = xai_engine._combine(ig_attr, shap_attr)
            else:
                combined = xai_engine.compute_combined(
                    sample_on_gpu, pred_class, n_steps=n_steps
                )
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
                subject_id=subject_id,
            )
            save_arg = (
                result,
                output_dir,
                sensor_names,
                sensor_groups,
                None,
                save_dpi,
                plot_every_n > 0 and n_saved % plot_every_n == 0,
                dataset_name,
                class_map,
            )
            future = save_executor.submit(_save_dashboard_and_json, save_arg)
            pending_futures.append((future, gi))
            saved_indices.append(gi)
            n_saved += 1
            if perf_monitor is not None:
                perf_monitor.record_sample(save_queue_depth=len(pending_futures))
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
            progress_tracker.update(split, correct=True)
            progress_tracker.write_periodic(every_n=50)
        pbar.close()
        del correct_data_tensor
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
    print(f"  [{split.upper()}] All {n_saved} PNGs + JSONs saved.")
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
        description="XAI Full Analysis Dashboard (Multi-Dataset)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucihar",
        choices=[
            "opportunity",
            "skoda",
            "pamap2",
            "hospital",
            "ucihar",
            "uschad",
            "capture24",
        ],
        help="HAR dataset to analyze",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data directory. For uschad: folder with processed .npz files (train.npz/val.npz/test.npz) OR omit to fall back to reading dataset/uschad.mat directly. For capture24: folder with processed .npz files OR omit to fall back to reading dataset/capture24.npz directly. For ucihar: folder with X_train.npy/y_train.npy or processed .npz files. For others: folder with train.npz/val.npz/test.npz. Default: uses path_processed from settings.py.",
    )
    parser.add_argument(
        "--mat_file",
        type=str,
        default=None,
        help="Path to pamap2.mat (our pipeline version). Only used when --dataset pamap2. For uschad, the .mat is found automatically via settings.py (dataset/uschad.mat). For capture24, the .npz is found automatically via settings.py (dataset/capture24.npz).",
    )
    parser.add_argument(
        "--opportunity_col_names",
        type=str,
        default=None,
        help="Path to a plain-text file listing Opportunity channel names, one per line (optionally preceded by an integer index). Only used when --dataset opportunity.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Which data splits to process. Default: train test. Options: train, val, test",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: ./xai_output/<dataset>/",
    )
    parser.add_argument("--mantis_checkpoint", type=str, default="paris-noah/MantisV2")
    parser.add_argument("--shap_bg_per_class", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(8, min(24, (os.cpu_count() or 16) // 2))
    parser.add_argument(
        "--plot_workers",
        type=int,
        default=_cpu_default,
        help=f"Threads for parallel PNG/JSON saving. Default {_cpu_default}.",
    )
    parser.add_argument("--save_dpi", type=int, default=100)
    parser.add_argument(
        "--plot_every_n",
        type=int,
        default=50,
        help="Save a PNG dashboard for every Nth correctly-predicted sample. Default 50. Set to 1 for all, 0 for none.",
    )
    parser.add_argument(
        "--max_plot_channels",
        type=int,
        default=MAX_PLOT_CHANNELS,
        help=f"Max sensor channels to show on dashboard plots. Default {MAX_PLOT_CHANNELS}. Only the top-N by importance are plotted; all channels are still in JSON.",
    )
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    return parser.parse_args()


def find_checkpoint(dataset_name, explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    search_patterns = [
        f"./models/{dataset_name}/train_*/checkpoints/checkpoint_best.pth",
        f"./weights/checkpoint_{dataset_name}.pth",
    ]
    if dataset_name == "ucihar":
        search_patterns.insert(
            0, "./models/ucihar/train_*/checkpoints/checkpoint_best.pth"
        )
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    return None


def main():
    args = parse_args()
    ds_config = _get_dataset_config(args.dataset)
    class_map = ds_config["class_map"]
    input_dim = ds_config["input_dim"]
    num_class = ds_config["num_class"]
    config_model_from_settings = ds_config["config_model"]
    if args.output_dir is None:
        args.output_dir = f"./xai_output/{args.dataset}/"
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
    print(f"cuDNN:         ON")
    print(
        f"torch.compile: {('OFF (--no_compile)' if args.no_compile else 'ON (mode=default)')}"
    )
    print(f"AMP float16:   {('ON (--use_amp)' if args.use_amp else 'OFF')}")
    print(f"IG n_steps:    {args.n_steps}")
    print(f"SHAP bg/class: {args.shap_bg_per_class}")
    print(f"Save DPI:      {args.save_dpi}")
    print(f"Plot every N:  {args.plot_every_n}")
    print(f"Max plot ch:   {args.max_plot_channels}")
    print(f"Min confidence:{args.min_confidence}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU:           {gpu_name}")
    print()
    print("[1/6] Loading data ...")
    data_path = args.data_path or ds_config["path_processed"]
    split_data = {}
    split_subjects = {}
    for split_name in args.splits:
        d, t, sids = load_data(
            args.dataset, data_path, split=split_name, ds_config=ds_config
        )
        split_data[split_name] = (d, t)
        split_subjects[split_name] = sids
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
        print("  Train split not in --splits; loading it for baseline only ...")
        train_data, train_labels, _ = load_data(
            args.dataset, data_path, split="train", ds_config=ds_config
        )
    if null_class_indices:
        null_mask_train = np.isin(train_labels.astype(int), list(null_class_indices))
        n_null_train = int(null_mask_train.sum())
        if n_null_train > 0:
            print(
                f"  Excluding {n_null_train} null-class samples from IG baseline and SHAP background computation."
            )
            train_data_bg = train_data[~null_mask_train]
            train_labels_bg = train_labels[~null_mask_train]
        else:
            train_data_bg = train_data
            train_labels_bg = train_labels
    else:
        train_data_bg = train_data
        train_labels_bg = train_labels
    baseline_np = train_data_bg.mean(axis=0).astype(np.float32)
    ig_baseline = torch.tensor(baseline_np, dtype=torch.float32).unsqueeze(0).to(device)
    rng = np.random.RandomState(args.seed)
    bg_indices = []
    for cls in sorted(np.unique(train_labels_bg)):
        cls_idxs = np.where(train_labels_bg == cls)[0]
        n = min(args.shap_bg_per_class, len(cls_idxs))
        chosen = rng.choice(cls_idxs, size=n, replace=False)
        bg_indices.extend(chosen.tolist())
    shap_background = torch.tensor(train_data_bg[bg_indices], dtype=torch.float32).to(
        device
    )
    print(f"  IG baseline shape: {tuple(ig_baseline.shape)}")
    print(
        f"  SHAP background: {shap_background.shape[0]} samples ({args.shap_bg_per_class}/class)"
    )
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
        print("  WARNING: No checkpoint found.")
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
    print(f"  Monitor: watch -n2 cat {progress_tracker.path}")
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
                subject_ids=split_subjects.get(split_name),
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
                print(
                    f"  Merged tracking from worker {wid} ({len(all_tracking_merged)} total)"
                )
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
        print(f"  Worker {args.worker_id}: skipping summary (worker 0 handles it).")
        elapsed = progress_tracker.state["overall"]["elapsed_seconds"]
        print(
            f"  Worker {args.worker_id} done. Time: {elapsed:.1f}s ({elapsed / 60:.1f}m)"
        )
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
    print(f"  {total_saved} dashboard PNGs + JSONs")
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
                        "run/incorrect_skipped": ov["incorrect_predictions"],
                        "run/accuracy_pct": ov["overall_accuracy_percent"],
                        "run/total_time_s": elapsed,
                        "run/samples_per_sec": round(
                            ov["total_samples"] / max(elapsed, 1), 2
                        ),
                    }
                )
                rows = [
                    [
                        pc["class_name"],
                        pc["total_samples"],
                        pc["correct_predictions"],
                        round(100 - pc["error_rate_percent"], 2),
                    ]
                    for pc in ov["per_class"]
                ]
                _wandb.log(
                    {
                        "per_class_accuracy": _wandb.Table(
                            columns=["class", "total", "correct", "accuracy_pct"],
                            data=rows,
                        )
                    }
                )
                _wandb.finish()
                print(
                    f"  W&B run finished: {(_wandb.run.url if _wandb.run else 'closed')}"
                )
        except Exception as e:
            print(f"  W&B finish failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
