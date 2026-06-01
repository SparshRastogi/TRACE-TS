"""
TRACE — Configuration, dataset registry, CLI argument parser.

All user-facing parameters are defined in parse_args(). CONFIG holds only
internal logging intervals and dataset-specific lists that are populated by
apply_dataset_config() from the DATASET_CONFIGS registry at startup.
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ===========================================================================
# DATASET CONFIGS REGISTRY
# Keys: activity_classes, valid_sensors, sensor_terms
# apply_dataset_config() fills any of these into cfg ONLY if the user
# did not explicitly provide them via CLI.
# ===========================================================================

_UCIHAR_SENSORS = {
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
    "gyro_x", "gyro_y", "gyro_z",
}
_PAMAP2_SENSORS = {
    'heart_rate', 'hand_temp', 'hand_acc16g_x', 'hand_acc16g_y', 'hand_acc16g_z',
    'hand_acc6g_x', 'hand_acc6g_y', 'hand_acc6g_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'hand_mag_x', 'hand_mag_y', 'hand_mag_z',
    'hand_orient_0', 'hand_orient_1', 'hand_orient_2', 'hand_orient_3',
    'chest_temp', 'chest_acc16g_x', 'chest_acc16g_y', 'chest_acc16g_z',
    'chest_acc6g_x', 'chest_acc6g_y', 'chest_acc6g_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
    'chest_orient_0', 'chest_orient_1', 'chest_orient_2', 'chest_orient_3',
    'ankle_temp', 'ankle_acc16g_x', 'ankle_acc16g_y', 'ankle_acc16g_z',
    'ankle_acc6g_x', 'ankle_acc6g_y', 'ankle_acc6g_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'ankle_orient_0', 'ankle_orient_1', 'ankle_orient_2', 'ankle_orient_3',
}
_USCHAD_SENSORS = {
    "body_acc_x", "body_acc_y", "body_acc_z",
    "gyro_x", "gyro_y", "gyro_z",
}
_CAPTURE24_SENSORS = {
    "body_acc_x", "body_acc_y", "body_acc_z",
}
_OPPORTUNITY_SENSORS = {
    "IMU_BACK_Quat3", "Lshoe_Euler_pitch", "IMU_BACK_accZ", "IMU_BACK_magY",
    "IMU_RLA_gyrX", "Acc_LUA^_accZ", "Acc_LUA^_accX", "Acc_LUA^_accY",
    "IMU_RLA_Quat1", "IMU_RUA_Quat4", "IMU_BACK_gyrZ", "IMU_BACK_gyrY",
    "Acc_RKN^_accX", "IMU_RUA_Quat1", "IMU_LLA_gyrX", "IMU_LLA_magZ",
    "IMU_RUA_accX", "IMU_BACK_Quat4", "Acc_HIP_accY", "IMU_RUA_magY",
    "IMU_RLA_Quat2", "Acc_LH_accY", "Acc_RKN^_accZ", "IMU_RLA_magZ",
    "IMU_BACK_gyrX", "IMU_LUA_Quat1", "IMU_RUA_magZ", "IMU_BACK_Quat1",
    "Acc_RUA_accY", "IMU_RUA_Quat3", "IMU_RUA_Quat2", "Acc_LH_accX",
    "IMU_RUA_accZ", "Acc_RUA_accX", "IMU_BACK_magX", "Acc_HIP_accZ",
    "IMU_RLA_Quat4", "IMU_RUA_gyrX", "IMU_BACK_accY", "IMU_BACK_Quat2",
    "Lshoe_gyrX", "Acc_RKN^_accY", "IMU_LLA_Quat3", "IMU_BACK_magZ",
    "IMU_LUA_accY", "Acc_LH_accZ", "Lshoe_Quat4", "IMU_RLA_accX",
    "Acc_HIP_accX", "IMU_LLA_Quat1", "IMU_LLA_Quat4", "IMU_BACK_accX",
    "Acc_RUA_accZ", "IMU_RUA_accY", "IMU_LLA_Quat2", "Lshoe_accX",
    "IMU_RUA_magX", "IMU_LUA_Quat4", "Lshoe_Quat2", "Lshoe_Quat3",
    "IMU_LLA_accZ", "Lshoe_Quat1", "IMU_LUA_Quat3", "IMU_RLA_accY",
    "Lshoe_accZ", "Lshoe_Euler_yaw", "IMU_RLA_accZ", "Lshoe_accY",
    "IMU_RLA_Quat3", "IMU_LLA_accY", "IMU_RUA_gyrZ", "IMU_RUA_gyrY",
    "IMU_LLA_accX", "Lshoe_magZ", "IMU_LUA_accX", "IMU_LUA_accZ",
    "IMU_LUA_Quat2", "IMU_LUA_gyrX", "IMU_LUA_magZ",
}
_SHOAIB_SENSORS = {
    "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
    "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z",
    "wrist_linacc_x", "wrist_linacc_y", "wrist_linacc_z",
    "leftpocket_acc_x", "leftpocket_acc_y", "leftpocket_acc_z",
    "leftpocket_gyro_x", "leftpocket_gyro_y", "leftpocket_gyro_z",
    "leftpocket_linacc_x", "leftpocket_linacc_y", "leftpocket_linacc_z",
    "rightpocket_acc_x", "rightpocket_acc_y", "rightpocket_acc_z",
    "rightpocket_gyro_x", "rightpocket_gyro_y", "rightpocket_gyro_z",
    "rightpocket_linacc_x", "rightpocket_linacc_y", "rightpocket_linacc_z",
    "belt_acc_x", "belt_acc_y", "belt_acc_z",
    "belt_gyro_x", "belt_gyro_y", "belt_gyro_z",
    "belt_linacc_x", "belt_linacc_y", "belt_linacc_z",
    "upperarm_acc_x", "upperarm_acc_y", "upperarm_acc_z",
    "upperarm_gyro_x", "upperarm_gyro_y", "upperarm_gyro_z",
    "upperarm_linacc_x", "upperarm_linacc_y", "upperarm_linacc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "linacc_x", "linacc_y", "linacc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "acc_x", "acc_y", "acc_z",
}
_MHEALTH_SENSORS = {
    "lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
    "lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
    "lankle_mag_x", "lankle_mag_y", "lankle_mag_z",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_ecg_i", "chest_ecg_ii", "chest_ecg_iii",
    "rwrist_acc_x", "rwrist_acc_y", "rwrist_acc_z",
    "rwrist_gyro_x", "rwrist_gyro_y", "rwrist_gyro_z",
    "rwrist_mag_x", "rwrist_mag_y", "rwrist_mag_z",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "ecg_i", "ecg_ii", "ecg_iii",
    "body_acc_x", "body_acc_y", "body_acc_z",
}

DATASET_CONFIGS = {
    "ucihar": {
        "activity_classes": [
            "walking", "walking upstairs", "walking downstairs",
            "sitting", "standing", "laying",
        ],
        "valid_sensors": _UCIHAR_SENSORS,
        "sensor_terms": [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "accelerometer", "gyroscope", "acceleration", "angular velocity",
            "x-axis", "y-axis", "z-axis",
        ],
    },
    "pamap2": {
        "activity_classes": [
            'ascending stairs', 'cycling', 'descending stairs', 'ironing',
            'lying', 'nordic walking', 'rope jumping', 'running',
            'sitting', 'standing', 'vacuum cleaning', 'walking',
        ],
        "valid_sensors": _PAMAP2_SENSORS,
        "sensor_terms": sorted(_PAMAP2_SENSORS) + [
            "accelerometer", "gyroscope", "magnetometer", "temperature",
            "heart rate", "acc", "gyro", "mag",
        ],
    },
    "uschad": {
        "activity_classes": [
            'walking forward', 'walking left', 'walking right', 'walking upstairs',
            'elevator up', 'elevator down', 'walking downstairs',
            'running forward', 'jumping', 'sitting', 'standing', 'sleeping',
        ],
        "valid_sensors": _USCHAD_SENSORS,
        "sensor_terms": [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "accelerometer", "gyroscope", "acceleration", "angular velocity",
            "x-axis", "y-axis", "z-axis",
        ],
    },
    "capture24": {
        "activity_classes": [
            'bicycling', 'mixed', 'sit-stand', 'sleep', 'vehicle', 'walking',
        ],
        "valid_sensors": _CAPTURE24_SENSORS,
        "sensor_terms": [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "accelerometer", "acceleration", "x-axis", "y-axis", "z-axis",
        ],
    },
    "opportunity": {
        "activity_classes": [
            'clean table', 'close dishwasher', 'close door 1', 'close door 2',
            'close drawer 1', 'close drawer 2', 'close drawer 3', 'close fridge',
            'drink from cup', 'open dishwasher', 'open door 1', 'open door 2',
            'open drawer 1', 'open drawer 2', 'open drawer 3', 'open fridge',
            'toggle switch',
        ],
        "valid_sensors": _OPPORTUNITY_SENSORS,
        "sensor_terms": list(_OPPORTUNITY_SENSORS) + [
            "accelerometer", "gyroscope", "magnetometer",
        ],
    },
    "shoaib": {
        "activity_classes": [
            'biking', 'jogging', 'sitting', 'standing', 'walking',
            'walking downstairs', 'walking upstairs',
        ],
        "valid_sensors": _SHOAIB_SENSORS,
        "sensor_terms": sorted(_SHOAIB_SENSORS) + [
            "accelerometer", "gyroscope", "linear acceleration",
            "wrist", "belt", "pocket", "upper arm",
            "x-axis", "y-axis", "z-axis",
        ],
    },
    "mhealth": {
        "activity_classes": [
            'climbing stairs', 'cycling', 'frontal elevation of arms',
            'jogging', 'jump front & back', 'knees bending', 'lying down',
            'running', 'sitting and relaxing', 'standing still',
            'waist bends forward', 'walking',
        ],
        "valid_sensors": _MHEALTH_SENSORS,
        "sensor_terms": [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
            "ecg_i", "ecg_ii", "ecg_iii",
            "chest_ecg_i", "chest_ecg_ii", "chest_ecg_iii",
            "chest_acc_x", "chest_acc_y", "chest_acc_z",
            "lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
            "lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
            "lankle_mag_x", "lankle_mag_y", "lankle_mag_z",
            "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
            "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
            "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
            "rwrist_acc_x", "rwrist_acc_y", "rwrist_acc_z",
            "rwrist_gyro_x", "rwrist_gyro_y", "rwrist_gyro_z",
            "rwrist_mag_x", "rwrist_mag_y", "rwrist_mag_z",
            "body_acc_x", "body_acc_y", "body_acc_z",
            "accelerometer", "gyroscope", "magnetometer", "ecg",
            "chest", "ankle", "wrist", "x-axis", "y-axis", "z-axis",
        ],
    },
}


def apply_dataset_config(cfg: dict, cli_provided_keys: set) -> None:
    """Fill activity_classes / valid_sensors / sensor_terms from DATASET_CONFIGS
    for the current dataset, but ONLY for keys not explicitly provided via CLI.
    """
    dataset = cfg.get("dataset", "").lower()
    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        print(f"[config] WARNING: dataset '{dataset}' not in DATASET_CONFIGS. "
              "Using CONFIG defaults for sensor_terms/valid_sensors/activity_classes.")
        return
    for key in ("activity_classes", "valid_sensors", "sensor_terms"):
        if key not in cli_provided_keys:
            cfg[key] = ds_cfg[key]
            print(f"[config] {key}: loaded from DATASET_CONFIGS['{dataset}'] "
                  f"({len(ds_cfg[key])} entries)")
        else:
            print(f"[config] {key}: using CLI override ({len(cfg[key])} entries)")


# Internal logging intervals — not exposed as CLI args.
# Dataset-specific lists are populated by apply_dataset_config() at startup.
CONFIG = {
    "adapter_log_every":   50,
    "vram_log_every":      100,
    "crossattn_log_every": 200,
    "activity_classes": ["walking", "sitting", "standing"],
    "sensor_terms":     ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    "valid_sensors":    {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"},
}

STRUCTURED_TEMPLATE_PREFIX = "\n[OBSERVATION | id: O1]\nsensor:"

_OBS_RE = re.compile(
    r'\[OBSERVATION\s*\|\s*id:\s*(O\d+)\]\s*\n'
    r'sensor:\s*(.+?)\n'
    r'temporal:\s*(.+?)\n'
    r'pattern:\s*(.+?)\n'
    r'confidence:\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_INF_RE = re.compile(
    r'\[INFERENCE\s*\|\s*id:\s*(I\d+)\]\s*\n'
    r'based_on:\s*(.+?)\n'
    r'inference:\s*(.+?)\n'
    r'confidence:\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_SYN_RE = re.compile(
    r'\[SYNTHESIS\]\s*\n'
    r'based_on:\s*(.+?)\n'
    r'([\s\S]+?)(?=\[ACTIVITY\]|\Z)',
    re.IGNORECASE,
)
_ACT_RE = re.compile(
    r'\[ACTIVITY\]\s*:\s*(.+?)$',
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> dict:
    """Single source of truth for all user-facing parameters.

    PATH CONVENTION (all derived from --data_root + --dataset if not overridden):
      train_json_dir : <data_root>/outputs/<dataset>/<reasoning_subdir>/train
      test_json_dir  : <data_root>/outputs/<dataset>/<reasoning_subdir>/test
      raw_embed_dir  : <data_root>/data/raw_embeds/<dataset>/test
      ig_json_dir    : <data_root>/data/<dataset>
      output_dir     : <data_root>/evaluate/<dataset>
      checkpoint_dir : <data_root>/checkpoints/<dataset>
    """
    p = argparse.ArgumentParser(
        description="TRACE — Train / Evaluate / Figure3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "evaluate", "figure3"], required=True)

    p.add_argument("--data_root",
                   default="",
                   help="Root data directory.")
    p.add_argument("--reasoning_subdir",
                   default="reasoning_outputs_structured_qwen122b_a10b/reasoning_json",
                   help="Subdir under <data_root>/outputs/<dataset>/ containing train/ and test/.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write inference JSONs, eval JSONs, and CSV. "
                        "Default: <data_root>/evaluate/<dataset>")

    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--train_json_dir", default=None)
    p.add_argument("--test_json_dir", default=None)
    p.add_argument("--raw_embed_dir", default=None,
                   help="Dir with raw embed JSONs for ALL test samples. "
                        "Default: <data_root>/data/raw_embeds/<dataset>/test")
    p.add_argument("--ig_json_dir", default=None,
                   help="Dir with IG attribution JSONs. Pass 'none' to skip GS.")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--projector_checkpoint", default=None,
                   help="Explicit .pt path; omit to auto-resolve latest matching checkpoint")
    p.add_argument("--checkpoint_path", default=None,
                   help="Alias for --projector_checkpoint")

    p.add_argument("--dataset", default="ucihar")
    p.add_argument("--data_version", default="structured")

    p.add_argument("--n_tokens",          type=int,   default=8)
    p.add_argument("--adapter_rank",      type=int,   default=128)
    p.add_argument("--adapter_num_heads", type=int,   default=8)
    p.add_argument("--adapter_dropout",   type=float, default=0.1)
    p.add_argument("--adapter_layers",    default="all")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--instruction",
                   default="Analyze the sensor embeddings, generate reasoning and explain the activity:")

    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--max_seq_len",  type=int,   default=768)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--val_split",    type=float, default=0.2)
    p.add_argument("--blindfold_every_n_epochs", type=int, default=1)
    p.add_argument("--activity_token_weight", type=float, default=1.0,
                   help="Upweight [ACTIVITY]: token positions in the LM loss (1.0 = no upweighting).")
    p.add_argument("--early_stopping_patience",  type=int,   default=3)
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-4)

    p.add_argument("--max_new_tokens",        type=int,   default=768)
    p.add_argument("--num_inference_samples", type=int,   default=-1)
    p.add_argument("--num_eval_samples",      type=int,   default=-1)
    p.add_argument("--inference_batch_size",  type=int,   default=256)
    p.add_argument("--eval_batch_size",       type=int,   default=4)
    p.add_argument("--temperature",           type=float, default=0.3)
    p.add_argument("--repetition_penalty",    type=float, default=1.1)
    p.add_argument("--use_structured_prefix", type=int,   default=1)
    p.add_argument("--json_key",              default="overall_reasoning")
    p.add_argument("--activity_label_key",    default="predicted_activity")

    p.add_argument("--constrained_decoding", type=int, default=0,
                   help="Force [ACTIVITY] tokens to match a valid class string (trie). "
                        "Set to 0 to restore unconstrained generation.")
    p.add_argument("--gs_top_k", type=int, default=3)
    p.add_argument("--skip_model_metrics", type=int, default=0)
    p.add_argument("--n_eval_runs", type=int, default=1)

    p.add_argument("--use_tensorboard",   type=int, default=0)
    p.add_argument("--use_wandb",         type=int, default=1)
    p.add_argument("--wandb_project",     default="trace-ts")
    p.add_argument("--log_every_n_steps", type=int, default=10)

    p.add_argument("--figure3_n_samples", type=int, default=50)
    p.add_argument("--figure3_output", default=None)

    args = p.parse_args()

    cfg = dict(CONFIG)
    cfg.update(vars(args))

    data_root = Path(cfg["data_root"])
    dataset   = cfg["dataset"].lower()

    if cfg.get("train_json_dir") is None:
        cfg["train_json_dir"] = data_root / "outputs" / dataset / cfg["reasoning_subdir"] / "train"
    if cfg.get("test_json_dir") is None:
        cfg["test_json_dir"]  = data_root / "outputs" / dataset / cfg["reasoning_subdir"] / "test"
    if cfg.get("raw_embed_dir") is None:
        cfg["raw_embed_dir"]  = data_root / "data" / "raw_embeds" / dataset / "test"
    if cfg.get("ig_json_dir") is None:
        cfg["ig_json_dir"]    = data_root / "data" / dataset
    elif str(cfg["ig_json_dir"]).lower() == "none":
        cfg["ig_json_dir"] = None
    if cfg.get("checkpoint_dir") is None:
        cfg["checkpoint_dir"] = data_root / "checkpoints" / dataset
    if cfg.get("output_dir") is None:
        cfg["output_dir"] = data_root / "evaluate" / dataset

    apply_dataset_config(cfg, cli_provided_keys=set())

    for flag in ("use_tensorboard", "use_wandb", "gradient_checkpointing",
                 "use_structured_prefix", "skip_model_metrics", "constrained_decoding"):
        if isinstance(cfg.get(flag), int):
            cfg[flag] = bool(cfg[flag])

    if cfg.get("checkpoint_path") and not cfg.get("projector_checkpoint"):
        cfg["projector_checkpoint"] = cfg["checkpoint_path"]

    for key in ("train_json_dir", "test_json_dir", "checkpoint_dir",
                 "raw_embed_dir", "output_dir"):
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key])
    if cfg.get("ig_json_dir") is not None:
        cfg["ig_json_dir"] = Path(cfg["ig_json_dir"])

    print("[config] Resolved paths:")
    for k in ("train_json_dir", "test_json_dir", "raw_embed_dir",
               "ig_json_dir", "checkpoint_dir", "output_dir"):
        print(f"  {k:20s}: {cfg.get(k)}")

    return cfg
