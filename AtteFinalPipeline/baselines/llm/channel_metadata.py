from __future__ import annotations
from typing import Dict, List

_OPP_CHANNELS: List[str] = (
    [f"acc_RUA_{ax}" for ax in ("x", "y", "z")]
    + [f"acc_RLA_{ax}" for ax in ("x", "y", "z")]
    + [f"acc_LUA_{ax}" for ax in ("x", "y", "z")]
    + [f"acc_LLA_{ax}" for ax in ("x", "y", "z")]
    + [f"acc_BACK_{ax}" for ax in ("x", "y", "z")]
    + [
        f"imu_BACK_{ch}"
        for ch in (
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "quat1",
            "quat2",
            "quat3",
            "quat4",
        )
    ]
    + [
        f"imu_RUA_{ch}"
        for ch in (
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "quat1",
            "quat2",
            "quat3",
            "quat4",
        )
    ]
    + [
        f"imu_RLA_{ch}"
        for ch in (
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "quat1",
            "quat2",
            "quat3",
            "quat4",
        )
    ]
    + [
        f"imu_LUA_{ch}"
        for ch in (
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "quat1",
            "quat2",
            "quat3",
            "quat4",
        )
    ]
    + [
        f"imu_LLA_{ch}"
        for ch in (
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "quat1",
            "quat2",
            "quat3",
            "quat4",
        )
    ]
)
_OPP_CHANNELS = _OPP_CHANNELS[:79]
_PAMAP2_IMU_BLOCK = [
    "temp",
    "acc16g_x",
    "acc16g_y",
    "acc16g_z",
    "acc6g_x",
    "acc6g_y",
    "acc6g_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "orient_q1",
    "orient_q2",
    "orient_q3",
    "orient_q4",
]
_PAMAP2_CHANNELS: List[str] = (
    ["heart_rate"]
    + [f"hand_{n}" for n in _PAMAP2_IMU_BLOCK]
    + [f"chest_{n}" for n in _PAMAP2_IMU_BLOCK]
    + [f"ankle_{n}" for n in _PAMAP2_IMU_BLOCK]
)
_USCHAD_CHANNELS: List[str] = [
    "hip_acc_x",
    "hip_acc_y",
    "hip_acc_z",
    "hip_gyro_x",
    "hip_gyro_y",
    "hip_gyro_z",
]
_UCIHAR_CHANNELS: List[str] = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]
_MHEALTH_CHANNELS: List[str] = [
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "chest_ecg_lead1",
    "chest_ecg_lead2",
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "rwrist_acc_x",
    "rwrist_acc_y",
    "rwrist_acc_z",
    "rwrist_gyro_x",
    "rwrist_gyro_y",
    "rwrist_gyro_z",
    "rwrist_mag_x",
    "rwrist_mag_y",
    "rwrist_mag_z",
]
_SHOAIB_POSITIONS = ["lpocket", "rpocket", "wrist", "uarm", "belt"]
_SHOAIB_PER_POS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "linacc_x",
    "linacc_y",
    "linacc_z",
]
_SHOAIB_CHANNELS: List[str] = [
    f"{pos}_{ch}" for pos in _SHOAIB_POSITIONS for ch in _SHOAIB_PER_POS
]
_CAPTURE24_CHANNELS: List[str] = ["wrist_acc_x", "wrist_acc_y", "wrist_acc_z"]
CHANNEL_METADATA: Dict[str, dict] = {
    "opportunity": {
        "description": "Opportunity dataset: 18-class kitchen/morning-routine gesture recognition. 79 channels of body-worn IMU and accelerometer data from sensors on the upper arms, lower arms, and back. Sampling rate ~30 Hz. Class 0 is 'Null' (no labelled gesture).",
        "sample_rate_hz": 30,
        "channels": _OPP_CHANNELS,
    },
    "pamap2": {
        "description": "PAMAP2 Physical Activity Monitoring dataset: 12 classes of daily and sport activities. 52 channels = 1 heart-rate + three IMU blocks (hand, chest, ankle) of 17 channels each: temperature, two accelerometers (±16g and ±6g), gyroscope, magnetometer, and a 4-component orientation quaternion. Downsampled to ~33 Hz.",
        "sample_rate_hz": 33,
        "channels": _PAMAP2_CHANNELS,
    },
    "uschad": {
        "description": "USC-HAD: 12-class daily-activity dataset. 6 channels from a single hip-mounted IMU: 3-axis accelerometer + 3-axis gyroscope. Sampling rate 200 Hz; windows are 400 samples (2.0 s). For LLM input the timesteps are decimated 4x to 100 samples to fit in a reasonable token budget — this is documented in the prompt.",
        "sample_rate_hz": 200,
        "channels": _USCHAD_CHANNELS,
    },
    "ucihar": {
        "description": "UCI HAR: 6-class smartphone activity recognition. 9 channels = body acceleration (gravity removed) x/y/z, body angular velocity x/y/z, and total acceleration (with gravity) x/y/z, all from a waist-mounted smartphone. Pre-windowed at 128 samples (2.56 s) at 50 Hz.",
        "sample_rate_hz": 50,
        "channels": _UCIHAR_CHANNELS,
    },
    "mhealth": {
        "description": "MHEALTH: 13-class body sensor network dataset (1 null + 12 physical activities). 23 channels: chest accelerometer + 2-lead ECG, left-ankle accelerometer + gyroscope + magnetometer, right-wrist accelerometer + gyroscope + magnetometer. Sampling rate 50 Hz. Class 0 is 'Null'.",
        "sample_rate_hz": 50,
        "channels": _MHEALTH_CHANNELS,
    },
    "mhealth_nonull": {
        "description": "MHEALTH (12 classes, null removed): same as 'mhealth' but the null class is dropped and labels remapped 1-12 → 0-11.",
        "sample_rate_hz": 50,
        "channels": _MHEALTH_CHANNELS,
    },
    "shoaib": {
        "description": "Shoaib Sensors Activity Recognition: 7-class smartphone activity dataset with simultaneous data from 5 body positions (left pocket, right pocket, wrist, upper arm, belt). Each position contributes 9 channels: 3-axis accelerometer, 3-axis gyroscope, 3-axis linear acceleration. 45 channels total. Sampling rate 50 Hz.",
        "sample_rate_hz": 50,
        "channels": _SHOAIB_CHANNELS,
    },
    "capture24": {
        "description": "Capture-24 (Willetts2018 6-class subset): wrist-worn 3-axis accelerometer free-living dataset. 6 classes: Sleep, Sit-stand, Walking, Bicycling, Mixed, Vehicle. Sampling rate 100 Hz; windows of 200 samples (2.0 s).",
        "sample_rate_hz": 100,
        "channels": _CAPTURE24_CHANNELS,
    },
    "capture24_walmsley": {
        "description": "Capture-24 (Walmsley2020 4-class subset): wrist-worn 3-axis accelerometer free-living dataset. 4 classes: Sleep, Sedentary, Light, Moderate-Vigorous. Sampling rate 100 Hz; windows of 200 samples (2.0 s).",
        "sample_rate_hz": 100,
        "channels": _CAPTURE24_CHANNELS,
    },
}


def get_channel_names(dataset: str, expected_dim: int) -> List[str]:
    meta = CHANNEL_METADATA.get(dataset)
    if meta is None:
        return [f"ch_{i}" for i in range(expected_dim)]
    names = list(meta["channels"])
    if len(names) == expected_dim:
        return names
    print(
        f"[warn] llm_channel_metadata: registered {len(names)} channels for '{dataset}' but data has {expected_dim}. Falling back to generic channel names. Update CHANNEL_METADATA in llm_channel_metadata.py to silence this."
    )
    return [f"ch_{i}" for i in range(expected_dim)]


def get_dataset_description(dataset: str) -> str:
    return CHANNEL_METADATA.get(dataset, {}).get(
        "description", f"{dataset} HAR dataset."
    )


def get_sample_rate(dataset: str) -> int:
    return int(CHANNEL_METADATA.get(dataset, {}).get("sample_rate_hz", 50))
