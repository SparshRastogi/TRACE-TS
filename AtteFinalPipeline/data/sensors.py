import os

SENSORS_6CH = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
GROUPS_6CH = [
    ("Accelerometer Data", [0, 1, 2], ["Acc_X", "Acc_Y", "Acc_Z"]),
    ("Gyroscope Data", [3, 4, 5], ["Gyro_X", "Gyro_Y", "Gyro_Z"]),
]
SENSORS_9CH = [
    "Total_Acc_X",
    "Total_Acc_Y",
    "Total_Acc_Z",
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
]
GROUPS_9CH = [
    (
        "Total Acceleration Data",
        [0, 1, 2],
        ["Total_Acc_X", "Total_Acc_Y", "Total_Acc_Z"],
    ),
    ("Accelerometer Data", [3, 4, 5], ["Acc_X", "Acc_Y", "Acc_Z"]),
    ("Gyroscope Data", [6, 7, 8], ["Gyro_X", "Gyro_Y", "Gyro_Z"]),
]
SENSORS_HOSPITAL = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
GROUPS_HOSPITAL = [
    ("Accelerometer Data", [0, 1, 2], ["Acc_X", "Acc_Y", "Acc_Z"]),
    ("Gyroscope Data", [3, 4, 5], ["Gyro_X", "Gyro_Y", "Gyro_Z"]),
]
SENSORS_USCHAD = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
GROUPS_USCHAD = [
    ("Accelerometer (X / Y / Z)", [0, 1, 2], ["Acc_X", "Acc_Y", "Acc_Z"]),
    ("Gyroscope (X / Y / Z)", [3, 4, 5], ["Gyro_X", "Gyro_Y", "Gyro_Z"]),
]
SENSORS_CAPTURE24 = ["Acc_X", "Acc_Y", "Acc_Z"]
GROUPS_CAPTURE24 = [
    (
        "Accelerometer (X / Y / Z) — wrist, 100 Hz",
        [0, 1, 2],
        ["Acc_X", "Acc_Y", "Acc_Z"],
    )
]
SENSORS_MHEALTH = [
    "Chest_Acc_X",
    "Chest_Acc_Y",
    "Chest_Acc_Z",
    "Chest_ECG_I",
    "Chest_ECG_II",
    "Chest_ECG_III",
    "LAnkle_Acc_X",
    "LAnkle_Acc_Y",
    "LAnkle_Acc_Z",
    "LAnkle_Gyro_X",
    "LAnkle_Gyro_Y",
    "LAnkle_Gyro_Z",
    "LAnkle_Mag_X",
    "LAnkle_Mag_Y",
    "LAnkle_Mag_Z",
    "RWrist_Acc_X",
    "RWrist_Acc_Y",
    "RWrist_Acc_Z",
    "RWrist_Gyro_X",
    "RWrist_Gyro_Y",
    "RWrist_Gyro_Z",
    "RWrist_Mag_X",
    "RWrist_Mag_Y",
]
assert len(SENSORS_MHEALTH) == 23
GROUPS_MHEALTH = [
    ("Chest — Accelerometer (X / Y / Z)", [0, 1, 2], SENSORS_MHEALTH[0:3]),
    ("Chest — ECG (Leads I / II / III)", [3, 4, 5], SENSORS_MHEALTH[3:6]),
    (
        "Left Ankle — Acc / Gyro / Mag",
        [6, 7, 8, 9, 10, 11, 12, 13, 14],
        SENSORS_MHEALTH[6:15],
    ),
    (
        "Right Wrist — Acc / Gyro / Mag",
        [15, 16, 17, 18, 19, 20, 21, 22],
        SENSORS_MHEALTH[15:23],
    ),
]
_SHOAIB_POSITIONS = ["LeftPocket", "RightPocket", "Wrist", "UpperArm", "Belt"]
_SHOAIB_CHANS_PER_POS = [
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
    "LinAcc_X",
    "LinAcc_Y",
    "LinAcc_Z",
]


def _build_shoaib_names_and_groups():
    names = []
    groups = []
    for p_idx, pos in enumerate(_SHOAIB_POSITIONS):
        block_start = p_idx * len(_SHOAIB_CHANS_PER_POS)
        block_names = [f"{pos}_{c}" for c in _SHOAIB_CHANS_PER_POS]
        names.extend(block_names)
        block_indices = list(
            range(block_start, block_start + len(_SHOAIB_CHANS_PER_POS))
        )
        groups.append((f"{pos} (Acc / Gyro / LinAcc)", block_indices, block_names))
    assert len(names) == 45
    return (names, groups)


SENSORS_SHOAIB, GROUPS_SHOAIB = _build_shoaib_names_and_groups()
_IMU_FIELDS = [
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
    "orient_0",
    "orient_1",
    "orient_2",
    "orient_3",
]
_PAMAP2_FALLBACK_NAMES = (
    ["heart_rate"]
    + [f"hand_{f}" for f in _IMU_FIELDS]
    + [f"chest_{f}" for f in _IMU_FIELDS]
    + [f"ankle_{f}" for f in _IMU_FIELDS]
)
assert len(_PAMAP2_FALLBACK_NAMES) == 52
SENSORS_PAMAP2 = list(_PAMAP2_FALLBACK_NAMES)
GROUPS_PAMAP2 = [
    ("Heart rate + hand IMU", list(range(0, 18)), SENSORS_PAMAP2[0:18]),
    ("Chest IMU", list(range(18, 35)), SENSORS_PAMAP2[18:35]),
    ("Ankle IMU", list(range(35, 52)), SENSORS_PAMAP2[35:52]),
]


def _channel_names_from_mat(mat_path):
    import scipy.io as sio

    if not mat_path or not os.path.exists(mat_path):
        return None
    try:
        c = sio.loadmat(mat_path)
        if "channel_names" not in c:
            print(
                f"  [mat] No 'channel_names' key in {mat_path} — using fallback names"
            )
            return None
        raw = c["channel_names"].flatten()
        names = []
        for item in raw:
            if hasattr(item, "__len__") and (not isinstance(item, str)):
                names.append(str(item[0]) if len(item) > 0 else "?")
            else:
                names.append(str(item))
        print(f"  [mat] Read {len(names)} channel_names from {mat_path}")
        return names
    except Exception as e:
        print(f"  [mat] Could not read {mat_path}: {e} — using fallback names")
        return None


_OPP_RAW_COL_NAMES = {
    1: "Acc_RKN^_accX",
    2: "Acc_RKN^_accY",
    3: "Acc_RKN^_accZ",
    4: "Acc_HIP_accX",
    5: "Acc_HIP_accY",
    6: "Acc_HIP_accZ",
    7: "Acc_LUA^_accX",
    8: "Acc_LUA^_accY",
    9: "Acc_LUA^_accZ",
    10: "Acc_RUA_accX",
    11: "Acc_RUA_accY",
    12: "Acc_RUA_accZ",
    13: "Acc_LH_accX",
    14: "Acc_LH_accY",
    15: "Acc_LH_accZ",
    16: "IMU_BACK_accX",
    17: "IMU_BACK_accY",
    18: "IMU_BACK_accZ",
    19: "IMU_BACK_gyrX",
    20: "IMU_BACK_gyrY",
    21: "IMU_BACK_gyrZ",
    22: "IMU_BACK_magX",
    23: "IMU_BACK_magY",
    24: "IMU_BACK_magZ",
    25: "IMU_BACK_Quat1",
    26: "IMU_BACK_Quat2",
    27: "IMU_BACK_Quat3",
    28: "IMU_BACK_Quat4",
    29: "IMU_RUA_accX",
    30: "IMU_RUA_accY",
    31: "IMU_RUA_accZ",
    32: "IMU_RUA_gyrX",
    33: "IMU_RUA_gyrY",
    34: "IMU_RUA_gyrZ",
    35: "IMU_RUA_magX",
    36: "IMU_RUA_magY",
    37: "IMU_RUA_magZ",
    38: "IMU_RUA_Quat1",
    39: "IMU_RUA_Quat2",
    40: "IMU_RUA_Quat3",
    41: "IMU_RUA_Quat4",
    42: "IMU_RLA_accX",
    43: "IMU_RLA_accY",
    44: "IMU_RLA_accZ",
    45: "IMU_RLA_gyrX",
    46: "IMU_RLA_gyrY",
    47: "IMU_RLA_gyrZ",
    48: "IMU_RLA_magX",
    49: "IMU_RLA_magY",
    50: "IMU_RLA_magZ",
    51: "IMU_RLA_Quat1",
    52: "IMU_RLA_Quat2",
    53: "IMU_RLA_Quat3",
    54: "IMU_RLA_Quat4",
    55: "IMU_LUA_accX",
    56: "IMU_LUA_accY",
    57: "IMU_LUA_accZ",
    58: "IMU_LUA_gyrX",
    59: "IMU_LUA_gyrY",
    60: "IMU_LUA_gyrZ",
    61: "IMU_LUA_magX",
    62: "IMU_LUA_magY",
    63: "IMU_LUA_magZ",
    64: "IMU_LUA_Quat1",
    65: "IMU_LUA_Quat2",
    66: "IMU_LUA_Quat3",
    67: "IMU_LUA_Quat4",
    68: "IMU_LLA_accX",
    69: "IMU_LLA_accY",
    70: "IMU_LLA_accZ",
    71: "IMU_LLA_gyrX",
    72: "IMU_LLA_gyrY",
    73: "IMU_LLA_gyrZ",
    74: "IMU_LLA_magX",
    75: "IMU_LLA_magY",
    76: "IMU_LLA_magZ",
    77: "IMU_LLA_Quat1",
    78: "IMU_LLA_Quat2",
    79: "IMU_LLA_Quat3",
    80: "IMU_LLA_Quat4",
    81: "Lshoe_accX",
    82: "Lshoe_accY",
    83: "Lshoe_accZ",
    84: "Lshoe_gyrX",
    85: "Lshoe_gyrY",
    86: "Lshoe_gyrZ",
    87: "Lshoe_magX",
    88: "Lshoe_magY",
    89: "Lshoe_magZ",
    90: "Lshoe_Quat1",
    91: "Lshoe_Quat2",
    92: "Lshoe_Quat3",
    93: "Lshoe_Quat4",
    94: "Lshoe_Euler_yaw",
    95: "Lshoe_Euler_pitch",
    96: "Lshoe_Euler_roll",
    97: "Lshoe_Nav_accX",
    98: "Lshoe_Nav_accY",
    99: "Lshoe_Nav_accZ",
    100: "Lshoe_Compass",
    101: "Lshoe_zutc",
    102: "Rshoe_accX",
    103: "Rshoe_accY",
    104: "Rshoe_accZ",
    105: "Rshoe_gyrX",
    106: "Rshoe_gyrY",
    107: "Rshoe_gyrZ",
    108: "Rshoe_magX",
    109: "Rshoe_magY",
    110: "Rshoe_magZ",
    111: "Rshoe_Quat1",
    112: "Rshoe_Quat2",
    113: "Rshoe_Quat3",
    114: "Rshoe_Quat4",
    115: "Rshoe_Euler_yaw",
    116: "Rshoe_Euler_pitch",
    117: "Rshoe_Euler_roll",
    118: "Rshoe_Nav_accX",
    119: "Rshoe_Nav_accY",
    120: "Rshoe_Nav_accZ",
    121: "Rshoe_Compass",
    122: "Rshoe_zutc",
}


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
    names = [
        _OPP_RAW_COL_NAMES.get(raw_col, f"Col_{raw_col}")
        for raw_col in feature_raw_cols
    ]
    assert len(names) == 79
    return (names, feature_raw_cols)


_OPPORTUNITY_79_NAMES, _OPPORTUNITY_79_RAW_COLS = _build_opportunity_79_names()


def _build_opportunity_79_groups(names):
    groups_def = [
        ("Accelerometers (RKN/HIP/LUA/RUA/LH)", lambda n: n.startswith("Acc_")),
        ("IMU BACK", lambda n: n.startswith("IMU_BACK")),
        ("IMU RUA", lambda n: n.startswith("IMU_RUA")),
        ("IMU RLA", lambda n: n.startswith("IMU_RLA")),
        ("IMU LUA", lambda n: n.startswith("IMU_LUA")),
        ("IMU LLA", lambda n: n.startswith("IMU_LLA")),
        ("IMU L-Shoe", lambda n: n.startswith("Lshoe_")),
        ("IMU R-Shoe", lambda n: n.startswith("Rshoe_")),
    ]
    assigned = [False] * len(names)
    result = []
    for title, pred in groups_def:
        idxs = [i for i, n in enumerate(names) if pred(n) and (not assigned[i])]
        if idxs:
            for i in idxs:
                assigned[i] = True
            result.append((title, idxs, [names[i] for i in idxs]))
    remaining = [i for i, a in enumerate(assigned) if not a]
    if remaining:
        result.append(("Other", remaining, [names[i] for i in remaining]))
    return result


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
    assert len(names) == 113
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
            return None
        return names
    except Exception as e:
        print(f"  [opportunity] Could not read {col_names_path}: {e}")
        return None


def get_sensor_config(
    n_channels, dataset_name="ucihar", mat_path=None, opportunity_col_names_path=None
):
    ds_key = dataset_name
    if dataset_name.startswith("capture24"):
        ds_key = "capture24"
    if ds_key == "ucihar":
        if n_channels == 9:
            return (SENSORS_9CH, GROUPS_9CH)
        elif n_channels == 6:
            return (SENSORS_6CH, GROUPS_6CH)
    elif ds_key == "hospital":
        if n_channels == 6:
            return (SENSORS_HOSPITAL, GROUPS_HOSPITAL)
    elif ds_key == "uschad":
        if n_channels == 6:
            return (list(SENSORS_USCHAD), list(GROUPS_USCHAD))
        names = [f"Ch_{i}" for i in range(n_channels)]
        return (names, [("All Channels", list(range(n_channels)), names)])
    elif ds_key == "capture24":
        if n_channels == 3:
            return (list(SENSORS_CAPTURE24), list(GROUPS_CAPTURE24))
        names = [f"Ch_{i}" for i in range(n_channels)]
        return (names, [("All Channels", list(range(n_channels)), names)])
    elif ds_key == "shoaib":
        if n_channels == 45:
            return (list(SENSORS_SHOAIB), list(GROUPS_SHOAIB))
        names = [f"Ch_{i}" for i in range(n_channels)]
        groups = []
        for g_start in range(0, n_channels, 9):
            g_end = min(g_start + 9, n_channels)
            indices = list(range(g_start, g_end))
            groups.append(
                (f"Channels {g_start}–{g_end - 1}", indices, names[g_start:g_end])
            )
        return (names, groups)
    elif ds_key in ("mhealth", "mhealth_nonull"):
        if n_channels == 23:
            return (list(SENSORS_MHEALTH), list(GROUPS_MHEALTH))
        names = [f"Ch_{i}" for i in range(n_channels)]
        groups = []
        for g_start in range(0, n_channels, 8):
            g_end = min(g_start + 8, n_channels)
            indices = list(range(g_start, g_end))
            groups.append(
                (f"Channels {g_start}-{g_end - 1}", indices, names[g_start:g_end])
            )
        return (names, groups)
    elif ds_key == "pamap2":
        mat_names = _channel_names_from_mat(mat_path)
        sensor_names = (
            mat_names
            if mat_names and len(mat_names) == 52
            else list(_PAMAP2_FALLBACK_NAMES)
        )
        if n_channels == 52:
            groups = [
                ("Heart rate + hand IMU", list(range(0, 18)), sensor_names[0:18]),
                ("Chest IMU", list(range(18, 35)), sensor_names[18:35]),
                ("Ankle IMU", list(range(35, 52)), sensor_names[35:52]),
            ]
            return (sensor_names, groups)
    elif ds_key == "opportunity":
        file_names = _channel_names_from_opportunity_file(opportunity_col_names_path)
        if file_names and len(file_names) == n_channels:
            sensor_names = file_names
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
        if n_channels == 79:
            return (list(_OPPORTUNITY_79_NAMES), _OPPORTUNITY_79_GROUPS)
        if n_channels == 113:
            return (_OPPORTUNITY_113_NAMES, _OPPORTUNITY_113_GROUPS)
        sensor_names = [f"Ch_{i}" for i in range(n_channels)]
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
    elif ds_key == "skoda":
        names = [f"Ch_{i}" for i in range(n_channels)]
        groups = []
        for g_start in range(0, n_channels, 10):
            g_end = min(g_start + 10, n_channels)
            groups.append(
                (
                    f"Channels {g_start}-{g_end - 1}",
                    list(range(g_start, g_end)),
                    names[g_start:g_end],
                )
            )
        return (names, groups)
    names = [f"Ch_{i}" for i in range(n_channels)]
    return (names, [("All Channels", list(range(n_channels)), names)])


def get_null_class_indices(class_map, dataset_name):
    if dataset_name not in ("opportunity", "skoda", "mhealth"):
        return set()
    NULL_NAMES = {"null", "none", "background", "other", "unknown", "0"}
    null_indices = set()
    for idx, name in enumerate(class_map):
        if name.strip().lower() in NULL_NAMES or name.strip() == "0":
            null_indices.add(idx)
    null_indices.add(0)
    return null_indices
