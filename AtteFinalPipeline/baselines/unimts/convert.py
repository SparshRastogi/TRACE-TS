from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
import argparse
import json
import os
import numpy as np
import scipy.io as sio

OUT_ROOT = "./UniMTS_data"
SRC_DIR = "./dataset"
DATASET_CONFIGS = {
    "pamap2": dict(
        src_path=f"{SRC_DIR}/pamap2_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=24,
        stride=12,
        sampling_rate=33,
        joint_list=[21, 7, 11],
        joint_blocks=[(5, 8, 8, 11), (22, 25, 25, 28), (39, 42, 42, 45)],
        class_names=[
            "rope jumping",
            "lying",
            "sitting",
            "standing",
            "walking",
            "running",
            "cycling",
            "nordic walking",
            "ascending stairs",
            "descending stairs",
            "vacuum cleaning",
            "ironing",
        ],
        num_class=12,
    ),
    "uschad": dict(
        src_path=f"{SRC_DIR}/uschad_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=400,
        stride=200,
        sampling_rate=100,
        joint_list=[5],
        joint_blocks=[(0, 3, 3, 6)],
        class_names=[label.lower() for label in USCHAD_CLASS_MAP],
        num_class=12,
    ),
    "opportunity": dict(
        src_path=f"{SRC_DIR}/opportunity_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=24,
        stride=12,
        sampling_rate=30,
        joint_list=[16, 20, 15, 10, 19],
        joint_blocks=[
            (0, 3, 3, 6),
            (9, 12, 12, 15),
            (18, 21, 21, 24),
            (27, 30, 30, 33),
            (36, 39, 39, 42),
        ],
        class_names=[
            "null",
            "open door 1",
            "open door 2",
            "close door 1",
            "close door 2",
            "open fridge",
            "close fridge",
            "open dishwasher",
            "close dishwasher",
            "open drawer 1",
            "close drawer 1",
            "open drawer 2",
            "close drawer 2",
            "open drawer 3",
            "close drawer 3",
            "clean table",
            "drink from cup",
            "toggle switch",
        ],
        num_class=18,
    ),
    "opportunity_nonull": dict(
        src_path=f"{SRC_DIR}/opportunity_nonull_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=24,
        stride=12,
        sampling_rate=30,
        joint_list=[16, 20, 15, 10, 19],
        joint_blocks=[
            (0, 3, 3, 6),
            (9, 12, 12, 15),
            (18, 21, 21, 24),
            (27, 30, 30, 33),
            (36, 39, 39, 42),
        ],
        class_names=[
            "open door 1",
            "open door 2",
            "close door 1",
            "close door 2",
            "open fridge",
            "close fridge",
            "open dishwasher",
            "close dishwasher",
            "open drawer 1",
            "close drawer 1",
            "open drawer 2",
            "close drawer 2",
            "open drawer 3",
            "close drawer 3",
            "clean table",
            "drink from cup",
            "toggle switch",
        ],
        num_class=17,
    ),
    "shoaib": dict(
        src_path=f"{SRC_DIR}/shoaib_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=100,
        stride=50,
        sampling_rate=50,
        joint_list=[1, 5, 21, 20, 0],
        joint_blocks=[
            (0, 3, 3, 6),
            (9, 12, 12, 15),
            (18, 21, 21, 24),
            (27, 30, 30, 33),
            (36, 39, 39, 42),
        ],
        class_names=[
            "walking",
            "standing",
            "jogging",
            "sitting",
            "biking",
            "walking upstairs",
            "walking downstairs",
        ],
        num_class=7,
    ),
    "mhealth": dict(
        src_path=f"{SRC_DIR}/mhealth_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=100,
        stride=50,
        sampling_rate=50,
        joint_list=[11, 3, 21],
        joint_blocks=[(0, 3, None, None), (6, 9, 9, 12), (15, 18, 18, 21)],
        class_names=[
            "null",
            "standing still",
            "sitting and relaxing",
            "lying down",
            "walking",
            "climbing stairs",
            "waist bends forward",
            "frontal elevation of arms",
            "knees bending",
            "cycling",
            "jogging",
            "running",
            "jump front and back",
        ],
        num_class=13,
    ),
    "mhealth_nonull": dict(
        src_path=f"{SRC_DIR}/mhealth_nonull_unimts.mat",
        src_kind="mat",
        prewindowed=False,
        window=100,
        stride=50,
        sampling_rate=50,
        joint_list=[11, 3, 21],
        joint_blocks=[(0, 3, None, None), (6, 9, 9, 12), (15, 18, 18, 21)],
        class_names=[
            "standing still",
            "sitting and relaxing",
            "lying down",
            "walking",
            "climbing stairs",
            "waist bends forward",
            "frontal elevation of arms",
            "knees bending",
            "cycling",
            "jogging",
            "running",
            "jump front and back",
        ],
        num_class=12,
    ),
    "capture24": dict(
        src_path=f"{SRC_DIR}/capture24_Willetts2018_unimts.npz",
        src_kind="npz",
        prewindowed=True,
        window=200,
        stride=1,
        sampling_rate=100,
        joint_list=[21],
        joint_blocks=[(0, 3, None, None)],
        class_names=[
            "sleep",
            "sit-stand",
            "walking",
            "bicycling",
            "mixed activity",
            "vehicle",
        ],
        num_class=6,
    ),
    "capture24_walmsley": dict(
        src_path=f"{SRC_DIR}/capture24_Walmsley2020_unimts.npz",
        src_kind="npz",
        prewindowed=True,
        window=200,
        stride=1,
        sampling_rate=100,
        joint_list=[21],
        joint_blocks=[(0, 3, None, None)],
        class_names=[
            "sleep",
            "sedentary",
            "light physical activity",
            "moderate to vigorous physical activity",
        ],
        num_class=4,
    ),
    "capture24_full": dict(
        src_path=f"{SRC_DIR}/capture24_Willetts2018_full_unimts.npz",
        src_kind="npz",
        prewindowed=True,
        window=200,
        stride=1,
        sampling_rate=100,
        joint_list=[21],
        joint_blocks=[(0, 3, None, None)],
        class_names=[
            "sleep",
            "sit-stand",
            "walking",
            "bicycling",
            "mixed activity",
            "vehicle",
        ],
        num_class=6,
    ),
    "ucihar": dict(
        src_path=f"{SRC_DIR}/ucihar_unimts.mat",
        src_kind="mat",
        prewindowed=True,
        window=128,
        stride=1,
        sampling_rate=50,
        joint_list=[0],
        joint_blocks=[(0, 3, 3, 6)],
        class_names=[
            "walking",
            "walking upstairs",
            "walking downstairs",
            "sitting",
            "standing",
            "laying",
        ],
        num_class=6,
    ),
}


def _load_mat_split(path, split):
    m = sio.loadmat(path)
    Xkey = {"train": "X_train", "val": "X_valid", "test": "X_test"}[split]
    ykey = {"train": "y_train", "val": "y_valid", "test": "y_test"}[split]
    if Xkey not in m or ykey not in m:
        return (None, None)
    X = np.asarray(m[Xkey])
    y = np.asarray(m[ykey]).squeeze().astype(np.int64)
    return (X, y)


def _load_npz_split(path, split):
    npz = np.load(path, allow_pickle=True)
    Xkey = {"train": "X_train", "val": "X_valid", "test": "X_test"}[split]
    ykey = {"train": "y_train", "val": "y_valid", "test": "y_test"}[split]
    if Xkey not in npz.files or ykey not in npz.files:
        return (None, None)
    return (npz[Xkey], np.asarray(npz[ykey]).squeeze().astype(np.int64))


def sliding_window_last(x_2d, y_1d, window, stride):
    T, D = x_2d.shape
    data, target = ([], [])
    start = 0
    while start + window < T:
        end = start + window
        data.append(x_2d[start:end])
        target.append(y_1d[end - 1])
        start += stride
    if not data:
        return (np.empty((0, window, D), np.float32), np.empty((0,), np.int64))
    return (np.asarray(data, dtype=np.float32), np.asarray(target, dtype=np.int64))


def build_unimts_X(X_NTD, cfg):
    N, T, D = X_NTD.shape
    n_joints = len(cfg["joint_list"])
    out = np.zeros((N, T, 6 * n_joints), dtype=np.float32)
    for j, (a_lo, a_hi, g_lo, g_hi) in enumerate(cfg["joint_blocks"]):
        out[:, :, 6 * j + 0 : 6 * j + 3] = X_NTD[:, :, a_lo:a_hi].astype(np.float32)
        if g_lo is not None and g_hi is not None:
            out[:, :, 6 * j + 3 : 6 * j + 6] = X_NTD[:, :, g_lo:g_hi].astype(np.float32)
    return out


def process(name, cfg):
    print(f"\n=== {name} ===")
    print(f"  source       : {cfg['src_path']}")
    print(f"  joints       : {cfg['joint_list']}")
    print(
        f"  window/stride: {cfg['window']}/{cfg['stride']}  @ {cfg['sampling_rate']} Hz"
    )
    if not os.path.exists(cfg["src_path"]):
        print(
            f"  [skip] {cfg['src_path']} does not exist. Run the matching prepare_*_unimts.py first."
        )
        return
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    loader = _load_mat_split if cfg["src_kind"] == "mat" else _load_npz_split
    have_test = False
    for split in ("train", "val", "test"):
        X, y = loader(cfg["src_path"], split)
        if X is None:
            continue
        if cfg["prewindowed"]:
            assert X.ndim == 3, f"Expected pre-windowed 3-D, got {X.shape}"
            X_NTD, y_N = (X.astype(np.float32), y)
        else:
            assert X.ndim == 2, f"Expected raw 2-D, got {X.shape}"
            X_NTD, y_N = sliding_window_last(
                X.astype(np.float32), y, cfg["window"], cfg["stride"]
            )
            print(f"  {split:5s}: raw {X.shape} → windowed {X_NTD.shape}")
        X_unimts = build_unimts_X(X_NTD, cfg)
        np.save(os.path.join(out_dir, f"X_{split}.npy"), X_unimts)
        np.save(os.path.join(out_dir, f"y_{split}.npy"), y_N)
        if split == "test":
            have_test = True
        print(
            f"  {split:5s}: UniMTS X {X_unimts.shape}, y unique {sorted(np.unique(y_N).tolist())[:8]}..."
        )
    if not have_test and os.path.exists(os.path.join(out_dir, "X_val.npy")):
        Xv = np.load(os.path.join(out_dir, "X_val.npy"))
        yv = np.load(os.path.join(out_dir, "y_val.npy"))
        np.save(os.path.join(out_dir, "X_test.npy"), Xv)
        np.save(os.path.join(out_dir, "y_test.npy"), yv)
        print(f"  (no test split — copied val → test)")
    label_dict = {str(i): [cfg["class_names"][i]] for i in range(cfg["num_class"])}
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump({"label_dictionary": label_dict}, f, indent=2)
    print(f"  wrote {json_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", required=True, choices=list(DATASET_CONFIGS.keys()) + ["all"]
    )
    args = p.parse_args()
    targets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        process(name, DATASET_CONFIGS[name])


if __name__ == "__main__":
    main()
