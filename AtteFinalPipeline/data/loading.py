import os
import sys
import numpy as np
from AtteFinalPipeline.expert.settings import get_args as _settings_get_args


def _get_dataset_config(dataset_name):
    args, config_dataset, config_model = _settings_get_args(["--dataset", dataset_name])
    class_map = args.class_map
    return {
        "class_map": class_map,
        "input_dim": args.input_dim,
        "num_class": args.num_class,
        "window": args.window,
        "stride": args.stride,
        "path_data": args.path_data,
        "path_raw": args.path_raw,
        "path_processed": args.path_processed,
        "config_model": config_model,
    }


def _is_capture24_variant(dataset_name):
    return dataset_name.startswith("capture24")


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
                return (ds["data"].astype(np.float32), ds["target"].astype(np.int64))
        mat_path = (
            ds_config["path_data"] if ds_config else None
        ) or "./dataset/uschad.mat"
        if not os.path.exists(mat_path):
            print(f"ERROR: USC-HAD .mat not found at {mat_path}.")
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
        x_key, y_key = _key_map[split]
        return (
            contents[x_key].astype(np.float32),
            contents[y_key].reshape(-1).astype(np.int64),
        )
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
    if dataset_name == "ucihar":
        base = data_path or (ds_config["path_processed"] if ds_config else None)
        if ds_config and (
            not base
            or (
                base == ds_config["path_processed"]
                and not os.path.isfile(os.path.join(base, f"X_{split}.npy"))
                and not os.path.isfile(os.path.join(base, f"{split}.npz"))
            )
        ):
            base = ds_config["path_data"]
        x_path = os.path.join(base, f"X_{split}.npy")
        y_path = os.path.join(base, f"y_{split}.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            data = np.load(x_path).astype(np.float32)
            target = np.load(y_path).astype(np.longlong)
            if np.min(target) > 0:
                target = target - 1
            return (data, target)
        npz_path = os.path.join(base, f"{split}.npz")
        if os.path.exists(npz_path):
            ds = np.load(npz_path)
            data = ds["data"].astype(np.float32)
            target = ds["target"].astype(np.longlong)
            if np.min(target) > 0:
                target = target - 1
            return (data, target)
        print(f"ERROR: No data files found for ucihar split='{split}' in {base}")
        sys.exit(1)
    if dataset_name == "shoaib":
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
        mat_path = (
            ds_config["path_data"] if ds_config else None
        ) or "./dataset/shoaib.mat"
        if not os.path.exists(mat_path):
            print(
                f"ERROR: Shoaib processed .npz not found in '{processed_base}' and .mat fallback not found at '{mat_path}'.\n       Run prepare_shoaib.py then preprocess.py first."
            )
            sys.exit(1)
        print(
            f"  WARNING: Falling back to sample-level .mat at {mat_path} — the model expects (window={ds_config['window']}, D=45) windows. Run preprocess.py to generate the processed .npz files."
        )
        import scipy.io as sio

        contents = sio.loadmat(mat_path)
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
    if dataset_name in ("mhealth", "mhealth_nonull"):
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
        default_mat = (
            "./dataset/mhealth_nonull.mat"
            if dataset_name == "mhealth_nonull"
            else "./dataset/mhealth.mat"
        )
        mat_path = (ds_config["path_data"] if ds_config else None) or default_mat
        if not os.path.exists(mat_path):
            print(
                f"ERROR: MHEALTH processed .npz not found in '{processed_base}' and .mat fallback not found at '{mat_path}'.\n       Run prepare_mhealth.py then preprocess.py first."
            )
            sys.exit(1)
        print(
            f"  WARNING: Falling back to sample-level .mat at {mat_path} — the model expects (window={ds_config['window']}, D=23) windows. Run preprocess.py to generate the processed .npz files."
        )
        import scipy.io as sio

        contents = sio.loadmat(mat_path)
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
