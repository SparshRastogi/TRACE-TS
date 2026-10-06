import sys
import os
import scipy.io as sio
import numpy as np
from AtteFinalPipeline.expert.utils.utils import Logger, paint
from AtteFinalPipeline.expert.utils.plot import plot_pie
from AtteFinalPipeline.expert.settings import get_args

__all__ = ["preprocess_pipeline"]


def load_mat(path_data, path_raw, class_map, dataset, filter_null_at_load=False):
    print(f"[*] Reading data files from {path_data}")
    if "Sleep" in class_map and len(class_map) in (4, 6):
        n_classes = len(class_map)
        schema_tag = (
            "Walmsley2020 (4-class)" if n_classes == 4 else "Willetts2018 (6-class)"
        )
        contents = np.load(path_data)
        x_train = contents["X_train"].astype(np.float32)
        y_train = contents["y_train"].reshape(-1).astype(np.int64)
        x_val = contents["X_valid"].astype(np.float32)
        y_val = contents["y_valid"].reshape(-1).astype(np.int64)
        x_test = contents["X_test"].astype(np.float32)
        y_test = contents["y_test"].reshape(-1).astype(np.int64)
        C = x_train.shape[-1]
        per_ch_mean = x_train.reshape(-1, C).mean(axis=0)
        per_ch_std = x_train.reshape(-1, C).std(axis=0)
        print(
            paint(
                f"[Capture-24 {schema_tag}] .npz stats — per-channel mean={per_ch_mean.tolist()}, std={per_ch_std.tolist()} (should be ~0, ~1)",
                "blue",
            )
        )
    else:
        contents = sio.loadmat(path_data)
        if dataset in ("opportunity", "opportunity_nonull"):
            x_train = contents["trainingData"].astype(np.float32).T
            y_train = contents["trainingLabels"].reshape(-1).astype(np.int64) - 1
            x_val = contents["valData"].astype(np.float32).T
            y_val = contents["valLabels"].reshape(-1).astype(np.int64) - 1
            x_test = contents["testingData"].astype(np.float32).T
            y_test = contents["testingLabels"].reshape(-1).astype(np.int64) - 1
            if filter_null_at_load:
                print(
                    paint(
                        "[Opportunity-nonull] Dropping label==0 (Null) rows BEFORE normalisation+windowing ...",
                        "blue",
                    )
                )

                def _drop_null(x, y, split_tag):
                    if y.size == 0:
                        return (x, y)
                    keep = y != 0
                    n_before = y.size
                    n_after = int(keep.sum())
                    x_f = x[keep]
                    y_f = (y[keep] - 1).astype(np.int64)
                    pct = 100.0 * (n_before - n_after) / max(n_before, 1)
                    print(
                        paint(
                            f"  [drop_null] {split_tag:<5s}: {n_before:>8d} → {n_after:>8d}   (dropped {n_before - n_after:>8d} null rows, {pct:.2f}%)",
                            "blue",
                        )
                    )
                    return (x_f, y_f)

                x_train, y_train = _drop_null(x_train, y_train, "train")
                x_val, y_val = _drop_null(x_val, y_val, "val")
                x_test, y_test = _drop_null(x_test, y_test, "test")
            mean_train = np.mean(x_train, axis=0)
            std_train = np.std(x_train, axis=0)
            std_train[std_train == 0] = 1.0
            x_train = (x_train - mean_train) / std_train
            x_val = (x_val - mean_train) / std_train
            x_test = (x_test - mean_train) / std_train
        elif len(class_map) == 12 and "Walking Forward" in class_map:
            x_train = contents["X_train"].astype(np.float32)
            y_train = contents["y_train"].reshape(-1).astype(np.int64)
            x_val = contents["X_valid"].astype(np.float32)
            y_val = contents["y_valid"].reshape(-1).astype(np.int64)
            x_test = contents["X_test"].astype(np.float32)
            y_test = contents["y_test"].reshape(-1).astype(np.int64)
            train_mean = np.mean(x_train)
            train_std = np.std(x_train)
            print(
                paint(
                    f"[USC-HAD] .mat stats — mean={train_mean:.4f}, std={train_std:.4f} (should be ~0, ~1)",
                    "blue",
                )
            )
        elif dataset in ("mhealth", "mhealth_nonull"):
            x_train = contents["X_train"].astype(np.float32)
            y_train = contents["y_train"].reshape(-1).astype(np.int64)
            x_val = contents["X_valid"].astype(np.float32)
            y_val = contents["y_valid"].reshape(-1).astype(np.int64)
            x_test = contents["X_test"].astype(np.float32)
            y_test = contents["y_test"].reshape(-1).astype(np.int64)
            train_mean = np.mean(x_train)
            train_std = np.std(x_train)
            print(
                paint(
                    f"[MHEALTH] .mat stats — mean={train_mean:.4f}, std={train_std:.4f} (should be ~0 and ~1 if prepare_mhealth.py was used)",
                    "blue",
                )
            )
            if abs(train_mean) > 0.5 or abs(train_std - 1.0) > 0.5:
                print(
                    paint(
                        "[WARN] MHEALTH data does not look pre-normalised. Applying z-score normalisation now.",
                        "blue",
                    )
                )
                mean_tr = np.mean(x_train, axis=0)
                std_tr = np.std(x_train, axis=0)
                std_tr[std_tr == 0] = 1.0
                x_train = (x_train - mean_tr) / std_tr
                x_val = (x_val - mean_tr) / std_tr
                x_test = (x_test - mean_tr) / std_tr
            else:
                print(
                    paint(
                        "[OK] MHEALTH data appears pre-normalised — skipping second normalisation.",
                        "blue",
                    )
                )
        else:
            x_train = contents["X_train"].astype(np.float32)
            y_train = contents["y_train"].reshape(-1).astype(np.int64)
            x_val = contents["X_valid"].astype(np.float32)
            y_val = contents["y_valid"].reshape(-1).astype(np.int64)
            x_test = contents["X_test"].astype(np.float32)
            y_test = contents["y_test"].reshape(-1).astype(np.int64)
            train_mean = np.mean(x_train)
            train_std = np.std(x_train)
            print(
                paint(
                    f"[PAMAP2/Skoda] Raw .mat stats — mean={train_mean:.4f}, std={train_std:.4f} (should be ~0 and ~1 if prepare_datasets.py was used)",
                    "blue",
                )
            )
            if abs(train_mean) > 0.5 or abs(train_std - 1.0) > 0.5:
                print(
                    paint(
                        "[WARN] Data does not look pre-normalised. Applying z-score normalisation now.",
                        "blue",
                    )
                )
                mean_tr = np.mean(x_train, axis=0)
                std_tr = np.std(x_train, axis=0)
                std_tr[std_tr == 0] = 1.0
                x_train = (x_train - mean_tr) / std_tr
                x_val = (x_val - mean_tr) / std_tr
                x_test = (x_test - mean_tr) / std_tr
            else:
                print(
                    paint(
                        "[OK] Data appears pre-normalised — skipping second normalisation.",
                        "blue",
                    )
                )
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique_labels = np.unique(y)
        n_expected = len(class_map)
        print(
            paint(
                f"[LABEL CHECK] {split_name}: unique labels = {unique_labels.tolist()}  (expected 0..{n_expected - 1})",
                "blue",
            )
        )
        if unique_labels.size > 0 and (
            unique_labels.min() < 0 or unique_labels.max() >= n_expected
        ):
            print(
                paint(
                    f"[WARN] {split_name} labels out of expected range [0, {n_expected - 1}]!",
                    "blue",
                )
            )
    print(
        "[-] Train data : {} {}, target {} {}".format(
            x_train.shape, x_train.dtype, y_train.shape, y_train.dtype
        )
    )
    print(
        "[-] Valid data : {} {}, target {} {}".format(
            x_val.shape, x_val.dtype, y_val.shape, y_val.dtype
        )
    )
    print(
        "[-] Test data  : {} {}, target {} {}".format(
            x_test.shape, x_test.dtype, y_test.shape, y_test.dtype
        )
    )
    plot_pie(y_train, "train", path_raw, class_map)
    plot_pie(y_val, "val", path_raw, class_map)
    plot_pie(y_test, "test", path_raw, class_map)
    np.savez_compressed(os.path.join(path_raw, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(path_raw, "val.npz"), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(path_raw, "test.npz"), x=x_test, y=y_test)
    print("[+] Raw sample datasets successfully saved!")
    print(paint("--" * 50, "blue"))


def partition(path_raw, path_processed, window, stride, class_map):
    print(f"[*] Reading raw files from {path_raw}")
    dataset_train = np.load(os.path.join(path_raw, "train.npz"))
    x_train, y_train = (dataset_train["x"], dataset_train["y"])
    dataset_val = np.load(os.path.join(path_raw, "val.npz"))
    x_val, y_val = (dataset_val["x"], dataset_val["y"])
    dataset_test = np.load(os.path.join(path_raw, "test.npz"))
    x_test, y_test = (dataset_test["x"], dataset_test["y"])
    data_train, target_train = sliding_window(x_train, y_train, window, stride)
    data_val, target_val = sliding_window(x_val, y_val, window, stride)
    data_test, target_test = sliding_window(x_test, y_test, window, stride)
    data_test_sw, target_test_sw = sliding_window(x_test, y_test, window, 1)
    print(
        "[-] Train data : {} {}, target {} {}".format(
            data_train.shape, data_train.dtype, target_train.shape, target_train.dtype
        )
    )
    print(
        "[-] Valid data : {} {}, target {} {}".format(
            data_val.shape, data_val.dtype, target_val.shape, target_val.dtype
        )
    )
    print(
        "[-] Test data  : {} {}, target {} {}".format(
            data_test.shape, data_test.dtype, target_test.shape, target_test.dtype
        )
    )
    print(
        "[-] Test data sample-wise : {} {}, target sample-wise {} {}".format(
            data_test_sw.shape,
            data_test_sw.dtype,
            target_test_sw.shape,
            target_test_sw.dtype,
        )
    )
    plot_pie(target_train, "train", path_processed, class_map)
    plot_pie(target_val, "val", path_processed, class_map)
    plot_pie(target_test, "test", path_processed, class_map)
    plot_pie(target_test_sw, "test_sample_wise", path_processed, class_map)
    np.savez_compressed(
        os.path.join(path_processed, "train.npz"), data=data_train, target=target_train
    )
    np.savez_compressed(
        os.path.join(path_processed, "val.npz"), data=data_val, target=target_val
    )
    np.savez_compressed(
        os.path.join(path_processed, "test.npz"), data=data_test, target=target_test
    )
    np.savez_compressed(
        os.path.join(path_processed, "test_sample_wise.npz"),
        data=data_test_sw,
        target=target_test_sw,
    )
    print("[+] Processed segment datasets successfully saved!")
    print(paint("--" * 50, "blue"))


def sliding_window(x, y, window, stride, scheme="last"):
    data, target = ([], [])
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            y_segment = y[start:end][-1]
        elif scheme == "max":
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)
    return (data, target)


def _is_capture24(class_map):
    return "Sleep" in class_map and len(class_map) in (4, 6)


def _is_mhealth(dataset):
    return dataset in ("mhealth", "mhealth_nonull")


def preprocess_pipeline(args):
    if _is_capture24(args.class_map):
        _preprocess_capture24(args)
        return
    filter_null_at_load = getattr(args, "filter_null_at_load", False)
    if not os.path.exists(args.path_raw):
        os.makedirs(args.path_raw, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.path_raw, "log_raw.txt"))
        print(paint("[STEP 0] Loading the data files..."))
        if filter_null_at_load:
            print(
                paint(
                    f"[STEP 0] filter_null_at_load=True for dataset '{args.dataset}' — null class will be dropped IN MEMORY before normalisation+windowing. Source .mat is NOT modified.",
                    "blue",
                )
            )
        load_mat(
            path_data=args.path_data,
            path_raw=args.path_raw,
            class_map=args.class_map,
            dataset=args.dataset,
            filter_null_at_load=filter_null_at_load,
        )
    else:
        print(paint("[STEP 0] Files already loaded!"))
    w, s = (args.window, args.stride)
    if not os.path.exists(args.path_processed):
        os.makedirs(args.path_processed, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.path_processed, f"log_{w}_{s}.txt"))
        print(
            paint(f"[STEP 1] Partitioning the dataset (window,stride) = ({w},{s})...")
        )
        partition(
            path_raw=args.path_raw,
            path_processed=args.path_processed,
            window=w,
            stride=s,
            class_map=args.class_map,
        )
    else:
        print(
            paint(f"[STEP 1] Dataset already partitioned (window,stride) = ({w},{s})!")
        )


def _preprocess_capture24(args):
    import shutil

    if not os.path.exists(args.path_raw):
        os.makedirs(args.path_raw, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.path_raw, "log_raw.txt"))
        print(paint("[STEP 0] Loading Capture-24 .npz file..."))
        load_mat(
            path_data=args.path_data,
            path_raw=args.path_raw,
            class_map=args.class_map,
            dataset=args.dataset,
        )
    else:
        print(paint("[STEP 0] Capture-24 raw files already loaded!"))
    w, s = (args.window, args.stride)
    if not os.path.exists(args.path_processed):
        os.makedirs(args.path_processed, exist_ok=True)
        print(
            paint(
                f"[STEP 1] Repackaging Capture-24 segments into processed format (window={w}, stride={s})..."
            )
        )
        for split in ("train", "val", "test"):
            raw_npz = np.load(os.path.join(args.path_raw, f"{split}.npz"))
            data = raw_npz["x"]
            target = raw_npz["y"]
            np.savez_compressed(
                os.path.join(args.path_processed, f"{split}.npz"),
                data=data,
                target=target,
            )
            print(f"  {split}: data={data.shape}, target={target.shape}")
        shutil.copy(
            os.path.join(args.path_processed, "test.npz"),
            os.path.join(args.path_processed, "test_sample_wise.npz"),
        )
        print("[+] Capture-24 processed segments saved!")
        print(paint("--" * 50, "blue"))
    else:
        print(paint(f"[STEP 1] Capture-24 already processed (window={w}, stride={s})!"))


def main():
    args, _, _ = get_args()
    preprocess_pipeline(args)


if __name__ == "__main__":
    main()
