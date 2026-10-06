from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
import os
import argparse
import glob
import numpy as np
import scipy.io as sio

USC_HAD_CLASS_NAMES = [label.lower() for label in USCHAD_CLASS_MAP]
NUM_CLASSES = 12
G_TO_MS2 = 9.80665
DEG_TO_RAD = np.pi / 180.0


def load_subject(subject_dir):
    x_list, y_list = ([], [])
    mat_files = sorted(glob.glob(os.path.join(subject_dir, "*.mat")))
    for fpath in mat_files:
        try:
            data = sio.loadmat(fpath)
        except Exception as e:
            print(f"  [WARN] {fpath}: {e}")
            continue
        readings = None
        for k in ("sensor_readings", "Sensor_readings"):
            if k in data:
                readings = data[k].astype(np.float32)
                break
        if readings is None:
            continue
        if readings.shape[0] == 6 and readings.ndim == 2:
            readings = readings.T
        if "activity_number" in data:
            act_num = int(np.squeeze(data["activity_number"]))
        else:
            fname = os.path.basename(fpath)
            digits = "".join((c for c in fname.split("t")[0] if c.isdigit()))
            if not digits:
                continue
            act_num = int(digits)
        label = act_num - 1
        if label < 0 or label >= NUM_CLASSES:
            continue
        readings = readings.copy()
        readings[:, :3] *= G_TO_MS2
        readings[:, 3:] *= DEG_TO_RAD
        labels = np.full(len(readings), label, dtype=np.int64)
        x_list.append(readings)
        y_list.append(labels)
    return (x_list, y_list)


def load_subjects(data_dir, subject_ids, name):
    Xs, ys = ([], [])
    for sid in subject_ids:
        sd = os.path.join(data_dir, f"Subject{sid}")
        if not os.path.isdir(sd):
            print(f"  [WARN] {sd} missing")
            continue
        print(f"  Loading Subject{sid}...")
        xl, yl = load_subject(sd)
        Xs.extend(xl)
        ys.extend(yl)
    if not Xs:
        raise RuntimeError(f"No data for {name} {subject_ids}")
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    print(f"  [{name}] {X.shape[0]} samples, {len(np.unique(y))} classes")
    return (X, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir", required=True, help="USC-HAD root (Subject1..Subject14 folders)"
    )
    p.add_argument("--out_dir", default="./dataset")
    p.add_argument("--val_subjects", nargs="+", type=int, default=[13, 14])
    p.add_argument("--test_subjects", nargs="+", type=int, default=[11, 12])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "uschad_unimts.mat")
    all_ids = list(range(1, 15))
    val_set, test_set = (set(args.val_subjects), set(args.test_subjects))
    train_set = set(all_ids) - val_set - test_set
    print(f"Train: {sorted(train_set)}")
    print(f"Val:   {sorted(val_set)}")
    print(f"Test:  {sorted(test_set)}")
    print(f"Units: acc → m/s² (× 9.80665), gyro → rad/s (× π/180)")
    print(f"Sampling rate: 100 Hz (USC-HAD native; matches prepare_uschad.py)")
    X_train, y_train = load_subjects(args.data_dir, sorted(train_set), "train")
    X_val, y_val = load_subjects(args.data_dir, sorted(val_set), "val")
    X_test, y_test = load_subjects(args.data_dir, sorted(test_set), "test")
    print(f"\nFinal shapes (physical units, raw 2-D):")
    print(f"  X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    print(
        f"  Sanity: |X_train acc| mean {np.abs(X_train[:, :3]).mean():.2f} (expect ~5–15 m/s²)"
    )
    sio.savemat(
        out_path,
        {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_valid": X_val,
            "y_valid": y_val.reshape(-1, 1),
            "X_test": X_test,
            "y_test": y_test.reshape(-1, 1),
            "sampling_rate": 100,
            "class_names": np.array(USC_HAD_CLASS_NAMES, dtype=object),
        },
        do_compression=True,
    )
    print(f"\n[+] Saved -> {out_path}")


if __name__ == "__main__":
    main()
