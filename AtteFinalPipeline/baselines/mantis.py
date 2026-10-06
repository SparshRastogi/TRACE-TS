from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
from AtteFinalPipeline.embeddings.encoders import MANTIS_OUTPUT_TOKEN
import os
import sys
import argparse
import datetime
import time
import numpy as np
from sklearn import metrics as skmetrics
import torch
import torch.nn.functional as F

MANTIS_CHECKPOINT = "paris-noah/MantisV2"
MANTIS_SEQ_LEN = 512
MANTIS_RETURN_LAYER = 2
DROP_NULL_SUPPORTED = ("opportunity", "mhealth")
_DATASET_CONFIGS = {
    "pamap2": dict(
        processed_subdir="pamap2/processed/24_12",
        num_class=12,
        window=24,
        class_map=[
            "Rope Jumping",
            "Lying",
            "Sitting",
            "Standing",
            "Walking",
            "Running",
            "Cycling",
            "Nordic Walking",
            "Ascending Stairs",
            "Descending Stairs",
            "Vacuum Cleaning",
            "Ironing",
        ],
    ),
    "capture24": dict(
        processed_subdir="capture24_Willetts2018_subset/processed/200_1",
        num_class=6,
        window=200,
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
    ),
    "uschad": dict(
        processed_subdir="uschad/processed/400_200",
        num_class=12,
        window=400,
        class_map=list(USCHAD_CLASS_MAP),
    ),
    "opportunity": dict(
        processed_subdir="opportunity/processed/24_12",
        num_class=18,
        window=24,
        class_map=[
            "Null",
            "Open Door 1",
            "Open Door 2",
            "Close Door 1",
            "Close Door 2",
            "Open Fridge",
            "Close Fridge",
            "Open Dishwasher",
            "Close Dishwasher",
            "Open Drawer 1",
            "Close Drawer 1",
            "Open Drawer 2",
            "Close Drawer 2",
            "Open Drawer 3",
            "Close Drawer 3",
            "Clean Table",
            "Drink from Cup",
            "Toggle Switch",
        ],
    ),
    "opportunity_nonull": dict(
        processed_subdir="opportunity/processed/24_12",
        num_class=17,
        window=24,
        class_map=[
            "Open Door 1",
            "Open Door 2",
            "Close Door 1",
            "Close Door 2",
            "Open Fridge",
            "Close Fridge",
            "Open Dishwasher",
            "Close Dishwasher",
            "Open Drawer 1",
            "Close Drawer 1",
            "Open Drawer 2",
            "Close Drawer 2",
            "Open Drawer 3",
            "Close Drawer 3",
            "Clean Table",
            "Drink from Cup",
            "Toggle Switch",
        ],
        filter_null_at_load=True,
    ),
    "shoaib": dict(
        processed_subdir="shoaib/processed/100_50",
        num_class=7,
        window=100,
        class_map=[
            "Walking",
            "Standing",
            "Jogging",
            "Sitting",
            "Biking",
            "Walking Upstairs",
            "Walking Downstairs",
        ],
    ),
    "mhealth": dict(
        processed_subdir="mhealth/processed/100_50",
        num_class=13,
        window=100,
        class_map=[
            "Null",
            "Standing still",
            "Sitting and relaxing",
            "Lying down",
            "Walking",
            "Climbing stairs",
            "Waist bends forward",
            "Frontal elevation of arms",
            "Knees bending",
            "Cycling",
            "Jogging",
            "Running",
            "Jump front & back",
        ],
    ),
    "mhealth_nonull": dict(
        processed_subdir="mhealth_nonull/processed/100_50",
        num_class=12,
        window=100,
        class_map=[
            "Standing still",
            "Sitting and relaxing",
            "Lying down",
            "Walking",
            "Climbing stairs",
            "Waist bends forward",
            "Frontal elevation of arms",
            "Knees bending",
            "Cycling",
            "Jogging",
            "Running",
            "Jump front & back",
        ],
        filter_null_at_load=False,
    ),
    "ucihar": dict(
        processed_subdir=None,
        num_class=6,
        window=128,
        class_map=[
            "Walking",
            "Walking Upstairs",
            "Walking Downstairs",
            "Sitting",
            "Standing",
            "Laying",
        ],
    ),
}
UCIHAR_BASE_PATH = str(UCIHAR_DATA_DIR)
UCIHAR_VAL_RATIO = 0.2


class _Tee:
    def __init__(self, fh, orig):
        self._fh, self._orig = (fh, orig)

    def write(self, msg):
        self._fh.write(msg)
        self._orig.write(msg)

    def flush(self):
        self._fh.flush()
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()


def _redirect(log_path):
    orig = sys.stdout
    fh = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(fh, orig)
    return (fh, orig)


def _restore(fh, orig):
    sys.stdout = orig
    fh.close()


def _load_ucihar_split(split):
    x_train_full = np.load(os.path.join(UCIHAR_BASE_PATH, "X_train.npy")).astype(
        np.float32
    )
    y_train_full = np.load(os.path.join(UCIHAR_BASE_PATH, "y_train.npy")).astype(
        np.int64
    )
    split_idx = int(len(x_train_full) * (1 - UCIHAR_VAL_RATIO))
    if split == "train":
        data, target = (x_train_full[:split_idx], y_train_full[:split_idx])
    elif split == "val":
        data, target = (x_train_full[split_idx:], y_train_full[split_idx:])
    elif split == "test":
        data = np.load(os.path.join(UCIHAR_BASE_PATH, "X_test.npy")).astype(np.float32)
        target = np.load(os.path.join(UCIHAR_BASE_PATH, "y_test.npy")).astype(np.int64)
    else:
        raise ValueError(f"unknown split '{split}'")
    if target.size > 0 and target.min() > 0:
        target = target - 1
    return (data, target)


def load_ad_split(ds_name, split, data_root):
    if ds_name == "ucihar":
        return _load_ucihar_split(split)
    cfg = _DATASET_CONFIGS[ds_name]
    if split == "test":
        filename = "test_sample_wise.npz"
    else:
        filename = f"{split}.npz"
    npz_path = os.path.join(data_root, cfg["processed_subdir"], filename)
    if not os.path.isfile(npz_path):
        if split == "test":
            alt = os.path.join(data_root, cfg["processed_subdir"], "test.npz")
            raise FileNotFoundError(
                f"{filename} not found at {npz_path}.  A&D evaluates test on test_sample_wise.npz (stride=1).  Either: (a) re-run A&D's preprocess.py to regenerate it, or (b) confirm stride_test == stride for this dataset and rename test.npz → test_sample_wise.npz manually.  (test.npz {('exists' if os.path.isfile(alt) else 'missing')} too.)"
            )
        raise FileNotFoundError(
            f"{filename} not found at {npz_path}. Run the A&D preprocessing pipeline (preprocess.py) first."
        )
    npz = np.load(npz_path, allow_pickle=False)
    data = npz["data"].astype(np.float32)
    target = npz["target"].astype(np.int64)
    return (data, target)


def drop_null_class(data, target, split_name=""):
    if target.size == 0:
        return (data, target)
    keep_mask = target != 0
    n_before = target.size
    n_after = int(keep_mask.sum())
    n_dropped = n_before - n_after
    data_f = data[keep_mask]
    target_f = (target[keep_mask] - 1).astype(np.int64)
    print(
        f"    [drop_null] {split_name:<5s}: {n_before:>7d} → {n_after:>7d}  (dropped {n_dropped:>7d} null rows, {100.0 * n_dropped / max(n_before, 1):.2f}%)",
        flush=True,
    )
    return (data_f, target_f)


def prepare_for_mantis(samples_np, chunk_size=4096, verbose=True):
    if samples_np.dtype != np.float32:
        samples_np = samples_np.astype(np.float32, copy=False)
    if not samples_np.flags["C_CONTIGUOUS"]:
        samples_np = np.ascontiguousarray(samples_np)
    N, T, C = samples_np.shape
    out = np.empty((N, C, MANTIS_SEQ_LEN), dtype=np.float32)
    if verbose:
        in_gb = N * T * C * 4 / 1000000000.0
        out_gb = N * C * MANTIS_SEQ_LEN * 4 / 1000000000.0
        print(
            f"    resize: ({N},{T},{C}) → ({N},{C},{MANTIS_SEQ_LEN})  [{in_gb:.2f} GB in / {out_gb:.2f} GB out]",
            flush=True,
        )
    n_chunks = (N + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        a = ci * chunk_size
        b = min(N, a + chunk_size)
        chunk = torch.from_numpy(samples_np[a:b]).permute(0, 2, 1).contiguous()
        chunk = F.interpolate(
            chunk, size=MANTIS_SEQ_LEN, mode="linear", align_corners=False
        )
        out[a:b] = chunk.numpy()
        del chunk
        if verbose and (ci % max(1, n_chunks // 10) == 0 or ci == n_chunks - 1):
            print(f"      chunk {ci + 1}/{n_chunks}", flush=True)
    return out


def _predict_in_chunks(trainer, x, kind="predict", batch_size=512, verbose=True):
    fn = trainer.predict if kind == "predict" else trainer.predict_proba
    N = x.shape[0]
    if N == 0:
        return (
            np.empty((0,), dtype=np.int64)
            if kind == "predict"
            else np.empty((0, 0), dtype=np.float32)
        )
    pieces = []
    n_chunks = (N + batch_size - 1) // batch_size
    if verbose:
        print(f"    {kind}: N={N} in {n_chunks} chunks of {batch_size}", flush=True)
    for ci in range(n_chunks):
        a = ci * batch_size
        b = min(N, a + batch_size)
        pieces.append(fn(x[a:b]))
        if verbose and (ci % max(1, n_chunks // 10) == 0 or ci == n_chunks - 1):
            print(f"      chunk {ci + 1}/{n_chunks}", flush=True)
    return np.concatenate(pieces, axis=0)


def finetune_dataset(ds_name, args, device):
    if ds_name.endswith("_nonull"):
        out_subdir = ds_name
    elif args.drop_null:
        out_subdir = f"{ds_name}_nonull"
    else:
        out_subdir = ds_name
    out_root = os.path.join(args.output_root, out_subdir)
    ckpt_dir = os.path.join(out_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    fh, orig = _redirect(os.path.join(out_root, "train.log"))
    try:
        result = _inner(ds_name, args, device, out_root, ckpt_dir)
    finally:
        _restore(fh, orig)
    return result


def _inner(ds_name, args, device, out_root, ckpt_dir):
    from mantis.trainer import MantisTrainer
    from mantis.architecture import MantisV2

    sep = "=" * 70
    print(sep)
    cfg = dict(_DATASET_CONFIGS[ds_name])
    cfg["class_map"] = list(cfg["class_map"])
    cfg_says_filter = cfg.get("filter_null_at_load", False)
    cli_says_filter = bool(args.drop_null)
    filter_at_load = cfg_says_filter or cli_says_filter
    mode_str = ""
    if filter_at_load:
        mode_str = "  [LOAD-TIME NULL FILTER]"
    elif ds_name.endswith("_nonull"):
        mode_str = "  [PRE-WINDOWED NULL-FREE]"
    print(f"  Fine-tuning MantisV2 — dataset : {ds_name}{mode_str}")
    print(f"  Checkpoint              : {args.mantis_checkpoint}")
    print(f"  Output                  : {out_root}")
    print(f"  Data root               : {args.data_root}")
    print(sep)
    if cli_says_filter and ds_name == "mhealth":
        print()
        print("  [WARN] --drop_null on 'mhealth' is the DEPRECATED path.")
        print("  [WARN] It filters AFTER windowing, which concentrates the")
        print("  [WARN] stride-1 test set on activity-interior near-duplicates")
        print("  [WARN] of training windows and inflates test accuracy.")
        print("  [WARN] Prefer:  --dataset mhealth_nonull")
        print("  [WARN] (reads pre-windowed null-free files from prepare_mhealth.py)")
        print()
    print(f"  Classes ({cfg['num_class']}): {cfg['class_map']}")
    print("\n  Loading data ...")
    try:
        x_train, y_train = load_ad_split(ds_name, "train", args.data_root)
        x_val, y_val = load_ad_split(ds_name, "val", args.data_root)
        x_test, y_test = load_ad_split(ds_name, "test", args.data_root)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None
    print(f"  Train raw : {x_train.shape}   classes: {np.unique(y_train).tolist()}")
    print(f"  Val   raw : {x_val.shape}")
    print(f"  Test  raw : {x_test.shape}")
    if filter_at_load:
        if cli_says_filter:
            assert ds_name in DROP_NULL_SUPPORTED, (
                f"--drop_null requested for '{ds_name}' but this dataset does not have label 0 = 'Null'. Supported: {DROP_NULL_SUPPORTED}"
            )
            assert cfg["class_map"][0] == "Null", (
                f"Expected class_map[0] == 'Null' for {ds_name}, got '{cfg['class_map'][0]}'. Refusing to filter."
            )
        labels_have_null = (
            y_train.size > 0
            and (y_train == 0).any()
            or (y_val.size > 0 and (y_val == 0).any())
            or (y_test.size > 0 and (y_test == 0).any())
        )
        if not labels_have_null:
            print(
                "\n  [filter_at_load] No label==0 rows present in loaded data; on-disk files are already null-free. Skipping filter."
            )
        else:
            print("\n  Dropping null class (label 0) in-memory ...")
            x_train, y_train = drop_null_class(x_train, y_train, "train")
            x_val, y_val = drop_null_class(x_val, y_val, "val")
            x_test, y_test = drop_null_class(x_test, y_test, "test")
            if cfg["class_map"][0] == "Null":
                cfg["num_class"] = cfg["num_class"] - 1
                cfg["class_map"] = cfg["class_map"][1:]
            print(f"  Post-filter classes ({cfg['num_class']}): {cfg['class_map']}")
            print(
                f"  Post-filter sizes : train={x_train.shape[0]}  val={x_val.shape[0]}  test={x_test.shape[0]}"
            )
            for split_name, arr in (
                ("train", y_train),
                ("val", y_val),
                ("test", y_test),
            ):
                if arr.size == 0:
                    print(f"  [SKIP] {split_name} split is empty after null-drop.")
                    return None
    for split_name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        if y.size == 0:
            continue
        if y.min() < 0 or y.max() >= cfg["num_class"]:
            print(
                f"  WARNING: {split_name} labels out of range — min={y.min()} max={y.max()} expected [0,{cfg['num_class'] - 1}]"
            )
    print("  Resizing to (N, C, 512) ...", flush=True)
    print("   train:", flush=True)
    x_train_r = prepare_for_mantis(x_train, chunk_size=args.resize_chunk)
    del x_train
    print("   val:", flush=True)
    x_val_r = prepare_for_mantis(x_val, chunk_size=args.resize_chunk)
    del x_val
    print("   test:", flush=True)
    x_test_r = prepare_for_mantis(x_test, chunk_size=args.resize_chunk)
    del x_test
    import gc

    gc.collect()
    print(f"  Train prepared : {x_train_r.shape}  (N, C, 512)")
    print(f"  Val   prepared : {x_val_r.shape}")
    print(f"  Test  prepared : {x_test_r.shape}")
    import json as _json

    with open(os.path.join(out_root, "class_map.json"), "w") as _fh:
        _json.dump(
            {
                "dataset": ds_name,
                "drop_null_cli": bool(cli_says_filter),
                "filter_null_at_load": bool(cfg_says_filter),
                "num_class": cfg["num_class"],
                "window": cfg["window"],
                "class_map": cfg["class_map"],
            },
            _fh,
            indent=2,
        )
    print(f"\n  Loading MantisV2 from '{args.mantis_checkpoint}' ...")
    network = MantisV2(
        return_transf_layer=MANTIS_RETURN_LAYER,
        output_token=MANTIS_OUTPUT_TOKEN,
        device=device,
    )
    network = network.from_pretrained(args.mantis_checkpoint)
    trainer = MantisTrainer(device=str(device), network=network)
    print("  MantisTrainer ready.")
    print(f"\n  trainer.fit() settings:")
    print(f"    fine_tuning_type        = 'full'")
    print(f"    num_epochs              = {args.num_epochs}")
    print(f"    batch_size              = {args.batch_size}")
    print(f"    base_learning_rate      = {args.base_learning_rate}")
    print(f"    learning_rate_adjusting = {not args.no_lr_schedule}")
    metrics_path = os.path.join(out_root, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(
            f"MantisV2 fine-tuning: {ds_name}{mode_str}  started {datetime.datetime.now()}\n"
        )
        f.write(f"checkpoint              : {args.mantis_checkpoint}\n")
        f.write(f"data_root               : {args.data_root}\n")
        f.write(f"processed_subdir        : {cfg['processed_subdir']}\n")
        f.write(f"drop_null (CLI)         : {bool(cli_says_filter)}\n")
        f.write(f"filter_null_at_load     : {bool(cfg_says_filter)}\n")
        if ds_name == "ucihar":
            f.write(
                f"test file               : {UCIHAR_BASE_PATH}/X_test.npy (special UCI-HAR loader)\n"
            )
        else:
            f.write(
                f"test file               : {os.path.join(args.data_root, cfg['processed_subdir'], 'test_sample_wise.npz')}\n"
            )
            f.write(f"  (matches A&D's stride_test=1 SensorDataset behaviour)\n")
        f.write(f"num_class               : {cfg['num_class']}\n")
        f.write(f"raw window (T)          : {cfg['window']}\n")
        f.write(f"class_map               : {cfg['class_map']}\n")
        f.write(
            f"train/val/test sizes    : {x_train_r.shape[0]}/{x_val_r.shape[0]}/{x_test_r.shape[0]}\n"
        )
        f.write(f"channels (C)            : {x_train_r.shape[1]}\n")
        f.write(f"resized seq_len         : {MANTIS_SEQ_LEN}\n")
        f.write(f"num_epochs              : {args.num_epochs}\n")
        f.write(f"batch_size              : {args.batch_size}\n")
        f.write(f"base_learning_rate      : {args.base_learning_rate}\n")
        f.write(f"learning_rate_adjusting : {not args.no_lr_schedule}\n\n")
    t0 = time.time()
    trainer.fit(
        x_train_r,
        y_train,
        fine_tuning_type="full",
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        base_learning_rate=args.base_learning_rate,
        learning_rate_adjusting=not args.no_lr_schedule,
    )
    elapsed = time.time() - t0
    elapsed_str = str(datetime.timedelta(seconds=round(elapsed)))
    print(f"\n  Training complete — {elapsed_str}")
    print("  Evaluating ...", flush=True)
    print("   predict TRAIN ...", flush=True)
    y_pred_tr = _predict_in_chunks(
        trainer, x_train_r, "predict", batch_size=args.predict_chunk
    )
    print("   predict VAL ...", flush=True)
    y_pred_va = _predict_in_chunks(
        trainer, x_val_r, "predict", batch_size=args.predict_chunk
    )
    print("   predict TEST ...", flush=True)
    y_pred_te = _predict_in_chunks(
        trainer, x_test_r, "predict", batch_size=args.predict_chunk
    )
    print("   predict_proba VAL ...", flush=True)
    y_prob_va = _predict_in_chunks(
        trainer, x_val_r, "proba", batch_size=args.predict_chunk
    )
    print("   predict_proba TEST ...", flush=True)
    y_prob_te = _predict_in_chunks(
        trainer, x_test_r, "proba", batch_size=args.predict_chunk
    )

    def report(y_true, y_pred, tag):
        acc = 100.0 * skmetrics.accuracy_score(y_true, y_pred)
        fm = 100.0 * skmetrics.f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        fw = 100.0 * skmetrics.f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        print(f"  {tag:<6s}  acc={acc:.2f}%  macro-F1={fm:.2f}%  weighted-F1={fw:.2f}%")
        return (acc, fm, fw)

    acc_tr, fm_tr, fw_tr = report(y_train, y_pred_tr, "TRAIN")
    acc_va, fm_va, fw_va = report(y_val, y_pred_va, "VAL")
    is_capture24 = "capture24" in ds_name
    if not is_capture24 and y_test.size > 0:
        ws = cfg["window"] - 1
        filler = y_test[0]
        y_true_te = np.concatenate([y_test, np.full(ws, filler, dtype=np.int64)])
        y_pred_te_padded = np.concatenate(
            [y_pred_te, np.full(ws, filler, dtype=y_pred_te.dtype)]
        )
        print(
            f"  [A&D-compat] padding test predictions with {ws} filler samples (label={filler}{(' — first non-null sample' if filter_at_load else '')}) to mirror A&D's eval_one_epoch."
        )
        acc_te_u, fm_te_u, fw_te_u = report(y_test, y_pred_te, "TEST*")
    else:
        y_true_te = y_test
        y_pred_te_padded = y_pred_te
        acc_te_u = fm_te_u = fw_te_u = None
    acc_te, fm_te, fw_te = report(y_true_te, y_pred_te_padded, "TEST")
    with open(metrics_path, "a") as f:
        f.write(f"TRAIN          acc={acc_tr:.2f}%  fm={fm_tr:.2f}%  fw={fw_tr:.2f}%\n")
        f.write(f"VAL            acc={acc_va:.2f}%  fm={fm_va:.2f}%  fw={fw_va:.2f}%\n")
        f.write(
            f"TEST  (A&D)    acc={acc_te:.2f}%  fm={fm_te:.2f}%  fw={fw_te:.2f}%   # padded with (window-1) filler samples to match A&D\n"
        )
        if acc_te_u is not None:
            f.write(
                f"TEST  (raw)    acc={acc_te_u:.2f}%  fm={fm_te_u:.2f}%  fw={fw_te_u:.2f}%   # unpadded; for reference only\n"
            )
        else:
            f.write(f"TEST  (raw)    same as A&D (capture24: no padding applied)\n")
        f.write(f"elapsed={elapsed_str}\n")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pth")
    torch.save(
        {
            "network_state_dict": trainer.network.state_dict(),
            "dataset": ds_name,
            "drop_null_cli": bool(cli_says_filter),
            "filter_null_at_load": bool(cfg_says_filter),
            "num_class": cfg["num_class"],
            "class_map": cfg["class_map"],
            "window": cfg["window"],
            "processed_subdir": cfg["processed_subdir"],
            "data_root": args.data_root,
            "mantis_checkpoint": args.mantis_checkpoint,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "base_learning_rate": args.base_learning_rate,
            "learning_rate_adjusting": not args.no_lr_schedule,
            "val_acc": acc_va,
            "val_fm": fm_va,
            "val_fw": fw_va,
            "test_acc": acc_te,
            "test_fm": fm_te,
            "test_fw": fw_te,
            "test_acc_unpadded": acc_te_u,
            "test_fm_unpadded": fm_te_u,
            "test_fw_unpadded": fw_te_u,
        },
        ckpt_path,
    )
    print(f"  Checkpoint    → {ckpt_path}")
    np.savez_compressed(
        os.path.join(out_root, "val_predictions.npz"),
        y_true=y_val,
        y_pred=y_pred_va,
        y_prob=y_prob_va,
    )
    np.savez_compressed(
        os.path.join(out_root, "test_predictions.npz"),
        y_true=y_test,
        y_pred=y_pred_te,
        y_prob=y_prob_te,
    )
    print(f"  Predictions   → {out_root}/{{val,test}}_predictions.npz")
    print(
        f"\n  DONE {ds_name}{mode_str} | VAL macro-F1={fm_va:.2f}% | TEST macro-F1={fm_te:.2f}% | {elapsed_str}"
    )
    if ds_name.endswith("_nonull"):
        result_name = ds_name
    elif cli_says_filter:
        result_name = ds_name + "_nonull"
    else:
        result_name = ds_name
    return dict(
        dataset=result_name,
        val_fm=round(fm_va, 2),
        test_acc=round(acc_te, 2),
        test_fm=round(fm_te, 2),
        test_fw=round(fw_te, 2),
        test_fm_unpadded=round(fm_te_u, 2) if fm_te_u is not None else None,
        elapsed=elapsed_str,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune MantisV2 on A&D-preprocessed HAR datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="all", choices=["all"] + list(_DATASET_CONFIGS.keys())
    )
    p.add_argument(
        "--data_root",
        default="./data",
        help="Root containing A&D processed dataset folders. Ignored for ucihar (uses hardcoded path).",
    )
    p.add_argument("--output_root", default="./mantis_finetuned")
    p.add_argument("--mantis_checkpoint", default=MANTIS_CHECKPOINT)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--base_learning_rate", type=float, default=0.0002)
    p.add_argument("--no_lr_schedule", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drop_null",
        action="store_true",
        default=False,
        help="[LEGACY] Drop label==0 ('Null') rows in-memory for opportunity and mhealth. Prefer --dataset opportunity_nonull / mhealth_nonull instead. For mhealth, this path inflates test accuracy and will print a warning.",
    )
    p.add_argument(
        "--resize_chunk",
        type=int,
        default=4096,
        help="Windows per chunk during CPU upsampling to 512.",
    )
    p.add_argument(
        "--predict_chunk",
        type=int,
        default=512,
        help="Windows per chunk during MantisTrainer.predict / predict_proba. trainer.fit() is unaffected.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Output : {args.output_root}")
    if args.drop_null:
        print(
            f"Mode   : --drop_null (LEGACY load-time filter, opportunity + mhealth only)"
        )
    print()
    if args.dataset == "all":
        datasets = [d for d in _DATASET_CONFIGS.keys() if not d.endswith("_nonull")]
    else:
        datasets = [args.dataset]
    if args.drop_null:
        skipped = [
            d
            for d in datasets
            if d not in DROP_NULL_SUPPORTED and (not d.endswith("_nonull"))
        ]
        datasets = [
            d for d in datasets if d in DROP_NULL_SUPPORTED or d.endswith("_nonull")
        ]
        if skipped:
            print(
                f"[--drop_null] Skipping datasets where label 0 is not 'Null': {skipped}"
            )
            print(f"[--drop_null] Running on: {datasets}\n")
        if not datasets:
            print(
                f"[--drop_null] Nothing to do. Supported datasets: {list(DROP_NULL_SUPPORTED)}"
            )
            return
    results, failed, t0 = ([], [], time.time())
    for ds in datasets:
        try:
            r = finetune_dataset(ds, args, device)
            if r:
                results.append(r)
        except Exception as exc:
            import traceback

            print(f"\n  [ERROR] {ds}: {exc}")
            traceback.print_exc()
            failed.append((ds, str(exc)))
    wall = time.time() - t0
    print(f"\n{'=' * 88}")
    print(
        f"  ALL DONE — {wall:.1f}s ({wall / 60:.1f}m){('  [--drop_null mode]' if args.drop_null else '')}"
    )
    print(f"  {'─' * 86}")
    print(
        f"  {'Dataset':<20s}  {'Val-F1':>7s}  {'Test-Acc':>9s}  {'Test-F1':>8s}  {'Test-WtF1':>10s}  {'F1*':>7s}  Elapsed"
    )
    print(f"  {'─' * 86}")
    print(f"  (Test-F1 = A&D-padded; F1* = unpadded raw, '-' for capture24)")
    for r in results:
        f1u = (
            f"{r['test_fm_unpadded']:.2f}%"
            if r["test_fm_unpadded"] is not None
            else "-"
        )
        print(
            f"  {r['dataset']:<20s}  {r['val_fm']:>6.2f}%  {r['test_acc']:>8.2f}%  {r['test_fm']:>7.2f}%  {r['test_fw']:>9.2f}%  {f1u:>7s}  {r['elapsed']}"
        )
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for ds, reason in failed:
            print(f"    {ds}: {reason}")
    print(f"{'=' * 88}")


if __name__ == "__main__":
    main()
