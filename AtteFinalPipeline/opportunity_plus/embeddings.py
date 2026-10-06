from AtteFinalPipeline.serialization import _json_dumps
from AtteFinalPipeline.embeddings.encoders import (
    MANTIS_OUTPUT_TOKEN,
    MANTIS_RETURN_TRANSF_LAYER,
    _resize_batch_for_mantis,
    load_mantis_model,
)
import os
import csv
import time
import argparse
import numpy as np
import scipy.io as sio
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

OPP_WINDOW = 24
OPP_NUM_CLASS = 18
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
assert len(OPP_CLASS_NAMES) == OPP_NUM_CLASS


def load_predictions_csv(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            row["window_idx"] = int(row["window_idx"])
            row["start_sample"] = int(row["start_sample"])
            row["end_sample"] = int(row["end_sample"])
            row["true_label"] = int(row["true_label"])
            row["pred_label"] = int(row["pred_label"])
            try:
                row["confidence"] = float(row["confidence"])
            except (KeyError, ValueError, TypeError):
                row["confidence"] = float("nan")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"predictions CSV at {path} is empty.")
    return rows


def build_windows_from_predictions(X, pred_rows, window=OPP_WINDOW):
    N = X.shape[0]
    C = X.shape[1]
    n_win = len(pred_rows)
    out = np.empty((n_win, window, C), dtype=np.float32)
    bad = []
    for i, row in enumerate(pred_rows):
        s = row["start_sample"]
        e = s + window
        if e > N:
            bad.append((i, s, e, N))
            slab = np.zeros((window, C), dtype=np.float32)
            avail = max(0, N - s)
            if avail > 0:
                slab[:avail] = X[s : s + avail]
            out[i] = slab
        else:
            out[i] = X[s:e]
    if bad:
        first = bad[0]
        raise RuntimeError(
            f"[FATAL] {len(bad)} prediction rows reference samples past the end of the .mat sensor array (e.g. row {first[0]}: start_sample={first[1]}, end={first[2]}, but N={first[3]}). The .mat and predictions.csv are out of sync — re-run prepare_opportunity_plus.py and infer_opportunity_plus.py together so they reference the same data."
        )
    return out


def _save_embedding_json(args_tuple):
    (
        output_dir,
        split,
        window_idx,
        true_label,
        pred_label,
        activity_name,
        embedding,
        embedding_dim,
        class_map,
    ) = args_tuple
    safe_pred_name = activity_name.replace(" ", "_").replace("/", "_")
    filename = f"{split}_class{pred_label}_{safe_pred_name}_s{window_idx:06d}.json"
    path = os.path.join(output_dir, filename)
    entry = {
        "sample_idx": int(window_idx),
        "split": split,
        "true_label": int(true_label),
        "activity": activity_name,
        "class_names": class_map,
        "mantis_embedding": {
            "model": f"MantisV2 (paris-noah/MantisV2, layer={MANTIS_RETURN_TRANSF_LAYER}, token={MANTIS_OUTPUT_TOKEN})",
            "embedding_dim": embedding_dim,
            "values": np.round(embedding, 6).tolist(),
        },
    }
    with open(path, "w") as f:
        f.write(_json_dumps(entry))
    return path


def save_combined_npz(
    split, sample_indices, true_labels, pred_labels, embeddings, output_dir
):
    path = os.path.join(output_dir, f"{split}_mantis_embeddings.npz")
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32),
        sample_indices=sample_indices.astype(np.int64),
        labels=true_labels.astype(np.int64),
        pred_labels=pred_labels.astype(np.int64),
    )
    print(f"  Combined NPZ → {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate MantisV2 embeddings for Opportunity++ windows, keyed by window_idx for downstream join with predictions.csv and video clips.",
    )
    p.add_argument(
        "--mat",
        required=True,
        help="Path to the .mat from prepare_opportunity_plus.py.",
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.csv from infer_opportunity_plus.py.",
    )
    p.add_argument(
        "--output_dir", required=True, help="Where to write per-window embedding JSONs."
    )
    p.add_argument("--mantis_checkpoint", type=str, default="paris-noah/MantisV2")
    p.add_argument("--mantis_batch_size", type=int, default=32)
    _cpu_default = max(4, min(16, (os.cpu_count() or 8) // 2))
    p.add_argument("--save_workers", type=int, default=_cpu_default)
    p.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save a combined .npz for fast loading.",
    )
    p.add_argument(
        "--keep_null",
        action="store_true",
        help="Generate embeddings for ALL windows including those the model predicted as Null. Default: skip Null predictions, since no clip exists for them.",
    )
    p.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Value to write into the 'split' field of each JSON and the filename prefix.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[1/5] Loading predictions: {args.predictions}")
    pred_rows = load_predictions_csv(args.predictions)
    print(f"    {len(pred_rows)} prediction rows")
    if args.keep_null:
        kept_rows = pred_rows
        print(f"    --keep_null set; processing all {len(kept_rows)} windows")
    else:
        kept_rows = [r for r in pred_rows if r["pred_label"] != 0]
        n_dropped = len(pred_rows) - len(kept_rows)
        print(
            f"    Dropping {n_dropped} Null-prediction windows (no clip available); processing {len(kept_rows)} windows."
        )
    if not kept_rows:
        print("[!] No windows to process. Exiting.")
        return
    print(f"\n[2/5] Loading {args.mat}")
    mat = sio.loadmat(args.mat)
    X = mat["testingData"].astype(np.float32).T
    print(f"    X (full sensor stream) = {X.shape}")
    print(f"\n[3/5] Building {len(kept_rows)} windows from start_sample positions")
    X_win = build_windows_from_predictions(X, kept_rows, window=OPP_WINDOW)
    print(
        f"    X_win = {X_win.shape}  (n_windows, window={OPP_WINDOW}, channels={X.shape[1]})"
    )
    print(f"\n[4/5] Loading Mantis and embedding {X_win.shape[0]} windows")
    mantis = load_mantis_model(args.mantis_checkpoint, device)
    n_win = X_win.shape[0]
    all_embeddings = []
    t0 = time.time()
    for start in tqdm(
        range(0, n_win, args.mantis_batch_size),
        desc="  Mantis",
        unit="batch",
        dynamic_ncols=True,
    ):
        end = min(start + args.mantis_batch_size, n_win)
        x_resized = _resize_batch_for_mantis(X_win[start:end])
        emb = mantis.transform(x_resized)
        all_embeddings.append(emb)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    emb_dim = int(all_embeddings.shape[1])
    print(
        f"    Embeddings done — dim={emb_dim}, n={n_win}, elapsed={time.time() - t0:.1f}s"
    )
    print(f"\n[5/5] Writing per-window JSONs to {args.output_dir}")
    save_args = []
    for local_i, row in enumerate(kept_rows):
        pred = int(row["pred_label"])
        true = int(row["true_label"])
        pred_name = (
            OPP_CLASS_NAMES[pred] if 0 <= pred < OPP_NUM_CLASS else f"Class_{pred}"
        )
        save_args.append(
            (
                args.output_dir,
                args.split_name,
                int(row["window_idx"]),
                true,
                pred,
                pred_name,
                all_embeddings[local_i],
                emb_dim,
                OPP_CLASS_NAMES,
            )
        )
    with ThreadPoolExecutor(max_workers=args.save_workers) as ex:
        list(
            tqdm(
                ex.map(_save_embedding_json, save_args),
                total=len(save_args),
                desc="  saving JSONs",
                unit="file",
                dynamic_ncols=True,
            )
        )
    if args.save_npz:
        sample_indices = np.array([r["window_idx"] for r in kept_rows], dtype=np.int64)
        true_labels = np.array([r["true_label"] for r in kept_rows], dtype=np.int64)
        pred_labels = np.array([r["pred_label"] for r in kept_rows], dtype=np.int64)
        save_combined_npz(
            args.split_name,
            sample_indices,
            true_labels,
            pred_labels,
            all_embeddings,
            args.output_dir,
        )
    pred_hist = {
        int(c): int((np.array([r["pred_label"] for r in kept_rows]) == c).sum())
        for c in range(OPP_NUM_CLASS)
    }
    summary = {
        "dataset": "opportunity_plus",
        "mat": os.path.abspath(args.mat),
        "predictions_csv": os.path.abspath(args.predictions),
        "splits": [args.split_name],
        "mantis_checkpoint": args.mantis_checkpoint,
        "mantis_layer": MANTIS_RETURN_TRANSF_LAYER,
        "mantis_token": MANTIS_OUTPUT_TOKEN,
        "embedding_dim": emb_dim,
        "total_saved": len(save_args),
        "total_null_skipped": 0 if args.keep_null else len(pred_rows) - len(kept_rows),
        "null_class_indices": [] if args.keep_null else [0],
        "class_map": OPP_CLASS_NAMES,
        "class_map_source": "opportunity_plus_corrected_ordering",
        "activity_field_source": "predicted_label",
        "sample_idx_source": "window_idx_from_predictions_csv",
        "predicted_class_histogram": pred_hist,
    }
    summary_path = os.path.join(args.output_dir, "embedding_summary.json")
    with open(summary_path, "w") as f:
        f.write(_json_dumps(summary))
    print(f"\n  Summary → {summary_path}")
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Windows embedded:     {len(save_args)}")
    print(f"  Output directory:     {args.output_dir}")
    print(f"  Embedding dimension:  {emb_dim}")
    print("=" * 70)


if __name__ == "__main__":
    main()
