import os
import csv
import argparse
import subprocess
import numpy as np
import scipy.io as sio
from collections import Counter
import torch
import torch.nn as nn
from AtteFinalPipeline.expert.model import create

OPP_INPUT_DIM = 79
OPP_NUM_CLASS = 18
OPP_WINDOW = 24
OPP_HIDDEN_DIM = 128
OPP_FILTER_NUM = 64
OPP_FILTER_SIZE = 5
OPP_ENC_LAYERS = 2
OPP_ENC_BIDIR = False
OPP_DROPOUT = 0.5
OPP_DROPOUT_RNN = 0.25
OPP_DROPOUT_CLS = 0.5
OPP_ACTIVATION = "ReLU"
OPP_SA_DIV = 1
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
DEFAULT_OPPORTUNITY_ROOTS = ["./models/opportunity", "./models/opportunity"]


def auto_find_checkpoint(explicit):
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = []
    if os.path.exists("./weights/checkpoint_opportunity.pth"):
        candidates.append("./weights/checkpoint_opportunity.pth")
    import glob

    for root in DEFAULT_OPPORTUNITY_ROOTS:
        if os.path.isdir(root):
            hits = glob.glob(
                os.path.join(root, "*", "checkpoints", "checkpoint_best.pth")
            )
            hits.sort(key=os.path.getmtime, reverse=True)
            candidates.extend(hits)
            short = os.path.join(root, "checkpoints", "checkpoint_best.pth")
            if os.path.exists(short):
                candidates.append(short)
    for c in candidates:
        if os.path.exists(c):
            print(f"[*] Using checkpoint: {c}")
            return c
    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint /path/to/checkpoint_best.pth, or place one at ./weights/checkpoint_opportunity.pth, or train a model so a checkpoint appears under "
        + " | ".join(DEFAULT_OPPORTUNITY_ROOTS)
    )


def load_index_csv(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            row["window_idx"] = int(row["window_idx"])
            row["start_sample"] = int(row["start_sample"])
            row["end_sample"] = int(row["end_sample"])
            row["run_local_start"] = int(row["run_local_start"])
            row["run_local_end"] = int(row["run_local_end"])
            row["start_time_s"] = float(row["start_time_s"])
            row["end_time_s"] = float(row["end_time_s"])
            row["start_frame"] = int(row["start_frame"])
            row["end_frame"] = int(row["end_frame"])
            row["true_label"] = int(row["true_label"])
            row["used_srt_anchor"] = row["used_srt_anchor"] == "True"
            row["video_fps"] = float(row.get("video_fps") or 0.0)
            if "sensor_file" not in row and "dat_file" in row:
                row["sensor_file"] = row["dat_file"]
            rows.append(row)
    return rows


def build_windows_from_index(X, index_rows, window=OPP_WINDOW):
    N = X.shape[0]
    C = X.shape[1]
    n_win = len(index_rows)
    out = np.empty((n_win, window, C), dtype=np.float32)
    bad_idx = []
    for i, row in enumerate(index_rows):
        s = row["start_sample"]
        e = s + window
        if e > N:
            bad_idx.append((i, s, e, N))
            slab = np.zeros((window, C), dtype=np.float32)
            avail = max(0, N - s)
            if avail > 0:
                slab[:avail] = X[s : s + avail]
            out[i] = slab
        else:
            out[i] = X[s:e]
    if bad_idx:
        first = bad_idx[0]
        raise RuntimeError(
            f"[FATAL] {len(bad_idx)} CSV rows reference samples past the end of the .mat sensor array (e.g. row {first[0]}: start_sample={first[1]}, end={first[2]}, but N={first[3]}). The .mat and CSV are out of sync — re-run prepare_opportunity_plus.py."
        )
    return out


def run_inference(model, X_windows, batch_size, device):
    model.eval()
    preds = np.empty(X_windows.shape[0], dtype=np.int64)
    confs = np.empty(X_windows.shape[0], dtype=np.float32)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i in range(0, X_windows.shape[0], batch_size):
            batch = torch.from_numpy(X_windows[i : i + batch_size]).to(device)
            _z, logits, _attn = model(batch)
            probs = softmax(logits)
            conf, pred = torch.max(probs, dim=1)
            preds[i : i + batch.shape[0]] = pred.cpu().numpy()
            confs[i : i + batch.shape[0]] = conf.cpu().numpy()
            if i // batch_size % 50 == 0:
                print(
                    f"    batch {i // batch_size + 1}/{(X_windows.shape[0] + batch_size - 1) // batch_size}",
                    flush=True,
                )
    return (preds, confs)


def extract_clip(video_path, start_frame, end_frame, fps, out_path):
    start_t = start_frame / float(fps)
    duration = max(1.0 / fps, (end_frame - start_frame + 1) / float(fps))
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_t:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-an",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    [WARN] ffmpeg failed for {out_path}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="./dataset/opportunity_plus_S4.mat")
    ap.add_argument("--index_csv", default="./dataset/opportunity_plus_S4_index.csv")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint_best.pth for the trained `opportunity` model. Auto-detected if omitted.",
    )
    ap.add_argument(
        "--opp_plus_root",
        required=True,
        help="Opportunity++ root, for resolving video paths.",
    )
    ap.add_argument("--output_dir", default="./results/opportunity_plus")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument(
        "--video_fps",
        type=float,
        default=30.0,
        help="Fallback frame rate if the index CSV's video_fps column is missing or zero. Modern prepare scripts embed per-run fps from ffprobe.",
    )
    ap.add_argument(
        "--no_clips",
        action="store_true",
        help="Skip clip extraction; only write predictions.csv.",
    )
    ap.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Optional cap on number of clips to extract (debug).",
    )
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Loading {args.mat}")
    mat = sio.loadmat(args.mat)
    X = mat["testingData"].astype(np.float32).T
    y = mat["testingLabels"].reshape(-1).astype(np.int64) - 1
    print(
        f"    X = {X.shape}, y = {y.shape}, classes = {sorted(set(y.tolist()))[:5]}..."
    )
    print(f"[*] Loading {args.index_csv}")
    index_rows = load_index_csv(args.index_csv)
    print(f"    {len(index_rows)} window rows")
    print(
        f"[*] Building {len(index_rows)} windows (window={OPP_WINDOW}) from index CSV positions"
    )
    X_win = build_windows_from_index(X, index_rows, window=OPP_WINDOW)
    print(f"    X_win = {X_win.shape}")
    ckpt_path = auto_find_checkpoint(args.checkpoint)
    print(f"[*] Loading checkpoint: {ckpt_path}")
    config_model = dict(
        model="AttendDiscriminate",
        dataset="opportunity",
        input_dim=OPP_INPUT_DIM,
        hidden_dim=OPP_HIDDEN_DIM,
        filter_num=OPP_FILTER_NUM,
        filter_size=OPP_FILTER_SIZE,
        enc_num_layers=OPP_ENC_LAYERS,
        enc_is_bidirectional=OPP_ENC_BIDIR,
        dropout=OPP_DROPOUT,
        dropout_rnn=OPP_DROPOUT_RNN,
        dropout_cls=OPP_DROPOUT_CLS,
        activation=OPP_ACTIVATION,
        sa_div=OPP_SA_DIV,
        num_class=OPP_NUM_CLASS,
        train_mode=False,
        experiment="inference_opportunity_plus",
    )
    model = create("AttendDiscriminate", config_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("    checkpoint loaded.")
    print(
        f"[*] Running inference on {X_win.shape[0]} windows (batch_size={args.batch_size})"
    )
    preds, confs = run_inference(model, X_win, args.batch_size, device)
    y_window = np.array([r["true_label"] for r in index_rows], dtype=np.int64)
    correct = preds == y_window
    acc = 100.0 * correct.mean()
    from sklearn import metrics as skm

    fm = 100.0 * skm.f1_score(y_window, preds, average="macro", zero_division=0)
    fw = 100.0 * skm.f1_score(y_window, preds, average="weighted", zero_division=0)
    print(f"\n[Inference summary]")
    print(f"  windows         = {preds.shape[0]}")
    print(f"  accuracy        = {acc:.2f}%")
    print(f"  F1-macro        = {fm:.2f}%")
    print(f"  F1-weighted     = {fw:.2f}%")
    pred_hist = Counter(preds.tolist())
    print(f"  pred class hist (top 10):")
    for c, n in pred_hist.most_common(10):
        print(f"    {c:2d} {OPP_CLASS_NAMES[c]:<22s} {n:>8d}")
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_csv = os.path.join(args.output_dir, "predictions.csv")
    print(f"\n[*] Writing {predictions_csv}")
    base_fields = list(index_rows[0].keys())
    extra_fields = ["pred_label", "pred_label_name", "confidence", "correct"]
    with open(predictions_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        w.writeheader()
        for i, row in enumerate(index_rows):
            out = dict(row)
            p = int(preds[i])
            out["pred_label"] = p
            out["pred_label_name"] = OPP_CLASS_NAMES[p]
            out["confidence"] = round(float(confs[i]), 4)
            out["correct"] = p == row["true_label"]
            w.writerow(out)
    print(f"    {predictions_csv} written.")
    if args.no_clips:
        print("[*] --no_clips set; skipping clip extraction.")
    else:
        clips_dir = os.path.join(args.output_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        targets = [(i, r) for i, r in enumerate(index_rows) if int(preds[i]) != 0]
        if args.max_clips is not None:
            targets = targets[: args.max_clips]
        print(
            f"[*] Extracting clips for {len(targets)} non-Null predictions (of {len(index_rows)} total windows)"
        )
        n_ok, n_skip, n_fail = (0, 0, 0)
        for k, (i, row) in enumerate(targets):
            if not row["video_file"]:
                n_skip += 1
                continue
            video_abspath = os.path.join(args.opp_plus_root, row["video_file"])
            if not os.path.exists(video_abspath):
                n_skip += 1
                continue
            run_dir = os.path.join(clips_dir, row["run"])
            os.makedirs(run_dir, exist_ok=True)
            pred_name = OPP_CLASS_NAMES[int(preds[i])].replace(" ", "")
            true_name = OPP_CLASS_NAMES[row["true_label"]].replace(" ", "")
            out_path = os.path.join(
                run_dir,
                f"{row['window_idx']:06d}_pred-{pred_name}_true-{true_name}.mp4",
            )
            ok = extract_clip(
                video_abspath,
                row["start_frame"],
                row["end_frame"],
                row["video_fps"] if row.get("video_fps") else args.video_fps,
                out_path,
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if (k + 1) % 500 == 0:
                print(
                    f"    progress: {k + 1}/{len(targets)}  ok={n_ok} skip={n_skip} fail={n_fail}"
                )
        print(f"\n  Clips: ok={n_ok}  skipped(no video)={n_skip}  failed={n_fail}")
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Opportunity++ inference summary\n")
        f.write(f"==================================\n\n")
        f.write(f"checkpoint:    {ckpt_path}\n")
        f.write(f"mat:           {args.mat}\n")
        f.write(f"index_csv:     {args.index_csv}\n")
        f.write(f"windows:       {preds.shape[0]}\n")
        f.write(f"accuracy:      {acc:.4f}%\n")
        f.write(f"F1 macro:      {fm:.4f}%\n")
        f.write(f"F1 weighted:   {fw:.4f}%\n\n")
        f.write(f"Predicted class histogram:\n")
        for c in range(OPP_NUM_CLASS):
            n = int((preds == c).sum())
            f.write(f"  {c:2d} {OPP_CLASS_NAMES[c]:<22s} {n:>8d}\n")
        f.write(f"\nTrue class histogram (over scored windows):\n")
        for c in range(OPP_NUM_CLASS):
            n = int((y_window == c).sum())
            f.write(f"  {c:2d} {OPP_CLASS_NAMES[c]:<22s} {n:>8d}\n")
    print(f"[*] Wrote {summary_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
