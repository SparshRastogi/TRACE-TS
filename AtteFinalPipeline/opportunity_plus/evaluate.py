import os
import sys
import csv
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from collections import defaultdict
import scipy.io as sio
from AtteFinalPipeline.expert.model import create

OPP_WINDOW = 24
INPUT_DIM = 79
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
OPP_NONULL_CLASS_NAMES = OPP_CLASS_NAMES[1:]


def make_config(num_class, experiment):
    return dict(
        model="AttendDiscriminate",
        dataset="opportunity",
        input_dim=INPUT_DIM,
        hidden_dim=128,
        filter_num=64,
        filter_size=5,
        enc_num_layers=2,
        enc_is_bidirectional=False,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
        activation="ReLU",
        sa_div=1,
        num_class=num_class,
        train_mode=False,
        experiment=experiment,
    )


def find_checkpoint(path_or_root):
    if os.path.isfile(path_or_root):
        return path_or_root
    if not os.path.isdir(path_or_root):
        raise FileNotFoundError(f"Path does not exist: {path_or_root}")
    direct = [
        os.path.join(path_or_root, "checkpoint_best.pth"),
        os.path.join(path_or_root, "checkpoints", "checkpoint_best.pth"),
    ]
    for p in direct:
        if os.path.isfile(p):
            return p
    hits = glob.glob(
        os.path.join(path_or_root, "*", "checkpoints", "checkpoint_best.pth")
    )
    if not hits:
        raise FileNotFoundError(
            f"No checkpoint_best.pth found at any of:\n  "
            + "\n  ".join(
                direct + [path_or_root + "/*/checkpoints/checkpoint_best.pth"]
            )
        )
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


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
            row["true_label"] = int(row["true_label"])
            rows.append(row)
    return rows


def build_windows_from_index(X, index_rows, window=OPP_WINDOW):
    N, C = X.shape
    out = np.empty((len(index_rows), window, C), dtype=np.float32)
    for i, row in enumerate(index_rows):
        s = row["start_sample"]
        e = s + window
        if e > N:
            raise RuntimeError(
                f"CSV row {i} references samples past the end of X (start={s}, end={e}, N={N}). Re-run prepare_opportunity_plus.py."
            )
        out[i] = X[s:e]
    return out


def run_inference(model, X_windows, batch_size, device):
    model.eval()
    preds = np.empty(X_windows.shape[0], dtype=np.int64)
    confs = np.empty(X_windows.shape[0], dtype=np.float32)
    softmax = nn.Softmax(dim=1)
    n_batches = (X_windows.shape[0] + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, X_windows.shape[0], batch_size):
            batch = torch.from_numpy(X_windows[i : i + batch_size]).to(device)
            _z, logits, _attn = model(batch)
            probs = softmax(logits)
            c, p = torch.max(probs, dim=1)
            preds[i : i + batch.shape[0]] = p.cpu().numpy()
            confs[i : i + batch.shape[0]] = c.cpu().numpy()
            if i // batch_size % 50 == 0:
                print(f"    batch {i // batch_size + 1}/{n_batches}", flush=True)
    return (preds, confs)


def report(
    name, y_true, y_pred, class_names, index_rows=None, keep_mask=None, top_confusions=5
):
    n_class = len(class_names)
    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(
        y_true, y_pred, average="macro", labels=list(range(n_class)), zero_division=0
    )
    fw = 100.0 * metrics.f1_score(
        y_true, y_pred, average="weighted", labels=list(range(n_class)), zero_division=0
    )
    print()
    print("━" * 72)
    print(f"  {name}")
    print("━" * 72)
    print(f"  windows scored : {len(y_true)}")
    print(f"  accuracy       : {acc:6.2f} %")
    print(f"  F1-macro       : {fm:6.2f} %")
    print(f"  F1-weighted    : {fw:6.2f} %")
    f1_per = metrics.f1_score(
        y_true, y_pred, average=None, labels=list(range(n_class)), zero_division=0
    )
    support_per = np.array([int((y_true == c).sum()) for c in range(n_class)])
    pred_count_per = np.array([int((y_pred == c).sum()) for c in range(n_class)])
    print(f"\n  Per-class F1:")
    print(f"  {'cls':>3}  {'name':<22s}  {'F1%':>7s}  {'support':>8s}  {'pred#':>8s}")
    for c in range(n_class):
        sup = support_per[c]
        flag = ""
        if sup > 0 and f1_per[c] == 0.0:
            flag = "  ← F1=0 with support>0!"
        elif sup == 0 and pred_count_per[c] > 0:
            flag = "  ← predicted but no truth"
        print(
            f"  {c:>3}  {class_names[c]:<22s}  {100 * f1_per[c]:6.2f}   {sup:>8d}  {pred_count_per[c]:>8d}{flag}"
        )
    if index_rows is not None:
        if keep_mask is not None:
            assert len(keep_mask) == len(index_rows)
            rows_used = [r for r, k in zip(index_rows, keep_mask) if k]
        else:
            rows_used = index_rows
        assert len(rows_used) == len(y_true), (
            f"index rows used ({len(rows_used)}) != metrics len ({len(y_true)})"
        )
        by_run = defaultdict(list)
        for idx, r in enumerate(rows_used):
            by_run[r["run"]].append(idx)
        print(f"\n  Per-run breakdown:")
        print(f"  {'run':<10s}  {'N':>8s}  {'acc%':>7s}  {'F1m%':>7s}  {'F1w%':>7s}")
        for run in sorted(by_run.keys()):
            ids = np.array(by_run[run])
            yt = y_true[ids]
            yp = y_pred[ids]
            if len(yt) == 0:
                continue
            a = 100.0 * metrics.accuracy_score(yt, yp)
            m = 100.0 * metrics.f1_score(
                yt, yp, average="macro", labels=list(range(n_class)), zero_division=0
            )
            w = 100.0 * metrics.f1_score(
                yt, yp, average="weighted", labels=list(range(n_class)), zero_division=0
            )
            print(f"  {run:<10s}  {len(yt):>8d}  {a:>6.2f}   {m:>6.2f}   {w:>6.2f}")
    if top_confusions > 0:
        cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_class)))
        off = []
        for i in range(n_class):
            for j in range(n_class):
                if i != j and cm[i, j] > 0:
                    off.append((cm[i, j], i, j))
        off.sort(reverse=True)
        if off:
            print(
                f"\n  Top {min(top_confusions, len(off))} confusions (true → pred, count):"
            )
            for cnt, ti, pj in off[:top_confusions]:
                print(f"    {class_names[ti]:<22s} → {class_names[pj]:<22s}  {cnt:>6d}")
    print()
    return dict(accuracy=acc, f1_macro=fm, f1_weighted=fw)


def evaluate_18class(mat_path, index_rows, ckpt_path, batch_size, device):
    print(f"\n[*] Loading {mat_path}")
    mat = sio.loadmat(mat_path)
    X = mat["testingData"].astype(np.float32).T
    y_sample = mat["testingLabels"].reshape(-1).astype(np.int64) - 1
    if X.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"opportunity_plus_*.mat has {X.shape[1]} channels, expected {INPUT_DIM}."
        )
    X_win = build_windows_from_index(X, index_rows, window=OPP_WINDOW)
    y_window = np.array([r["true_label"] for r in index_rows], dtype=np.int64)
    print(f"    X = {X.shape},  X_win = {X_win.shape},  y_window = {y_window.shape}")
    sampled_check = min(1000, len(index_rows))
    take = np.linspace(0, len(index_rows) - 1, sampled_check).astype(int)
    csv_labels = np.array([index_rows[i]["true_label"] for i in take])
    mat_labels = np.array([y_sample[index_rows[i]["end_sample"]] for i in take])
    if not np.array_equal(csv_labels, mat_labels):
        n_mismatch = int((csv_labels != mat_labels).sum())
        raise RuntimeError(
            f"CSV/MAT label mismatch on {n_mismatch}/{sampled_check} sampled windows — CSV true_label != y[end_sample] in the .mat. The index CSV is out of sync with the .mat. Re-run prepare_opportunity_plus.py."
        )
    print(
        f"    cross-check: {sampled_check} sampled windows have CSV.true_label == mat.y[end_sample] ✓"
    )
    print(f"[*] Loading 18-class checkpoint: {ckpt_path}")
    config = make_config(num_class=18, experiment="eval_oppplus_18cls")
    model = create("AttendDiscriminate", config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[*] Inference on {X_win.shape[0]} windows")
    preds, confs = run_inference(model, X_win, batch_size, device)
    r_all = report(
        "opportunity (18-class) on Opportunity++ — ALL windows",
        y_window,
        preds,
        OPP_CLASS_NAMES,
        index_rows=index_rows,
        keep_mask=None,
    )
    keep = y_window != 0
    yt = y_window[keep]
    yp = preds[keep]
    print()
    print("    (Bonus view: drop rows where true=Null. Counts pred=Null as error.)")
    r_nonull = report(
        "opportunity (18-class) on Opportunity++ — NON-NULL truths only",
        yt,
        yp,
        OPP_CLASS_NAMES,
        index_rows=index_rows,
        keep_mask=keep,
    )
    return {"all": r_all, "nonull": r_nonull, "preds": preds, "y": y_window}


def evaluate_17class(mat_path, index_rows, ckpt_path, batch_size, device):
    print(f"\n[*] Loading {mat_path}")
    mat = sio.loadmat(mat_path)
    X = mat["testingData"].astype(np.float32).T
    if X.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"opportunity_plus_*.mat has {X.shape[1]} channels, expected {INPUT_DIM}."
        )
    X_win = build_windows_from_index(X, index_rows, window=OPP_WINDOW)
    y_window = np.array([r["true_label"] for r in index_rows], dtype=np.int64)
    keep = y_window != 0
    X_win_kept = X_win[keep]
    y_remapped = (y_window[keep] - 1).astype(np.int64)
    print(f"    Kept {keep.sum()}/{len(keep)} windows (true != Null)")
    print(f"    Label range after remap: [{y_remapped.min()}, {y_remapped.max()}]")
    print(f"[*] Loading 17-class checkpoint: {ckpt_path}")
    config = make_config(num_class=17, experiment="eval_oppplus_17cls")
    model = create("AttendDiscriminate", config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[*] Inference on {X_win_kept.shape[0]} windows")
    preds, confs = run_inference(model, X_win_kept, batch_size, device)
    r = report(
        "opportunity_nonull (17-class) on Opportunity++ (null-truth windows dropped, labels remapped 1..17 → 0..16)",
        y_remapped,
        preds,
        OPP_NONULL_CLASS_NAMES,
        index_rows=index_rows,
        keep_mask=keep,
    )
    return {"main": r, "preds": preds, "y": y_remapped}


def main():
    ap = argparse.ArgumentParser(
        description="Sanity-check Opportunity++ preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--mat",
        default="./dataset/opportunity_plus_S2S3_ADL45.mat",
        help="Opportunity++ .mat (produced by prepare_opportunity_plus.py).",
    )
    ap.add_argument(
        "--index_csv",
        default="./dataset/opportunity_plus_S2S3_ADL45_index.csv",
        help="Per-window index CSV produced alongside the .mat.",
    )
    ap.add_argument(
        "--ckpt_opportunity",
        default="./models/opportunity",
        help="18-class checkpoint or its parent folder.",
    )
    ap.add_argument(
        "--ckpt_opportunity_nonull",
        default="./models/opportunity_nonull/train_opportunity_nonull/checkpoints/checkpoint_best.pth",
        help="17-class checkpoint or its parent folder.",
    )
    ap.add_argument(
        "--only", choices=["both", "opportunity", "opportunity_nonull"], default="both"
    )
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.mat):
        sys.exit(
            f"[FATAL] {args.mat} not found — run prepare_opportunity_plus.py first."
        )
    if not os.path.exists(args.index_csv):
        sys.exit(
            f"[FATAL] {args.index_csv} not found — run prepare_opportunity_plus.py first."
        )
    print(f"[*] Loading {args.index_csv}")
    index_rows = load_index_csv(args.index_csv)
    print(f"    {len(index_rows)} window rows")
    results = {}
    if args.only in ("both", "opportunity"):
        ckpt = find_checkpoint(args.ckpt_opportunity)
        results["opportunity"] = evaluate_18class(
            args.mat, index_rows, ckpt, args.batch_size, device
        )
    if args.only in ("both", "opportunity_nonull"):
        ckpt = find_checkpoint(args.ckpt_opportunity_nonull)
        results["opportunity_nonull"] = evaluate_17class(
            args.mat, index_rows, ckpt, args.batch_size, device
        )
    print()
    print("=" * 72)
    print("[SUMMARY — Opportunity++ sanity check]")
    print("=" * 72)
    print(f"  {'view':<58s}  {'acc%':>6s}  {'F1m%':>6s}")
    print("  " + "-" * 70)
    if "opportunity" in results:
        r = results["opportunity"]["all"]
        print(
            f"  {'opportunity 18-cls — all windows':<58s}  {r['accuracy']:>5.2f}   {r['f1_macro']:>5.2f}"
        )
        r = results["opportunity"]["nonull"]
        print(
            f"  {'opportunity 18-cls — non-Null truths only':<58s}  {r['accuracy']:>5.2f}   {r['f1_macro']:>5.2f}"
        )
    if "opportunity_nonull" in results:
        r = results["opportunity_nonull"]["main"]
        print(
            f"  {'opportunity_nonull 17-cls — null windows dropped':<58s}  {r['accuracy']:>5.2f}   {r['f1_macro']:>5.2f}"
        )
    print("=" * 72)
    print()
    print("How to read these numbers")
    print("-" * 72)
    print("On the original A&D test split (S2-ADL4, S2-ADL5, S3-ADL4, S3-ADL5):")
    print("  these should be close to the metrics each model reports on its own")
    print("  test set. A large gap (>~5 pct points) points at a preprocessing bug.")
    print()
    print("On the cross-subject S4 split:")
    print("  expect LOWER numbers (S4 was never seen during training). The per-")
    print("  class F1 distribution should still be sensible — not 16 of 17")
    print("  classes pinned at F1=0. Look at the per-class table to localise.")
    print()
    print("Class names ARE NOW CORRECT (numerically-sorted gesture-ID ordering).")
    print("Pair the predicted class index with OPP_CLASS_NAMES[idx] for video")
    print("annotation.")


if __name__ == "__main__":
    main()
