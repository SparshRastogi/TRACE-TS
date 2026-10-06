import os
import sys
import time
import json
import random
import datetime
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn import metrics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AtteFinalPipeline.baselines.deepconvlstm.settings import get_args
from AtteFinalPipeline.baselines.deepconvlstm.model import DeepConvLSTM, init_weights
from AtteFinalPipeline.baselines.chronos.dataset import load_split


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, prefix: str = "train"):
        assert x.ndim == 3, f"expected (N,T,C), got {x.shape}"
        assert x.shape[0] == y.shape[0]
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.len = x.shape[0]
        self.prefix = prefix
        if prefix == "train":
            self.weight_samples = self._compute_weights()
        print(
            f"[WindowDataset/{prefix}] x={self.x.shape}  y={self.y.shape}  classes={sorted(np.unique(self.y).tolist())}"
        )

    def _compute_weights(self):
        labels = self.y
        counts = np.array([np.sum(labels == c) for c in sorted(set(labels.tolist()))])
        inv = {c: 1.0 / cnt for c, cnt in zip(sorted(set(labels.tolist())), counts)}
        w = np.array([inv[int(t)] for t in labels], dtype=np.float64)
        return torch.from_numpy(w).double()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return (x, y, torch.tensor(idx, dtype=torch.long))


def make_results_dirs(results_dir: str):
    for sub in ("confusion", "checkpoints"):
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)


def append_metric(path: str, split: str, loss, acc, fm, fw, epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tag = f"epoch={epoch}" if epoch is not None else "final"
    with open(path, "a") as f:
        f.write(
            f"[{tag}] {split:5s}  loss={loss:.4f}  acc={acc:.2f}%  F1-macro={fm:.2f}%  F1-weighted={fw:.2f}%\n"
        )


def plot_cm(y_true, y_pred, class_map, out_path: str, title: str):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_map))))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(
        figsize=(max(6, len(class_map) * 0.5), max(5, len(class_map) * 0.5))
    )
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_map)))
    ax.set_yticks(range(len(class_map)))
    ax.set_xticklabels(class_map, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_map, fontsize=7)
    for i in range(len(class_map)):
        for j in range(len(class_map)):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def eval_loader(
    model, loader, criterion, device, args=None, prefix: str = "val"
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum, loss_n = (0.0, 0)
    y_true_all, y_pred_all = ([], [])
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y.view(-1))
            bs = x.shape[0]
            loss_sum += loss.item() * bs
            loss_n += bs
            preds = torch.argmax(logits, dim=1)
            y_pred_all.append(preds.cpu().numpy().reshape(-1))
            y_true_all.append(y.cpu().numpy().reshape(-1))
    if prefix == "test" and args is not None and ("capture24" not in args.dataset):
        ws = args.window - 1
        if ws > 0 and len(y_true_all) > 0:
            filler = np.full(ws, y_true_all[0][0], dtype=np.int64)
            y_true_all.append(filler)
            y_pred_all.append(filler)
    y_true = np.concatenate(y_true_all, 0)
    y_pred = np.concatenate(y_pred_all, 0)
    avg_loss = loss_sum / max(loss_n, 1)
    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    fw = 100.0 * metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return (avg_loss, acc, fm, fw, y_true, y_pred)


def train_one_epoch(model, loader, criterion, optimizer, device, log_every: int = 100):
    model.train()
    running, n = (0.0, 0)
    for batch_idx, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.shape[0]
        n += x.shape[0]
        if batch_idx % log_every == 0:
            print(f"    [batch {batch_idx:4d}/{len(loader)}]  loss={loss.item():.4f}")
    return running / max(n, 1)


def main():
    args = get_args()
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device = {device}  |  dataset = {args.dataset}")
    print(f"[main] num_class = {args.num_class}  |  class_map = {args.class_map}")
    print(f"[main] results_dir = {args.results_dir}")
    make_results_dirs(args.results_dir)
    metrics_log = os.path.join(args.results_dir, "metrics.txt")
    if os.path.exists(metrics_log):
        os.remove(metrics_log)
    with open(metrics_log, "w") as f:
        f.write(f"# DeepConvLSTM baseline  |  seed={args.seed}\n")
        f.write(f"# Run started: {datetime.datetime.now().isoformat()}\n")
        f.write(
            f"# args: {json.dumps({k: str(v) for k, v in sorted(vars(args).items())}, indent=2)}\n"
        )
    x_train, y_train = load_split(args, "train")
    x_val, y_val = load_split(args, "val")
    x_test, y_test = load_split(args, "test")
    ds_train = WindowDataset(x_train, y_train, prefix="train")
    ds_val = WindowDataset(x_val, y_val, prefix="val")
    ds_test = WindowDataset(x_test, y_test, prefix="test")
    if args.weighted_sampler:
        print("[main] using WeightedRandomSampler on train (matches A&D)")
        sampler = WeightedRandomSampler(
            ds_train.weight_samples, len(ds_train.weight_samples)
        )
        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=2,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    model_config = {
        "window_size": args.window,
        "nb_channels": args.input_dim,
        "nb_classes": args.num_class,
        "nb_conv_blocks": args.nb_conv_blocks,
        "nb_filters": args.nb_filters,
        "filter_width": args.filter_width,
        "dilation": args.dilation,
        "batch_norm": args.batch_norm,
        "nb_units_lstm": args.nb_units_lstm,
        "nb_layers_lstm": args.nb_layers_lstm,
        "drop_prob": args.drop_prob,
        "weights_init": args.weights_init,
        "seed": args.seed,
    }
    final_seq_len = args.window - (args.filter_width - 1) * (args.nb_conv_blocks * 2)
    if final_seq_len < 1:
        print(
            f"[ERROR] window={args.window} too short for nb_conv_blocks={args.nb_conv_blocks}, filter_width={args.filter_width} (final_seq_len={final_seq_len}). Reduce nb_conv_blocks or filter_width."
        )
        sys.exit(1)
    model = DeepConvLSTM(model_config)
    init_weights(model)
    model = model.to(device)
    n_params = model.number_of_parameters()
    print(f"[main] DeepConvLSTM params (trainable) = {n_params:,}")
    print(f"[main] final_seq_len after convolutions = {final_seq_len}")
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_decay
    )
    metric_best = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    print("=" * 72)
    print(
        f"[train] starting DeepConvLSTM training (epochs={args.epochs}, patience={args.patience})"
    )
    print("=" * 72)
    t_start = time.time()
    for epoch in range(args.epochs):
        print(
            f"\n── epoch {epoch:3d}/{args.epochs - 1}  lr={optimizer.param_groups[0]['lr']:.2e} ──"
        )
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_eval_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        loss_t, acc_t, fm_t, fw_t, _, _ = eval_loader(
            model, train_eval_loader, criterion, device, args=args, prefix="train"
        )
        loss_v, acc_v, fm_v, fw_v, _, _ = eval_loader(
            model, val_loader, criterion, device, args=args, prefix="val"
        )
        print(
            f"   train  loss={loss_t:.4f}  acc={acc_t:.2f}%  fm={fm_t:.2f}%  fw={fw_t:.2f}%"
        )
        print(
            f"   val    loss={loss_v:.4f}  acc={acc_v:.2f}%  fm={fm_v:.2f}%  fw={fw_v:.2f}%"
        )
        append_metric(metrics_log, "train", loss_t, acc_t, fm_t, fw_t, epoch=epoch)
        append_metric(metrics_log, "val", loss_v, acc_v, fm_v, fw_v, epoch=epoch)
        if fm_v > metric_best:
            print(f"   [best ↑] fm_val {metric_best:.2f} → {fm_v:.2f}")
            metric_best = fm_v
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {"model_state_dict": best_state, "epoch": epoch, "fm_val": fm_v},
                os.path.join(args.results_dir, "checkpoints", "checkpoint_best.pth"),
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"   [no improve] {epochs_no_improve}/{args.patience}  (best fm_val={metric_best:.2f} @ epoch {best_epoch})"
            )
        scheduler.step()
        if epochs_no_improve >= args.patience:
            print(
                f"\n[early stopping] no improvement for {args.patience} epochs at epoch {epoch}"
            )
            break
    elapsed = str(datetime.timedelta(seconds=round(time.time() - t_start)))
    print(
        f"\n[train] done in {elapsed}.  best fm_val = {metric_best:.2f} @ epoch {best_epoch}"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
        print("[test] loaded best checkpoint")
    loss_te, acc_te, fm_te, fw_te, y_true, y_pred = eval_loader(
        model, test_loader, criterion, device, args=args, prefix="test"
    )
    print()
    print("█" * 72)
    print(f"█  FINAL TEST RESULTS — {args.dataset} — DeepConvLSTM")
    print("█" * 72)
    print(f"█  loss        = {loss_te:.4f}")
    print(f"█  accuracy    = {acc_te:.2f}%")
    print(f"█  F1-macro    = {fm_te:.2f}%")
    print(f"█  F1-weighted = {fw_te:.2f}%")
    print(f"█  num_class   = {args.num_class}")
    print(f"█  best_epoch  = {best_epoch}   (best fm_val = {metric_best:.2f}%)")
    print("█" * 72)
    append_metric(metrics_log, "test", loss_te, acc_te, fm_te, fw_te)
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "model": "DeepConvLSTM (Bock et al.)",
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_fm_val": metric_best,
                "test_acc": acc_te,
                "test_fm": fm_te,
                "test_fw": fw_te,
                "test_loss": loss_te,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
                "num_class": args.num_class,
                "class_map": args.class_map,
                "n_params": n_params,
                "model_config": model_config,
            },
            f,
            indent=2,
        )
    print(f"[main] wrote {summary_path}")
    cm_title = f"{args.dataset} — DeepConvLSTM — test (acc={acc_te:.1f}, fm={fm_te:.1f}, fw={fw_te:.1f})"
    plot_cm(
        y_true,
        y_pred,
        args.class_map,
        out_path=os.path.join(args.results_dir, "confusion", "test_cm.png"),
        title=cm_title,
    )
    print("[main] done.")


if __name__ == "__main__":
    main()
