import os
import time
import glob
import re
import csv
import random
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from AtteFinalPipeline.baselines.nst.dataset import NSTDataset
from AtteFinalPipeline.baselines.nst.model import NonStationaryTransformer
from AtteFinalPipeline.baselines.nst.settings import get_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_results_dirs(results_dir):
    for sub in [
        "confusion/train",
        "confusion/val",
        "confusion/test",
        "tensorboard/train",
        "tensorboard/val",
        "tensorboard/test",
        "checkpoints",
        "logs",
    ]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)


def save_metrics_txt(path, split, loss, acc, fm, fw, epoch=None):
    tag = f"epoch={epoch}" if epoch is not None else "final"
    with open(path, "a") as f:
        f.write(
            f"[{tag}] {split:5s}  loss={loss:.4f}  acc={acc:.2f}%  F1-macro={fm:.2f}%  F1-weighted={fw:.2f}%\n"
        )


def append_epoch_csv(path, epoch, split, loss, acc, fm, fw):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch", "split", "loss", "acc", "f1_macro", "f1_weighted"])
        w.writerow(
            [epoch, split, f"{loss:.6f}", f"{acc:.4f}", f"{fm:.4f}", f"{fw:.4f}"]
        )


def append_summary_csv(path, row):
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "dataset",
                "drop_null",
                "num_class",
                "test_loss",
                "test_acc",
                "test_f1_macro",
                "test_f1_weighted",
                "best_val_f1_macro",
                "epochs_trained",
                "n_params",
                "results_dir",
            ],
        )
        if new_file:
            w.writeheader()
        w.writerow(row)


def find_latest_checkpoint(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth"))
    best_e, best_p = (-1, None)
    for p in paths:
        m = re.match("checkpoint_(\\d+)\\.pth$", os.path.basename(p))
        if m:
            e = int(m.group(1))
            if e > best_e:
                best_e, best_p = (e, p)
    return (best_p, best_e)


def plot_confusion(y_true, y_pred, out_dir, epoch, class_map):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = len(class_map)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_map, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_map, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion (epoch={epoch})")
    for i in range(n):
        for j in range(n):
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
    fig.savefig(os.path.join(out_dir, f"cm_epoch_{epoch}.png"), dpi=120)
    plt.close(fig)
    np.save(os.path.join(out_dir, f"cm_epoch_{epoch}.npy"), cm)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def __str__(self):
        return f"{self.avg:.4f}"


def train_one_epoch(model, loader, criterion, optimizer, args):
    model.train()
    losses = AverageMeter()
    for batch_idx, (data, target, _) in enumerate(loader):
        data = data.cuda(non_blocking=True)
        target = target.view(-1).cuda(non_blocking=True)
        _, logits = model(data)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))
        if batch_idx % args.print_freq == 0:
            print(f"[-] Batch {batch_idx}/{len(loader)}  Loss: {losses}")


def eval_one_epoch(model, loader, criterion, epoch, logger, args, cm_dir):
    model.eval()
    losses = AverageMeter()
    y_true_chunks, y_pred_chunks = ([], [])
    with torch.no_grad():
        for data, target, _ in loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            _, logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.size(0))
            preds = torch.argmax(logits, dim=1)
            y_pred_chunks.append(preds.cpu().numpy().reshape(-1))
            y_true_chunks.append(target.cpu().numpy().reshape(-1))
    is_test = loader.dataset.prefix == "test"
    is_capture24 = args.dataset == "capture24"
    if is_test and (not is_capture24):
        ws = loader.dataset.window - 1
        first_label = y_true_chunks[0][0]
        pad = np.full(ws, first_label, dtype=np.int64)
        y_true_chunks.append(pad)
        y_pred_chunks.append(pad)
    y_true = np.concatenate(y_true_chunks, 0)
    y_pred = np.concatenate(y_pred_chunks, 0)
    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    fw = 100.0 * metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if logger is not None:
        logger.add_scalar("Loss", losses.avg, epoch)
        logger.add_scalar("Acc", acc, epoch)
        logger.add_scalar("Fm", fm, epoch)
        logger.add_scalar("Fw", fw, epoch)
    save_cm = epoch == -1 or epoch % 10 == 0
    if save_cm and cm_dir is not None:
        plot_confusion(y_true, y_pred, cm_dir, epoch, args.class_map)
    return (losses.avg, acc, fm, fw)


def main():
    args = get_args()
    set_seed(args.seed)
    make_results_dirs(args.results_dir)
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    metrics_log = os.path.join(args.results_dir, "metrics.txt")
    metrics_csv = os.path.join(args.results_dir, "metrics.csv")
    config_dump = os.path.join(args.results_dir, "config.json")
    summary_csv = os.path.join(
        os.path.dirname(args.results_dir.rstrip("/")), "results.csv"
    )
    with open(config_dump, "w") as f:
        cfg = {
            k: v
            for k, v in vars(args).items()
            if not k.startswith("_")
            and isinstance(v, (int, float, str, bool, list, type(None)))
        }
        json.dump(cfg, f, indent=2)
    print("=" * 60)
    print(f"NST training on {args.dataset}  (drop_null={args.drop_null})")
    print(f"  input_dim={args.input_dim}  num_class={args.num_class}")
    print(
        f"  window={args.window}  stride={args.stride}  stride_test={args.stride_test}"
    )
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  e_layers={args.e_layers}")
    print(f"  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}")
    print(f"  results_dir={args.results_dir}")
    print("=" * 60)
    common_kwargs = dict(
        dataset=args.dataset,
        window=args.window,
        stride=args.stride,
        stride_test=args.stride_test,
        path_processed=args.path_processed,
        drop_null=args.drop_null,
    )
    ds_train = NSTDataset(**common_kwargs, prefix="train")
    ds_val = NSTDataset(**common_kwargs, prefix="val")
    ds_test = NSTDataset(**common_kwargs, prefix="test")
    print(f"[*] train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
    for split_name, ds in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        if len(ds) > 0:
            tmin, tmax = (int(ds.target.min()), int(ds.target.max()))
            assert tmin >= 0 and tmax < args.num_class, (
                f"{split_name} label range [{tmin}, {tmax}] doesn't fit num_class={args.num_class}"
            )
    if args.weighted_sampler:
        sampler = WeightedRandomSampler(
            ds_train.weight_samples, len(ds_train.weight_samples)
        )
        loader_train = DataLoader(
            ds_train, args.batch_size, sampler=sampler, pin_memory=True, num_workers=2
        )
    else:
        loader_train = DataLoader(
            ds_train, args.batch_size, shuffle=True, pin_memory=True, num_workers=2
        )
    loader_val = DataLoader(
        ds_val, args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )
    loader_test = DataLoader(
        ds_test, args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )
    model = NonStationaryTransformer(
        input_dim=args.input_dim,
        seq_len=args.window,
        num_class=args.num_class,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        projector_hidden=args.projector_hidden,
    ).cuda()
    n_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[*] NST params: {n_params:,}")
    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_decay
    )
    tb_train = SummaryWriter(
        log_dir=os.path.join(args.results_dir, "tensorboard/train")
    )
    tb_val = SummaryWriter(log_dir=os.path.join(args.results_dir, "tensorboard/val"))
    tb_test = SummaryWriter(log_dir=os.path.join(args.results_dir, "tensorboard/test"))
    start_epoch = 0
    metric_best = 0.0
    if args.resume:
        ckpt_path, ckpt_epoch = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            print(f"[*] Resuming from {ckpt_path} (epoch {ckpt_epoch})")
            ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])
            for _ in range(ckpt_epoch + 1):
                scheduler.step()
            start_epoch = ckpt_epoch + 1
            best_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
            if os.path.exists(best_path):
                tmp = {k: v.clone() for k, v in model.state_dict().items()}
                bckpt = torch.load(best_path, map_location="cuda", weights_only=False)
                model.load_state_dict(bckpt["model"])
                _, _, fm_best, _ = eval_one_epoch(
                    model,
                    loader_val,
                    criterion,
                    epoch=start_epoch - 1,
                    logger=None,
                    args=args,
                    cm_dir=None,
                )
                metric_best = fm_best
                model.load_state_dict(tmp)
                print(f"[*] metric_best (val fm_macro) = {metric_best:.2f}")
    end_epoch = (
        start_epoch + args.extra_epochs
        if args.resume and start_epoch > 0
        else args.epochs
    )
    last_epoch_run = start_epoch - 1
    if args.train_mode:
        epochs_no_improve = 0
        t0 = time.time()
        print(f"[*] Training from epoch {start_epoch} to {end_epoch - 1}")
        for epoch in range(start_epoch, end_epoch):
            print("--" * 50)
            print(f"[-] Epoch {epoch}  lr={optimizer.param_groups[0]['lr']:.2e}")
            train_one_epoch(model, loader_train, criterion, optimizer, args)
            tr_loss, tr_acc, tr_fm, tr_fw = eval_one_epoch(
                model,
                loader_train,
                criterion,
                epoch,
                tb_train,
                args,
                cm_dir=os.path.join(args.results_dir, "confusion/train"),
            )
            v_loss, v_acc, v_fm, v_fw = eval_one_epoch(
                model,
                loader_val,
                criterion,
                epoch,
                tb_val,
                args,
                cm_dir=os.path.join(args.results_dir, "confusion/val"),
            )
            tb_train.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            print(
                f"[-] Train loss={tr_loss:.4f} acc={tr_acc:.2f} fm={tr_fm:.2f} fw={tr_fw:.2f}"
            )
            print(
                f"[-] Val   loss={v_loss:.4f} acc={v_acc:.2f} fm={v_fm:.2f} fw={v_fw:.2f}"
            )
            save_metrics_txt(metrics_log, "train", tr_loss, tr_acc, tr_fm, tr_fw, epoch)
            save_metrics_txt(metrics_log, "val", v_loss, v_acc, v_fm, v_fw, epoch)
            append_epoch_csv(metrics_csv, epoch, "train", tr_loss, tr_acc, tr_fm, tr_fw)
            append_epoch_csv(metrics_csv, epoch, "val", v_loss, v_acc, v_fm, v_fw)
            ckpt = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
            }
            if v_fm >= metric_best:
                print(f"[*] new best fm_val: {metric_best:.2f} -> {v_fm:.2f}")
                metric_best = v_fm
                epochs_no_improve = 0
                torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best.pth"))
            else:
                epochs_no_improve += 1
                print(
                    f"[-] no improvement ({epochs_no_improve}/{args.patience}) best={metric_best:.2f}"
                )
            if epoch % 5 == 0:
                torch.save(ckpt, os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth"))
            scheduler.step()
            last_epoch_run = epoch
            if epochs_no_improve >= args.patience:
                print(
                    f"[!] early stopping at epoch {epoch}, best fm_val={metric_best:.2f}"
                )
                torch.save(ckpt, os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth"))
                break
        elapsed = str(datetime.timedelta(seconds=round(time.time() - t0)))
        print(f"[*] training time: {elapsed}")
    print("=" * 60)
    print("[*] final test evaluation (loading checkpoint_best.pth)")
    best_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
    if os.path.exists(best_path):
        bckpt = torch.load(best_path, map_location="cuda", weights_only=False)
        model.load_state_dict(bckpt["model"])
    else:
        print("[!] no best checkpoint found, using current weights")
    t_loss, t_acc, t_fm, t_fw = eval_one_epoch(
        model,
        loader_test,
        criterion,
        epoch=-1,
        logger=tb_test,
        args=args,
        cm_dir=os.path.join(args.results_dir, "confusion/test"),
    )
    print("=" * 60)
    print(f"[TEST] dataset={args.dataset}  drop_null={args.drop_null}")
    print(
        f"[TEST] loss={t_loss:.4f}  acc={t_acc:.2f}%  f1_macro={t_fm:.2f}%  f1_weighted={t_fw:.2f}%"
    )
    print("=" * 60)
    save_metrics_txt(metrics_log, "test", t_loss, t_acc, t_fm, t_fw)
    append_epoch_csv(metrics_csv, -1, "test", t_loss, t_acc, t_fm, t_fw)
    append_summary_csv(
        summary_csv,
        {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "drop_null": bool(args.drop_null),
            "num_class": args.num_class,
            "test_loss": f"{t_loss:.6f}",
            "test_acc": f"{t_acc:.4f}",
            "test_f1_macro": f"{t_fm:.4f}",
            "test_f1_weighted": f"{t_fw:.4f}",
            "best_val_f1_macro": f"{metric_best:.4f}",
            "epochs_trained": last_epoch_run + 1,
            "n_params": n_params,
            "results_dir": args.results_dir,
        },
    )
    print(f"[*] summary appended to {summary_csv}")
    tb_train.close()
    tb_val.close()
    tb_test.close()


if __name__ == "__main__":
    main()
