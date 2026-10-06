import os
import time
import json
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from AtteFinalPipeline.baselines.patchtst.settings import get_args
from AtteFinalPipeline.baselines.patchtst.dataset import PatchTSTSensorDataset
from AtteFinalPipeline.baselines.patchtst.model import create_patchtst
from AtteFinalPipeline.expert.utils.mixup import mixup_data, MixUpLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_results_dirs(results_dir: str):
    for sub in (
        "checkpoints",
        "tensorboard/train",
        "tensorboard/val",
        "tensorboard/test",
        "confusion",
    ):
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)


def save_metrics_line(path: str, split: str, loss, acc, fm, fw, epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tag = f"epoch={epoch}" if epoch is not None else "final"
    with open(path, "a") as f:
        f.write(
            f"[{tag}] {split:5s}  loss={loss:.4f}  acc={acc:.2f}%  F1-macro={fm:.2f}%  F1-weighted={fw:.2f}%\n"
        )


def train_one_epoch(model, loader, criterion, optimizer, args, device):
    model.train()
    losses = []
    for batch_idx, (data, target, _) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.view(-1).to(device, non_blocking=True)
        if args.mixup:
            data, y_a_y_b_lam = mixup_data(data, target, args.alpha)
        logits = model(data)
        if args.mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(logits, y_a_y_b_lam)
        else:
            loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        losses.append(loss.item())
        if args.mixup:
            criterion = criterion.get_old()
        if batch_idx % args.print_freq == 0:
            print(f"  [batch {batch_idx:4d}/{len(loader)}]  loss={loss.item():.4f}")
    return (float(np.mean(losses)) if losses else 0.0, criterion)


@torch.no_grad()
def eval_one_epoch(
    model, loader, criterion, args, device, apply_invalid_prefix: bool = False
):
    model.eval()
    losses = []
    y_true_chunks, y_pred_chunks = ([], [])
    last_data_T = None
    for data, target, _ in loader:
        data = data.to(device, non_blocking=True)
        target = target.view(-1).to(device, non_blocking=True)
        logits = model(data)
        loss = criterion(logits, target)
        losses.append(loss.item())
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        y_true_chunks.append(target.detach().cpu().numpy().reshape(-1))
        y_pred_chunks.append(pred.detach().cpu().numpy().reshape(-1))
        last_data_T = data.shape[1]
    if apply_invalid_prefix and last_data_T is not None and (len(y_true_chunks) > 0):
        ws = last_data_T - 1
        first_true = int(y_true_chunks[0][0])
        samples_invalid = np.full(ws, first_true, dtype=np.int64)
        y_true_chunks.append(samples_invalid)
        y_pred_chunks.append(samples_invalid)
    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(y_true, y_pred, average="macro")
    fw = 100.0 * metrics.f1_score(y_true, y_pred, average="weighted")
    loss_avg = float(np.mean(losses)) if losses else 0.0
    return (loss_avg, acc, fm, fw, y_true, y_pred)


def run_training(model, ds_train, ds_val, args, device):
    print("\n" + "=" * 70)
    print(f"[STEP 4] Training PatchTST on {args.dataset} (seed={args.seed})")
    print("=" * 70)
    if args.weighted_sampler:
        print("[-] Using WeightedRandomSampler")
        sampler = WeightedRandomSampler(
            ds_train.weight_samples, len(ds_train.weight_samples)
        )
        train_loader = DataLoader(
            ds_train, args.batch_size, sampler=sampler, pin_memory=True, num_workers=2
        )
    else:
        train_loader = DataLoader(
            ds_train, args.batch_size, shuffle=True, pin_memory=True, num_workers=2
        )
    val_loader = DataLoader(
        ds_val, args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args.lr)
    else:
        raise ValueError(args.optimizer)
    scheduler = None
    if args.lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_decay
        )
    metrics_log = os.path.join(args.results_dir, "metrics.txt")
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    metric_best = 0.0
    epochs_no_improve = 0
    history = []
    start_time = time.time()
    for epoch in range(args.epochs):
        print("-" * 70)
        print(
            f"Epoch {epoch}/{args.epochs - 1}  LR={optimizer.param_groups[0]['lr']:.6g}"
        )
        train_loss, criterion = train_one_epoch(
            model, train_loader, criterion, optimizer, args, device
        )
        _, acc_t, fm_t, fw_t, _, _ = eval_one_epoch(
            model, train_loader, criterion, args, device, apply_invalid_prefix=False
        )
        loss_v, acc_v, fm_v, fw_v, _, _ = eval_one_epoch(
            model, val_loader, criterion, args, device, apply_invalid_prefix=False
        )
        print(
            f"  TRAIN  loss={train_loss:.4f}  acc={acc_t:.2f}%  F1m={fm_t:.2f}%  F1w={fw_t:.2f}%"
        )
        print(
            f"  VAL    loss={loss_v:.4f}  acc={acc_v:.2f}%  F1m={fm_v:.2f}%  F1w={fw_v:.2f}%"
        )
        save_metrics_line(metrics_log, "train", train_loss, acc_t, fm_t, fw_t, epoch)
        save_metrics_line(metrics_log, "val", loss_v, acc_v, fm_v, fw_v, epoch)
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=acc_t,
                train_fm=fm_t,
                train_fw=fw_t,
                val_loss=loss_v,
                val_acc=acc_v,
                val_fm=fm_v,
                val_fw=fw_v,
                lr=optimizer.param_groups[0]["lr"],
            )
        )
        if fm_v >= metric_best:
            print(f"  [*] checkpoint improved {metric_best:.2f} -> {fm_v:.2f}")
            metric_best = fm_v
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "metric_best": metric_best,
                    "args": vars(args),
                },
                os.path.join(ckpt_dir, "checkpoint_best.pth"),
            )
        else:
            epochs_no_improve += 1
            print(
                f"  [-] no improvement ({epochs_no_improve}/{args.patience})  best fm_val={metric_best:.2f}"
            )
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth"),
            )
        if scheduler is not None:
            scheduler.step()
        if epochs_no_improve >= args.patience:
            print(
                f"\n[!] Early stopping at epoch {epoch}  best fm_val={metric_best:.2f}"
            )
            break
    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(f"\n[STEP 4] Training finished — elapsed {elapsed}")
    with open(os.path.join(args.results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return metric_best


def run_evaluation(model, ds_test, args, device):
    print("\n" + "=" * 70)
    print(f"[STEP 5] Evaluating PatchTST on {args.dataset}")
    print("=" * 70)
    apply_padding = ds_test.prefix == "test" and "capture24" not in args.dataset
    if apply_padding:
        print(
            f"[*] Replicating A&D's invalid-prefix padding (window-1 = {args.window - 1} free correct samples)"
        )
    else:
        print("[*] No invalid-prefix padding (Capture-24 / pre-windowed dataset)")
    test_loader = DataLoader(
        ds_test, args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    ckpt_path = os.path.join(args.results_dir, "checkpoints", "checkpoint_best.pth")
    print(f"[*] Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    loss_te, acc_te, fm_te, fw_te, y_true, y_pred = eval_one_epoch(
        model, test_loader, criterion, args, device, apply_invalid_prefix=apply_padding
    )
    print("\n--- TEST RESULTS ---")
    print(f"  loss        = {loss_te:.4f}")
    print(f"  Accuracy    = {acc_te:.2f}%")
    print(f"  F1-macro    = {fm_te:.2f}%")
    print(f"  F1-weighted = {fw_te:.2f}%")
    metrics_log = os.path.join(args.results_dir, "metrics.txt")
    save_metrics_line(metrics_log, "test", loss_te, acc_te, fm_te, fw_te)
    report = metrics.classification_report(
        y_true, y_pred, target_names=args.class_map, digits=4, zero_division=0
    )
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(args.num_class)))
    with open(
        os.path.join(args.results_dir, "test_classification_report.txt"), "w"
    ) as f:
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))
    np.savez(
        os.path.join(args.results_dir, "test_predictions.npz"),
        y_true=y_true,
        y_pred=y_pred,
        cm=cm,
    )
    drop_null = getattr(args, "drop_null", False)
    null_label = getattr(args, "null_label", None)
    summary = dict(
        dataset=args.dataset,
        seed=args.seed,
        accuracy=acc_te,
        f1_macro=fm_te,
        f1_weighted=fw_te,
        loss=loss_te,
        n_test=int(len(y_true)),
        applied_invalid_prefix=apply_padding,
        drop_null=bool(drop_null),
        null_label=null_label,
        num_class=args.num_class,
        class_map=args.class_map,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        mixup=args.mixup,
        weighted_sampler=args.weighted_sampler,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    with open(os.path.join(args.results_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Wrote artefacts to {args.results_dir}")
    return summary


def main():
    args, config_dataset, config_model = get_args()
    set_seed(args.seed)
    make_results_dirs(args.results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}  (cuda available: {torch.cuda.is_available()})")
    print(f"[*] Experiment: {args.experiment}")
    print(f"[*] Seed:       {args.seed}")
    print(f"[*] Results dir: {args.results_dir}")
    print(f"[*] Dataset config: {config_dataset}")
    print(f"[*] Model   config: {config_model}")
    if config_dataset.get("drop_null", False):
        print(
            f"[*] >>> NO-NULL BASELINE: dropping label {config_dataset['null_label']} at load time, num_class={args.num_class}"
        )
    if args.dataset != "ucihar":
        required = ["train.npz", "val.npz"]
        if args.stride_test == 1 and "capture24" not in args.dataset:
            required.append("test_sample_wise.npz")
        else:
            required.append("test.npz")
        for fname in required:
            fp = os.path.join(args.path_processed, fname)
            if not os.path.exists(fp):
                raise FileNotFoundError(
                    f"Required file missing: {fp}\nRun A&D's preprocess.py on dataset '{args.dataset}' first to generate it."
                )
    if args.train_mode:
        ds_train = PatchTSTSensorDataset(**config_dataset, prefix="train")
        ds_val = PatchTSTSensorDataset(**config_dataset, prefix="val")
    ds_test = PatchTSTSensorDataset(**config_dataset, prefix="test")
    model = create_patchtst(config_model).to(device)
    n_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[*] Model trainable params: {n_params:,}")
    if args.train_mode:
        run_training(model, ds_train, ds_val, args, device)
    summary = run_evaluation(model, ds_test, args, device)
    print("\n" + "=" * 70)
    print("DONE — final test summary:")
    print(json.dumps(summary, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
