import sys
import os
import time
import datetime
import glob
import re
import random
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from AtteFinalPipeline.data.preprocess import preprocess_pipeline
from AtteFinalPipeline.expert.dataset import SensorDataset
from AtteFinalPipeline.expert.model import create
from AtteFinalPipeline.expert.utils.utils import paint, Logger, AverageMeter
from AtteFinalPipeline.expert.utils.plot import plot_confusion
from AtteFinalPipeline.expert.utils.pytorch import (
    get_info_params,
    get_info_layers,
    init_weights_orthogonal,
)
from AtteFinalPipeline.expert.utils.mixup import mixup_data, MixUpLoss
from AtteFinalPipeline.expert.utils.centerloss import (
    compute_center_loss,
    get_center_delta,
)
from AtteFinalPipeline.expert.settings import get_args
import warnings

warnings.filterwarnings("ignore")
_SKIP_GENERIC_PREPROCESS = {"ucihar"}


def make_results_dirs(results_dir):
    for sub in [
        "confusion",
        "tensorboard/train",
        "tensorboard/val",
        "tensorboard/test",
    ]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)


def save_metrics_txt(path, split, loss, acc, fm, fw, epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tag = f"epoch={epoch}" if epoch is not None else "final"
    with open(path, "a") as f:
        f.write(
            f"[{tag}] {split:5s}  loss={loss:.4f}  acc={acc:.2f}%  F1-macro={fm:.2f}%  F1-weighted={fw:.2f}%\n"
        )


def _find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return (None, -1)
    paths = glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth"))
    best_epoch, best_path = (-1, None)
    for p in paths:
        m = re.match("checkpoint_(\\d+)\\.pth$", os.path.basename(p))
        if m:
            e = int(m.group(1))
            if e > best_epoch:
                best_epoch, best_path = (e, p)
    return (best_path, best_epoch)


def model_train(model, dataset, dataset_val, args):
    print(paint("[STEP 4] Running HAR training loop ..."))
    results_dir = args.results_dir
    logger = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard", "train"))
    logger_val = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard", "val"))
    _ = SummaryWriter(log_dir=os.path.join(model.path_logs, "train"))
    if args.weighted_sampler:
        print(paint("[-] Using weighted sampler (balanced batch)..."))
        sampler = WeightedRandomSampler(
            dataset.weight_samples, len(dataset.weight_samples)
        )
        loader = DataLoader(dataset, args.batch_size, sampler=sampler, pin_memory=True)
    else:
        loader = DataLoader(dataset, args.batch_size, True, pin_memory=True)
    loader_val = DataLoader(dataset_val, args.batch_size, False, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args.lr)
    if args.lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_decay
        )
    if args.init_weights == "orthogonal" and (not args.resume):
        print(paint("[-] Initializing weights (orthogonal)..."))
        model.apply(init_weights_orthogonal)
    start_epoch = 0
    metric_best = 0.0
    if args.resume:
        ckpt_path, ckpt_epoch = _find_latest_checkpoint(model.path_checkpoints)
        if ckpt_path is None:
            print(
                paint(
                    f"[!] --resume set but no numbered checkpoint found in {model.path_checkpoints}. Starting from scratch.",
                    "blue",
                )
            )
            if args.init_weights == "orthogonal":
                print(paint("[-] Initializing weights (orthogonal)..."))
                model.apply(init_weights_orthogonal)
        else:
            print(paint(f"[*] Resuming from {ckpt_path} (epoch {ckpt_epoch})", "blue"))
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optim_state_dict"])
            criterion.load_state_dict(ckpt["criterion_state_dict"])
            random.setstate(ckpt["random_rnd_state"])
            np.random.set_state(ckpt["numpy_rnd_state"])
            torch.set_rng_state(ckpt["torch_rnd_state"])
            start_epoch = ckpt_epoch + 1
            if args.lr_step > 0:
                for _ in range(start_epoch):
                    scheduler.step()
                print(
                    paint(
                        f"[*] LR scheduler advanced to epoch {start_epoch}; current LR = {optimizer.param_groups[0]['lr']:.6g}",
                        "blue",
                    )
                )
            best_path = os.path.join(model.path_checkpoints, "checkpoint_best.pth")
            if os.path.exists(best_path):
                print(
                    paint(
                        "[*] Bootstrapping metric_best from checkpoint_best.pth ...",
                        "blue",
                    )
                )
                best_ckpt = torch.load(best_path, weights_only=False)
                tmp_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_ckpt["model_state_dict"])
                _, _, fm_best_val, _ = eval_one_epoch(
                    model,
                    loader_val,
                    criterion,
                    epoch=start_epoch - 1,
                    logger=None,
                    args=args,
                    cm_dir=None,
                )
                metric_best = fm_best_val
                model.load_state_dict(tmp_model_state)
                print(
                    paint(
                        f"[*] metric_best (fm_val of checkpoint_best) = {metric_best:.2f}",
                        "blue",
                    )
                )
            else:
                print(
                    paint(
                        "[*] No checkpoint_best.pth found; evaluating latest checkpoint for baseline.",
                        "blue",
                    )
                )
                _, _, fm_latest, _ = eval_one_epoch(
                    model,
                    loader_val,
                    criterion,
                    epoch=start_epoch - 1,
                    logger=None,
                    args=args,
                    cm_dir=None,
                )
                metric_best = fm_latest
                print(paint(f"[*] metric_best baseline = {metric_best:.2f}", "blue"))
    if args.resume and start_epoch > 0:
        end_epoch = start_epoch + args.extra_epochs
    else:
        end_epoch = args.epochs
    print(
        paint(
            f"[*] Training from epoch {start_epoch} to {end_epoch - 1} (early-stopping patience = {args.patience})",
            "blue",
        )
    )
    start_time = time.time()
    metrics_log = os.path.join(results_dir, "metrics.txt")
    epochs_no_improve = 0
    for epoch in range(start_epoch, end_epoch):
        print("--" * 50)
        print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, epoch, args)
        loss, acc, fm, fw = eval_one_epoch(
            model,
            loader,
            criterion,
            epoch,
            logger,
            args,
            cm_dir=os.path.join(results_dir, "confusion", "train"),
        )
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(
            model,
            loader_val,
            criterion,
            epoch,
            logger_val,
            args,
            cm_dir=os.path.join(results_dir, "confusion", "val"),
        )
        logger.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        print(
            paint(
                f"[-] Epoch {epoch}/{end_epoch - 1}\tTrain loss: {loss:.2f} \tacc: {acc:.2f}(%)\tfm: {fm:.2f}(%)\tfw: {fw:.2f}(%)"
            )
        )
        print(
            paint(
                f"[-] Epoch {epoch}/{end_epoch - 1}\tVal loss: {loss_val:.2f} \tacc: {acc_val:.2f}(%)\tfm: {fm_val:.2f}(%)\tfw: {fw_val:.2f}(%)"
            )
        )
        save_metrics_txt(metrics_log, "train", loss, acc, fm, fw, epoch)
        save_metrics_txt(metrics_log, "val", loss_val, acc_val, fm_val, fw_val, epoch)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }
        metric = fm_val
        if metric >= metric_best:
            print(
                paint(
                    f"[*] Saving checkpoint... ({metric_best:.2f}->{metric:.2f})",
                    "blue",
                )
            )
            metric_best = metric
            epochs_no_improve = 0
            torch.save(
                checkpoint, os.path.join(model.path_checkpoints, "checkpoint_best.pth")
            )
        else:
            epochs_no_improve += 1
            print(
                paint(
                    f"[-] No improvement ({epochs_no_improve}/{args.patience})  best fm_val = {metric_best:.2f}"
                )
            )
        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints, f"checkpoint_{epoch}.pth"),
            )
        if args.lr_step > 0:
            scheduler.step()
        if epochs_no_improve >= args.patience:
            print(
                paint(
                    f"[!] Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs). Best fm_val = {metric_best:.2f}",
                    "blue",
                )
            )
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints, f"checkpoint_{epoch}.pth"),
            )
            break
    logger.close()
    logger_val.close()
    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(paint(f"[STEP 4] Finished HAR training loop (h:m:s): {elapsed}"))
    print(paint("--" * 50, "blue"))


def train_one_epoch(model, loader, criterion, optimizer, epoch, args):
    losses = AverageMeter("Loss")
    model.train()
    for batch_idx, (data, target, idx) in enumerate(loader):
        data = data.cuda()
        target = target.view(-1).cuda()
        centers = model.centers
        if args.mixup:
            data, y_a_y_b_lam = mixup_data(data, target, args.alpha)
        z, logits, _ = model(data)
        if args.mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(logits, y_a_y_b_lam)
        else:
            loss = criterion(logits, target)
        center_loss = compute_center_loss(z, centers, target)
        loss = loss + args.beta * center_loss
        losses.update(loss.item(), data.shape[0])
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        center_deltas = get_center_delta(z.data, centers, target, args.lr_cent)
        model.centers = centers - center_deltas
        if batch_idx % args.print_freq == 0:
            print(f"[-] Batch {batch_idx}/{len(loader)}\t Loss: {str(losses)}")
        if args.mixup:
            criterion = criterion.get_old()


def eval_one_epoch(model, loader, criterion, epoch, logger, args, cm_dir=None):
    losses = AverageMeter("Loss")
    y_true, y_pred = ([], [])
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data = data.cuda()
            target = target.cuda()
            z, logits, _ = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])
            probabilities = nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)
            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))
    if loader.dataset.prefix == "test" and "capture24" not in args.dataset:
        ws = data.shape[1] - 1
        samples_invalid = [y_true[0][0]] * ws
        y_true.append(samples_invalid)
        y_pred.append(samples_invalid)
    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)
    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(y_true, y_pred, average="macro")
    fw = 100.0 * metrics.f1_score(y_true, y_pred, average="weighted")
    if logger:
        logger.add_scalars("Loss", {"CrossEntropy": losses.avg}, epoch)
        logger.add_scalar("Acc", acc, epoch)
        logger.add_scalar("Fm", fm, epoch)
        logger.add_scalar("Fw", fw, epoch)
    save_cm = epoch == -1 or epoch % 10 == 0
    if save_cm and cm_dir is not None:
        split_tag = loader.dataset.prefix
        out_dir = (
            cm_dir if cm_dir else os.path.join(model.path_visuals, f"cm/{split_tag}")
        )
        plot_confusion(y_true, y_pred, out_dir, epoch, class_map=args.class_map)
    return (losses.avg, acc, fm, fw)


def model_eval(model, dataset_test, args):
    print(paint("[STEP 5] Running HAR evaluation loop ..."))
    results_dir = args.results_dir
    logger_test = SummaryWriter(
        log_dir=os.path.join(results_dir, "tensorboard", "test")
    )
    metrics_log = os.path.join(results_dir, "metrics.txt")
    loader_test = DataLoader(dataset_test, args.batch_size, False, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
    print("[-] Loading checkpoint ...")
    if args.train_mode:
        path_checkpoint = os.path.join(model.path_checkpoints, "checkpoint_best.pth")
    else:
        path_checkpoint = os.path.join(f"./weights/checkpoint_{args.dataset}.pth")
    checkpoint = torch.load(path_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])
    start_time = time.time()
    loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
        model,
        loader_test,
        criterion,
        epoch=-1,
        logger=logger_test,
        args=args,
        cm_dir=os.path.join(results_dir, "confusion", "test"),
    )
    logger_test.add_scalar("Acc", acc_test, 0)
    logger_test.add_scalar("Fm", fm_test, 0)
    logger_test.add_scalar("Fw", fw_test, 0)
    logger_test.add_scalar("Loss", loss_test, 0)
    logger_test.close()
    save_metrics_txt(metrics_log, "test", loss_test, acc_test, fm_test, fw_test)
    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}\tacc: {acc_test:.2f}(%)\tfm: {fm_test:.2f}(%)\tfw: {fw_test:.2f}(%)"
        )
    )
    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(paint(f"[STEP 5] Finished HAR evaluation loop (h:m:s): {elapsed}"))


def main():
    args, config_dataset, config_model = get_args()
    make_results_dirs(args.results_dir)
    if args.dataset not in _SKIP_GENERIC_PREPROCESS:
        preprocess_pipeline(args)
    if args.train_mode:
        dataset = SensorDataset(**config_dataset, prefix="train")
        dataset_val = SensorDataset(**config_dataset, prefix="val")
        if torch.cuda.is_available():
            model = create(args.model, config_model).cuda()
            torch.backends.cudnn.benchmark = True
            sys.stdout = Logger(
                os.path.join(model.path_logs, f"log_main_{args.experiment}.txt")
            )
        print("##" * 50)
        print(paint(f"Experiment: {model.experiment}", "blue"))
        print(paint(f"Results dir: {args.results_dir}", "blue"))
        if args.resume:
            print(
                paint(
                    f"[*] Resume mode ON — will look for checkpoints in {model.path_checkpoints}",
                    "blue",
                )
            )
        print(
            paint(
                f"[-] Using {torch.cuda.device_count()} GPU: {torch.cuda.is_available()}"
            )
        )
        print(args)
        get_info_params(model)
        get_info_layers(model)
        print("##" * 50)
        model_train(model, dataset, dataset_val, args)
    dataset_test = SensorDataset(**config_dataset, prefix="test")
    if not args.train_mode:
        config_model["experiment"] = "inference"
        model = create(args.model, config_model).cuda()
    model_eval(model, dataset_test, args)


if __name__ == "__main__":
    main()
