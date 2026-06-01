"""Main training loop."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from trace.model.backbone import load_model_and_tokenizer, _get_llm_layers, _get_llm_dim
from trace.model.projector import SensorProjector
from trace.model.adapter import SensorCrossAttentionAdapter
from trace.model.sensor_llm import SensorLLMCrossAttn
from trace.data.loader import load_data, make_stratified_split
from trace.data.dataset import ReasoningDataset, make_collate_fn
from trace.utils.naming import _resolve_adapter_layer_indices, _run_name, _dataset_dir
from trace.training.logging_utils import (
    init_loggers, log_step_metrics, log_epoch_metrics, close_loggers,
)
from trace.training.checkpoint import save_checkpoint, TopKCheckpointManager
from trace.training.diagnostics import _log_adapter_diagnostics, _log_vram
from trace.training.callbacks import (
    BlindfoldMonitor, EarlyStopping,
    run_startup_checks, run_crossattn_startup_checks,
)


def _compute_val_losses(model, val_loader, cfg, device):
    model.eval()
    lm_sum, lm_unw_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            lm_out = model(
                batch["sensor_embed"].to(device), batch["input_ids"].to(device),
                batch["attention_mask"].to(device), batch["labels"].to(device),
                token_weights=batch["token_weights"].to(device) if "token_weights" in batch else None)
            if not lm_out.loss.isfinite():
                raise RuntimeError(f"[SA8] Val lm_loss={lm_out.loss.item()}")
            lm_sum += lm_out.loss.item()
            lm_unw_sum += getattr(lm_out, "unweighted_lm_loss", lm_out.loss).item()
            n += 1
    n = max(n, 1)
    return {
        "val_lm_loss": lm_sum / n,
        "val_total_loss": lm_sum / n,
        "val_lm_loss_unweighted": lm_unw_sum / n,
    }


def run_train(cfg: dict):
    # ── DDP setup ────────────────────────────────────────────────────────
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main = (local_rank == 0)
    else:
        local_rank = 0
        world_size = 1
        is_main = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = device.startswith("cuda")

    if is_main:
        print(f"[train] Device: {device}  |  Model: {cfg['model_id']}")
        print(f"[train] n_tokens={cfg['n_tokens']}  adapter_rank={cfg['adapter_rank']}")
        if is_ddp:
            print(f"[train] DDP enabled: {world_size} GPUs")

    tokenizer, llm = load_model_and_tokenizer(cfg["model_id"], local_rank=local_rank)
    n_llm_layers = len(_get_llm_layers(llm))
    layer_indices = _resolve_adapter_layer_indices(cfg["adapter_layers"], n_llm_layers)
    n_adapter_layers = len(layer_indices)
    cfg["_n_adapter_layers"] = n_adapter_layers
    if is_main:
        print(f"[train] LLM: {n_llm_layers} layers, adapters on {n_adapter_layers}")

    if cfg["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if is_main:
            print("[train] Gradient checkpointing enabled (use_reentrant=False)")

    embeddings, texts, activity_labels, _ = load_data(
        cfg["train_json_dir"], cfg["json_key"], cfg["activity_label_key"])
    input_dim, llm_dim = embeddings.shape[1], _get_llm_dim(llm)
    n_classes, n_tokens = len(cfg["activity_classes"]), cfg["n_tokens"]
    if is_main:
        print(f"[train] Sensor dim: {input_dim}  |  LLM hidden: {llm_dim}  |  "
              f"Classes: {n_classes}  |  Tokens: {n_tokens}")

    rank = cfg["adapter_rank"]
    proj_params = input_dim * n_tokens * llm_dim + n_tokens * llm_dim + llm_dim * llm_dim + llm_dim + 2 * llm_dim
    adapter_params_each = 2 * llm_dim * rank * 3 + llm_dim * llm_dim + 2 * llm_dim + 1
    total_adapter_params = adapter_params_each * n_adapter_layers
    total_trainable = proj_params + total_adapter_params
    if is_main:
        print(f"[train] Trainable params: {total_trainable / 1e6:.1f}M "
              f"(projector: {proj_params / 1e6:.1f}M, "
              f"adapters: {total_adapter_params / 1e6:.1f}M x {n_adapter_layers} layers)")

    train_idx, val_idx = make_stratified_split(embeddings, texts, activity_labels, cfg["val_split"])
    full_dataset = ReasoningDataset(
        embeddings, texts, activity_labels, cfg["activity_classes"],
        tokenizer, cfg["instruction"], cfg["max_seq_len"],
        activity_token_weight=cfg.get("activity_token_weight", 1.0))
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    train_sampler = DistributedSampler(train_subset, shuffle=True)  if is_ddp else None
    val_sampler   = DistributedSampler(val_subset,   shuffle=False) if is_ddp else None
    train_loader = DataLoader(train_subset,
        batch_size=cfg["batch_size"], shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset,
        batch_size=cfg["batch_size"], shuffle=False,
        sampler=val_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    projector = SensorProjector(input_dim=input_dim, llm_dim=llm_dim, n_tokens=n_tokens).to(device)
    try:
        projector = torch.compile(projector, mode="reduce-overhead")
        if is_main:
            print("[train] torch.compile on projector")
    except Exception as e:
        if is_main:
            print(f"[train] WARNING: torch.compile failed ({e})")
    adapters = nn.ModuleList([
        SensorCrossAttentionAdapter(hidden_dim=llm_dim, num_heads=cfg["adapter_num_heads"],
                                     rank=cfg["adapter_rank"], dropout=cfg["adapter_dropout"])
        for _ in range(n_adapter_layers)
    ]).to(device)
    model = SensorLLMCrossAttn(llm, projector, adapters, layer_indices)

    _diag_raw = next(iter(train_loader))
    diag_batch = {k: v.to(device) for k, v in _diag_raw.items()
                  if k in ("sensor_embed", "input_ids", "attention_mask", "labels")}
    monitor = BlindfoldMonitor(diag_batch)

    if is_main:
        run_startup_checks(cfg, llm, projector, embeddings, device)
        run_crossattn_startup_checks(model, adapters, diag_batch, cfg, device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    _raw_model = model.module if is_ddp else model
    trainable = list(_raw_model.projector.parameters()) + list(_raw_model.adapters.parameters())
    optimizer = AdamW(trainable, lr=cfg["lr"])
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and len(adapters) > 0)
    if use_cuda:
        initial_scale = scaler.get_scale()
        if initial_scale <= 0:
            raise RuntimeError(f"[SA3] GradScaler initial scale={initial_scale} not positive.")
        if is_main:
            print(f"[checks] [SA3] GradScaler initial scale: {initial_scale}")

    run_name = _run_name(cfg)
    run_dir = _dataset_dir(cfg["checkpoint_dir"], cfg["dataset"]) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] Run dir: {run_dir}")
    tb_writer, wandb_run = (init_loggers(cfg, run_dir) if is_main else (None, None))

    def _make_save_fn(epoch):
        def _save(tag):
            return save_checkpoint(_raw_model.projector, _raw_model.adapters, epoch, cfg,
                                   input_dim, llm_dim, run_dir, tag=tag)
        return _save

    ckpt_mgr     = TopKCheckpointManager(run_dir, k=3) if is_main else None
    early_stopper = EarlyStopping(patience=cfg["early_stopping_patience"],
                                   min_delta=cfg["early_stopping_min_delta"]) if is_main else None

    global_step, _first_step = 0, True
    if is_main:
        print("[train] Starting training...")

    for epoch in range(1, cfg["epochs"] + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}") if is_main else train_loader
        epoch_lm, epoch_total, epoch_lm_unw, n_batches = 0.0, 0.0, 0.0, 0

        for batch in pbar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_cuda):
                lm_out = model(
                    batch["sensor_embed"].to(device), batch["input_ids"].to(device),
                    batch["attention_mask"].to(device), batch["labels"].to(device),
                    token_weights=batch["token_weights"].to(device) if "token_weights" in batch else None)
                lm_loss = lm_out.loss
                total_loss = lm_loss
                unweighted_lm_loss = getattr(lm_out, "unweighted_lm_loss", lm_loss)

            if not lm_loss.isfinite():
                raise RuntimeError(f"[SA2] lm_loss={lm_loss.item()} at step {global_step}.")
            if _first_step:
                if is_main:
                    print(f"[checks] [SA2] First-step loss: lm={lm_loss.item():.4f} total={total_loss.item():.4f}")
                _first_step = False

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_lm     += lm_loss.item()
            epoch_total  += total_loss.item()
            epoch_lm_unw += unweighted_lm_loss.item()
            n_batches    += 1
            global_step  += 1

            if is_main and global_step % cfg["log_every_n_steps"] == 0:
                step_metrics = {
                    "lm_loss": lm_loss.item(),
                    "lm_loss_unweighted": unweighted_lm_loss.item(),
                    "total_loss": total_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "activity_token_weight": cfg.get("activity_token_weight", 1.0),
                }
                if use_cuda:
                    cs = scaler.get_scale()
                    step_metrics["grad_scaler_scale"] = cs
                    if cs < 1.0:
                        print(f"[SA3] WARNING: GradScaler scale={cs:.1f} at step {global_step}.")
                log_step_metrics(step_metrics, global_step, tb_writer, wandb_run)

            if is_main:
                _log_adapter_diagnostics(_raw_model.adapters, global_step, wandb_run, tb_writer,
                                         log_every=cfg.get("adapter_log_every", 50))
                _log_vram(global_step, wandb_run, log_every=cfg.get("vram_log_every", 100))
                pbar.set_postfix(lm=f"{lm_loss.item():.4f}", total=f"{total_loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        n_batches   = max(n_batches, 1)
        avg_lm      = epoch_lm     / n_batches
        avg_total   = epoch_total  / n_batches
        avg_lm_unw  = epoch_lm_unw / n_batches

        val_losses = _compute_val_losses(model, val_loader, cfg, device)
        if is_ddp:
            for key in val_losses:
                t = torch.tensor(val_losses[key], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_losses[key] = t.item()
        avg_val_total = val_losses["val_total_loss"]

        if is_main:
            print(f"[train] Epoch {epoch}  train_lm={avg_lm:.4f} train_total={avg_total:.4f}  "
                  f"val_lm={val_losses['val_lm_loss']:.4f} val_total={avg_val_total:.4f}")
            gate_vals = [a.gate.item() for a in _raw_model.adapters]
            if gate_vals:
                print(f"[adapters] gates: mean={np.mean(gate_vals):.4f} max={np.max(gate_vals):.4f} "
                      f"active={sum(1 for g in gate_vals if abs(g) > 0.01)}/{len(gate_vals)}")
            else:
                print("[adapters] gates: no adapters (none mode)")
            if torch.cuda.is_available():
                print(f"[vram] peak={torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

        epoch_metrics = {
            "train_lm_loss": avg_lm,
            "train_lm_loss_unweighted": avg_lm_unw,
            "train_total_loss": avg_total,
            "val_lm_loss": val_losses["val_lm_loss"],
            "val_lm_loss_unweighted": val_losses["val_lm_loss_unweighted"],
            "val_total_loss": avg_val_total,
            "activity_token_weight": cfg.get("activity_token_weight", 1.0),
        }
        if is_main and epoch % cfg["blindfold_every_n_epochs"] == 0 and len(_raw_model.adapters) > 0:
            stats = monitor.check(_raw_model)
            print(f"[blindfold] ratio={stats['ratio']:.3f} adapter={stats['adapter_contribution_ratio']:.3f}"
                  + ("  sensor used" if stats["ratio"] < 0.95 else "  sensor ignored!"))
            epoch_metrics["blindfold_ratio"] = stats["ratio"]
            epoch_metrics["adapter_contribution_ratio"] = stats["adapter_contribution_ratio"]

        if is_main:
            epoch_metrics["early_stopping_counter"] = early_stopper.counter
            log_epoch_metrics(epoch_metrics, epoch, tb_writer, wandb_run)
            ckpt_mgr.update(avg_val_total, epoch, _make_save_fn(epoch))
            should_stop = early_stopper.step(avg_val_total, epoch)
        else:
            should_stop = False

        if is_ddp:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            if is_main:
                print(f"\n{'='*60}\nEARLY STOPPING at epoch {epoch}")
                print(f"  Best val_total_loss={early_stopper.best_loss:.4f} @ epoch {early_stopper.best_epoch}")
                print(f"{'='*60}\n")
            break
        else:
            if is_main:
                print(f"[early_stop] {early_stopper.status_str()}")

    if is_ddp:
        dist.barrier()
    if is_main:
        save_checkpoint(_raw_model.projector, _raw_model.adapters, epoch, cfg, input_dim, llm_dim,
                        run_dir, tag="last")
        print(f"[train] Done. Best: {ckpt_mgr.best_loss:.4f} @ epoch {ckpt_mgr.best_epoch}")
    close_loggers(tb_writer, wandb_run, finish_wandb=False)
    if is_ddp:
        dist.destroy_process_group()
