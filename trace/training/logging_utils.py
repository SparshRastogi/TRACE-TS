"""W&B and TensorBoard logging helpers."""

import json
from pathlib import Path

from trace.utils.naming import _model_short, _run_name


_WANDB_RUN_ID_FILE = ".wandb_run_id.json"


def _get_active_wandb_run_id() -> str | None:
    try:
        import wandb
        if wandb.run is not None:
            return wandb.run.id
    except Exception:
        pass
    return None


def _load_wandb_run_id_from_checkpoint(cfg: dict) -> str | None:
    try:
        from trace.training.checkpoint import resolve_checkpoint
        import torch
        ckpt_path = resolve_checkpoint(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            run_id = ckpt.get("wandb_run_id")
            if run_id:
                print(f"[logging] Found wandb_run_id={run_id} in checkpoint")
                return run_id
    except Exception:
        pass
    return None


def _build_wandb_tags(cfg: dict) -> list:
    tags = ["cross-attention", f"{cfg['n_tokens']}tok", f"R{cfg['adapter_rank']}",
            cfg["dataset"], _model_short(cfg["model_id"]),
            cfg.get("data_version", "structured")]
    tags.extend([f"bs{cfg['batch_size']}", f"lr{cfg['lr']}"])
    if cfg.get("gradient_checkpointing"):
        tags.append("gc-on")
    if cfg.get("use_structured_prefix"):
        tags.append("prefix-enforced")
    return tags


def _define_wandb_axes(wandb):
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("loss/step/*", step_metric="global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("loss/epoch/*", step_metric="epoch")
        wandb.define_metric("diag/*", step_metric="epoch")
        for m in ("adapter_gate_mean", "adapter_gate_max", "adapter_n_active",
                  "adapter_grad_norm_mean", "vram_allocated_gb", "vram_peak_gb"):
            wandb.define_metric(f"diag/{m}", step_metric="global_step")
        wandb.define_metric("diag/crossattn/*", step_metric="global_step")
        wandb.define_metric("diag/gate/*", step_metric="global_step")
    except Exception:
        pass


def _save_wandb_run_id(run_id: str, run_name: str, cfg: dict):
    try:
        with open(_WANDB_RUN_ID_FILE, "w") as f:
            json.dump({"run_id": run_id, "run_name": run_name,
                       "project": cfg.get("wandb_project", "trace-ts")}, f)
    except Exception:
        pass


def _get_or_create_wandb_run(cfg: dict, run_name: str = None, save_id: bool = False):
    if not cfg.get("use_wandb"):
        return None
    try:
        import wandb
        if wandb.run is not None:
            return wandb.run
        wandb_cfg = {k: str(v) if isinstance(v, Path) else v
                     for k, v in cfg.items()
                     if k not in ("sensor_terms", "activity_classes", "valid_sensors")}
        tags = _build_wandb_tags(cfg)
        if cfg.get("mode") == "train":
            if run_name is None:
                run_name = _run_name(cfg)
            wb_run = wandb.init(project=cfg.get("wandb_project", "trace-ts"),
                                name=run_name, config=wandb_cfg, tags=tags)
            print(f"[logging] W&B NEW run: '{run_name}' -> {wb_run.url}")
            _define_wandb_axes(wandb)
            if save_id:
                _save_wandb_run_id(wb_run.id, run_name, cfg)
            return wb_run
        ckpt_run_id = _load_wandb_run_id_from_checkpoint(cfg)
        if ckpt_run_id is not None:
            try:
                wb_run = wandb.init(project=cfg.get("wandb_project", "trace-ts"),
                                    id=ckpt_run_id, resume="allow", config=wandb_cfg, tags=tags)
                print(f"[logging] W&B: Rejoined training run -> {wb_run.url}")
                _define_wandb_axes(wandb)
                return wb_run
            except Exception as e:
                print(f"[logging] W&B: Could not rejoin ({e}), creating new.")
        if run_name is None:
            run_name = _run_name(cfg)
        wb_run = wandb.init(project=cfg.get("wandb_project", "trace-ts"),
                            name=run_name, config=wandb_cfg, tags=tags)
        print(f"[logging] W&B run: '{run_name}' -> {wb_run.url}")
        _define_wandb_axes(wandb)
        return wb_run
    except ImportError:
        print("[logging] WARNING: wandb not installed")
        return None
    except Exception as e:
        print(f"[logging] WARNING: W&B init failed ({e})")
        return None


def init_loggers(cfg: dict, run_dir: Path):
    tb_writer = None
    if cfg.get("use_tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = run_dir / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"[logging] TensorBoard -> {tb_dir}")
        except ImportError:
            print("[logging] WARNING: tensorboard not installed")
    wandb_run = _get_or_create_wandb_run(cfg, run_name=run_dir.name, save_id=True)
    return tb_writer, wandb_run


def log_step_metrics(metrics: dict, step: int, tb_writer, wandb_run):
    wb_payload = {f"loss/step/{k}": v for k, v in metrics.items()
                  if isinstance(v, (int, float)) and v == v}
    if tb_writer is not None:
        for k, v in wb_payload.items():
            tb_writer.add_scalar(k.replace("loss/step/", "train/step/"), v, global_step=step)
    if wandb_run is not None:
        try:
            wandb_run.log({**wb_payload, "global_step": step}, step=step)
        except Exception:
            pass


def log_epoch_metrics(metrics: dict, epoch: int, tb_writer, wandb_run):
    wb_payload = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)) or v != v:
            continue
        prefix = "diag/" if any(x in k for x in ("blindfold", "early_stop", "adapter")) else "loss/epoch/"
        wb_payload[f"{prefix}{k}"] = v
    if tb_writer is not None:
        for k, v in wb_payload.items():
            tb_writer.add_scalar(k, v, global_step=epoch)
    if wandb_run is not None:
        try:
            wandb_run.log({**wb_payload, "epoch": epoch})
        except Exception:
            pass


def close_loggers(tb_writer, wandb_run, finish_wandb: bool = False):
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None and finish_wandb:
        try:
            wandb_run.finish()
        except Exception:
            pass
