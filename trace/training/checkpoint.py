"""Checkpoint save/load and TopK manager."""

import glob
import torch
import torch.nn as nn
from pathlib import Path

from trace.model.projector import SensorProjector
from trace.model.adapter import SensorCrossAttentionAdapter
from trace.training.logging_utils import _get_active_wandb_run_id
from trace.utils.naming import _run_prefix, _dataset_dir


def _clean_state_dict(module):
    raw_sd = module.state_dict()
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in raw_sd.items()}


def save_checkpoint(projector, adapters, epoch, cfg, input_dim, llm_dim,
                    run_dir, tag=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    fname = f"checkpoint_{tag}.pt" if tag else f"checkpoint_epoch_{epoch}.pt"
    save_path = run_dir / fname
    payload = {
        "projector_state_dict": _clean_state_dict(projector),
        "adapters_state_dict":  adapters.state_dict(),
        "input_dim": input_dim, "llm_dim": llm_dim,
        "n_tokens": cfg["n_tokens"],
        "n_adapter_layers": len(adapters),
        "adapter_rank": cfg["adapter_rank"],
        "adapter_num_heads": cfg["adapter_num_heads"],
        "adapter_layers": cfg["adapter_layers"],
        "epoch": epoch, "model_id": cfg["model_id"],
        "dataset": cfg["dataset"],
        "wandb_run_id": _get_active_wandb_run_id(),
        "num_activity_classes": len(cfg["activity_classes"]),
        "run_name": run_dir.name,
        "config": {
            "n_tokens": cfg["n_tokens"],
            "adapter_rank": cfg["adapter_rank"],
            "adapter_num_heads": cfg["adapter_num_heads"],
            "adapter_layers": cfg["adapter_layers"],
            "model_id": cfg["model_id"], "dataset": cfg["dataset"],
            "batch_size": cfg["batch_size"], "lr": cfg["lr"],
            "max_seq_len": cfg["max_seq_len"],
        },
    }
    torch.save(payload, save_path)
    print(f"[save] {save_path}")
    return save_path


class TopKCheckpointManager:
    """Keeps only top-K best checkpoints (by val_total_loss) + checkpoint_last.pt."""
    _RANK_TAGS = ["best", "2nd_best", "3rd_best"]

    def __init__(self, run_dir: Path, k: int = 3):
        self.run_dir = run_dir
        self.k = k
        self._best: list[tuple[float, int, Path]] = []

    def _tag_for_rank(self, rank: int) -> str:
        return self._RANK_TAGS[rank] if rank < len(self._RANK_TAGS) else f"{rank+1}th_best"

    def update(self, val_loss: float, epoch: int, save_fn) -> bool:
        if len(self._best) >= self.k and val_loss >= self._best[-1][0]:
            return False
        temp_tag = f"_epoch{epoch}_temp"
        new_path = save_fn(tag=temp_tag)
        self._best.append((val_loss, epoch, new_path))
        self._best.sort(key=lambda x: x[0])
        if len(self._best) > self.k:
            _, _, evicted_path = self._best.pop()
            if evicted_path is not None and evicted_path.exists():
                evicted_path.unlink()
                print(f"[checkpoint] Evicted: {evicted_path.name}")
        rename_plan = []
        for i, (loss, ep, old_path) in enumerate(self._best):
            final_path = self.run_dir / f"checkpoint_{self._tag_for_rank(i)}.pt"
            if old_path != final_path:
                rename_plan.append((i, old_path, final_path))
        temp_paths = []
        for i, old_path, final_path in rename_plan:
            tmp = old_path.with_suffix(f".rank{i}.tmp")
            if old_path.exists():
                old_path.rename(tmp)
            temp_paths.append((i, tmp, final_path))
        for i, tmp_path, final_path in temp_paths:
            if tmp_path.exists():
                tmp_path.rename(final_path)
            self._best[i] = (self._best[i][0], self._best[i][1], final_path)
        is_new_best = (self._best[0][1] == epoch)
        rank = next(i for i, (_, ep, _) in enumerate(self._best) if ep == epoch)
        if is_new_best:
            print(f"[checkpoint] * New best at epoch {epoch} (val_total={val_loss:.4f})")
        else:
            print(f"[checkpoint] Saved as rank {rank+1} at epoch {epoch} (val_total={val_loss:.4f})")
        return is_new_best

    @property
    def best_loss(self) -> float:
        return self._best[0][0] if self._best else float("inf")

    @property
    def best_epoch(self) -> int:
        return self._best[0][1] if self._best else 0


def load_checkpoint(checkpoint_path: Path, input_dim: int, llm_dim: int,
                    n_tokens: int, adapter_rank: int, adapter_num_heads: int,
                    n_adapter_layers: int, device: str, adapter_dropout: float = 0.1):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(ckpt)}")
    saved_in  = ckpt.get("input_dim")
    saved_llm = ckpt.get("llm_dim")
    saved_n   = ckpt.get("n_tokens")
    if saved_in is not None and saved_in != input_dim:
        raise ValueError(f"Checkpoint input_dim={saved_in} != current {input_dim}")
    if saved_llm is not None and saved_llm != llm_dim:
        raise ValueError(f"Checkpoint llm_dim={saved_llm} != current {llm_dim}")
    if saved_n is not None and saved_n != n_tokens:
        raise ValueError(f"Checkpoint n_tokens={saved_n} != current --n_tokens={n_tokens}.")
    projector = SensorProjector(input_dim=input_dim, llm_dim=llm_dim, n_tokens=n_tokens)
    projector.load_state_dict(ckpt["projector_state_dict"])
    projector = projector.to(device=device)
    projector.eval()
    saved_n_adapters = ckpt.get("n_adapter_layers", n_adapter_layers)
    saved_rank = ckpt.get("adapter_rank", adapter_rank)
    saved_heads = ckpt.get("adapter_num_heads", adapter_num_heads)
    adapters = nn.ModuleList([
        SensorCrossAttentionAdapter(hidden_dim=llm_dim, num_heads=saved_heads,
                                     rank=saved_rank, dropout=adapter_dropout)
        for _ in range(saved_n_adapters)
    ])
    adapters.load_state_dict(ckpt["adapters_state_dict"])
    adapters = adapters.to(device=device)
    adapters.eval()
    return projector, adapters


def resolve_checkpoint(cfg: dict) -> Path:
    if cfg.get("projector_checkpoint") is not None:
        p = Path(cfg["projector_checkpoint"])
        if not p.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {p}")
        print(f"[checkpoint] >> Using explicit checkpoint: {p}")
        return p
    prefix = _run_prefix(cfg) + "_"
    ckpt_dir = _dataset_dir(cfg["checkpoint_dir"], cfg["dataset"])
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {ckpt_dir}")
    run_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    if not run_dirs:
        raise FileNotFoundError(f"No checkpoint dirs matching '{prefix}*' in {ckpt_dir}. Run training first.")
    latest = run_dirs[-1]
    best = latest / "checkpoint_best.pt"
    if not best.exists():
        candidates = sorted(latest.glob("checkpoint_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints in {latest}.")
        best = candidates[-1]
        print(f"[checkpoint] checkpoint_best.pt not found; using {best.name}")
    print(f"[checkpoint] >> Auto-resolved checkpoint: {best}")
    print(f"[checkpoint]    Run dir: {latest}")
    return best
