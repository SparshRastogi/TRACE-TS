"""Adapter gate diagnostics and VRAM logging."""

import numpy as np
import torch


def _log_adapter_diagnostics(adapters, global_step, wandb_run, tb_writer, log_every: int = 50):
    if len(adapters) == 0 or global_step % log_every != 0:
        return
    gate_values, gate_grads, adapter_grad_norms = [], [], []
    for adapter in adapters:
        gate_values.append(adapter.gate.item())
        gate_grads.append(adapter.gate.grad.item() if adapter.gate.grad is not None else 0.0)
        total_norm = sum(p.grad.data.norm(2).item() ** 2
                         for p in adapter.parameters() if p.grad is not None) ** 0.5
        adapter_grad_norms.append(total_norm)
    metrics = {
        "diag/adapter_gate_mean": np.mean(gate_values),
        "diag/adapter_gate_max": np.max(gate_values),
        "diag/adapter_gate_min": np.min(gate_values),
        "diag/adapter_gate_std": np.std(gate_values),
        "diag/adapter_grad_norm_mean": np.mean(adapter_grad_norms),
        "diag/adapter_grad_norm_max": np.max(adapter_grad_norms),
        "diag/adapter_gate_grad_mean": np.mean(np.abs(gate_grads)),
        "diag/adapter_n_active": sum(1 for g in gate_values if abs(g) > 0.01),
        "diag/adapter_n_total": len(gate_values),
    }
    for i, gv in enumerate(gate_values):
        metrics[f"diag/gate/layer_{i:02d}"] = gv
    if wandb_run is not None:
        try:
            wandb_run.log({**metrics, "global_step": global_step})
        except Exception:
            pass
    if tb_writer is not None:
        for k, v in metrics.items():
            if not k.startswith("diag/gate/"):
                tb_writer.add_scalar(k, v, global_step=global_step)
    if max(abs(g) for g in gate_values) < 1e-6 and global_step > 200:
        print(f"[WARNING] Step {global_step}: ALL adapter gates near-zero.")
    if max(adapter_grad_norms) < 1e-10:
        print(f"[WARNING] Step {global_step}: ALL adapter gradient norms zero.")


def _log_vram(global_step, wandb_run, log_every: int = 100):
    if global_step % log_every != 0 or not torch.cuda.is_available():
        return
    metrics = {
        "diag/vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "diag/vram_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "diag/vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    if wandb_run is not None:
        try:
            wandb_run.log({**metrics, "global_step": global_step})
        except Exception:
            pass
    if metrics["diag/vram_peak_gb"] > 70.0:
        print(f"[VRAM] WARNING: Peak {metrics['diag/vram_peak_gb']:.1f} GB")
