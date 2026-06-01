"""Training callbacks: BlindfoldMonitor, EarlyStopping, startup checks."""

import torch
import torch.nn as nn

from trace.model.backbone import _get_llm_layers, _get_llm_dim
from trace.model.adapter import AdapterWrappedLayer


class BlindfoldMonitor:
    def __init__(self, diagnostic_batch: dict):
        self.batch = diagnostic_batch

    @torch.no_grad()
    def check(self, model) -> dict:
        model.eval()
        try:
            b = self.batch
            device = next(model.llm.parameters()).device
            llm_dtype = next(model.llm.parameters()).dtype
            sensor_mem_real = model.projector(b["sensor_embed"].to(device))
            sensor_mem_cast = sensor_mem_real.to(dtype=llm_dtype)
            ids, mask, lbl = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
            model._sensor_memory_ref[0] = sensor_mem_cast
            loss_real = model.llm(input_ids=ids, attention_mask=mask, labels=lbl).loss.item()
            model._sensor_memory_ref[0] = torch.zeros_like(sensor_mem_cast)
            loss_blind = model.llm(input_ids=ids, attention_mask=mask, labels=lbl).loss.item()
            saved_gates = [a.gate.data.clone() for a in model.adapters]
            for a in model.adapters:
                a.gate.data.zero_()
            model._sensor_memory_ref[0] = sensor_mem_cast
            loss_no_adapter = model.llm(input_ids=ids, attention_mask=mask, labels=lbl).loss.item()
            for a, sg in zip(model.adapters, saved_gates):
                a.gate.data.copy_(sg)
            return {
                "loss_real": loss_real, "loss_blind": loss_blind,
                "loss_no_adapter": loss_no_adapter,
                "ratio": loss_real / (loss_blind + 1e-8),
                "adapter_contribution_ratio": loss_real / (loss_no_adapter + 1e-8),
            }
        finally:
            model.train()


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.best_epoch, self.counter, self.triggered = float("inf"), 0, 0, False

    def step(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.counter = val_loss, epoch, 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False

    def status_str(self) -> str:
        return f"patience={self.counter}/{self.patience}  best_val_total_loss={self.best_loss:.4f} @ epoch {self.best_epoch}"


def run_startup_checks(cfg: dict, llm, projector, embeddings, device: str):
    print("[checks] Running startup checks...")
    errors, warnings = [], []
    input_dim, llm_dim, n_tokens = embeddings.shape[1], _get_llm_dim(llm), cfg["n_tokens"]
    if input_dim <= 0:
        errors.append(f"[SA7] input_dim={input_dim} not positive.")
    if llm_dim <= 0:
        errors.append(f"[SA7] llm_dim={llm_dim} not positive.")
    try:
        proj_mod = getattr(projector, "_orig_mod", projector)
        if proj_mod.fc_expand.in_features != input_dim:
            errors.append(f"[SA7] Projector in={proj_mod.fc_expand.in_features} != input_dim={input_dim}")
        if proj_mod.fc_expand.out_features != n_tokens * llm_dim:
            errors.append(f"[SA7] Projector out={proj_mod.fc_expand.out_features} != {n_tokens}*{llm_dim}")
    except AttributeError as e:
        warnings.append(f"[SA7] Could not verify projector dims ({e}).")
    try:
        projector.eval()
        with torch.no_grad():
            test_out = projector(torch.randn(1, input_dim, device=device, dtype=torch.float32))
        if test_out.dtype != torch.float32:
            errors.append(f"[SA4] Projector output dtype={test_out.dtype}, expected float32.")
        if test_out.shape != (1, n_tokens, llm_dim):
            errors.append(f"[SA4] Projector output shape={test_out.shape}")
        projector.train()
    except Exception as e:
        warnings.append(f"[SA4] Could not verify projector output ({e}).")
    try:
        import transformers
        parts = transformers.__version__.split(".")
        if int(parts[0]) < 4 or (int(parts[0]) == 4 and int(parts[1]) < 36):
            errors.append(f"[SA1] transformers=={transformers.__version__} too old.")
    except Exception as e:
        warnings.append(f"[SA1] Could not check transformers version ({e}).")
    try:
        llm.eval()
        P = 8
        emb = torch.randn(1, P, llm_dim, dtype=next(llm.parameters()).dtype, device=device)
        mask_full = torch.ones(1, P, dtype=torch.long, device=device)
        mask_half = mask_full.clone()
        mask_half[0, P // 2:] = 0
        with torch.no_grad():
            out_full = llm(inputs_embeds=emb, attention_mask=mask_full)
            out_half = llm(inputs_embeds=emb, attention_mask=mask_half)
        diff = (out_full.logits - out_half.logits).abs().max().item()
        if diff < 1e-6:
            errors.append("[SA1] FlashAttn2 mask check FAILED.")
        else:
            print(f"[checks] [SA1] FlashAttn2 mask OK (diff={diff:.4f})")
        llm.train()
    except Exception as e:
        warnings.append(f"[SA1] Could not run FlashAttn2 mask check ({e}).")
    for w in warnings:
        print(f"[checks] WARNING: {w}")
    if errors:
        raise RuntimeError(f"\n{'='*70}\nSTARTUP CHECK FAILED\n{'='*70}\n" + "\n\n".join(errors))
    print("[checks] All basic startup checks passed.")


def run_crossattn_startup_checks(model, adapters: nn.ModuleList,
                                  sample_batch: dict, cfg: dict, device: str):
    print("[checks] Running cross-attention startup checks...")
    n_adapters = len(adapters)
    print(f"[checks] [CA1] LLM layers: {len(_get_llm_layers(model.llm))}, Adapters: {n_adapters}")
    if n_adapters == 0:
        print("[checks] [CA2-CA6] Skipped — no adapters (none mode).")
        return
    bad_gates = [(i, a.gate.item()) for i, a in enumerate(adapters) if abs(a.gate.item() - 0.01) > 1e-6]
    if bad_gates:
        print(f"[checks] WARNING [CA2]: {len(bad_gates)} unexpected gate values: {bad_gates[:5]}")
    else:
        print(f"[checks] [CA2] Gate init: all {n_adapters} at 0.01")
    model.train()
    saved_gates = [a.gate.data.clone() for a in adapters]
    for a in adapters:
        a.gate.data.fill_(0.1)
        for name, p in a.named_parameters():
            if not p.requires_grad:
                raise RuntimeError(f"[CA3] Adapter param '{name}' frozen!")
    print(f"[checks] [CA3] All adapter params have requires_grad=True")
    with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        lm_out = model(sample_batch["sensor_embed"], sample_batch["input_ids"],
                       sample_batch["attention_mask"], sample_batch["labels"])
    print(f"[checks] [CA3] loss={lm_out.loss.item():.4f}")
    lm_out.loss.backward()
    dead_adapters = []
    for i, adapter in enumerate(adapters):
        has_grad = any(p.grad is not None and p.grad.abs().max().item() > 1e-15
                       for p in adapter.parameters())
        if not has_grad:
            dead_adapters.append(i)
    for a, sg in zip(adapters, saved_gates):
        a.gate.data.copy_(sg)
    model.zero_grad()
    if dead_adapters:
        print(f"[checks] WARNING [CA3]: {len(dead_adapters)}/{n_adapters} adapters showed no gradients.")
        print(f"  Monitor 'diag/adapter_gate_mean' in W&B. Dead: {dead_adapters[:10]}")
    else:
        print(f"[checks] [CA3] All {n_adapters} adapters receiving gradients")
    with torch.no_grad():
        test_mem = model.projector(sample_batch["sensor_embed"].to(device))
    if test_mem.shape[-1] != _get_llm_dim(model.llm):
        raise RuntimeError(f"[CA4] Sensor memory dim mismatch")
    print(f"[checks] [CA4] Sensor memory dim matches LLM")
    n_wrapped = len(model._adapter_manager.layer_indices)
    if n_wrapped != n_adapters:
        raise RuntimeError(f"[CA5] Wrapped={n_wrapped} != adapters={n_adapters}")
    llm_layers = _get_llm_layers(model.llm)
    for idx in model._adapter_manager.layer_indices:
        if not isinstance(llm_layers[idx], AdapterWrappedLayer):
            raise RuntimeError(f"[CA5] Layer {idx} not wrapped")
    print(f"[checks] [CA5] {n_wrapped} layers wrapped")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.amp.autocast(enabled=True):
            out = model(sample_batch["sensor_embed"], sample_batch["input_ids"],
                        sample_batch["attention_mask"], sample_batch["labels"])
        out.loss.backward()
        model.zero_grad()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[checks] [CA6] Peak VRAM: {peak_gb:.1f} GB "
              f"(GC: {'ON' if cfg.get('gradient_checkpointing') else 'OFF'})")
        if peak_gb > 75.0:
            print(f"[checks] WARNING [CA6]: Close to 80 GB limit.")
    print("[checks] All cross-attention startup checks passed")
