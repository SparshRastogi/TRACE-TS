"""Top-level model combining frozen LLM + SensorProjector + cross-attention adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.model.projector import SensorProjector
from trace.model.adapter import SensorCrossAttentionAdapter, SensorAdapterManager


class SensorLLMCrossAttn(nn.Module):
    def __init__(self, llm, projector: SensorProjector,
                 adapters: nn.ModuleList, layer_indices: list[int]):
        super().__init__()
        self.llm       = llm
        self.projector = projector
        self.adapters  = adapters
        for param in self.llm.parameters():
            param.requires_grad = False
        self._sensor_memory_ref = [None]
        self._adapter_manager = SensorAdapterManager(
            llm, adapters, layer_indices, self._sensor_memory_ref
        )

    def forward(self, sensor_embeds, input_ids, attention_mask,
                labels=None, token_weights=None):
        device = next(self.llm.parameters()).device
        llm_dtype = next(self.llm.parameters()).dtype
        sensor_memory = self.projector(sensor_embeds.to(device)).to(dtype=llm_dtype)
        self._sensor_memory_ref[0] = sensor_memory

        # When token_weights provided, get logits only and compute weighted loss manually.
        # Otherwise let the LLM compute the standard mean cross-entropy internally.
        lm_out = self.llm(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device) if (labels is not None and token_weights is None) else None,
        )

        if labels is not None and token_weights is not None:
            shift_logits  = lm_out.logits[..., :-1, :].contiguous()
            shift_labels  = labels[..., 1:].to(device).contiguous()
            shift_weights = token_weights[..., 1:].to(device=device, dtype=torch.float32).contiguous()
            per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100, reduction="none",
            )
            active = shift_labels.view(-1) != -100
            denom  = shift_weights.view(-1)[active].sum().clamp(min=1.0)
            lm_out.loss = (per_tok * shift_weights.view(-1)).sum() / denom
            # Unweighted loss for diagnostic logging (not used in backprop).
            lm_out.unweighted_lm_loss = per_tok[active].mean()

        return lm_out
