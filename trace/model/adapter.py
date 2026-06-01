"""Cross-attention adapter components.

SensorCrossAttentionAdapter — gated cross-attention from text hidden states to sensor memory.
AdapterWrappedLayer         — wraps an LLM decoder layer to inject adapter output.
SensorAdapterManager        — patches LLM layer list with wrapped layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.model.backbone import _get_llm_layers


class SensorCrossAttentionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 rank: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.q_proj_down = nn.Linear(hidden_dim, rank, bias=False)
        self.q_proj_up   = nn.Linear(rank, hidden_dim, bias=False)
        self.k_proj_down = nn.Linear(hidden_dim, rank, bias=False)
        self.k_proj_up   = nn.Linear(rank, hidden_dim, bias=False)
        self.v_proj_down = nn.Linear(hidden_dim, rank, bias=False)
        self.v_proj_up   = nn.Linear(rank, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.full((1,), 0.01))
        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj_down, self.q_proj_up, self.k_proj_down,
                     self.k_proj_up, self.v_proj_down, self.v_proj_up, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)

    def forward(self, text_hidden: torch.Tensor, sensor_memory: torch.Tensor) -> torch.Tensor:
        B, T, D = text_hidden.shape
        N = sensor_memory.shape[1]
        input_dtype = text_hidden.dtype
        adapter_dtype = self.norm.weight.dtype
        text_hidden_cast = text_hidden.to(adapter_dtype)
        sensor_memory_cast = sensor_memory.to(adapter_dtype)
        normed = self.norm(text_hidden_cast)
        Q = self.q_proj_up(self.q_proj_down(normed))
        K = self.k_proj_up(self.k_proj_down(sensor_memory_cast))
        V = self.v_proj_up(self.v_proj_down(sensor_memory_cast))
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout.p if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output).to(input_dtype)
        return text_hidden + self.gate.to(input_dtype) * attn_output


class AdapterWrappedLayer(nn.Module):
    def __init__(self, original_layer: nn.Module,
                 adapter: SensorCrossAttentionAdapter,
                 sensor_memory_ref: list):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter
        self.sensor_memory_ref = sensor_memory_ref

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, *args, **kwargs):
        output = self.original_layer(*args, **kwargs)
        sensor_mem = self.sensor_memory_ref[0]
        if sensor_mem is None:
            return output
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Move sensor_mem to this layer's device (supports device_map="auto").
            modified = self.adapter(hidden_states, sensor_mem.to(hidden_states.device))
            return (modified,) + output[1:]
        else:
            return self.adapter(output, sensor_mem.to(output.device))


class SensorAdapterManager:
    def __init__(self, llm, adapters: nn.ModuleList,
                 layer_indices: list[int], sensor_memory_ref: list):
        self.layer_indices = layer_indices
        self.original_layers = {}
        self.sensor_memory_ref = sensor_memory_ref
        layers = _get_llm_layers(llm)
        if len(adapters) != len(layer_indices):
            raise ValueError(f"Adapters ({len(adapters)}) != layer indices ({len(layer_indices)})")
        if not layer_indices:
            print("[adapters] No adapter layers requested (none mode) — cross-attention disabled.")
            return
        for adapter, layer_idx in zip(adapters, layer_indices):
            self.original_layers[layer_idx] = layers[layer_idx]
            # Co-locate adapter with its layer (supports device_map="auto").
            try:
                layer_device = next(layers[layer_idx].parameters()).device
                adapter = adapter.to(layer_device)
            except StopIteration:
                pass
            layers[layer_idx] = AdapterWrappedLayer(layers[layer_idx], adapter, sensor_memory_ref)
        print(f"[adapters] Wrapped {len(layer_indices)} layers with adapters: {layer_indices}")

    def restore_layers(self, llm):
        layers = _get_llm_layers(llm)
        for layer_idx, original in self.original_layers.items():
            layers[layer_idx] = original
        self.original_layers.clear()
