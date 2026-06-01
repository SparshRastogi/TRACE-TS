"""SensorProjector: maps flat sensor embedding to N tokens in LLM space."""

import torch
import torch.nn as nn


class SensorProjector(nn.Module):
    def __init__(self, input_dim: int, llm_dim: int, n_tokens: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.llm_dim   = llm_dim
        self.n_tokens  = n_tokens
        self.fc_expand = nn.Linear(input_dim, n_tokens * llm_dim)
        self.act       = nn.GELU()
        self.dropout   = nn.Dropout(dropout)
        self.fc_refine = nn.Linear(llm_dim, llm_dim)
        self.ln        = nn.LayerNorm(llm_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_expand.weight)
        nn.init.zeros_(self.fc_expand.bias)
        nn.init.zeros_(self.fc_refine.weight)
        nn.init.zeros_(self.fc_refine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.fc_expand(x).view(B, self.n_tokens, self.llm_dim)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc_refine(h)
        return self.ln(h)
