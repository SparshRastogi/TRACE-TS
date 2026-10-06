import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, output_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, stats):
        B = x.shape[0]
        flat = stats.reshape(B, -1)
        return self.backbone(flat)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe


class DSAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tau, delta):
        B, T, _ = x.shape
        H = self.n_heads
        Q = self.q_proj(x).view(B, T, H, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        tau_b = tau.unsqueeze(1).unsqueeze(1)
        delta_b = delta.unsqueeze(1).unsqueeze(1)
        scores = scores * tau_b + delta_b
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = DSAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tau, delta):
        x = self.norm1(x + self.dropout(self.attn(x, tau, delta)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class NonStationaryTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        num_class,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        projector_hidden=64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_class = num_class
        self.value_embed = nn.Linear(input_dim, d_model)
        self.pos_enc = LearnablePositionalEncoding(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.tau_proj = Projector(input_dim, seq_len, projector_hidden, output_dim=1)
        self.delta_proj = Projector(
            input_dim, seq_len, projector_hidden, output_dim=seq_len
        )
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_class)

    def forward(self, x):
        B, T, C = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got T={T}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got C={C}"
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True, unbiased=False) + 1e-05
        x_norm = (x - mu) / sigma
        stats = torch.cat([mu, sigma], dim=1)
        tau_log = self.tau_proj(x, stats)
        tau = torch.exp(tau_log)
        delta = self.delta_proj(x, stats)
        h = self.value_embed(x_norm)
        h = self.pos_enc(h)
        h = self.embed_dropout(h)
        for layer in self.layers:
            h = layer(h, tau, delta)
        h = self.norm(h)
        feat = h.mean(dim=1)
        logits = self.head(feat)
        return (feat, logits)
