import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosPositionalEmbedding(nn.Module):
    def __init__(self, n_pos: int, d_model: int):
        super().__init__()
        pe = torch.zeros(n_pos, d_model)
        pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(-2), :]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, n_pos: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_pos, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, : x.size(-2), :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class RevIN(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-05, affine: bool = True):
        super().__init__()
        self.eps, self.affine = (eps, affine)
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))
        self._mean = None
        self._stdev = None

    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(dim=-1, keepdim=True).detach()
            self._stdev = torch.sqrt(
                x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias.view(1, -1, 1)) / (
                    self.weight.view(1, -1, 1) + self.eps
                )
            return x * self._stdev + self._mean
        raise ValueError(mode)


class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_class: int,
        window: int,
        patch_len: int,
        patch_stride: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.0,
        pos_embed: str = "learnable",
        revin: bool = False,
    ):
        super().__init__()
        assert patch_len <= window, (
            f"patch_len ({patch_len}) must be <= window ({window})"
        )
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.input_dim = input_dim
        self.num_class = num_class
        self.window = window
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        if window <= patch_len:
            self.pad_len = 0
            self.n_patches = 1
        else:
            remainder = (window - patch_len) % patch_stride
            self.pad_len = patch_stride - remainder if remainder != 0 else 0
            padded_T = window + self.pad_len
            self.n_patches = (padded_T - patch_len) // patch_stride + 1
        self.use_revin = revin
        if revin:
            self.revin = RevIN(input_dim)
        self.patch_embed = nn.Linear(patch_len, d_model)
        if pos_embed == "learnable":
            self.pos = LearnablePositionalEmbedding(self.n_patches, d_model)
        elif pos_embed == "sincos":
            self.pos = SinCosPositionalEmbedding(self.n_patches, d_model)
        else:
            raise ValueError(pos_embed)
        self.pos_drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, num_class),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, C = x.shape
        assert T == self.window, f"Expected window length {self.window}, got {T}"
        assert C == self.input_dim, f"Expected {self.input_dim} channels, got {C}"
        x = x.transpose(1, 2)
        if self.use_revin:
            x = self.revin(x, mode="norm")
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        x = self.patch_embed(x)
        x = self.pos(x)
        x = self.pos_drop(x)
        BC = B * C
        x = x.reshape(BC, self.n_patches, self.d_model)
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        x = x.reshape(B, C, self.n_patches, self.d_model)
        x = x.mean(dim=2)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits


def create_patchtst(config_model):
    return PatchTSTClassifier(**config_model)
