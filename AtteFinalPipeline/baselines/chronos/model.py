import os
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class ChronosEmbedder:
    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "cuda"):
        try:
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting >= 2.1.0 is required for Chronos-2 embeddings.\nInstall with:  pip install 'chronos-forecasting>=2.1.0'"
            ) from e
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        print(
            f"[ChronosEmbedder] Loading {model_id} (dtype={dtype}, device={device}) ..."
        )
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=torch_dtype
        )
        with torch.no_grad():
            dummy = torch.randn(8, dtype=torch.float32)
            emb_list, _ = self.pipeline.embed([dummy])
        self.d_model = int(emb_list[0].shape[-1])
        print(f"[ChronosEmbedder] d_model = {self.d_model}")

    @torch.no_grad()
    def embed_windows(
        self,
        windows: np.ndarray,
        batch_size: int = 256,
        context_length: int = 512,
        pool: str = "mean",
        show_progress: bool = True,
    ) -> np.ndarray:
        N, T, C = windows.shape
        D = C * self.d_model
        bs_w = max(1, batch_size // max(C, 1))
        out = np.empty((N, D), dtype=np.float32)
        iters = range(0, N, bs_w)
        if show_progress:
            iters = tqdm(iters, desc=f"embed (bs_w={bs_w}, C={C})")
        for start in iters:
            end = min(start + bs_w, N)
            chunk = windows[start:end]
            series_list = []
            for w in range(end - start):
                for c in range(C):
                    series_list.append(
                        torch.from_numpy(chunk[w, :, c].astype(np.float32))
                    )
            emb_list, _ = self.pipeline.embed(
                series_list, batch_size=batch_size, context_length=context_length
            )
            for i, e in enumerate(emb_list):
                e = e.squeeze(0).float()
                if pool == "mean":
                    v = e.mean(dim=0)
                elif pool == "last":
                    v = e[-1]
                else:
                    raise ValueError(f"unknown pool: {pool}")
                w_idx = i // C
                c_idx = i % C
                out[
                    start + w_idx, c_idx * self.d_model : (c_idx + 1) * self.d_model
                ] = v.cpu().numpy().astype(np.float32)
        return out


class MLPHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_class: int, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class),
        )

    def forward(self, x):
        return self.net(x)


def cache_key(args, prefix: str) -> str:
    parts = [
        args.dataset,
        prefix,
        str(args.window),
        str(args.stride),
        str(args.stride_test),
        args.path_processed,
        getattr(args, "path_data", ""),
        args.chronos_model_id,
        args.chronos_dtype,
        str(args.chronos_context_len),
        args.embed_pool,
        f"drop_null={int(bool(getattr(args, 'drop_null', False)))}",
    ]
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return f"{args.dataset}_{prefix}_{h}"


def cache_paths(args, prefix: str):
    os.makedirs(args.cache_dir, exist_ok=True)
    key = cache_key(args, prefix)
    return (
        os.path.join(args.cache_dir, f"{key}.x.npy"),
        os.path.join(args.cache_dir, f"{key}.y.npy"),
        os.path.join(args.cache_dir, f"{key}.meta.txt"),
    )


def load_or_build_embeddings(args, prefix: str, embedder, force: bool = False):
    from AtteFinalPipeline.baselines.chronos.dataset import load_split

    x_path, y_path, meta_path = cache_paths(args, prefix)
    if not force and os.path.exists(x_path) and os.path.exists(y_path):
        emb = np.load(x_path)
        y = np.load(y_path)
        print(f"[cache] Loaded {prefix}: emb={emb.shape}  y={y.shape}  ({x_path})")
        return (emb, y)
    print(f"[cache] Building {prefix} embeddings (target: {x_path})")
    x, y = load_split(args, prefix)
    t0 = time.time()
    emb = embedder.embed_windows(
        x,
        batch_size=args.chronos_batch_size,
        context_length=args.chronos_context_len,
        pool=args.embed_pool,
        show_progress=True,
    )
    dt = time.time() - t0
    print(f"[cache] Built {prefix} in {dt:.1f}s  → emb={emb.shape}")
    np.save(x_path, emb)
    np.save(y_path, y)
    with open(meta_path, "w") as f:
        f.write(
            f"dataset        = {args.dataset}\nprefix         = {prefix}\nwindow         = {args.window}\nstride         = {args.stride}\nstride_test    = {args.stride_test}\npath_processed = {args.path_processed}\nchronos_model  = {args.chronos_model_id}\ndtype          = {args.chronos_dtype}\ncontext_length = {args.chronos_context_len}\nembed_pool     = {args.embed_pool}\ndrop_null      = {bool(getattr(args, 'drop_null', False))}\nd_model        = {embedder.d_model}\nemb_shape      = {emb.shape}\ny_shape        = {y.shape}\nbuild_seconds  = {dt:.1f}\n"
        )
    return (emb, y)
