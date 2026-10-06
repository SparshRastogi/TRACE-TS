import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_ucihar(args, prefix):
    base = args.path_data
    if prefix in ("train", "val"):
        x_full = np.load(os.path.join(base, "X_train.npy")).astype(np.float32)
        y_full = np.load(os.path.join(base, "y_train.npy")).astype(np.int64)
        val_ratio = 0.2
        split_idx = int(len(x_full) * (1 - val_ratio))
        if prefix == "train":
            x = x_full[:split_idx]
            y = y_full[:split_idx]
        else:
            x = x_full[split_idx:]
            y = y_full[split_idx:]
    elif prefix == "test":
        x = np.load(os.path.join(base, "X_test.npy")).astype(np.float32)
        y = np.load(os.path.join(base, "y_test.npy")).astype(np.int64)
    else:
        raise ValueError(f"Unknown prefix: {prefix}")
    if y.min() > 0:
        y = y - 1
    return (x.astype(np.float32), y.astype(np.int64))


def _load_npz(args, prefix):
    if prefix == "test" and args.stride_test == 1:
        path = os.path.join(args.path_processed, "test_sample_wise.npz")
    else:
        path = os.path.join(args.path_processed, f"{prefix}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed split not found: {path}\nRun A&D's preprocess.py first to generate it."
        )
    z = np.load(path)
    x = z["data"].astype(np.float32)
    y = z["target"].astype(np.int64).reshape(-1)
    return (x, y)


def _drop_null_and_remap(x: np.ndarray, y: np.ndarray, prefix: str, dataset: str):
    mask = y != 0
    n_total = int(y.shape[0])
    n_keep = int(mask.sum())
    n_drop = n_total - n_keep
    if n_keep == 0:
        raise ValueError(
            f"[drop_null/{dataset}/{prefix}] All {n_total} samples had label 0; nothing left after filtering. Check the dataset config."
        )
    x_kept = x[mask]
    y_kept = y[mask] - 1
    print(
        f"[drop_null/{dataset}/{prefix}] kept {n_keep}/{n_total} samples (dropped {n_drop} null = {100.0 * n_drop / max(n_total, 1):.1f}%); label range now {int(y_kept.min())}..{int(y_kept.max())}"
    )
    return (x_kept.astype(np.float32, copy=False), y_kept.astype(np.int64, copy=False))


def load_split(args, prefix):
    if args.dataset == "ucihar":
        x, y = _load_ucihar(args, prefix)
    else:
        x, y = _load_npz(args, prefix)
    if getattr(args, "drop_null", False):
        x, y = _drop_null_and_remap(x, y, prefix=prefix, dataset=args.dataset)
    return (x, y)


class ChronosWindowDataset(Dataset):
    def __init__(self, args, prefix):
        self.args = args
        self.prefix = prefix
        self.x, self.y = load_split(args, prefix)
        assert self.x.ndim == 3, f"expected (N,T,C), got {self.x.shape}"
        assert self.x.shape[0] == self.y.shape[0]
        self.len = self.x.shape[0]
        if prefix == "train":
            self.weight_samples = self._compute_weights()
        n_class = args.num_class
        if self.y.min() < 0 or self.y.max() >= n_class:
            raise ValueError(
                f"[{args.dataset}/{prefix}] labels {self.y.min()}..{self.y.max()} outside [0, {n_class - 1}]"
            )
        print(
            f"[ChronosWindowDataset/{args.dataset}/{prefix}] x={self.x.shape}  y={self.y.shape}  classes={sorted(np.unique(self.y).tolist())}"
        )

    def _compute_weights(self):
        labels = self.y
        counts = np.array([np.sum(labels == c) for c in sorted(set(labels.tolist()))])
        inv = {c: 1.0 / cnt for c, cnt in zip(sorted(set(labels.tolist())), counts)}
        w = np.array([inv[int(t)] for t in labels], dtype=np.float64)
        return torch.from_numpy(w).double()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return (x, y, torch.tensor(idx, dtype=torch.long))


class ChronosEmbeddingDataset(Dataset):
    def __init__(self, emb: np.ndarray, y: np.ndarray, prefix: str = "train"):
        assert emb.ndim == 2, f"expected (N,D), got {emb.shape}"
        assert emb.shape[0] == y.shape[0]
        self.emb = emb.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.len = self.emb.shape[0]
        self.prefix = prefix
        if prefix == "train":
            self.weight_samples = self._compute_weights()

    def _compute_weights(self):
        labels = self.y
        counts = np.array([np.sum(labels == c) for c in sorted(set(labels.tolist()))])
        inv = {c: 1.0 / cnt for c, cnt in zip(sorted(set(labels.tolist())), counts)}
        w = np.array([inv[int(t)] for t in labels], dtype=np.float64)
        return torch.from_numpy(w).double()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.from_numpy(self.emb[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return (x, y, torch.tensor(idx, dtype=torch.long))
