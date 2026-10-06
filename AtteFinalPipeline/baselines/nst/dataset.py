import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NSTDataset(Dataset):
    def __init__(
        self,
        dataset,
        window,
        stride,
        stride_test,
        path_processed,
        prefix,
        drop_null=False,
        null_label=0,
    ):
        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.stride_test = stride_test
        self.prefix = prefix
        self.path_processed = path_processed
        self.drop_null = drop_null
        self.null_label = null_label
        if dataset == "ucihar":
            base = path_processed
            x_full = np.load(os.path.join(base, "X_train.npy")).astype(np.float32)
            y_full = np.load(os.path.join(base, "y_train.npy")).astype(np.int64)
            val_ratio = 0.2
            split_idx = int(len(x_full) * (1 - val_ratio))
            if prefix == "train":
                self.data, self.target = (x_full[:split_idx], y_full[:split_idx])
            elif prefix == "val":
                self.data, self.target = (x_full[split_idx:], y_full[split_idx:])
            elif prefix == "test":
                self.data = np.load(os.path.join(base, "X_test.npy")).astype(np.float32)
                self.target = np.load(os.path.join(base, "y_test.npy")).astype(np.int64)
            else:
                raise ValueError(f"Unknown prefix: {prefix}")
            if self.target.min() > 0:
                self.target = self.target - 1
        else:
            if prefix == "test" and stride_test == 1:
                fname = "test_sample_wise.npz"
                path = os.path.join(path_processed, fname)
                if not os.path.exists(path):
                    print(f"[!] {fname} not found, falling back to test.npz")
                    fname = "test.npz"
            else:
                fname = f"{prefix}.npz"
            path = os.path.join(path_processed, fname)
            arr = np.load(path)
            self.data = arr["data"].astype(np.float32)
            self.target = arr["target"].astype(np.int64).reshape(-1)
        if self.drop_null:
            n_before = self.data.shape[0]
            mask = self.target != self.null_label
            self.data = self.data[mask]
            self.target = self.target[mask]
            if self.null_label == 0:
                self.target = self.target - 1
            else:
                unique_sorted = sorted(set(self.target.tolist()))
                remap = {lbl: i for i, lbl in enumerate(unique_sorted)}
                self.target = np.array(
                    [remap[int(t)] for t in self.target], dtype=np.int64
                )
            n_after = self.data.shape[0]
            print(
                f"[drop_null] {prefix}: kept {n_after}/{n_before} windows ({100.0 * n_after / max(1, n_before):.1f}%)"
            )
        self.len = self.data.shape[0]
        assert self.data.shape[0] == self.target.shape[0]
        if prefix == "train":
            self.weight_samples = self._compute_weights()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()
        y = torch.tensor(int(self.target[index]), dtype=torch.long)
        return (x, y, index)

    def _compute_weights(self):
        target = self.target
        labels = sorted(set(target.tolist()))
        counts = np.array([np.sum(target == lbl) for lbl in labels], dtype=np.float64)
        weight_per_class = 1.0 / counts
        lbl_to_w = {lbl: w for lbl, w in zip(labels, weight_per_class)}
        weights = np.array([lbl_to_w[int(t)] for t in target], dtype=np.float64)
        return torch.from_numpy(weights).double()
