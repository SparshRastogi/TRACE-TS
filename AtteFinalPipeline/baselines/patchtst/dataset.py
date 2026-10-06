import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

__all__ = ["PatchTSTSensorDataset"]


class PatchTSTSensorDataset(Dataset):
    def __init__(
        self,
        dataset,
        window,
        stride,
        stride_test,
        path_processed,
        prefix,
        drop_null=False,
        null_label=None,
    ):
        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.stride_test = stride_test
        self.prefix = prefix
        self.path_processed = path_processed
        self.drop_null = drop_null
        self.null_label = null_label if null_label is not None else 0
        if self.dataset == "ucihar":
            base_path = path_processed
            x_train_full = np.load(os.path.join(base_path, "X_train.npy")).astype(
                np.float32
            )
            y_train_full = np.load(os.path.join(base_path, "y_train.npy")).astype(
                np.int64
            )
            val_ratio = 0.2
            split_idx = int(len(x_train_full) * (1 - val_ratio))
            if prefix == "train":
                self.data = x_train_full[:split_idx]
                self.target = y_train_full[:split_idx]
            elif prefix == "val":
                self.data = x_train_full[split_idx:]
                self.target = y_train_full[split_idx:]
            elif prefix == "test":
                self.data = np.load(os.path.join(base_path, "X_test.npy")).astype(
                    np.float32
                )
                self.target = np.load(os.path.join(base_path, "y_test.npy")).astype(
                    np.int64
                )
            else:
                raise ValueError(f"Unknown prefix: {prefix}")
            if self.target.size > 0 and int(self.target.min()) > 0:
                self.target = self.target - 1
        else:
            if prefix == "test" and stride_test == 1:
                npz_name = "test_sample_wise.npz"
            else:
                npz_name = f"{prefix}.npz"
            self.path_dataset = os.path.join(path_processed, npz_name)
            with np.load(self.path_dataset) as f:
                self.data = f["data"].astype(np.float32)
                self.target = f["target"].astype(np.int64)
        if self.drop_null:
            n_before = self.data.shape[0]
            keep_mask = self.target != self.null_label
            self.data = self.data[keep_mask]
            self.target = self.target[keep_mask]
            shift_mask = self.target > self.null_label
            self.target[shift_mask] = self.target[shift_mask] - 1
            n_after = self.data.shape[0]
            pct_dropped = 100.0 * (1.0 - n_after / max(n_before, 1))
            print(
                f"[Dataset][drop_null] {self.dataset} {self.prefix}: removed label=={self.null_label}  {n_before} → {n_after} samples ({pct_dropped:.1f}% dropped); labels remapped (shift down by 1 for labels > {self.null_label})"
            )
        self.len = self.data.shape[0]
        assert self.data.shape[0] == self.target.shape[0]
        if self.len == 0:
            raise RuntimeError(
                f"[Dataset] {self.dataset} {self.prefix} is EMPTY after drop_null filtering (null_label={self.null_label}). This split has no non-null samples."
            )
        print(
            f"[Dataset] {self.dataset} {self.prefix}: {self.len} samples, data={self.data.shape}, target dtype={self.target.dtype}, label range=[{int(self.target.min())}, {int(self.target.max())}]"
        )
        if prefix == "train":
            self.weight_samples = self._get_weights()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        target = torch.LongTensor([int(self.target[index])])
        idx = torch.from_numpy(np.array(index))
        return (data, target, idx)

    def _get_weights(self):
        target = self.target
        target_count = np.array(
            [np.sum(target == lbl) for lbl in sorted(set(target.tolist()))]
        )
        weight_target = 1.0 / target_count
        class_to_w = {
            lbl: weight_target[i] for i, lbl in enumerate(sorted(set(target.tolist())))
        }
        weight_samples = np.array([class_to_w[int(t)] for t in target])
        weight_samples = torch.from_numpy(weight_samples).double()
        return weight_samples
