from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from AtteFinalPipeline.expert.utils.utils import paint
from AtteFinalPipeline.expert.utils.plot import plot_pie, plot_segment
from AtteFinalPipeline.data.preprocess import preprocess_pipeline
from AtteFinalPipeline.expert.settings import get_args

__all__ = ["SensorDataset"]


class SensorDataset(Dataset):
    def __init__(
        self,
        dataset,
        window,
        stride,
        stride_test,
        path_processed,
        prefix,
        transform=None,
        verbose=False,
        path_data=None,
    ):
        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.prefix = prefix
        self.transform = transform
        self.path_processed = path_processed
        self.verbose = verbose
        if self.dataset == "ucihar":
            base_path = path_data or str(UCIHAR_DATA_DIR)
            x_train_full = np.load(os.path.join(base_path, "X_train.npy")).astype(
                np.float32
            )
            y_train_full = np.load(os.path.join(base_path, "y_train.npy")).astype(
                np.longlong
            )
            val_ratio = 0.2
            split_idx = int(len(x_train_full) * (1 - val_ratio))
            if prefix == "train":
                self.data = x_train_full[:split_idx]
                self.target = y_train_full[:split_idx]
                print(paint(f"[UCI HAR] Loaded TRAIN split: {len(self.data)} samples"))
            elif prefix == "val":
                self.data = x_train_full[split_idx:]
                self.target = y_train_full[split_idx:]
                print(paint(f"[UCI HAR] Loaded VAL split: {len(self.data)} samples"))
            elif prefix == "test":
                self.data = np.load(os.path.join(base_path, "X_test.npy")).astype(
                    np.float32
                )
                self.target = np.load(os.path.join(base_path, "y_test.npy")).astype(
                    np.longlong
                )
                print(paint(f"[UCI HAR] Loaded TEST set: {len(self.data)} samples"))
            if np.min(self.target) > 0:
                self.target = self.target - 1
            self.len = self.data.shape[0]
        else:
            if prefix == "test" and stride_test == 1:
                self.path_dataset = os.path.join(path_processed, "test_sample_wise.npz")
            else:
                self.path_dataset = os.path.join(
                    path_processed, "{}.npz".format(prefix)
                )
            dataset = np.load(self.path_dataset)
            self.data = dataset["data"]
            self.target = dataset["target"]
            self.len = self.data.shape[0]
        assert self.data.shape[0] == self.target.shape[0]
        print(
            paint(
                f"[STEP 2] Creating {self.dataset} {self.prefix} HAR dataset of size {self.len} ..."
            )
        )
        if self.verbose:
            self.get_info()
            self.get_distribution()
        if prefix == "train":
            self.weight_samples = self.get_weights()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.transform is None:
            data = torch.FloatTensor(self.data[index])
            target = torch.LongTensor([int(self.target[index])])
            idx = torch.from_numpy(np.array(index))
        return (data, target, idx)

    def get_info(self, n_samples=3):
        print(paint(f"[-] Information on {self.prefix} dataset:"))
        print("\t data: ", self.data.shape, self.data.dtype, type(self.data))
        print("\t target: ", self.target.shape, self.target.dtype, type(self.target))
        target_idx = [np.where(self.target == label)[0] for label in set(self.target)]
        target_idx_samples = np.array(
            [np.random.choice(idx, n_samples, replace=False) for idx in target_idx]
        ).flatten()
        for i, random_idx in enumerate(target_idx_samples):
            data, target, index = self.__getitem__(random_idx)
            if i == 0:
                print(paint(f"[-] Information on segment #{random_idx}/{self.len}:"))
                print("\t data: ", data.shape, data.dtype, type(data))
                print("\t target: ", target.shape, target.dtype, type(target))
                print("\t index: ", index, index.shape, index.dtype, type(index))
            path_save = os.path.join(self.path_processed, "segments")
            plot_segment(
                data,
                target,
                index=index,
                prefix=self.prefix,
                path_save=path_save,
                num_class=len(target_idx),
            )

    def get_distribution(self):
        plot_pie(
            self.target, self.prefix, os.path.join(self.path_processed, "distribution")
        )

    def get_weights(self):
        target = self.target
        target_count = np.array([np.sum(target == label) for label in set(target)])
        weight_target = 1.0 / target_count
        weight_samples = np.array([weight_target[t] for t in target])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.double()
        if self.verbose:
            (print(paint("[-] Target sampling weights:")),)
            print(weight_target)
        return weight_samples


def main():
    args, config_dataset, _ = get_args()
    if args.dataset not in ("ucihar",):
        preprocess_pipeline(args)
    dataset = SensorDataset(**config_dataset, prefix="train", verbose=True)


if __name__ == "__main__":
    main()
