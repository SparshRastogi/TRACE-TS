from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import argparse

_DATASET_CONFIGS = {
    "capture24": dict(
        dataset="capture24",
        input_dim=3,
        num_class=6,
        window=200,
        stride=1,
        stride_test=1,
        path_data="./dataset/capture24_Willetts2018_subset.npz",
        path_raw="./data/capture24_Willetts2018_subset/raw",
        path_processed="./data/capture24_Willetts2018_subset/processed/200_1",
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
        results_dir="./results_deepconvlstm/capture24",
    ),
    "opportunity": dict(
        dataset="opportunity",
        input_dim=79,
        num_class=18,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/opportunity.mat",
        path_raw="./data/opportunity/raw",
        path_processed="./data/opportunity/processed/24_12",
        class_map=[
            "Null",
            "Open Door 1",
            "Open Door 2",
            "Close Door 1",
            "Close Door 2",
            "Open Fridge",
            "Close Fridge",
            "Open Dishwasher",
            "Close Dishwasher",
            "Open Drawer 1",
            "Close Drawer 1",
            "Open Drawer 2",
            "Close Drawer 2",
            "Open Drawer 3",
            "Close Drawer 3",
            "Clean Table",
            "Drink from Cup",
            "Toggle Switch",
        ],
        results_dir="./results_deepconvlstm/opportunity",
    ),
    "pamap2": dict(
        dataset="pamap2",
        input_dim=52,
        num_class=12,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/pamap2.mat",
        path_raw="./data/pamap2/raw",
        path_processed="./data/pamap2/processed/24_12",
        class_map=[
            "Rope Jumping",
            "Lying",
            "Sitting",
            "Standing",
            "Walking",
            "Running",
            "Cycling",
            "Nordic Walking",
            "Ascending Stairs",
            "Descending Stairs",
            "Vacuum Cleaning",
            "Ironing",
        ],
        results_dir="./results_deepconvlstm/pamap2",
    ),
    "uschad": dict(
        dataset="uschad",
        input_dim=6,
        num_class=12,
        window=400,
        stride=200,
        stride_test=1,
        path_data="./dataset/uschad.mat",
        path_raw="./data/uschad/raw",
        path_processed="./data/uschad/processed/400_200",
        class_map=list(USCHAD_CLASS_MAP),
        results_dir="./results_deepconvlstm/uschad",
    ),
    "ucihar": dict(
        dataset="ucihar",
        input_dim=9,
        num_class=6,
        window=128,
        stride=64,
        stride_test=64,
        path_data=str(UCIHAR_DATA_DIR),
        path_raw="./data/ucihar/raw",
        path_processed="./data/ucihar/processed/128_64",
        class_map=[
            "Walking",
            "Walking Upstairs",
            "Walking Downstairs",
            "Sitting",
            "Standing",
            "Laying",
        ],
        results_dir="./results_deepconvlstm/ucihar",
    ),
    "mhealth": dict(
        dataset="mhealth",
        input_dim=23,
        num_class=13,
        window=100,
        stride=50,
        stride_test=1,
        path_data="./dataset/mhealth.mat",
        path_raw="./data/mhealth/raw",
        path_processed="./data/mhealth/processed/100_50",
        class_map=[
            "Null",
            "Standing still",
            "Sitting and relaxing",
            "Lying down",
            "Walking",
            "Climbing stairs",
            "Waist bends forward",
            "Frontal elevation of arms",
            "Knees bending",
            "Cycling",
            "Jogging",
            "Running",
            "Jump front & back",
        ],
        results_dir="./results_deepconvlstm/mhealth",
    ),
    "shoaib": dict(
        dataset="shoaib",
        input_dim=45,
        num_class=7,
        window=100,
        stride=50,
        stride_test=1,
        path_data="./dataset/shoaib.mat",
        path_raw="./data/shoaib/raw",
        path_processed="./data/shoaib/processed/100_50",
        class_map=[
            "Walking",
            "Standing",
            "Jogging",
            "Sitting",
            "Biking",
            "Walking Upstairs",
            "Walking Downstairs",
        ],
        results_dir="./results_deepconvlstm/shoaib",
    ),
}
_MODEL_DEFAULTS = dict(
    nb_conv_blocks=2,
    nb_filters=64,
    filter_width=5,
    dilation=1,
    batch_norm=False,
    nb_units_lstm=128,
    nb_layers_lstm=2,
    drop_prob=0.5,
    weights_init="xavier_normal",
)
_TRAIN_DEFAULTS = dict(
    optimizer="Adam",
    lr=0.0001,
    weight_decay=1e-06,
    lr_step=50,
    lr_decay=0.5,
    epochs=100,
    patience=20,
    batch_size=256,
    weighted_sampler=True,
    seed=42,
)


def get_args():
    p = argparse.ArgumentParser(
        description="DeepConvLSTM baseline — settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="capture24", choices=list(_DATASET_CONFIGS.keys())
    )
    p.add_argument("--path_data", default=None)
    p.add_argument("--path_processed", default=None)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--nb_conv_blocks", type=int, default=None)
    p.add_argument("--nb_filters", type=int, default=None)
    p.add_argument("--filter_width", type=int, default=None)
    p.add_argument("--dilation", type=int, default=None)
    p.add_argument("--nb_units_lstm", type=int, default=None)
    p.add_argument("--nb_layers_lstm", type=int, default=None)
    p.add_argument("--drop_prob", type=float, default=None)
    p.add_argument(
        "--weights_init",
        default=None,
        choices=[
            "xavier_normal",
            "orthogonal",
            "kaiming_normal",
            "xavier_uniform",
            "kaiming_uniform",
            "normal",
        ],
    )
    p.add_argument("--optimizer", default=None, choices=["Adam", "AdamW", "RMSprop"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr_step", type=int, default=None)
    p.add_argument("--lr_decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--batch_norm_flag", dest="batch_norm", action="store_true", default=None
    )
    p.add_argument("--no_batch_norm", dest="batch_norm", action="store_false")
    p.add_argument(
        "--weighted_sampler", dest="weighted_sampler", action="store_true", default=None
    )
    p.add_argument(
        "--no_weighted_sampler", dest="weighted_sampler", action="store_false"
    )
    args = p.parse_args()
    ds_cfg = dict(_DATASET_CONFIGS[args.dataset])
    for key in ("path_data", "path_processed", "results_dir"):
        v = getattr(args, key, None)
        if v is not None:
            ds_cfg[key] = v
    for k, v in ds_cfg.items():
        setattr(args, k, v)
    model_cfg = dict(_MODEL_DEFAULTS)
    for k in list(model_cfg.keys()):
        v = getattr(args, k, None)
        if v is not None:
            model_cfg[k] = v
    for k, v in model_cfg.items():
        setattr(args, k, v)
    train_cfg = dict(_TRAIN_DEFAULTS)
    for k in list(train_cfg.keys()):
        v = getattr(args, k, None)
        if v is not None:
            train_cfg[k] = v
    for k, v in train_cfg.items():
        setattr(args, k, v)
    args.input_dim = ds_cfg["input_dim"]
    args.num_class = ds_cfg["num_class"]
    args.window = ds_cfg["window"]
    args.stride = ds_cfg["stride"]
    args.stride_test = ds_cfg["stride_test"]
    args.class_map = list(ds_cfg["class_map"])
    return args


if __name__ == "__main__":
    args = get_args()
    print("\n── DeepConvLSTM settings ──")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:25s} = {v}")
