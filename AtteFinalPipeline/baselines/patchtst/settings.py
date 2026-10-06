from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import argparse
import os

DATASET_VARIANT = "capture24"
_DATASET_CONFIGS = {
    "capture24": dict(
        dataset="capture24",
        input_dim=3,
        num_class=6,
        window=200,
        stride=1,
        stride_test=1,
        path_data="./dataset/capture24_Willetts2018_subset.npz",
        path_processed="./data/capture24_Willetts2018_subset/processed/200_1",
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
        results_dir="./results_patchtst/capture24_Willetts2018_subset",
        drop_null=False,
        null_label=None,
    ),
    "capture24_walmsley": dict(
        dataset="capture24",
        input_dim=3,
        num_class=4,
        window=200,
        stride=1,
        stride_test=1,
        path_data="./dataset/capture24_Walmsley2020_subset.npz",
        path_processed="./data/capture24_Walmsley2020_subset/processed/200_1",
        class_map=["Sleep", "Sedentary", "Light", "Moderate-Vigorous"],
        results_dir="./results_patchtst/capture24_Walmsley2020_subset",
        drop_null=False,
        null_label=None,
    ),
    "capture24_full": dict(
        dataset="capture24",
        input_dim=3,
        num_class=6,
        window=200,
        stride=1,
        stride_test=1,
        path_data="./dataset/capture24_Willetts2018.npz",
        path_processed="./data/capture24_Willetts2018/processed/200_1",
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
        results_dir="./results_patchtst/capture24_Willetts2018_full",
        drop_null=False,
        null_label=None,
    ),
    "opportunity": dict(
        dataset="opportunity",
        input_dim=79,
        num_class=18,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/opportunity.mat",
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
        results_dir="./results_patchtst/opportunity",
        drop_null=False,
        null_label=None,
    ),
    "opportunity_nonull": dict(
        dataset="opportunity",
        input_dim=79,
        num_class=17,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/opportunity.mat",
        path_processed="./data/opportunity/processed/24_12",
        class_map=[
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
        results_dir="./results_patchtst/opportunity_nonull",
        drop_null=True,
        null_label=0,
    ),
    "pamap2": dict(
        dataset="pamap2",
        input_dim=52,
        num_class=12,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/pamap2.mat",
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
        results_dir="./results_patchtst/pamap2",
        drop_null=False,
        null_label=None,
    ),
    "ucihar": dict(
        dataset="ucihar",
        input_dim=9,
        num_class=6,
        window=128,
        stride=64,
        stride_test=64,
        path_data=str(UCIHAR_DATA_DIR),
        path_processed=str(UCIHAR_DATA_DIR),
        class_map=[
            "Walking",
            "Walking Upstairs",
            "Walking Downstairs",
            "Sitting",
            "Standing",
            "Laying",
        ],
        results_dir="./results_patchtst/ucihar",
        drop_null=False,
        null_label=None,
    ),
    "uschad": dict(
        dataset="uschad",
        input_dim=6,
        num_class=12,
        window=400,
        stride=200,
        stride_test=1,
        path_data="./dataset/uschad.mat",
        path_processed="./data/uschad/processed/400_200",
        class_map=list(USCHAD_CLASS_MAP),
        results_dir="./results_patchtst/uschad",
        drop_null=False,
        null_label=None,
    ),
    "shoaib": dict(
        dataset="shoaib",
        input_dim=45,
        num_class=7,
        window=100,
        stride=50,
        stride_test=1,
        path_data="./dataset/shoaib.mat",
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
        results_dir="./results_patchtst/shoaib",
        drop_null=False,
        null_label=None,
    ),
    "mhealth": dict(
        dataset="mhealth",
        input_dim=23,
        num_class=13,
        window=100,
        stride=50,
        stride_test=1,
        path_data="./dataset/mhealth.mat",
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
        results_dir="./results_patchtst/mhealth",
        drop_null=False,
        null_label=None,
    ),
    "mhealth_nonull": dict(
        dataset="mhealth",
        input_dim=23,
        num_class=12,
        window=100,
        stride=50,
        stride_test=1,
        path_data="./dataset/mhealth_nonull.mat",
        path_processed="./data/mhealth_nonull/processed/100_50",
        class_map=[
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
        results_dir="./results_patchtst/mhealth_nonull",
        drop_null=False,
        null_label=None,
    ),
}
_PATCHTST_MODEL_DEFAULTS = {
    "capture24": dict(patch_len=16, patch_stride=8),
    "capture24_walmsley": dict(patch_len=16, patch_stride=8),
    "capture24_full": dict(patch_len=16, patch_stride=8),
    "opportunity": dict(patch_len=4, patch_stride=2),
    "opportunity_nonull": dict(patch_len=4, patch_stride=2),
    "pamap2": dict(patch_len=4, patch_stride=2),
    "uschad": dict(patch_len=16, patch_stride=8),
    "ucihar": dict(patch_len=16, patch_stride=8),
    "shoaib": dict(patch_len=10, patch_stride=5),
    "mhealth": dict(patch_len=10, patch_stride=5),
    "mhealth_nonull": dict(patch_len=10, patch_stride=5),
}
_PATCHTST_GLOBAL_DEFAULTS = dict(
    d_model=128,
    n_heads=8,
    n_layers=3,
    ff_dim=256,
    dropout=0.2,
    head_dropout=0.0,
    pos_embed="learnable",
    revin=False,
)
_TRAIN_DEFAULTS = dict(
    optimizer="Adam",
    lr=0.0001,
    lr_step=50,
    lr_decay=0.5,
    clip_grad=0.0,
    epochs=100,
    patience=20,
    batch_size=16,
    weighted_sampler=True,
    mixup=False,
    alpha=0.2,
    print_freq=100,
)
_DATASET_TRAIN_OVERRIDES = {"ucihar": dict(batch_size=64)}


def get_args():
    parser = argparse.ArgumentParser(
        description="PatchTST HAR pipeline — settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default=DATASET_VARIANT, choices=list(_DATASET_CONFIGS.keys())
    )
    parser.add_argument("--path_data", default=None)
    parser.add_argument("--path_processed", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_step", type=int, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--print_freq", type=int, default=None)
    parser.add_argument("--mixup", dest="mixup", action="store_true", default=None)
    parser.add_argument("--no_mixup", dest="mixup", action="store_false")
    parser.add_argument(
        "--weighted_sampler", dest="weighted_sampler", action="store_true", default=None
    )
    parser.add_argument(
        "--no_weighted_sampler", dest="weighted_sampler", action="store_false"
    )
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--train_mode", action="store_true", default=True)
    parser.add_argument("--no_train_mode", dest="train_mode", action="store_false")
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()
    ds_cfg = dict(_DATASET_CONFIGS[args.dataset])
    for key in ("path_data", "path_processed", "results_dir"):
        if getattr(args, key, None) is not None:
            ds_cfg[key] = getattr(args, key)
    for key, val in ds_cfg.items():
        setattr(args, key, val)
    train_cfg = dict(_TRAIN_DEFAULTS)
    train_cfg.update(_DATASET_TRAIN_OVERRIDES.get(args.dataset, {}))
    for key in list(train_cfg.keys()):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            train_cfg[key] = cli_val
    for key, val in train_cfg.items():
        setattr(args, key, val)
    model_cfg = dict(_PATCHTST_GLOBAL_DEFAULTS)
    model_cfg.update(_PATCHTST_MODEL_DEFAULTS[args.dataset])
    for key in (
        "patch_len",
        "patch_stride",
        "d_model",
        "n_heads",
        "n_layers",
        "ff_dim",
        "dropout",
        "head_dropout",
    ):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            model_cfg[key] = cli_val
    for key, val in model_cfg.items():
        setattr(args, key, val)
    if args.experiment is None:
        args.experiment = f"{args.dataset}_seed{args.seed}"
    args.results_dir = os.path.join(args.results_dir, f"seed{args.seed}")
    config_dataset = dict(
        dataset=args.dataset,
        window=args.window,
        stride=args.stride,
        stride_test=args.stride_test,
        path_processed=args.path_processed,
        drop_null=ds_cfg["drop_null"],
        null_label=ds_cfg["null_label"],
    )
    config_model = dict(
        input_dim=args.input_dim,
        num_class=args.num_class,
        window=args.window,
        patch_len=model_cfg["patch_len"],
        patch_stride=model_cfg["patch_stride"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ff_dim=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
        head_dropout=model_cfg["head_dropout"],
        pos_embed=model_cfg["pos_embed"],
        revin=model_cfg["revin"],
    )
    return (args, config_dataset, config_model)


if __name__ == "__main__":
    args, config_dataset, config_model = get_args()
    print("\n── args ──────────────────────────────────────────────────")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:30s} = {v}")
    print("\n── config_dataset ────────────────────────────────────────")
    for k, v in config_dataset.items():
        print(f"  {k:30s} = {v}")
    print("\n── config_model ──────────────────────────────────────────")
    for k, v in config_model.items():
        print(f"  {k:30s} = {v}")
