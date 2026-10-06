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
        results_dir="./results_chronos/capture24",
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
        results_dir="./results_chronos/opportunity",
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
        results_dir="./results_chronos/pamap2",
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
        results_dir="./results_chronos/uschad",
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
        results_dir="./results_chronos/ucihar",
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
        results_dir="./results_chronos/mhealth",
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
        results_dir="./results_chronos/shoaib",
    ),
}
_CHRONOS_DEFAULTS = dict(
    chronos_model_id="amazon/chronos-2",
    chronos_dtype="bfloat16",
    chronos_batch_size=256,
    chronos_context_len=512,
    mlp_hidden_dim=512,
    mlp_dropout=0.3,
    optimizer="AdamW",
    lr=0.001,
    weight_decay=0.0001,
    lr_step=50,
    lr_decay=0.5,
    epochs=100,
    patience=20,
    train_batch_size=512,
    weighted_sampler=True,
    seed=42,
    cache_dir="./chronos_cache",
    embed_pool="mean",
)


def get_args():
    p = argparse.ArgumentParser(
        description="Chronos-2 + MLP linear probing — settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="capture24", choices=list(_DATASET_CONFIGS.keys())
    )
    p.add_argument("--path_data", default=None)
    p.add_argument("--path_processed", default=None)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--chronos_model_id", default=None)
    p.add_argument("--chronos_batch_size", type=int, default=None)
    p.add_argument("--chronos_context_len", type=int, default=None)
    p.add_argument("--mlp_hidden_dim", type=int, default=None)
    p.add_argument("--mlp_dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr_step", type=int, default=None)
    p.add_argument("--lr_decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--weighted_sampler", dest="weighted_sampler", action="store_true", default=None
    )
    p.add_argument(
        "--no_weighted_sampler", dest="weighted_sampler", action="store_false"
    )
    p.add_argument(
        "--drop_null",
        action="store_true",
        default=False,
        help="Drop label==0 ('Null') samples and remap labels.",
    )
    p.add_argument("--force_recompute", action="store_true", default=False)
    p.add_argument("--cache_only", action="store_true", default=False)
    args = p.parse_args()
    ds_cfg = dict(_DATASET_CONFIGS[args.dataset])
    for key in ("path_data", "path_processed", "results_dir"):
        v = getattr(args, key, None)
        if v is not None:
            ds_cfg[key] = v
    for k, v in ds_cfg.items():
        setattr(args, k, v)
    cfg = dict(_CHRONOS_DEFAULTS)
    for k in list(cfg.keys()):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    for k, v in cfg.items():
        setattr(args, k, v)
    args.input_dim = ds_cfg["input_dim"]
    args.num_class = ds_cfg["num_class"]
    args.window = ds_cfg["window"]
    args.stride = ds_cfg["stride"]
    args.stride_test = ds_cfg["stride_test"]
    args.class_map = list(ds_cfg["class_map"])
    if args.drop_null:
        if args.num_class < 2:
            raise ValueError(
                f"--drop_null on dataset with num_class={args.num_class} would leave fewer than 1 class."
            )
        if args.class_map and args.class_map[0].strip().lower() != "null":
            print(
                f"[WARN] --drop_null set, but class 0 is '{args.class_map[0]}', not 'Null'. Proceeding anyway (class 0 will still be dropped)."
            )
        args.num_class = args.num_class - 1
        args.class_map = args.class_map[1:]
        if getattr(args, "_results_dir_user_override", False) is False:
            default_results_dir = _DATASET_CONFIGS[args.dataset]["results_dir"]
            if args.results_dir == default_results_dir:
                args.results_dir = default_results_dir.rstrip("/") + "_no_null"
    return args


if __name__ == "__main__":
    args = get_args()
    print("\n── Chronos-2 + MLP settings ──")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:25s} = {v}")
