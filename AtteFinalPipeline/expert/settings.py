from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import argparse

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
        path_raw="./data/capture24_Willetts2018_subset/raw",
        path_processed="./data/capture24_Willetts2018_subset/processed/200_1",
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
        results_dir="./results/capture24_Willetts2018_subset",
    ),
    "capture24_walmsley": dict(
        dataset="capture24",
        input_dim=3,
        num_class=4,
        window=200,
        stride=1,
        stride_test=1,
        path_data="./dataset/capture24_Walmsley2020_subset.npz",
        path_raw="./data/capture24_Walmsley2020_subset/raw",
        path_processed="./data/capture24_Walmsley2020_subset/processed/200_1",
        class_map=["Sleep", "Sedentary", "Light", "Moderate-Vigorous"],
        results_dir="./results/capture24_Walmsley2020_subset",
    ),
    "capture24_full": dict(
        dataset="capture24",
        input_dim=3,
        num_class=6,
        window=200,
        stride=1,
        stride_test=1,
        path_data="./dataset/capture24_Willetts2018.npz",
        path_raw="./data/capture24_Willetts2018/raw",
        path_processed="./data/capture24_Willetts2018/processed/200_1",
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
        results_dir="./results/capture24_Willetts2018_full",
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
        results_dir="./results/opportunity",
    ),
    "opportunity_nonull": dict(
        dataset="opportunity_nonull",
        input_dim=79,
        num_class=17,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/opportunity.mat",
        path_raw="./data/opportunity_nonull/raw",
        path_processed="./data/opportunity_nonull/processed/24_12",
        filter_null_at_load=True,
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
        results_dir="./results/opportunity_nonull",
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
        results_dir="./results/pamap2",
    ),
    "skoda": dict(
        dataset="skoda",
        input_dim=60,
        num_class=11,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/skoda.mat",
        path_raw="./data/skoda/raw",
        path_processed="./data/skoda/processed/24_12",
        class_map=[
            "Null",
            "Write on Notepad",
            "Open Hood",
            "Close Hood",
            "Check Door Gaps",
            "Open Left Front Door",
            "Close Left Front Door",
            "Close Both Left Doors",
            "Check Trunk Gaps",
            "Open and Close Trunk",
            "Check Steering Wheel",
        ],
        results_dir="./results/skoda",
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
        results_dir="./results/ucihar",
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
        results_dir="./results/uschad",
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
        results_dir="./results/shoaib",
    ),
    "hospital": dict(
        dataset="hospital",
        input_dim=6,
        num_class=7,
        window=24,
        stride=12,
        stride_test=1,
        path_data="./dataset/hospital.mat",
        path_raw="./data/hospital/raw",
        path_processed="./data/hospital/processed/24_12",
        class_map=[
            "Lying",
            "Standing Up",
            "Sitting",
            "Walking",
            "Lying Down",
            "Sitting Down",
            "Getting Up",
        ],
        results_dir="./results/hospital",
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
        results_dir="./results/mhealth",
    ),
    "mhealth_nonull": dict(
        dataset="mhealth_nonull",
        input_dim=23,
        num_class=12,
        window=100,
        stride=50,
        stride_test=1,
        path_data="./dataset/mhealth_nonull.mat",
        path_raw="./data/mhealth_nonull/raw",
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
        results_dir="./results/mhealth_nonull",
    ),
}
_MODEL_DEFAULTS = dict(
    model="AttendDiscriminate",
    hidden_dim=128,
    filter_num=64,
    filter_size=5,
    enc_num_layers=2,
    enc_is_bidirectional=False,
    dropout=0.5,
    dropout_rnn=0.25,
    dropout_cls=0.5,
    activation="ReLU",
    sa_div=1,
)
_DATASET_TRAIN_OVERRIDES = {
    "opportunity": dict(
        init_weights="orthogonal",
        beta=0.0003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
    "opportunity_nonull": dict(
        init_weights="orthogonal",
        beta=0.0003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
    "pamap2": dict(
        init_weights=None, beta=0.003, dropout=0.9, dropout_rnn=0.0, dropout_cls=0.5
    ),
    "skoda": dict(
        init_weights="orthogonal",
        beta=0.3,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.0,
    ),
    "ucihar": dict(
        init_weights="orthogonal",
        beta=0.003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
        batch_size=64,
    ),
    "uschad": dict(
        init_weights="orthogonal",
        beta=0.003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
    "shoaib": dict(
        init_weights="orthogonal",
        beta=0.003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
    "hospital": dict(
        init_weights="orthogonal",
        beta=0.3,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
    "mhealth": dict(
        init_weights="orthogonal",
        beta=0.003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
    "mhealth_nonull": dict(
        init_weights="orthogonal",
        beta=0.003,
        dropout=0.5,
        dropout_rnn=0.25,
        dropout_cls=0.5,
    ),
}
_TRAIN_DEFAULTS = dict(
    optimizer="Adam",
    lr=0.0001,
    lr_step=50,
    lr_decay=0.5,
    lr_cent=0.001,
    beta=0.01,
    clip_grad=0.0,
    epochs=100,
    extra_epochs=50,
    patience=20,
    batch_size=256,
    mixup=False,
    alpha=0.2,
    weighted_sampler=True,
    init_weights="orthogonal",
    print_freq=100,
)


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="AttendDiscriminate HAR pipeline — settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_VARIANT,
        choices=list(_DATASET_CONFIGS.keys()),
        help="Dataset / variant to use.",
    )
    parser.add_argument("--path_data", default=None)
    parser.add_argument("--path_raw", default=None)
    parser.add_argument("--path_processed", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--filter_num", type=int, default=None)
    parser.add_argument("--filter_size", type=int, default=None)
    parser.add_argument("--enc_num_layers", type=int, default=None)
    parser.add_argument("--enc_is_bidirectional", type=bool, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dropout_rnn", type=float, default=None)
    parser.add_argument("--dropout_cls", type=float, default=None)
    parser.add_argument("--activation", default=None, choices=["ReLU", "Tanh"])
    parser.add_argument("--sa_div", type=int, default=None)
    parser.add_argument("--optimizer", default=None, choices=["Adam", "RMSprop"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_step", type=int, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--lr_cent", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--extra_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--init_weights", default=None, choices=["orthogonal", "random"]
    )
    parser.add_argument("--print_freq", type=int, default=None)
    parser.add_argument("--mixup", dest="mixup", action="store_true", default=None)
    parser.add_argument("--no_mixup", dest="mixup", action="store_false")
    parser.add_argument(
        "--weighted_sampler", dest="weighted_sampler", action="store_true", default=None
    )
    parser.add_argument(
        "--no_weighted_sampler", dest="weighted_sampler", action="store_false"
    )
    parser.add_argument("--experiment", default="capture24_Willetts2018_subset")
    parser.add_argument("--train_mode", action="store_true", default=True)
    parser.add_argument("--no_train_mode", dest="train_mode", action="store_false")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--model", default="AttendDiscriminate", choices=["AttendDiscriminate"]
    )
    args = parser.parse_args(argv)
    ds_cfg = dict(_DATASET_CONFIGS[args.dataset])
    for key in ("path_data", "path_raw", "path_processed", "results_dir"):
        val = getattr(args, key, None)
        if val is not None:
            ds_cfg[key] = val
    for key, val in ds_cfg.items():
        setattr(args, key, val)
    if not hasattr(args, "filter_null_at_load"):
        args.filter_null_at_load = ds_cfg.get("filter_null_at_load", False)
    train_cfg = dict(_TRAIN_DEFAULTS)
    for key, val in _DATASET_TRAIN_OVERRIDES.get(args.dataset, {}).items():
        train_cfg[key] = val
    for key in list(train_cfg.keys()):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            train_cfg[key] = cli_val
    model_cfg = dict(_MODEL_DEFAULTS)
    for key in list(model_cfg.keys()):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            model_cfg[key] = cli_val
    for key, val in model_cfg.items():
        setattr(args, key, val)
    for key, val in train_cfg.items():
        setattr(args, key, val)
    args.input_dim = ds_cfg["input_dim"]
    args.num_class = ds_cfg["num_class"]
    args.window = ds_cfg["window"]
    args.stride = ds_cfg["stride"]
    args.stride_test = ds_cfg["stride_test"]
    args.class_map = ds_cfg["class_map"]
    config_dataset = dict(
        path_data=args.path_data,
        dataset=args.dataset,
        window=args.window,
        stride=args.stride,
        stride_test=args.stride_test,
        path_processed=args.path_processed,
    )
    config_model = dict(
        model=args.model,
        dataset=args.dataset,
        input_dim=args.input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        filter_num=model_cfg["filter_num"],
        filter_size=model_cfg["filter_size"],
        enc_num_layers=model_cfg["enc_num_layers"],
        enc_is_bidirectional=model_cfg["enc_is_bidirectional"],
        dropout=model_cfg["dropout"],
        dropout_rnn=model_cfg["dropout_rnn"],
        dropout_cls=model_cfg["dropout_cls"],
        activation=model_cfg["activation"],
        sa_div=model_cfg["sa_div"],
        num_class=args.num_class,
        train_mode=args.train_mode,
        experiment=args.experiment,
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
