from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
import argparse

_DATASET_CONFIGS = {
    "capture24": dict(
        dataset="capture24",
        input_dim=3,
        num_class=6,
        window=200,
        stride=1,
        stride_test=1,
        path_processed="./data/capture24_Willetts2018_subset/processed/200_1",
        class_map=["Sleep", "Sit-stand", "Walking", "Bicycling", "Mixed", "Vehicle"],
        results_dir="./results_nst/capture24",
    ),
    "opportunity": dict(
        dataset="opportunity",
        input_dim=79,
        num_class=18,
        window=24,
        stride=12,
        stride_test=1,
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
        results_dir="./results_nst/opportunity",
    ),
    "pamap2": dict(
        dataset="pamap2",
        input_dim=52,
        num_class=12,
        window=24,
        stride=12,
        stride_test=1,
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
        results_dir="./results_nst/pamap2",
    ),
    "uschad": dict(
        dataset="uschad",
        input_dim=6,
        num_class=12,
        window=400,
        stride=200,
        stride_test=1,
        path_processed="./data/uschad/processed/400_200",
        class_map=list(USCHAD_CLASS_MAP),
        results_dir="./results_nst/uschad",
    ),
    "ucihar": dict(
        dataset="ucihar",
        input_dim=9,
        num_class=6,
        window=128,
        stride=64,
        stride_test=64,
        path_processed=str(UCIHAR_DATA_DIR),
        class_map=[
            "Walking",
            "Walking Upstairs",
            "Walking Downstairs",
            "Sitting",
            "Standing",
            "Laying",
        ],
        results_dir="./results_nst/ucihar",
    ),
    "mhealth": dict(
        dataset="mhealth",
        input_dim=23,
        num_class=13,
        window=100,
        stride=50,
        stride_test=1,
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
        results_dir="./results_nst/mhealth",
    ),
    "shoaib": dict(
        dataset="shoaib",
        input_dim=45,
        num_class=7,
        window=100,
        stride=50,
        stride_test=1,
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
        results_dir="./results_nst/shoaib",
    ),
}
_TRAIN_DEFAULTS = dict(
    optimizer="Adam",
    lr=0.0001,
    lr_step=50,
    lr_decay=0.5,
    epochs=100,
    extra_epochs=50,
    patience=20,
    batch_size=512,
    weighted_sampler=True,
    print_freq=100,
    seed=42,
)
_DATASET_TRAIN_OVERRIDES = {"ucihar": dict(batch_size=64)}
_MODEL_DEFAULTS = dict(
    d_model=128, n_heads=8, e_layers=2, d_ff=256, dropout=0.1, projector_hidden=64
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(_DATASET_CONFIGS.keys()))
    p.add_argument("--results_dir", default=None)
    p.add_argument("--path_processed", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no_weighted_sampler",
        dest="weighted_sampler",
        action="store_false",
        default=None,
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--train_mode", action="store_true", default=True)
    p.add_argument("--no_train_mode", dest="train_mode", action="store_false")
    p.add_argument(
        "--drop_null",
        action="store_true",
        help="Filter windows whose label == 0 (Null) from every split; remap remaining labels and shrink num_class.",
    )
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--e_layers", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--projector_hidden", type=int, default=None)
    args = p.parse_args()
    ds = dict(_DATASET_CONFIGS[args.dataset])
    train = dict(_TRAIN_DEFAULTS)
    train.update(_DATASET_TRAIN_OVERRIDES.get(args.dataset, {}))
    model = dict(_MODEL_DEFAULTS)
    for k in ("results_dir", "path_processed"):
        if getattr(args, k) is not None:
            ds[k] = getattr(args, k)
    for k in list(train.keys()):
        v = getattr(args, k, None)
        if v is not None:
            train[k] = v
    for k in list(model.keys()):
        v = getattr(args, k, None)
        if v is not None:
            model[k] = v
    if args.drop_null:
        cm = ds["class_map"]
        if len(cm) == 0 or cm[0].strip().lower() != "null":
            print(
                f"[!] --drop_null was passed but class_map[0] for '{args.dataset}' is '{(cm[0] if cm else None)}', not 'Null'. Proceeding anyway: dropping label 0 and shrinking num_class by 1."
            )
        ds["class_map"] = cm[1:]
        ds["num_class"] = ds["num_class"] - 1
        ds["results_dir"] = ds["results_dir"].rstrip("/") + "_nonull"
    for k, v in ds.items():
        setattr(args, k, v)
    for k, v in train.items():
        setattr(args, k, v)
    for k, v in model.items():
        setattr(args, k, v)
    return args
