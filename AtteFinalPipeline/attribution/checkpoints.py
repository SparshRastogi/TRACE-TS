import os
import glob


def find_checkpoint(dataset_name, explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    capture24_experiment_map = {
        "capture24": "capture24_Willetts2018_subset",
        "capture24_walmsley": "capture24_Walmsley2020_subset",
        "capture24_full": "capture24_Willetts2018_full",
    }
    search_patterns = []
    if dataset_name in capture24_experiment_map:
        exp = capture24_experiment_map[dataset_name]
        search_patterns.extend(
            [
                f"./models/capture24/train_{exp}/checkpoints/checkpoint_best.pth",
                f"./models/capture24/train_{exp}/checkpoints/checkpoint_*.pth",
                f"./models/{exp}/train_{exp}/checkpoints/checkpoint_best.pth",
                f"./models/{exp}/checkpoints/checkpoint_best.pth",
                f"./results/{exp}/checkpoints/checkpoint_best.pth",
            ]
        )
    if dataset_name == "shoaib":
        search_patterns.extend(
            [
                "./models/shoaib/train_shoaib/checkpoints/checkpoint_best.pth",
                "./models/shoaib/train_*/checkpoints/checkpoint_best.pth",
                "./models/shoaib/train_*/checkpoints/checkpoint_*.pth",
                "./results/shoaib/checkpoints/checkpoint_best.pth",
                "./results/shoaib/train_*/checkpoints/checkpoint_best.pth",
                "./weights/checkpoint_shoaib.pth",
            ]
        )
    if dataset_name in ("mhealth", "mhealth_nonull"):
        exp_dir = "mhealth_nonull" if dataset_name == "mhealth_nonull" else "mhealth"
        search_patterns.extend(
            [
                f"./models/{exp_dir}/train_{exp_dir}/checkpoints/checkpoint_best.pth",
                f"./models/{exp_dir}/train_*/checkpoints/checkpoint_best.pth",
                f"./models/{exp_dir}/train_*/checkpoints/checkpoint_*.pth",
                f"./results/{exp_dir}/checkpoints/checkpoint_best.pth",
                f"./results/{exp_dir}/train_*/checkpoints/checkpoint_best.pth",
                f"./weights/checkpoint_{exp_dir}.pth",
            ]
        )
    search_patterns.extend(
        [
            f"./models/{dataset_name}/train_*/checkpoints/checkpoint_best.pth",
            f"./weights/checkpoint_{dataset_name}.pth",
        ]
    )
    if dataset_name == "pamap2":
        search_patterns.insert(
            0,
            "./models/pamap2/train_08_04_2026_06_11_32/checkpoints/checkpoint_best.pth",
        )
    if dataset_name == "ucihar":
        search_patterns.insert(
            0, "./models/ucihar/train_*/checkpoints/checkpoint_best.pth"
        )
    best_matches = []
    numbered_matches = []
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for m in matches:
            if os.path.basename(m) == "checkpoint_best.pth":
                best_matches.append(m)
            else:
                numbered_matches.append(m)
    if best_matches:
        best_matches.sort(key=lambda p: os.path.getmtime(p))
        return best_matches[-1]
    if numbered_matches:
        numbered_matches.sort(key=lambda p: os.path.getmtime(p))
        return numbered_matches[-1]
    return None
