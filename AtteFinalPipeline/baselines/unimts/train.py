from __future__ import annotations
from AtteFinalPipeline.paths import UCIHAR_DATA_DIR
import os
import shlex
import subprocess
import sys
from pathlib import Path

UNIMTS_DIR = Path(os.environ.get("UNIMTS_DIR", "./UniMTS"))
os.environ.setdefault("WANDB_MODE", "disabled")
USCHAD_DIR = os.environ.get("USCHAD_DIR", "./USC-HAD")
UCI_HAR_NUMPY_DIR = os.environ.get("UCI_HAR_NUMPY_DIR", str(UCIHAR_DATA_DIR))
EPOCHS = os.environ.get("EPOCHS", "50")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "512")
MODE = os.environ.get("MODE", "full")
GYRO = os.environ.get("GYRO", "0")
STFT = os.environ.get("STFT", "0")
RUN_FEW_SHOT = os.environ.get("RUN_FEW_SHOT", "0")
SKIP_INSTALL = os.environ.get("SKIP_INSTALL", "0")
FORCE = os.environ.get("FORCE", "0") == "1"
FORCE_INSTALL = FORCE or os.environ.get("FORCE_INSTALL", "0") == "1"
FORCE_PREP = FORCE or os.environ.get("FORCE_PREP", "0") == "1"
FORCE_CONVERT = FORCE or os.environ.get("FORCE_CONVERT", "0") == "1"
DATASETS = ["opportunity_nonull"]
JOINTS = {
    "pamap2": ["21", "7", "11"],
    "uschad": ["5"],
    "opportunity": ["16", "20", "15", "10", "19"],
    "opportunity_nonull": ["16", "20", "15", "10", "19"],
    "shoaib": ["1", "5", "21", "20", "0"],
    "mhealth": ["11", "3", "21"],
    "mhealth_nonull": ["11", "3", "21"],
    "capture24": ["21"],
    "capture24_full": ["21"],
    "ucihar": ["0"],
}
SR = {
    "pamap2": 33,
    "uschad": 100,
    "opportunity": 30,
    "opportunity_nonull": 30,
    "shoaib": 50,
    "mhealth": 50,
    "mhealth_nonull": 50,
    "capture24": 100,
    "capture24_full": 100,
    "ucihar": 50,
}
NC = {
    "pamap2": 12,
    "uschad": 12,
    "opportunity": 18,
    "opportunity_nonull": 17,
    "shoaib": 7,
    "mhealth": 13,
    "mhealth_nonull": 12,
    "capture24": 6,
    "capture24_full": 6,
    "ucihar": 6,
}
PREP_OUTPUTS = {
    "pamap2": Path("./dataset/pamap2_unimts.mat"),
    "uschad": Path("./dataset/uschad_unimts.mat"),
    "opportunity": Path("./dataset/opportunity_unimts.mat"),
    "opportunity_nonull": Path("./dataset/opportunity_nonull_unimts.mat"),
    "shoaib": Path("./dataset/shoaib_unimts.mat"),
    "mhealth": Path("./dataset/mhealth_unimts.mat"),
    "mhealth_nonull": Path("./dataset/mhealth_nonull_unimts.mat"),
    "capture24": Path("./dataset/capture24_unimts.mat"),
    "capture24_full": Path("./dataset/capture24_full_unimts.mat"),
    "ucihar": Path("./dataset/ucihar_unimts.mat"),
}


def run(cmd, *, cwd=None, check=True):
    if isinstance(cmd, (list, tuple)):
        printable = " ".join((shlex.quote(str(c)) for c in cmd))
    else:
        printable = cmd
    print(f"$ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    subprocess.run(cmd, cwd=cwd, check=check)


def have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def step1_clone_and_install():
    if not UNIMTS_DIR.exists():
        print(f"\n[Step 1] Cloning UniMTS into {UNIMTS_DIR}...")
        run(["git", "clone", "https://github.com/xiyuanzh/UniMTS.git", str(UNIMTS_DIR)])
    else:
        print(f"\n[Step 1] {UNIMTS_DIR} already exists — skipping clone.")
    if SKIP_INSTALL == "1":
        print("[Step 1b] SKIP_INSTALL=1 — skipping dependency install.")
        return
    deps_ok = (
        have_module("clip") and have_module("torch") and have_module("transformers")
    )
    if deps_ok and (not FORCE_INSTALL):
        print("[Step 1b] UniMTS dependencies already installed — skipping.")
        print("          (set FORCE_INSTALL=1 to reinstall)")
        return
    print("\n[Step 1b] Installing UniMTS dependencies into the active environment...")
    print("          (skip with SKIP_INSTALL=1 or FORCE_INSTALL=1 to force reinstall)")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(UNIMTS_DIR / "requirements.txt"),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/openai/CLIP.git",
        ],
        check=False,
    )


def prep_one(name: str):
    expected = PREP_OUTPUTS.get(name)
    if expected is not None and expected.exists() and (not FORCE_PREP):
        print(f"  [skip prep] {expected} already exists (set FORCE_PREP=1 to re-run)")
        return
    if name == "pamap2":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_pamap2_unimts",
            ]
        )
    elif name == "uschad":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_uschad_unimts",
                "--data_dir",
                USCHAD_DIR,
            ]
        )
    elif name == "opportunity":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_opportunity_unimts",
            ]
        )
    elif name == "opportunity_nonull":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_opportunity_unimts",
                "--drop_null",
            ]
        )
    elif name == "shoaib":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_shoaib_unimts",
            ]
        )
    elif name == "mhealth":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_mhealth_unimts",
            ]
        )
    elif name == "mhealth_nonull":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_mhealth_unimts",
                "--drop_null",
            ]
        )
    elif name == "capture24":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_capture24_unimts",
                "--label_schema",
                "Willetts2018",
            ]
        )
    elif name == "capture24_full":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_capture24_unimts",
                "--label_schema",
                "Willetts2018",
                "--n_subjects",
                "151",
            ]
        )
    elif name == "ucihar":
        run(
            [
                sys.executable,
                "-m",
                "AtteFinalPipeline.baselines.unimts.prepare_uci_har_unimts",
                "--src_dir",
                UCI_HAR_NUMPY_DIR,
            ]
        )
    else:
        sys.exit(f"[ERROR] no prep recipe for {name}")


def step2_prepare_datasets():
    print("\n[Step 2] Re-preparing datasets in physical units (no z-score)...")
    for d in DATASETS:
        print(f"\n─── prep: {d} ───")
        prep_one(d)


def convert_one(name: str):
    out_dir = Path(f"./UniMTS_data/{name}")
    sentinels = [
        out_dir / "X_train.npy",
        out_dir / "y_train.npy",
        out_dir / "X_test.npy",
        out_dir / "y_test.npy",
        out_dir / f"{name}.json",
    ]
    if all((p.exists() for p in sentinels)) and (not FORCE_CONVERT):
        print(
            f"  [skip convert] {out_dir} already populated (set FORCE_CONVERT=1 to re-run)"
        )
        return
    run(
        [
            sys.executable,
            "-m",
            "AtteFinalPipeline.baselines.unimts.convert",
            "--dataset",
            name,
        ]
    )


def step3_convert_layout():
    print("\n[Step 3] Converting to UniMTS layout...")
    for d in DATASETS:
        convert_one(d)
    print(f"\n[Step 3b] Linking ./UniMTS_data into {UNIMTS_DIR}/...")
    src = Path("./UniMTS_data").resolve()
    dst = UNIMTS_DIR / "UniMTS_data"
    if dst.is_symlink():
        try:
            if dst.resolve() == src:
                print(f"  [skip link] {dst} already -> {src}")
                return
        except OSError:
            pass
        dst.unlink()
    elif dst.exists():
        if dst.is_file():
            dst.unlink()
        else:
            print(f"[WARN] {dst} exists and is not a symlink; leaving it in place.")
            return
    dst.symlink_to(src, target_is_directory=True)


def run_finetune(name: str):
    joints = JOINTS[name]
    sr = SR[name]
    nc = NC[name]
    print()
    print("=" * 61)
    print(f"  Fine-tuning UniMTS on: {name}")
    print(f"    joints={' '.join(joints)}  sr={sr} Hz  num_class={nc}")
    print("=" * 61)
    args_common = [
        "--mode",
        MODE,
        "--batch_size",
        BATCH_SIZE,
        "--num_epochs",
        EPOCHS,
        "--checkpoint",
        "./checkpoint/UniMTS.pth",
        "--X_train_path",
        f"UniMTS_data/{name}/X_train.npy",
        "--y_train_path",
        f"UniMTS_data/{name}/y_train.npy",
        "--X_test_path",
        f"UniMTS_data/{name}/X_test.npy",
        "--y_test_path",
        f"UniMTS_data/{name}/y_test.npy",
        "--config_path",
        f"UniMTS_data/{name}/{name}.json",
        "--joint_list",
        *joints,
        "--original_sampling_rate",
        str(sr),
        "--num_class",
        str(nc),
        "--gyro",
        GYRO,
        "--stft",
        STFT,
    ]
    if RUN_FEW_SHOT == "1":
        for k in (1, 2, 3, 5, 10):
            run(
                [
                    sys.executable,
                    "finetune_custom.py",
                    *args_common,
                    "--k",
                    str(k),
                    "--run_tag",
                    f"{name}_k{k}",
                ],
                cwd=UNIMTS_DIR,
            )
    run(
        [
            sys.executable,
            "finetune_custom.py",
            *args_common,
            "--run_tag",
            f"{name}_fullshot",
        ],
        cwd=UNIMTS_DIR,
    )


def step4_finetune():
    print("\n[Step 4] Fine-tuning...")
    for d in DATASETS:
        run_finetune(d)


def main():
    import argparse

    global DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(PREP_OUTPUTS), default=DATASETS
    )
    args = parser.parse_args()
    DATASETS = args.datasets
    step1_clone_and_install()
    step2_prepare_datasets()
    step3_convert_layout()
    step4_finetune()
    print("\nAll UniMTS fine-tuning runs complete.")
    print(f"  Checkpoints saved under: {UNIMTS_DIR}/checkpoint/<run_tag>/")
    print("  Logs streamed to wandb (set WANDB_MODE=offline to disable).")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"[FAIL] command exited with status {e.returncode}: {e.cmd}")
