import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

TEST_PREFIX = "test_"


def _transfer_one(args_tuple):
    src, dst, mode = args_tuple
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return dst


def _list_test_jsons(src_dir):
    if not os.path.isdir(src_dir):
        return []
    return sorted(
        (
            os.path.join(src_dir, name)
            for name in os.listdir(src_dir)
            if name.startswith(TEST_PREFIX) and name.endswith(".json")
        )
    )


def process_dataset_dir(dataset_name, src_root, dst_root, mode, save_workers):
    src_dir = os.path.join(src_root, dataset_name)
    dst_dir = os.path.join(dst_root, dataset_name)
    test_files = _list_test_jsons(src_dir)
    if not test_files:
        print(f"  [{dataset_name}] no test_*.json files found, skipping")
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    transfer_args = [
        (src, os.path.join(dst_dir, os.path.basename(src)), mode) for src in test_files
    ]
    with ThreadPoolExecutor(max_workers=save_workers) as ex:
        list(
            tqdm(
                ex.map(_transfer_one, transfer_args),
                total=len(transfer_args),
                desc=f"  [{dataset_name}] {mode}",
                unit="file",
                dynamic_ncols=True,
            )
        )
    print(f"  [{dataset_name}] {len(test_files)} test JSONs -> {dst_dir}")
    return len(test_files)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract test-split JSONs from a mantis_and_raw output tree into a parallel folder."
    )
    p.add_argument(
        "--src",
        type=str,
        default="./mantis_and_raw",
        help="Source root containing per-dataset subfolders. Default: ./mantis_and_raw",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="./mantis_and_raw_test",
        help="Destination root. Default: ./mantis_and_raw_test",
    )
    p.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these datasets (subfolder names). Default: every subdirectory under --src.",
    )
    p.add_argument(
        "--mode",
        choices=["copy", "move", "symlink"],
        default="copy",
        help="How to place files in the destination. Default: copy.",
    )
    _cpu_default = max(4, min(16, (os.cpu_count() or 8) // 2))
    p.add_argument(
        "--workers",
        type=int,
        default=_cpu_default,
        help=f"Parallel file-transfer workers. Default: {_cpu_default}",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.src):
        print(f"ERROR: source directory does not exist: {args.src}")
        sys.exit(1)
    if args.dataset:
        datasets = list(args.dataset)
    else:
        datasets = sorted(
            (
                name
                for name in os.listdir(args.src)
                if os.path.isdir(os.path.join(args.src, name))
            )
        )
    if not datasets:
        print(f"ERROR: no dataset subdirectories found under {args.src}")
        sys.exit(1)
    print(f"Source:     {args.src}")
    print(f"Destination:{args.dst}")
    print(f"Mode:       {args.mode}")
    print(f"Datasets:   {datasets}")
    print()
    os.makedirs(args.dst, exist_ok=True)
    grand_total = 0
    for ds in datasets:
        n = process_dataset_dir(ds, args.src, args.dst, args.mode, args.workers)
        grand_total += n
    print()
    print(
        f"Done. Transferred {grand_total} test JSONs across {len(datasets)} dataset folder(s)."
    )


if __name__ == "__main__":
    main()
