import os
import sys
import json
import csv
import argparse
import subprocess
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--best_config", default=None, help="Path to grid_search best_config.json"
    )
    ap.add_argument("--patch_len", type=int)
    ap.add_argument("--patch_stride", type=int)
    ap.add_argument("--d_model", type=int)
    ap.add_argument("--n_layers", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument(
        "--results_root",
        default="./results_patchtst",
        help="Root for per-seed artefacts.",
    )
    ap.add_argument(
        "--summary_csv",
        default="./results_patchtst/all_seeds_summary.csv",
        help="CSV file to append the aggregated row to.",
    )
    args = ap.parse_args()
    if args.best_config and os.path.exists(args.best_config):
        with open(args.best_config) as f:
            best = json.load(f)
        cfg = best["cfg"]
        print(f"[*] Loaded best config from {args.best_config}: {cfg}")
    else:
        cfg = {
            "patch_len": args.patch_len,
            "patch_stride": args.patch_stride,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        }
        cfg = {k: v for k, v in cfg.items() if v is not None}
        if not cfg:
            print("[*] No config specified — using settings_patchtst.py defaults.")
        else:
            print(f"[*] Using CLI config: {cfg}")
    summaries = []
    for seed in args.seeds:
        run_dir = os.path.join(args.results_root, args.dataset)
        cmd = [
            sys.executable,
            "-m",
            "AtteFinalPipeline.baselines.patchtst.train",
            "--dataset",
            args.dataset,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--results_dir",
            run_dir,
        ]
        for k, v in cfg.items():
            cmd += [f"--{k}", str(v)]
        print("\n" + "#" * 70)
        print(f"# {args.dataset}  seed={seed}")
        print("#" * 70)
        print("Cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        actual_dir = os.path.join(run_dir, f"seed{seed}")
        with open(os.path.join(actual_dir, "test_summary.json")) as f:
            summaries.append(json.load(f))
    accs = np.array([s["accuracy"] for s in summaries])
    fms = np.array([s["f1_macro"] for s in summaries])
    fws = np.array([s["f1_weighted"] for s in summaries])
    print("\n" + "=" * 70)
    print(f"AGGREGATED RESULTS — {args.dataset}  (n={len(args.seeds)} seeds)")
    print("=" * 70)
    print(
        f"  Accuracy    : {accs.mean():.2f} ± {accs.std(ddof=1):.2f}    (per-seed: {accs.tolist()})"
    )
    print(
        f"  F1-macro    : {fms.mean():.2f} ± {fms.std(ddof=1):.2f}    (per-seed: {fms.tolist()})"
    )
    print(
        f"  F1-weighted : {fws.mean():.2f} ± {fws.std(ddof=1):.2f}    (per-seed: {fws.tolist()})"
    )
    print("=" * 70)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    file_exists = os.path.exists(args.summary_csv)
    with open(args.summary_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(
                [
                    "dataset",
                    "n_seeds",
                    "seeds",
                    "patch_len",
                    "patch_stride",
                    "d_model",
                    "n_layers",
                    "dropout",
                    "acc_mean",
                    "acc_std",
                    "fm_mean",
                    "fm_std",
                    "fw_mean",
                    "fw_std",
                    "acc_per_seed",
                    "fm_per_seed",
                    "fw_per_seed",
                ]
            )
        w.writerow(
            [
                args.dataset,
                len(args.seeds),
                ",".join((str(s) for s in args.seeds)),
                cfg.get("patch_len", ""),
                cfg.get("patch_stride", ""),
                cfg.get("d_model", ""),
                cfg.get("n_layers", ""),
                cfg.get("dropout", ""),
                f"{accs.mean():.2f}",
                f"{accs.std(ddof=1):.2f}",
                f"{fms.mean():.2f}",
                f"{fms.std(ddof=1):.2f}",
                f"{fws.mean():.2f}",
                f"{fws.std(ddof=1):.2f}",
                ",".join((f"{a:.2f}" for a in accs)),
                ",".join((f"{a:.2f}" for a in fms)),
                ",".join((f"{a:.2f}" for a in fws)),
            ]
        )
    print(f"[+] Appended row to {args.summary_csv}")


if __name__ == "__main__":
    main()
