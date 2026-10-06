import os
import sys
import json
import argparse
import itertools
import subprocess

GRIDS = {
    "capture24": dict(
        patch_len_stride=[(8, 4), (16, 8), (25, 12), (40, 20)],
        d_model=[128, 256],
        n_layers=[2, 3],
        dropout=[0.2, 0.3],
    ),
    "opportunity": dict(
        patch_len_stride=[(4, 2), (6, 3), (8, 4)],
        d_model=[128],
        n_layers=[2, 3],
        dropout=[0.3, 0.5],
    ),
    "pamap2": dict(
        patch_len_stride=[(4, 2), (6, 3), (8, 4)],
        d_model=[128],
        n_layers=[2, 3],
        dropout=[0.3, 0.5, 0.7],
    ),
    "uschad": dict(
        patch_len_stride=[(16, 8), (25, 12), (40, 20)],
        d_model=[128, 256],
        n_layers=[2, 3],
        dropout=[0.2, 0.3],
    ),
    "ucihar": dict(
        patch_len_stride=[(8, 4), (16, 8), (32, 16)],
        d_model=[128],
        n_layers=[2, 3],
        dropout=[0.2, 0.3],
    ),
    "shoaib": dict(
        patch_len_stride=[(5, 5), (10, 5), (10, 10), (20, 10)],
        d_model=[128],
        n_layers=[2, 3],
        dropout=[0.2, 0.3],
    ),
    "mhealth": dict(
        patch_len_stride=[(5, 5), (10, 5), (10, 10), (20, 10)],
        d_model=[128],
        n_layers=[2, 3],
        dropout=[0.2, 0.3],
    ),
}


def build_configs(dataset):
    g = GRIDS[dataset]
    configs = []
    for (P, S), dm, nl, dp in itertools.product(
        g["patch_len_stride"], g["d_model"], g["n_layers"], g["dropout"]
    ):
        configs.append(
            dict(patch_len=P, patch_stride=S, d_model=dm, n_layers=nl, dropout=dp)
        )
    return configs


def cfg_tag(cfg):
    return f"P{cfg['patch_len']}_S{cfg['patch_stride']}_D{cfg['d_model']}_L{cfg['n_layers']}_dp{cfg['dropout']}"


def run_one_config(dataset, cfg, grid_epochs, grid_root, seed):
    run_dir = os.path.join(grid_root, dataset, cfg_tag(cfg))
    cmd = [
        sys.executable,
        "-m",
        "AtteFinalPipeline.baselines.patchtst.train",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--epochs",
        str(grid_epochs),
        "--patience",
        str(grid_epochs),
        "--results_dir",
        run_dir,
        "--patch_len",
        str(cfg["patch_len"]),
        "--patch_stride",
        str(cfg["patch_stride"]),
        "--d_model",
        str(cfg["d_model"]),
        "--n_layers",
        str(cfg["n_layers"]),
        "--dropout",
        f"{cfg['dropout']}",
    ]
    print("\n" + "#" * 70)
    print(f"# GRID  {dataset}  cfg={cfg_tag(cfg)}")
    print("#" * 70)
    print("Cmd:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    hist_path = os.path.join(run_dir, "history.json")
    with open(hist_path) as f:
        hist = json.load(f)
    best_fm_v = max((epoch["val_fm"] for epoch in hist))
    return (best_fm_v, run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(GRIDS.keys()))
    ap.add_argument(
        "--grid_epochs",
        type=int,
        default=30,
        help="Epochs per grid run (kept short for speed).",
    )
    ap.add_argument(
        "--grid_root",
        default="./results_patchtst_grid",
        help="Where to store grid-run artefacts.",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Project-fixed seed (default 42)."
    )
    args = ap.parse_args()
    configs = build_configs(args.dataset)
    print(
        f"[*] Grid for {args.dataset}: {len(configs)} configs, {args.grid_epochs} epochs each"
    )
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Config: {cfg}")
        try:
            best_fm_v, run_dir = run_one_config(
                args.dataset, cfg, args.grid_epochs, args.grid_root, args.seed
            )
            results.append(dict(cfg=cfg, val_fm_best=best_fm_v, run_dir=run_dir))
            print(f"  → val_fm_best = {best_fm_v:.2f}")
        except subprocess.CalledProcessError as e:
            print(f"  [!] Run failed: {e}")
            results.append(dict(cfg=cfg, val_fm_best=None, error=str(e)))
    valid = [r for r in results if r["val_fm_best"] is not None]
    if not valid:
        print("[!] No grid runs succeeded.")
        return
    valid.sort(key=lambda r: r["val_fm_best"], reverse=True)
    best = valid[0]
    print("\n" + "=" * 70)
    print(f"BEST {args.dataset}: val_fm={best['val_fm_best']:.2f}")
    print(f"  config: {best['cfg']}")
    print("=" * 70)
    out_dir = os.path.join(args.grid_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grid_search.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "best_config.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n[+] Wrote grid_search.json and best_config.json to {out_dir}")
    bc = best["cfg"]
    final_cmd = f"python train_patchtst.py --dataset {args.dataset} --patch_len {bc['patch_len']} --patch_stride {bc['patch_stride']} --d_model {bc['d_model']} --n_layers {bc['n_layers']} --dropout {bc['dropout']}"
    print("\nFINAL HEADLINE RUN (full 100 epochs, default patience=20):")
    print(f"  {final_cmd}")


if __name__ == "__main__":
    main()
