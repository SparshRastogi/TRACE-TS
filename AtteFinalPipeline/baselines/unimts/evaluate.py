import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

EVAL_CONFIGS = {
    "pamap2": ("pamap2_fullshot", [21, 7, 11], 33, 12),
    "uschad": ("uschad_fullshot", [5], 100, 12),
    "opportunity": ("opportunity_fullshot", [16, 20, 15, 10, 19], 30, 18),
    "opportunity_nonull": ("opportunity_nonull_fullshot", [16, 20, 15, 10, 19], 30, 17),
    "shoaib": ("shoaib_fullshot", [1, 5, 21, 20, 0], 50, 7),
    "mhealth": ("mhealth_fullshot", [11, 3, 21], 50, 13),
    "mhealth_nonull": ("mhealth_nonull_fullshot", [11, 3, 21], 50, 12),
    "capture24": ("capture24_fullshot", [21], 100, 6),
    "capture24_walmsley": ("capture24_walmsley_fullshot", [21], 100, 4),
    "capture24_full": ("capture24_full_fullshot", [21], 100, 6),
    "ucihar": ("ucihar_fullshot", [0], 50, 6),
}
FEW_SHOT_KS = [1, 2, 3, 5, 10]


def setup_unimts_imports(unimts_dir: str):
    unimts_abs = os.path.abspath(unimts_dir)
    if not os.path.isdir(unimts_abs):
        sys.exit(f"[ERROR] UniMTS dir not found: {unimts_abs}")
    sys.path.insert(0, unimts_abs)
    from data import load_custom_data
    from contrastive import ContrastiveModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (load_custom_data, ContrastiveModule, device)


def build_and_load(ContrastiveModule, ckpt_path, num_class, device, gyro, stft):
    from types import SimpleNamespace

    args_ns = SimpleNamespace(
        stage="finetune", gyro=gyro, stft=stft, num_class=num_class
    )
    model = ContrastiveModule(args_ns).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and (not any((k.startswith(("model.", "fc.")) for k in state)))
    ):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    crit = [k for k in missing if k.startswith(("fc.", "model.acc."))]
    if crit:
        raise RuntimeError(
            f"Checkpoint missing critical keys (fc.* or model.acc.*): {crit[:5]}. Did you fine-tune with the same --gyro/--stft flags?"
        )
    if missing:
        print(
            f"   [info] {len(missing)} non-critical missing keys (likely CLIP text buffers — fine for eval)."
        )
    if unexpected:
        print(
            f"   [warn] {len(unexpected)} unexpected keys (showing 3): {unexpected[:3]}"
        )
    model.eval()
    return model


@torch.no_grad()
def evaluate_loader(model, test_loader, device, gyro, stft):
    if stft:
        raise NotImplementedError(
            "stft=1 path not implemented in eval; matches finetune-time stft=0."
        )
    all_preds, all_labels = ([], [])
    for batch in test_loader:
        inp = batch[0].to(device).float()
        labels = batch[-1].to(device).long().view(-1)
        b, t, c = inp.shape
        if not gyro:
            idx = np.concatenate([np.arange(i, i + 3) for i in range(0, c, 6)])
            inp = inp[:, :, idx]
            b, t, c = inp.shape
        inp = inp.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
        logits = model.classifier(inp)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return (np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0))


def evaluate_one(
    name,
    run_tag,
    joints,
    sr,
    num_class,
    unimts_dir,
    data_dir,
    load_custom_data,
    ContrastiveModule,
    device,
    batch_size=128,
    gyro=0,
    stft=0,
    padding_size=200,
    k_value=None,
):
    suffix = "k=None" if k_value is None else f"k={k_value}"
    ckpt_dir = os.path.join(unimts_dir, "checkpoint", run_tag)
    ckpt_path = os.path.join(ckpt_dir, f"{suffix}_best_loss.pth")
    row = dict(
        dataset=name,
        run_tag=run_tag,
        k="" if k_value is None else str(k_value),
        n_test="",
        accuracy="",
        f1_macro="",
        f1_weighted="",
        ckpt_path=ckpt_path,
        status="",
    )
    if not os.path.isfile(ckpt_path):
        row["status"] = "missing checkpoint"
        return row
    print(f"\n──── {name}  ({run_tag}, {suffix}) ────")
    print(f"   ckpt: {ckpt_path}")
    X_path = os.path.join(data_dir, name, "X_test.npy")
    y_path = os.path.join(data_dir, name, "y_test.npy")
    cfg = os.path.join(data_dir, name, f"{name}.json")
    for p in (X_path, y_path, cfg):
        if not os.path.exists(p):
            row["status"] = f"missing data file: {p}"
            return row
    try:
        result = load_custom_data(
            X_path,
            y_path,
            cfg,
            joints,
            sr,
            padding_size=padding_size,
            split="test",
            k=None,
            few_shot_path=None,
        )
    except Exception as e:
        row["status"] = f"load_custom_data failed: {type(e).__name__}: {e}"
        return row
    if len(result) == 5:
        test_inputs, test_masks, test_labels, _, _ = result
    elif len(result) == 4:
        test_inputs, test_masks, test_labels, _ = result
    elif len(result) == 3:
        test_inputs, test_labels, _ = result
        test_masks = torch.ones(test_inputs.shape[0], dtype=torch.long)
    elif len(result) == 2:
        test_inputs, test_labels = result
        test_masks = torch.ones(test_inputs.shape[0], dtype=torch.long)
    else:
        row["status"] = (
            f"unexpected load_custom_data return arity: got {len(result)} items"
        )
        return row
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    try:
        model = build_and_load(
            ContrastiveModule, ckpt_path, num_class, device, gyro=gyro, stft=stft
        )
    except Exception as e:
        row["status"] = f"build/load failed: {type(e).__name__}: {e}"
        return row
    try:
        preds, labels = evaluate_loader(
            model, test_loader, device, gyro=gyro, stft=stft
        )
    except Exception as e:
        row["status"] = f"inference failed: {type(e).__name__}: {e}"
        return row
    acc = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    print(f"   n_test       = {len(labels)}")
    print(f"   accuracy     = {acc:.4f}")
    print(f"   F1 (macro)   = {f1_mac:.4f}")
    print(f"   F1 (weighted)= {f1_w:.4f}")
    row.update(
        n_test=len(labels),
        accuracy=round(float(acc), 6),
        f1_macro=round(float(f1_mac), 6),
        f1_weighted=round(float(f1_w), 6),
        status="ok",
    )
    return row


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--unimts_dir", default="./UniMTS")
    p.add_argument("--data_dir", default="./UniMTS_data")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--out", default="./unimts_eval_resultsCap.csv")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--padding_size",
        type=int,
        default=200,
        help="Must match what was used at fine-tune time.",
    )
    p.add_argument(
        "--gyro",
        type=int,
        default=0,
        help="Must match the --gyro used at fine-tune time.",
    )
    p.add_argument(
        "--stft",
        type=int,
        default=0,
        help="Must match the --stft used at fine-tune time.",
    )
    p.add_argument(
        "--few_shot",
        action="store_true",
        help="Also evaluate k=1,2,3,5,10 if checkpoints exist.",
    )
    args = p.parse_args()
    print(f"[*] UniMTS dir : {os.path.abspath(args.unimts_dir)}")
    print(f"[*] data dir   : {os.path.abspath(args.data_dir)}")
    print(f"[*] gyro={args.gyro}  stft={args.stft}  padding_size={args.padding_size}")
    print(f"    (these MUST match the values you fine-tuned with)")
    load_custom_data, ContrastiveModule, device = setup_unimts_imports(args.unimts_dir)
    print(f"[*] device = {device}")
    targets = (
        list(EVAL_CONFIGS.keys())
        if not args.datasets
        else [d for d in args.datasets if d in EVAL_CONFIGS]
    )
    rows = []
    for name in targets:
        run_tag, joints, sr, num_class = EVAL_CONFIGS[name]
        rows.append(
            evaluate_one(
                name,
                run_tag,
                joints,
                sr,
                num_class,
                args.unimts_dir,
                args.data_dir,
                load_custom_data,
                ContrastiveModule,
                device,
                batch_size=args.batch_size,
                gyro=args.gyro,
                stft=args.stft,
                padding_size=args.padding_size,
                k_value=None,
            )
        )
        if args.few_shot:
            for k in FEW_SHOT_KS:
                fs_tag = run_tag.replace("_fullshot", f"_k{k}")
                rows.append(
                    evaluate_one(
                        name,
                        fs_tag,
                        joints,
                        sr,
                        num_class,
                        args.unimts_dir,
                        args.data_dir,
                        load_custom_data,
                        ContrastiveModule,
                        device,
                        batch_size=args.batch_size,
                        gyro=args.gyro,
                        stft=args.stft,
                        padding_size=args.padding_size,
                        k_value=k,
                    )
                )
    fieldnames = [
        "dataset",
        "run_tag",
        "k",
        "n_test",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "status",
        "ckpt_path",
    ]
    out_path = os.path.abspath(args.out)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print("\n" + "=" * 78)
    print(f"Wrote {len(rows)} rows → {out_path}")
    print("=" * 78)
    print(
        f"{'dataset':<22s} {'k':>4s} {'n_test':>8s} {'acc':>8s} {'F1_mac':>8s} {'F1_wgt':>8s}  status"
    )
    for r in rows:
        if r["status"] == "ok":
            print(
                f"{r['dataset']:<22s} {str(r['k']):>4s} {r['n_test']:>8d} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} {r['f1_weighted']:>8.4f}  ok"
            )
        else:
            print(
                f"{r['dataset']:<22s} {str(r['k']):>4s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s}  {r['status']}"
            )


if __name__ == "__main__":
    main()
