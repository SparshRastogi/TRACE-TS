import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

_JSON_PAT = re.compile(
    "^(?P<split>[a-z]+)_class(?P<true_label>\\d+)_(?P<label_name>.+?)_s(?P<sample_idx>\\d{4,})\\.json$"
)


def load_subject_lookup(npz_path: str, splits: list[str]) -> dict[str, dict[int, str]]:
    npz_key_map = {
        "train": "subject_ids_train",
        "val": "subject_ids_valid",
        "valid": "subject_ids_valid",
        "test": "subject_ids_test",
    }
    if not os.path.exists(npz_path):
        print(f"ERROR: .npz not found at {npz_path}", file=sys.stderr)
        print(
            "       Pass --npz_path explicitly, or ensure the path matches settings.py → path_data for your dataset variant.",
            file=sys.stderr,
        )
        sys.exit(1)
    lookup: dict[str, dict[int, str]] = {}
    with np.load(npz_path, allow_pickle=False) as f:
        available_keys = f.files
        for split in splits:
            key = npz_key_map.get(split)
            if key is None:
                print(f"WARNING: unknown split '{split}' — skipping subject lookup.")
                lookup[split] = {}
                continue
            if key not in available_keys:
                print(
                    f"WARNING: '{key}' not found in {npz_path}. Subject stratification will be skipped for split='{split}'. Available keys: {available_keys}"
                )
                lookup[split] = {}
                continue
            ids = f[key]
            lookup[split] = {int(i): str(sid) for i, sid in enumerate(ids)}
    return lookup


def scan_json_files(xai_dir: str, splits: set[str]) -> list[dict]:
    records = []
    for fname in os.listdir(xai_dir):
        m = _JSON_PAT.match(fname)
        if m is None:
            continue
        split = m.group("split")
        if split not in splits:
            continue
        records.append(
            {
                "filename": fname,
                "split": split,
                "true_label": int(m.group("true_label")),
                "label_name": m.group("label_name"),
                "sample_idx": int(m.group("sample_idx")),
            }
        )
    return records


def _png_for_json(fname: str) -> str:
    return fname.replace(".json", ".png")


def stratified_sample(
    records: list[dict],
    subject_lookup: dict[int, str],
    fraction: float,
    floor: int,
    cap: int,
    rng: np.random.Generator,
) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        sid = subject_lookup.get(rec["sample_idx"], "UNKNOWN")
        cell_key = (sid, rec["true_label"])
        cells[cell_key].append(rec)
    sampled = []
    cell_log = []
    for (sid, label), cell_records in sorted(cells.items()):
        n_cell = len(cell_records)
        n_take = int(round(fraction * n_cell))
        n_take = max(n_take, floor)
        n_take = min(n_take, cap)
        n_take = min(n_take, n_cell)
        chosen_indices = rng.choice(len(cell_records), size=n_take, replace=False)
        chosen = [cell_records[i] for i in sorted(chosen_indices)]
        sampled.extend(chosen)
        cell_log.append(
            {
                "subject_id": sid,
                "true_label": label,
                "label_name": cell_records[0]["label_name"],
                "n_before": n_cell,
                "n_after": n_take,
                "kept_pct": round(100.0 * n_take / n_cell, 2) if n_cell > 0 else 0.0,
            }
        )
    return (sampled, cell_log)


def copy_files(
    sampled: list[dict], xai_dir: str, out_dir: str, copy_png: bool, dry_run: bool
) -> int:
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
    n_copied = 0
    n_png = 0
    for rec in sampled:
        src_json = os.path.join(xai_dir, rec["filename"])
        dst_json = os.path.join(out_dir, rec["filename"])
        if not dry_run:
            shutil.copy2(src_json, dst_json)
        n_copied += 1
        if copy_png:
            png_fname = _png_for_json(rec["filename"])
            src_png = os.path.join(xai_dir, png_fname)
            if os.path.exists(src_png):
                dst_png = os.path.join(out_dir, png_fname)
                if not dry_run:
                    shutil.copy2(src_png, dst_png)
                n_png += 1
    return (n_copied, n_png)


def write_manifest(sampled: list[dict], cell_log: list[dict], out_dir: str, args):
    manifest_path = os.path.join(out_dir, "sample_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        fieldnames = [
            "filename",
            "split",
            "true_label",
            "label_name",
            "sample_idx",
            "subject_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sampled:
            writer.writerow(
                {
                    "filename": rec["filename"],
                    "split": rec["split"],
                    "true_label": rec["true_label"],
                    "label_name": rec["label_name"],
                    "sample_idx": rec["sample_idx"],
                    "subject_id": rec.get("subject_id", "UNKNOWN"),
                }
            )
    summary = {
        "sampling_params": {
            "fraction": args.fraction,
            "floor": args.floor,
            "cap": args.cap,
            "seed": args.seed,
            "splits": args.splits,
            "xai_dir": args.xai_dir,
            "npz_path": args.npz_path,
        },
        "totals": {
            "n_input": sum((c["n_before"] for c in cell_log)),
            "n_sampled": sum((c["n_after"] for c in cell_log)),
        },
        "per_cell": cell_log,
    }
    summary_path = os.path.join(out_dir, "sample_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return (manifest_path, summary_path)


def print_report(cell_log: list[dict], splits: list[str], fraction: float):
    all_splits = sorted({c.get("split", "?") for c in cell_log})
    print(f"\n{'=' * 70}")
    print(f"  Stratified sampling report  (target fraction = {fraction * 100:.1f}%)")
    print(f"{'=' * 70}")
    for sp in all_splits:
        sp_cells = [c for c in cell_log if c.get("split") == sp]
        if not sp_cells:
            continue
        n_before = sum((c["n_before"] for c in sp_cells))
        n_after = sum((c["n_after"] for c in sp_cells))
        n_subjects = len({c["subject_id"] for c in sp_cells})
        n_classes = len({c["true_label"] for c in sp_cells})
        print(f"\n  Split: {sp.upper()}")
        print(f"  {'─' * 65}")
        print(
            f"  Subjects: {n_subjects}  |  Classes: {n_classes}  |  Cells: {len(sp_cells)}"
        )
        print(
            f"  Total before: {n_before:6d}  →  after: {n_after:6d}  ({100 * n_after / n_before:.2f}% kept)"
        )
        print(
            f"\n  {'Subject':<8} {'Class':<6} {'Label':<20} {'Before':>8} {'After':>7} {'Kept%':>7}"
        )
        print(f"  {'─' * 65}")
        for c in sorted(sp_cells, key=lambda x: (x["subject_id"], x["true_label"])):
            print(
                f"  {c['subject_id']:<8} {c['true_label']:<6} {c['label_name']:<20} {c['n_before']:>8} {c['n_after']:>7} {c['kept_pct']:>6.2f}%"
            )
    print(f"\n{'=' * 70}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified (subject × class) sampler for XAI attribution outputs. Copies a `--fraction` sample of JSONs (+ optional PNGs) from --xai_dir into --out_dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xai_dir",
        required=True,
        help="Directory containing the generated attribution JSONs/PNGs.",
    )
    parser.add_argument(
        "--npz_path",
        default="./dataset/capture24_Willetts2018_subset.npz",
        help="Path to the source .npz used during XAI generation. Must contain subject_ids_train / subject_ids_valid / subject_ids_test. Defaults to the Willetts2018 6-class subset path from settings.py.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Where to write the sampled outputs. Defaults to <xai_dir>_sampled_<fraction*100>pct.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of each (subject, class) cell to keep.",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=1,
        help="Minimum samples per non-empty (subject, class) cell.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=200,
        help="Maximum samples per (subject, class) cell.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducibility."
    )
    parser.add_argument(
        "--copy_png",
        action="store_true",
        default=True,
        help="Also copy matching PNG dashboards when present.",
    )
    parser.add_argument(
        "--no_copy_png",
        dest="copy_png",
        action="store_false",
        help="Skip PNG copying (JSON only).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print stats and manifest without copying any files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        pct_tag = f"{int(round(args.fraction * 100))}pct"
        args.out_dir = str(
            Path(args.xai_dir).resolve().parent
            / f"{Path(args.xai_dir).name}_sampled_{pct_tag}"
        )
    if args.dry_run:
        print("[DRY RUN] No files will be copied.")
    print(f"XAI input dir : {args.xai_dir}")
    print(f"Source .npz   : {args.npz_path}")
    print(f"Output dir    : {args.out_dir}")
    print(f"Splits        : {args.splits}")
    print(f"Fraction      : {args.fraction * 100:.1f}%")
    print(f"Floor / Cap   : {args.floor} / {args.cap}")
    print(f"Seed          : {args.seed}")
    print(f"Copy PNGs     : {args.copy_png}")
    print(f"\n[1/4] Loading subject IDs from {args.npz_path} ...")
    subject_lookup_by_split = load_subject_lookup(args.npz_path, args.splits)
    for sp, lut in subject_lookup_by_split.items():
        print(f"      {sp}: {len(lut)} subject-ID entries loaded")
    print(f"\n[2/4] Scanning {args.xai_dir} for attribution JSONs ...")
    splits_set = set(args.splits)
    all_records = scan_json_files(args.xai_dir, splits_set)
    print(
        f"      Found {len(all_records)} matching JSON files across splits: {sorted(splits_set)}"
    )
    if len(all_records) == 0:
        print("ERROR: No JSON files found. Check --xai_dir and --splits.")
        sys.exit(1)
    print(f"\n[3/4] Stratified sampling ...")
    rng = np.random.default_rng(args.seed)
    all_sampled = []
    all_cell_log = []
    for split in args.splits:
        split_records = [r for r in all_records if r["split"] == split]
        if not split_records:
            print(f"      WARNING: no files found for split='{split}', skipping.")
            continue
        sampled, cell_log = stratified_sample(
            records=split_records,
            subject_lookup=subject_lookup_by_split.get(split, {}),
            fraction=args.fraction,
            floor=args.floor,
            cap=args.cap,
            rng=rng,
        )
        for rec in sampled:
            rec["subject_id"] = subject_lookup_by_split.get(split, {}).get(
                rec["sample_idx"], "UNKNOWN"
            )
        for c in cell_log:
            c["split"] = split
        all_sampled.extend(sampled)
        all_cell_log.extend(cell_log)
        n_before = sum((c["n_before"] for c in cell_log))
        n_after = sum((c["n_after"] for c in cell_log))
        print(
            f"      {split}: {n_before} → {n_after} ({100 * n_after / n_before:.2f}% kept)"
        )
    print(f"\n[4/4] Copying files to {args.out_dir} ...")
    n_copied, n_png = copy_files(
        all_sampled, args.xai_dir, args.out_dir, args.copy_png, args.dry_run
    )
    if not args.dry_run:
        manifest_path, summary_path = write_manifest(
            all_sampled, all_cell_log, args.out_dir, args
        )
        print(f"      Manifest CSV  → {manifest_path}")
        print(f"      Summary JSON  → {summary_path}")
    else:
        print("      [DRY RUN] Skipped file copy and manifest write.")
    print_report(all_cell_log, args.splits, args.fraction)
    print(f"Done.")
    print(f"  JSONs copied : {n_copied}" + (" (dry run)" if args.dry_run else ""))
    if args.copy_png:
        print(f"  PNGs  copied : {n_png}" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
