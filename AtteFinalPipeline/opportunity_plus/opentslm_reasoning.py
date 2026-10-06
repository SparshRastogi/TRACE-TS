from __future__ import annotations
from AtteFinalPipeline.paths import PIPELINE_ROOT
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.io as sio

SCRIPT_DIR = PIPELINE_ROOT
REPO_PATH = Path(
    os.environ.get(
        "OPENTSLM_ROOT", SCRIPT_DIR / "external_benchmarks" / "opentslm" / "repo"
    )
)
OPENTSLM_SRC = REPO_PATH / "src"
BENCH_PATH = SCRIPT_DIR / "external_benchmarks" / "opentslm"
OPP_WINDOW = 24
OPP_NUM_CLASS = 18
OPP_CLASS_NAMES = [
    "Null",
    "Close Dishwasher",
    "Close Drawer 3",
    "Close Drawer 2",
    "Close Door 1",
    "Close Door 2",
    "Close Drawer 1",
    "Close Fridge",
    "Toggle Switch",
    "Open Dishwasher",
    "Open Drawer 3",
    "Open Drawer 2",
    "Open Door 1",
    "Open Door 2",
    "Open Drawer 1",
    "Open Fridge",
    "Drink from Cup",
    "Clean Table",
]
assert len(OPP_CLASS_NAMES) == OPP_NUM_CLASS
OPP_ACCEL_X_IDX = 0
OPP_ACCEL_Y_IDX = 1
OPP_ACCEL_Z_IDX = 2
OPP_ACTIVITIES = OPP_CLASS_NAMES[1:]
PRETRAINED_REPO = "OpenTSLM/llama-3.2-1b-har-sp"
_EPS = 1e-08


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run OpenTSLM reasoning over Opportunity++ raw sensor windows and write per-window {reasoning.json + clip.mp4} folders.",
    )
    p.add_argument(
        "--mat",
        required=True,
        help="Path to the .mat from prepare_opportunity_plus.py.",
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.csv from infer_opportunity_plus.py.",
    )
    p.add_argument(
        "--clips_dir",
        required=True,
        help="Directory containing run-subfolders of .mp4 clips (e.g. results/opportunity_plus/clips/).",
    )
    p.add_argument(
        "--reasoning_dir",
        required=True,
        help="Output root for per-window reasoning folders.",
    )
    p.add_argument(
        "--mode",
        choices=["zero_shot", "finetuned"],
        default="zero_shot",
        help="zero_shot: load pretrained OpenTSLM/llama-3.2-1b-har-sp. finetuned: also load --checkpoint on top.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned checkpoint (best.pt) — required when --mode finetuned.",
    )
    p.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token if needed to download the model.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size (samples per forward pass).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per window.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Cap on windows to process (-1 = all non-Null predicted).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip windows that already have reasoning.json on disk.",
    )
    p.add_argument(
        "--require_clip",
        action="store_true",
        help="Skip windows with no matching clip on disk.",
    )
    p.add_argument(
        "--keep_null",
        action="store_true",
        help="Also process windows where the HAR model predicted Null (they won't have clips, but reasoning will still be written).",
    )
    return p.parse_args()


def load_predictions_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["window_idx"] = int(row["window_idx"])
            row["start_sample"] = int(row["start_sample"])
            row["end_sample"] = int(row["end_sample"])
            row["true_label"] = int(row["true_label"])
            row["pred_label"] = int(row["pred_label"])
            row["start_frame"] = int(row.get("start_frame") or 0)
            row["end_frame"] = int(row.get("end_frame") or 0)
            row["video_fps"] = float(row.get("video_fps") or 0.0)
            row["start_time_s"] = float(row.get("start_time_s") or 0.0)
            row["end_time_s"] = float(row.get("end_time_s") or 0.0)
            try:
                row["confidence"] = float(row["confidence"])
            except (KeyError, ValueError, TypeError):
                row["confidence"] = float("nan")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"predictions CSV at {path} is empty.")
    return rows


def build_windows(
    X: np.ndarray, pred_rows: list[dict], window: int = OPP_WINDOW
) -> np.ndarray:
    N, C = X.shape
    out = np.empty((len(pred_rows), window, C), dtype=np.float32)
    bad = []
    for i, row in enumerate(pred_rows):
        s = row["start_sample"]
        e = s + window
        if e > N:
            bad.append((i, s, e, N))
            slab = np.zeros((window, C), dtype=np.float32)
            avail = max(0, N - s)
            if avail > 0:
                slab[:avail] = X[s : s + avail]
            out[i] = slab
        else:
            out[i] = X[s:e]
    if bad:
        first = bad[0]
        raise RuntimeError(
            f"[FATAL] {len(bad)} prediction rows reference samples past the end of the .mat (e.g. row {first[0]}: start_sample={first[1]}, end={first[2]}, N={first[3]}). Re-run prepare_opportunity_plus.py and infer_opportunity_plus.py."
        )
    return out


def build_clip_index(clips_dir: str) -> dict[int, str]:
    if not os.path.isdir(clips_dir):
        print(f"  [WARN] clips_dir {clips_dir} not found; no clips will be copied.")
        return {}
    t0 = time.time()
    files = glob.glob(os.path.join(clips_dir, "**", "*.mp4"), recursive=True)
    rx = re.compile("^(\\d{6})_")
    index: dict[int, str] = {}
    for fp in files:
        m = rx.match(os.path.basename(fp))
        if m:
            index[int(m.group(1))] = os.path.abspath(fp)
    print(f"  Indexed {len(index)} clips under {clips_dir} ({time.time() - t0:.1f}s)")
    return index


_TS_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]


def _normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(arr.mean())
    std = float(arr.std())
    return ((arr - mean) / (std + _EPS), mean, std)


def _build_pre_prompt() -> str:
    label_str = ", ".join(OPP_ACTIVITIES)
    return f'You are given accelerometer data in all three dimensions. Your task is to classify the activity based on analysis of the data.\n\nInstructions:\n- Begin by analyzing the time series without assuming a specific label.\n- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.\n- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.\n- Do **not** mention any class label until the final sentence.\n\nPossible activity labels are:\n{label_str}.\n\n- Make sure that your last word is the answer. You MUST end your response with "Answer: "'


def build_opentslm_sample(window: np.ndarray, activity: str, eos_token: str) -> dict:
    x_raw = window[:, OPP_ACCEL_X_IDX].astype(float)
    y_raw = window[:, OPP_ACCEL_Y_IDX].astype(float)
    z_raw = window[:, OPP_ACCEL_Z_IDX].astype(float)
    x_norm, x_mean, x_std = _normalize(x_raw)
    y_norm, y_mean, y_std = _normalize(y_raw)
    z_norm, z_mean, z_std = _normalize(z_raw)
    ts_texts = [
        f"{_TS_LABELS[0]}, it has mean {x_mean:.4f} and std {x_std:.4f}:",
        f"{_TS_LABELS[1]}, it has mean {y_mean:.4f} and std {y_std:.4f}:",
        f"{_TS_LABELS[2]}, it has mean {z_mean:.4f} and std {z_std:.4f}:",
    ]
    return {
        "pre_prompt": _build_pre_prompt(),
        "post_prompt": "",
        "answer": "",
        "time_series": [x_norm, y_norm, z_norm],
        "time_series_text": ts_texts,
        "label": activity,
        "x_axis": x_raw.tolist(),
        "y_axis": y_raw.tolist(),
        "z_axis": z_raw.tolist(),
    }


def _safe_activity_token(name: str) -> str:
    if not name:
        return "Unknown"
    return name.replace(" ", "").replace("/", "_")


def make_reasoning_folder(
    reasoning_dir: str, window_idx: int, predicted_activity_name: str
) -> str:
    token = _safe_activity_token(predicted_activity_name)
    folder = os.path.join(reasoning_dir, f"reasonings_{window_idx:06d}_{token}")
    os.makedirs(folder, exist_ok=True)
    return folder


def main() -> None:
    args = parse_args()
    print("\n" + "=" * 70)
    print("OpenTSLM Opportunity++ Reasoning")
    print("=" * 70)
    print(f"  mat            : {args.mat}")
    print(f"  predictions    : {args.predictions}")
    print(f"  clips_dir      : {args.clips_dir}")
    print(f"  reasoning_dir  : {args.reasoning_dir}")
    print(f"  mode           : {args.mode}")
    if args.mode == "finetuned":
        print(f"  checkpoint     : {args.checkpoint}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  max_new_tokens : {args.max_new_tokens}")
    print(f"  num_samples    : {args.num_samples}")
    print(f"  skip_existing  : {args.skip_existing}")
    print(f"  require_clip   : {args.require_clip}")
    print("=" * 70)
    if args.mode == "finetuned" and (not args.checkpoint):
        sys.exit("[FATAL] --mode finetuned requires --checkpoint.")
    os.makedirs(args.reasoning_dir, exist_ok=True)
    print("\n[1/5] Loading predictions")
    pred_rows = load_predictions_csv(args.predictions)
    print(f"  {len(pred_rows)} total prediction rows")
    if not args.keep_null:
        kept = [r for r in pred_rows if r["pred_label"] != 0]
        print(
            f"  Dropped {len(pred_rows) - len(kept)} Null-prediction windows; {len(kept)} remain."
        )
    else:
        kept = pred_rows
        print(f"  --keep_null: processing all {len(kept)} windows.")
    if args.num_samples > 0:
        kept = kept[: args.num_samples]
        print(f"  --num_samples cap applied: {len(kept)} windows.")
    if not kept:
        print("[!] Nothing to process. Exiting.")
        return
    print(f"\n[2/5] Loading sensor data from {args.mat}")
    mat = sio.loadmat(args.mat)
    X = mat["testingData"].astype(np.float32).T
    print(f"  X = {X.shape}")
    print(f"  Building {len(kept)} windows (window={OPP_WINDOW})")
    X_win = build_windows(X, kept, window=OPP_WINDOW)
    print(f"  X_win = {X_win.shape}")
    del X
    print(f"\n[3/5] Indexing clips")
    clip_map = build_clip_index(args.clips_dir)
    if args.require_clip:
        before = len(kept)
        paired = [(r, w) for r, w in zip(kept, X_win) if r["window_idx"] in clip_map]
        if paired:
            kept, X_win_list = zip(*paired)
            kept = list(kept)
            X_win = np.stack(X_win_list)
        else:
            kept = []
            X_win = X_win[:0]
        print(f"  --require_clip: kept {len(kept)}/{before} windows with clips.")
    if args.skip_existing:
        before = len(kept)
        filtered_rows = []
        filtered_wins = []
        for r, w in zip(kept, X_win):
            wid = r["window_idx"]
            if glob.glob(
                os.path.join(
                    args.reasoning_dir, f"reasonings_{wid:06d}_*", "reasoning.json"
                )
            ):
                continue
            filtered_rows.append(r)
            filtered_wins.append(w)
        kept = filtered_rows
        X_win = np.stack(filtered_wins) if filtered_wins else X_win[:0]
        print(
            f"  --skip_existing: dropped {before - len(kept)} processed; {len(kept)} remain."
        )
    if not kept:
        print("\n[!] Nothing to do after filters. Exiting.")
        return
    total = len(kept)
    print(f"\n[4/5] Loading OpenTSLM")
    sys.path.insert(0, str(BENCH_PATH))
    try:
        import importlib.util as _ilu

        if _ilu.find_spec("opentslm") is None:
            raise ImportError
    except (ImportError, ValueError):
        if not OPENTSLM_SRC.exists():
            sys.exit(
                f"[FATAL] opentslm is not importable and the cloned repo was not found at:\n  {OPENTSLM_SRC}Install OpenTSLM or set OPENTSLM_ROOT to a checkout containing its src directory."
            )
        sys.path.insert(0, str(OPENTSLM_SRC))
        print(f"  [info] opentslm not in venv; added repo/src to sys.path")
    else:
        print(f"  [info] opentslm importable from active environment")
    if args.hf_token:
        os.environ.setdefault("HF_TOKEN", args.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", args.hf_token)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    from opentslm.model_config import PATCH_SIZE
    from parse_utils import parse_label as _parse_label

    print(f"  Loading pretrained: {PRETRAINED_REPO}")
    model = OpenTSLM.load_pretrained(PRETRAINED_REPO, enable_lora=True, device=device)
    if args.mode == "finetuned":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"[FATAL] Checkpoint not found: {ckpt_path}")
        print(f"  Loading fine-tuned weights: {ckpt_path}")
        model.load_from_file(str(ckpt_path))
    model.eval()
    eos_token = model.get_eos_token()
    print(f"  EOS token: {eos_token!r}  |  PATCH_SIZE: {PATCH_SIZE}")
    print(
        f"\n[5/5] Running inference on {total} windows (batch_size={args.batch_size})"
    )
    results = []
    n_clips_copied = 0
    n_clips_missing = 0
    t0 = time.time()
    for batch_start in range(0, total, args.batch_size):
        batch_rows = kept[batch_start : batch_start + args.batch_size]
        batch_wins = X_win[batch_start : batch_start + args.batch_size]
        samples = []
        for row, win in zip(batch_rows, batch_wins):
            pred_name = (
                OPP_CLASS_NAMES[row["pred_label"]]
                if 0 <= row["pred_label"] < OPP_NUM_CLASS
                else "Unknown"
            )
            samples.append(build_opentslm_sample(win, pred_name, eos_token))
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            samples, patch_size=PATCH_SIZE
        )
        with torch.no_grad():
            outputs = model.generate(batch, max_new_tokens=args.max_new_tokens)
        for sample, generated, row in zip(samples, outputs, batch_rows):
            wid = row["window_idx"]
            pred_label_int = row["pred_label"]
            true_label_int = row["true_label"]
            pred_name = (
                OPP_CLASS_NAMES[pred_label_int]
                if 0 <= pred_label_int < OPP_NUM_CLASS
                else "Unknown"
            )
            true_name = (
                OPP_CLASS_NAMES[true_label_int]
                if 0 <= true_label_int < OPP_NUM_CLASS
                else ""
            )
            opentslm_pred = _parse_label(generated, OPP_ACTIVITIES) or "unknown"
            clip_src = clip_map.get(wid)
            result: dict = {
                "window_idx": wid,
                "run": row.get("run", ""),
                "har_predicted_activity": pred_name,
                "har_predicted_label": pred_label_int,
                "true_activity": true_name,
                "true_label": true_label_int,
                "har_confidence": row["confidence"],
                "opentslm_predicted_activity": opentslm_pred,
                "generated_text": generated,
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "video_fps": row["video_fps"],
                "start_time_s": row["start_time_s"],
                "end_time_s": row["end_time_s"],
                "source_clip": clip_src or "",
                "model_id": PRETRAINED_REPO,
                "mode": args.mode,
            }
            if args.mode == "finetuned":
                result["checkpoint"] = str(args.checkpoint)
            folder = make_reasoning_folder(args.reasoning_dir, wid, pred_name)
            if clip_src and os.path.isfile(clip_src):
                clip_dst = os.path.join(folder, "clip.mp4")
                if not os.path.exists(clip_dst):
                    shutil.copyfile(clip_src, clip_dst)
                result["clip_copied_to"] = os.path.abspath(clip_dst)
                n_clips_copied += 1
            else:
                result["clip_copied_to"] = ""
                n_clips_missing += 1
            with open(os.path.join(folder, "reasoning.json"), "w") as fh:
                json.dump(
                    result,
                    fh,
                    indent=2,
                    ensure_ascii=False,
                    default=lambda o: (
                        float(o)
                        if isinstance(o, (float, np.floating))
                        else int(o)
                        if isinstance(o, (int, np.integer))
                        else str(o)
                    ),
                )
            results.append(result)
        done = len(results)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / speed if speed > 0 else 0.0
        print(
            f"  [{done}/{total}]  {speed:.2f} samples/s  ETA {eta / 60:.1f} min  clips_copied={n_clips_copied}  missing={n_clips_missing}",
            flush=True,
        )
    print(f"\nWriting summary.json")
    opentslm_hist = Counter((r["opentslm_predicted_activity"] for r in results))
    har_hist = Counter((r["har_predicted_activity"] for r in results))
    agree = sum(
        (
            1
            for r in results
            if r["opentslm_predicted_activity"].lower().strip()
            == r["har_predicted_activity"].lower().strip()
        )
    )
    correct = sum(
        (
            1
            for r in results
            if r["opentslm_predicted_activity"].lower().strip()
            == r["true_activity"].lower().strip()
        )
    )
    n = len(results)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_id": PRETRAINED_REPO,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint) if args.mode == "finetuned" else None,
        "mat": os.path.abspath(args.mat),
        "predictions_csv": os.path.abspath(args.predictions),
        "clips_dir": os.path.abspath(args.clips_dir),
        "reasoning_dir": os.path.abspath(args.reasoning_dir),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "total_windows": total,
        "clips_copied": n_clips_copied,
        "clips_missing": n_clips_missing,
        "opentslm_vs_true_correct": correct,
        "opentslm_vs_true_accuracy": round(correct / n, 4) if n else 0.0,
        "agreement_har_vs_opentslm": agree,
        "agreement_rate": round(agree / n, 4) if n else 0.0,
        "opentslm_activity_histogram": dict(opentslm_hist.most_common()),
        "har_activity_histogram": dict(har_hist.most_common()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary_path = os.path.join(args.reasoning_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Reasoning folders  : {args.reasoning_dir}/reasonings_*/")
    print(f"  Summary            : {summary_path}")
    print(f"  Total windows      : {total}")
    print(f"  Clips copied       : {n_clips_copied}")
    print(f"  Clips missing      : {n_clips_missing}")
    print(f"  OpenTSLM vs true   : {correct}/{n} ({100 * correct / max(1, n):.1f}%)")
    print(f"  HAR↔OpenTSLM agree : {agree}/{n} ({100 * agree / max(1, n):.1f}%)")
    print(f"  Elapsed            : {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
