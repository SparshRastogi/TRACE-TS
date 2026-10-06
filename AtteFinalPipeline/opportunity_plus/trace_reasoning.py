import os
import re
import csv
import json
import glob
import time
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
OPP_NUM_CLASS = 18
OPP_ACTIVITY_CLASSES = [n.lower() for n in OPP_CLASS_NAMES[1:]]
INSTRUCTION = (
    "Analyze the sensor embeddings, generate reasoning and explain the activity:"
)
STRUCTURED_TEMPLATE_PREFIX = "\n[OBSERVATION | id: O1]\nsensor:"
_ACT_RE = re.compile("\\[ACTIVITY\\]\\s*:\\s*(.+?)$", re.IGNORECASE | re.MULTILINE)
DEFAULT_LOAD_WORKERS = 16


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run TRACE Gemma reasoning over Opportunity++ Mantis embeddings and write per-window {reasoning.json + clip.mp4} folders.",
    )
    p.add_argument(
        "--emb_dir",
        required=True,
        help="Directory containing per-window Mantis embedding JSONs (test_class*.json from gen_mantis_embeddings_opp_plus.py).",
    )
    p.add_argument(
        "--clips_dir",
        required=True,
        help="Directory containing the run-subfolders of .mp4 clips from infer_opportunity_plus.py (e.g. results/opportunity_plus/clips/).",
    )
    p.add_argument(
        "--predictions_csv",
        default=None,
        help="Optional but recommended: predictions.csv from infer_opportunity_plus.py. Gives true labels, run names, and confidence to enrich reasoning.json.",
    )
    p.add_argument(
        "--reasoning_dir",
        required=True,
        help="Output root. Each window becomes a subfolder named reasonings_{window_idx:06d}_{PredActivity}/.",
    )
    p.add_argument(
        "--gemma_model_id",
        default="google/gemma-4-4b-it",
        help="HuggingFace model ID for the Gemma backbone.",
    )
    p.add_argument(
        "--gemma_checkpoint",
        required=True,
        help="Path to the TRACE cross-attention checkpoint (.pt).",
    )
    p.add_argument("--n_tokens", type=int, default=8)
    p.add_argument("--adapter_rank", type=int, default=128)
    p.add_argument("--adapter_num_heads", type=int, default=8)
    p.add_argument("--adapter_dropout", type=float, default=0.1)
    p.add_argument("--adapter_layers", default="all")
    p.add_argument(
        "--inference_batch_size",
        type=int,
        default=128,
        help="Batch size for Gemma generation — Gemma-4 is large, keep this small unless on big GPUs.",
    )
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling (default). Pass --no_sample for greedy.",
    )
    p.add_argument("--no_sample", dest="do_sample", action="store_false")
    p.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Cap on number of embedding windows to process (-1 = all). Applied AFTER globbing but BEFORE actually reading any file contents, so this is a true cheap smoke-test knob.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="If a reasoning.json already exists for a given window, skip it (lets you resume an interrupted run cheaply).",
    )
    p.add_argument(
        "--require_clip",
        action="store_true",
        help="Skip windows that have no matching clip on disk. By default, reasoning is still produced and the missing clip is just noted in reasoning.json.",
    )
    p.add_argument(
        "--load_workers",
        type=int,
        default=DEFAULT_LOAD_WORKERS,
        help="Parallel threads used to read embedding JSONs from disk. Threads are correct here (I/O-bound). Set to 1 to force serial loading for debugging.",
    )
    return p.parse_args()


def _parse_one_embedding_json(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "window_idx": int(d["sample_idx"]),
        "har_predicted_activity": d["activity"],
        "true_label": int(d.get("true_label", -1)),
        "embedding": np.asarray(d["mantis_embedding"]["values"], dtype=np.float32),
        "source_path": os.path.abspath(path),
    }


def load_embedding_jsons(emb_dir, limit=-1, workers=DEFAULT_LOAD_WORKERS):
    pattern = os.path.join(emb_dir, "test_class*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No test_class*.json files found in {emb_dir}.\nDid you run gen_mantis_embeddings_opp_plus.py first?"
        )
    if limit > 0:
        files = files[:limit]
    n = len(files)
    print(
        f"  Found {n} embedding JSON files; loading with {workers} thread{('s' if workers != 1 else '')}…",
        flush=True,
    )
    out = []
    failures = []
    t0 = time.time()
    step = max(500, n // 20)
    if workers <= 1:
        for i, path in enumerate(files, 1):
            try:
                out.append(_parse_one_embedding_json(path))
            except Exception as e:
                failures.append((path, repr(e)))
            if i % step == 0 or i == n:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                print(
                    f"    [{i}/{n}] {rate:.0f} files/s elapsed {elapsed:.1f}s",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_parse_one_embedding_json, p): p for p in files}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    out.append(fut.result())
                except Exception as e:
                    failures.append((futures[fut], repr(e)))
                if done % step == 0 or done == n:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(
                        f"    [{done}/{n}] {rate:.0f} files/s elapsed {elapsed:.1f}s",
                        flush=True,
                    )
    out.sort(key=lambda r: r["window_idx"])
    if failures:
        print(f"  [WARN] {len(failures)} embedding JSON(s) failed to load; first 5:")
        for path, err in failures[:5]:
            print(f"    {path}: {err}")
    if not out:
        raise RuntimeError(
            f"All {n} embedding JSONs failed to load — aborting. See [WARN] lines above for the first failures."
        )
    elapsed = time.time() - t0
    print(
        f"  Loaded {len(out)}/{n} embedding JSONs in {elapsed:.1f}s ({len(out) / max(elapsed, 1e-06):.0f} files/s, embedding dim={out[0]['embedding'].shape[0]})"
    )
    return out


def load_predictions_csv_map(path):
    if not path:
        print(
            "  [info] --predictions_csv not given; reasoning.json will lack run / true_activity / har_confidence fields."
        )
        return {}
    if not os.path.isfile(path):
        print(f"  [WARN] predictions.csv not found at {path}; continuing without it.")
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            wid = int(row["window_idx"])
            out[wid] = {
                "run": row.get("run", ""),
                "true_label": int(row["true_label"]),
                "pred_label": int(row["pred_label"]),
                "confidence": float(row.get("confidence") or "nan"),
                "start_frame": int(row.get("start_frame") or 0),
                "end_frame": int(row.get("end_frame") or 0),
                "video_fps": float(row.get("video_fps") or 0.0),
                "start_time_s": float(row.get("start_time_s") or 0.0),
                "end_time_s": float(row.get("end_time_s") or 0.0),
                "video_file": row.get("video_file", ""),
            }
    print(f"  Loaded {len(out)} predictions rows from {path}")
    return out


def build_clip_index(clips_dir):
    if not os.path.isdir(clips_dir):
        print(
            f"  [WARN] clips_dir {clips_dir} not found; reasoning will run but no clips will be copied."
        )
        return {}
    pattern = os.path.join(clips_dir, "**", "*.mp4")
    t0 = time.time()
    files = glob.glob(pattern, recursive=True)
    rx = re.compile("^(\\d{6})_")
    index = {}
    for fp in files:
        m = rx.match(os.path.basename(fp))
        if m:
            index[int(m.group(1))] = os.path.abspath(fp)
    print(f"  Indexed {len(index)} clips under {clips_dir} ({time.time() - t0:.1f}s)")
    return index


def extract_predicted_activity(text, activity_classes):
    m = _ACT_RE.search(text)
    if m:
        raw = m.group(1).strip().lower().replace("_", " ")
        for cls in sorted(activity_classes, key=len, reverse=True):
            if cls == raw or cls in raw:
                return cls
        return raw if raw else "unknown"
    tn = text.lower().replace("_", " ")
    for cls in sorted(activity_classes, key=len, reverse=True):
        if cls in tn:
            return cls
    return "unknown"


def _safe_activity_token(name):
    if not name:
        return "Unknown"
    return name.replace(" ", "").replace("/", "_")


def make_reasoning_folder(reasoning_dir, window_idx, predicted_activity_name):
    token = _safe_activity_token(predicted_activity_name)
    folder = os.path.join(reasoning_dir, f"reasonings_{window_idx:06d}_{token}")
    os.makedirs(folder, exist_ok=True)
    return folder


def main():
    args = parse_args()
    print("\n" + "=" * 70)
    print("TRACE Opportunity++ Stage-4 Reasoning")
    print("=" * 70)
    print(f"  emb_dir         : {args.emb_dir}")
    print(f"  predictions_csv : {args.predictions_csv or '(none)'}")
    print(f"  clips_dir       : {args.clips_dir}")
    print(f"  reasoning_dir   : {args.reasoning_dir}")
    print(f"  gemma_model_id  : {args.gemma_model_id}")
    print(f"  gemma_checkpoint: {args.gemma_checkpoint}")
    print(f"  num_samples     : {args.num_samples}")
    print(f"  batch_size      : {args.inference_batch_size}")
    print(f"  load_workers    : {args.load_workers}")
    print(f"  skip_existing   : {args.skip_existing}")
    print(f"  require_clip    : {args.require_clip}")
    print("=" * 70)
    os.makedirs(args.reasoning_dir, exist_ok=True)
    print("\n[1/4] Loading inputs")
    embeddings = load_embedding_jsons(
        args.emb_dir, limit=args.num_samples, workers=args.load_workers
    )
    pred_map = load_predictions_csv_map(args.predictions_csv)
    clip_map = build_clip_index(args.clips_dir)
    if args.require_clip:
        before = len(embeddings)
        embeddings = [r for r in embeddings if r["window_idx"] in clip_map]
        print(
            f"  --require_clip: kept {len(embeddings)}/{before} windows that have a clip on disk."
        )
    if args.skip_existing:
        before = len(embeddings)
        kept = []
        for r in embeddings:
            wid = r["window_idx"]
            existing = glob.glob(
                os.path.join(
                    args.reasoning_dir, f"reasonings_{wid:06d}_*", "reasoning.json"
                )
            )
            if not existing:
                kept.append(r)
        embeddings = kept
        print(
            f"  --skip_existing: dropped {before - len(embeddings)} already-processed windows; {len(embeddings)} remain."
        )
    if not embeddings:
        print("\n[!] Nothing to do. Exiting.")
        return
    total = len(embeddings)
    input_dim = embeddings[0]["embedding"].shape[0]
    print(f"\n[2/4] Loading models")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from trace.model.backbone import (
        load_model_and_tokenizer,
        _get_llm_layers,
        _get_llm_dim,
    )
    from trace.model.sensor_llm import SensorLLMCrossAttn
    from trace.training.checkpoint import load_checkpoint
    from trace.utils.naming import _resolve_adapter_layer_indices

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Loading backbone: {args.gemma_model_id}")
    tokenizer, llm = load_model_and_tokenizer(args.gemma_model_id)
    llm.eval()
    n_llm_layers = len(_get_llm_layers(llm))
    llm_dim = _get_llm_dim(llm)
    layer_indices = _resolve_adapter_layer_indices(args.adapter_layers, n_llm_layers)
    print(
        f"  LLM layers: {n_llm_layers}, hidden_dim: {llm_dim}, adapter_layers: {len(layer_indices)}"
    )
    print(f"  Loading TRACE checkpoint: {args.gemma_checkpoint}")
    projector, adapters, _cls_head = load_checkpoint(
        checkpoint_path=Path(args.gemma_checkpoint),
        input_dim=input_dim,
        llm_dim=llm_dim,
        n_tokens=args.n_tokens,
        adapter_rank=args.adapter_rank,
        adapter_num_heads=args.adapter_num_heads,
        n_adapter_layers=len(layer_indices),
        device=device,
        adapter_dropout=args.adapter_dropout,
    )
    model = SensorLLMCrossAttn(llm, projector, adapters, layer_indices)
    model.eval()
    prompt_text = f"User: {INSTRUCTION}\nAssistant: " + STRUCTURED_TEMPLATE_PREFIX
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    prompt_ids = prompt_inputs["input_ids"].to(device)
    prompt_mask = prompt_inputs["attention_mask"].to(device)
    print(f"  Prompt: {prompt_ids.shape[1]} tokens")
    print(
        f"\n[3/4] Running inference on {total} windows (batch_size={args.inference_batch_size})"
    )
    bs = args.inference_batch_size
    emb_tensor = torch.tensor(
        np.stack([r["embedding"] for r in embeddings]), dtype=torch.float32
    )
    idx_tensor = torch.arange(total, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(emb_tensor, idx_tensor), batch_size=bs, shuffle=False
    )
    results = []
    n_clips_copied = 0
    n_clips_missing = 0
    t0 = time.time()
    for emb_batch, idx_batch in loader:
        B = emb_batch.shape[0]
        with torch.no_grad():
            llm_dtype = next(llm.parameters()).dtype
            sensor_memory = projector(emb_batch.to(device)).to(dtype=llm_dtype)
            model._sensor_memory_ref[0] = sensor_memory
            gen_ids = llm.generate(
                input_ids=prompt_ids.expand(B, -1),
                attention_mask=prompt_mask.expand(B, -1),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = prompt_ids.shape[1]
        for j in range(B):
            rec = embeddings[int(idx_batch[j].item())]
            tail_ids = gen_ids[j][prompt_len:]
            gen_text = STRUCTURED_TEMPLATE_PREFIX + tokenizer.decode(
                tail_ids, skip_special_tokens=True
            )
            reasoning_pred = extract_predicted_activity(gen_text, OPP_ACTIVITY_CLASSES)
            wid = rec["window_idx"]
            pred_row = pred_map.get(wid, {})
            true_label_int = (
                pred_row["true_label"]
                if "true_label" in pred_row
                else rec.get("true_label", -1)
            )
            true_activity = (
                OPP_CLASS_NAMES[true_label_int]
                if 0 <= true_label_int < OPP_NUM_CLASS
                else ""
            )
            clip_src = clip_map.get(wid)
            clip_dst = None
            result = {
                "window_idx": wid,
                "run": pred_row.get("run", ""),
                "har_predicted_activity": rec["har_predicted_activity"],
                "har_predicted_label": pred_row.get("pred_label", -1),
                "true_activity": true_activity,
                "true_label": true_label_int,
                "har_confidence": pred_row.get("confidence", float("nan")),
                "reasoning_predicted_activity": reasoning_pred,
                "generated_text": gen_text,
                "start_frame": pred_row.get("start_frame", -1),
                "end_frame": pred_row.get("end_frame", -1),
                "video_fps": pred_row.get("video_fps", 0.0),
                "source_clip": clip_src or "",
                "source_embedding_json": rec["source_path"],
            }
            folder = make_reasoning_folder(
                args.reasoning_dir, wid, rec["har_predicted_activity"]
            )
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
                        if isinstance(o, np.floating)
                        else int(o)
                        if isinstance(o, np.integer)
                        else str(o)
                    ),
                )
            results.append(result)
        done = len(results)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / speed if speed > 0 else 0.0
        print(
            f"  [{done}/{total}]  {speed:.2f} samples/s  ETA {eta / 60:.1f} min  clips_copied={n_clips_copied} missing={n_clips_missing}",
            flush=True,
        )
    print(f"\n[4/4] Writing summary")
    pred_hist = Counter((r["reasoning_predicted_activity"] for r in results))
    har_hist = Counter((r["har_predicted_activity"] for r in results))
    agree = sum(
        (
            1
            for r in results
            if r["reasoning_predicted_activity"].lower().strip()
            == r["har_predicted_activity"].lower().strip()
        )
    )
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gemma_model_id": args.gemma_model_id,
        "gemma_checkpoint": args.gemma_checkpoint,
        "n_tokens": args.n_tokens,
        "adapter_rank": args.adapter_rank,
        "adapter_num_heads": args.adapter_num_heads,
        "adapter_layers": args.adapter_layers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "emb_dir": os.path.abspath(args.emb_dir),
        "predictions_csv": os.path.abspath(args.predictions_csv)
        if args.predictions_csv
        else "",
        "clips_dir": os.path.abspath(args.clips_dir),
        "reasoning_dir": os.path.abspath(args.reasoning_dir),
        "total_windows": total,
        "clips_copied": n_clips_copied,
        "clips_missing": n_clips_missing,
        "agreement_har_vs_reasoning": agree,
        "agreement_rate": agree / len(results) if results else 0.0,
        "reasoning_activity_histogram": dict(pred_hist),
        "har_activity_histogram": dict(har_hist),
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary_path = os.path.join(args.reasoning_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Reasoning folders : {args.reasoning_dir}/reasonings_*")
    print(f"  Summary           : {summary_path}")
    print(f"  Total windows     : {total}")
    print(f"  Clips copied      : {n_clips_copied}")
    print(f"  Clips missing     : {n_clips_missing}")
    print(
        f"  HAR↔Reasoning agree: {agree}/{len(results)} ({100 * agree / max(1, len(results)):.1f}%)"
    )
    print(f"  Elapsed           : {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
