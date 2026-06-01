"""Inference / generation pipeline."""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from trace.config import STRUCTURED_TEMPLATE_PREFIX, _ACT_RE
from trace.model.backbone import load_model_and_tokenizer, _get_llm_layers, _get_llm_dim
from trace.model.sensor_llm import SensorLLMCrossAttn
from trace.data.loader import load_data, load_raw_embeds_data
from trace.data.dataset import InferenceDataset
from trace.training.checkpoint import resolve_checkpoint, load_checkpoint
from trace.utils.naming import _run_prefix, _resolve_adapter_layer_indices, _numeric_sort_key
from trace.eval.constrained import build_constrained_processor


def extract_predicted_activity(text: str, activity_classes: list) -> str:
    act_match = _ACT_RE.search(text)
    if act_match:
        raw = act_match.group(1).strip().lower().replace("_", " ")
        for cls in sorted(activity_classes, key=len, reverse=True):
            if cls == raw or cls in raw:
                return cls
        return raw if raw else "unknown"
    text_normalised = text.lower().replace("_", " ")
    for cls in sorted(activity_classes, key=len, reverse=True):
        if cls in text_normalised:
            return cls
    return "unknown"


def _eval_dir(cfg: dict) -> Path:
    d = Path(cfg["output_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inference_dir(cfg: dict) -> Path:
    d = _eval_dir(cfg) / "inference"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _evaluate_json_dir(cfg: dict) -> Path:
    d = _eval_dir(cfg) / "evaluate"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _csvs_dir(cfg: dict) -> Path:
    d = _eval_dir(cfg) / "csvs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inference_output_path(cfg: dict, ts: str) -> Path:
    run_idx = cfg.get("_run_idx", "")
    suffix = f"_run{run_idx}" if run_idx else ""
    return _inference_dir(cfg) / f"{_run_prefix(cfg)}_inference{suffix}_{ts}.json"


def _inference_partial_path(cfg: dict, rank: int = 0) -> Path:
    run_idx = cfg.get("_run_idx", "")
    suffix = f"_run{run_idx}" if run_idx else ""
    rank_suffix = f"_rank{rank}" if rank > 0 else ""
    return _inference_dir(cfg) / f"{_run_prefix(cfg)}_inference_partial{suffix}{rank_suffix}.json"


def _find_latest_inference_file(cfg: dict) -> Path | None:
    prefix = f"{_run_prefix(cfg)}_inference_"
    d = _inference_dir(cfg)
    if not d.exists():
        return None
    complete = sorted([f for f in d.glob(f"{prefix}*.json") if "_partial" not in f.name])
    return complete[-1] if complete else None


def _save_inference_partial(results: list, partial_path: Path, total_expected: int):
    payload = {"_metadata": {"status": "partial", "completed": len(results),
               "total_expected": total_expected, "last_updated": datetime.now().isoformat()},
               "results": results}
    tmp_path = partial_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.rename(partial_path)


def _load_inference_partial(partial_path: Path, total_expected: int) -> list:
    if not partial_path.exists():
        return []
    try:
        with open(partial_path) as f:
            payload = json.load(f)
        if payload.get("_metadata", {}).get("total_expected", -1) != total_expected:
            print("[inference] Partial file total mismatch. Starting fresh.")
            return []
        results = payload.get("results", [])
        if results:
            print(f"[inference] Resuming: {len(results)}/{total_expected} done")
        return results
    except Exception as e:
        print(f"[inference] WARNING: Partial file corrupt ({e}). Starting fresh.")
        return []


def _build_inference_prompt_ids(tokenizer, cfg: dict, device: str) -> tuple:
    base_text = f"User: {cfg['instruction']}\nAssistant: "
    use_prefix = (cfg.get("use_structured_prefix", True)
                  and cfg.get("data_version", "structured") == "structured")
    prompt_text = base_text + STRUCTURED_TEMPLATE_PREFIX if use_prefix else base_text
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prefix_len = len(tokenizer(STRUCTURED_TEMPLATE_PREFIX, add_special_tokens=False)["input_ids"])
    status = f"WITH structured prefix ({prefix_len} prefix tokens)" if use_prefix else "WITHOUT structured prefix"
    print(f"[inference] Prompt: {status} ({prompt_ids.shape[1]} tokens)")
    return prompt_ids, prompt_mask, use_prefix


def run_generation(cfg: dict, tokenizer=None, llm=None) -> "Path | None":
    """Generate reasoning for test samples and save results JSON.

    In DDP mode (LOCAL_RANK env var set by torch.distributed.run), each rank processes
    a data shard independently. Returns the merged output Path on rank 0, None elsewhere.

    tokenizer/llm may be passed in (single-process mode) to avoid a second model load.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        tokenizer = None  # each rank loads its own model on its own GPU
        llm = None
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    limit = cfg["num_inference_samples"]

    raw_embed_dir = cfg.get("raw_embed_dir")
    if raw_embed_dir is not None and Path(raw_embed_dir).exists():
        if local_rank == 0:
            print(f"[inference] Loading raw embeds from {raw_embed_dir}")
        test_embeds, gt_activity_labels, gt_sample_ids, _ = load_raw_embeds_data(
            raw_embed_dir, class_names_override=cfg["activity_classes"], limit=limit)

        reasoning_gt_map: dict[str, str] = {}
        if cfg["test_json_dir"].exists():
            rfiles = sorted(cfg["test_json_dir"].glob("*.json"),
                            key=lambda f: _numeric_sort_key(str(f)))
            for rfile in tqdm(rfiles, desc="[inference] Indexing reasoning GT",
                              leave=False, disable=(local_rank != 0)):
                try:
                    with open(rfile) as f:
                        rd = json.load(f)
                    sid = str(rd.get("sample_idx", rd.get("sample_id", "")))
                    gt_text = rd.get(cfg["json_key"], "")
                    if sid and gt_text:
                        reasoning_gt_map[sid] = gt_text
                except Exception:
                    pass
            if local_rank == 0:
                print(f"[inference] Indexed {len(reasoning_gt_map)} reasoning GT samples "
                      f"from {cfg['test_json_dir'].name}")
        else:
            if local_rank == 0:
                print(f"[inference] WARNING: test_json_dir not found: {cfg['test_json_dir']}. "
                      "Reasoning metrics will be empty.")
        ground_truths = [reasoning_gt_map.get(sid, "") for sid in gt_sample_ids]
        if local_rank == 0:
            n_with_reasoning = sum(1 for g in ground_truths if g)
            print(f"[inference] {n_with_reasoning}/{len(gt_sample_ids)} samples have reasoning GT")
    else:
        if local_rank == 0:
            print("[inference] raw_embed_dir not found or not set — loading from test_json_dir.")
        test_embeds, ground_truths, gt_activity_labels, gt_sample_ids = load_data(
            cfg["test_json_dir"], cfg["json_key"], cfg["activity_label_key"], limit=limit)

    total_samples = len(test_embeds)

    _owns_model = tokenizer is None or llm is None
    if _owns_model:
        device_map = {"": local_rank} if is_ddp else "auto"
        tokenizer, llm = load_model_and_tokenizer(cfg["model_id"], device_map=device_map)
    llm.eval()
    n_llm_layers = len(_get_llm_layers(llm))
    layer_indices = _resolve_adapter_layer_indices(cfg["adapter_layers"], n_llm_layers)
    cfg["_n_adapter_layers"] = len(layer_indices)
    input_dim = test_embeds.shape[1]
    llm_dim = _get_llm_dim(llm)

    ckpt_path = resolve_checkpoint(cfg)
    projector, adapters = load_checkpoint(
        ckpt_path, input_dim, llm_dim, cfg["n_tokens"], cfg["adapter_rank"],
        cfg["adapter_num_heads"], len(layer_indices), device, cfg["adapter_dropout"])
    model = SensorLLMCrossAttn(llm, projector, adapters, layer_indices)
    model.eval()

    constrained_proc = build_constrained_processor(tokenizer, cfg)
    prompt_ids, prompt_mask, prefix_active = _build_inference_prompt_ids(tokenizer, cfg, device)

    # Each rank processes its own shard of the data
    rank_indices = list(range(local_rank, total_samples, world_size))
    n_rank = len(rank_indices)
    partial_path = _inference_partial_path(cfg, rank=local_rank)
    rank_results = _load_inference_partial(partial_path, n_rank)
    resume_from = len(rank_results)

    if resume_from < n_rank:
        remaining_indices = rank_indices[resume_from:]
        remaining_embeds = test_embeds[remaining_indices]
        remaining_gts    = [ground_truths[i] for i in remaining_indices]
        remaining_acts   = [gt_activity_labels[i] for i in remaining_indices]
        remaining_ids    = [gt_sample_ids[i] for i in remaining_indices]
        loader = DataLoader(InferenceDataset(remaining_embeds),
                            batch_size=cfg["inference_batch_size"], shuffle=False)
        batch_local_idx = 0
        if local_rank == 0:
            mode = f"DDP {world_size} GPUs" if is_ddp else "single GPU"
            print(f"[inference] Generating {total_samples} samples ({mode})  "
                  f"resume={resume_from}/{n_rank} on rank 0...")
        for emb_batch in tqdm(loader, desc=f"Generating [r{local_rank}]",
                              disable=(local_rank != 0)):
            B = emb_batch.shape[0]
            with torch.no_grad():
                llm_dtype = next(llm.parameters()).dtype
                sensor_memory = projector(emb_batch.to(device)).to(dtype=llm_dtype)
                model._sensor_memory_ref[0] = sensor_memory
                output_ids = llm.generate(
                    input_ids=prompt_ids.expand(B, -1),
                    attention_mask=prompt_mask.expand(B, -1),
                    max_new_tokens=cfg["max_new_tokens"],
                    do_sample=True, temperature=cfg["temperature"],
                    top_p=0.9, repetition_penalty=cfg["repetition_penalty"],
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=constrained_proc)
            prompt_len = prompt_ids.shape[1]
            for j in range(B):
                generated_ids = output_ids[j][prompt_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if prefix_active:
                    generated_text = STRUCTURED_TEMPLATE_PREFIX + generated_text
                rank_results.append({
                    "sample_id": remaining_ids[batch_local_idx],
                    "gt_activity": remaining_acts[batch_local_idx],
                    "predicted_activity": extract_predicted_activity(
                        generated_text, cfg["activity_classes"]),
                    "ground_truth_reasoning": remaining_gts[batch_local_idx],
                    "generated": generated_text,
                    "prefix_active": prefix_active,
                })
                batch_local_idx += 1
            _save_inference_partial(rank_results, partial_path, n_rank)
    else:
        if local_rank == 0:
            print(f"[inference] All {n_rank} samples already completed on rank 0.")

    # All ranks rendezvous before rank 0 merges
    if is_ddp:
        dist.barrier()

    # Non-rank-0 processes: leave partial file for rank 0 to read, then exit
    if local_rank != 0:
        return None

    # Rank 0 (or single-process): merge shards and save
    if is_ddp:
        results_by_idx: dict[int, dict] = {}
        for r in range(world_size):
            r_indices = list(range(r, total_samples, world_size))
            if r == 0:
                r_results = rank_results
            else:
                r_partial = _inference_partial_path(cfg, rank=r)
                with open(r_partial) as f:
                    payload = json.load(f)
                r_results = payload.get("results", [])
                r_partial.unlink()
            for idx, res in zip(r_indices[:len(r_results)], r_results):
                results_by_idx[idx] = res
        results = [results_by_idx[i] for i in range(total_samples) if i in results_by_idx]
        partial_path.unlink()
        print(f"[inference] DDP merge: {len(results)}/{total_samples} samples from {world_size} ranks")
    else:
        results = rank_results
        if partial_path.exists():
            partial_path.unlink()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = _inference_output_path(cfg, ts)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[inference] Saved {len(results)} results -> {out_path}")
    return out_path
