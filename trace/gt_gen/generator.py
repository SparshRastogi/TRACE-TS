"""ReasoningGenerator: vLLM-based batch ground-truth generation.

CLI usage:
    python gt_generate.py \\
        --input_dir  /path/to/ig_jsons \\
        --output_dir /path/to/outputs \\
        --model_id   Qwen/Qwen3.5-122B-A10B \\
        --dataset    ucihar \\
        [--batch_size 128] [--max_samples N] [--tensor_parallel_size 4]
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from trace.gt_gen.prompt import (
    REFUSAL_RE, NUMERIC_LEAK_RE, format_attention_data,
)
from trace.gt_gen.output import parse_filename, ReasoningOutput, parse_response
from trace.gt_gen.viz import save_reasoning_graph


class ReasoningGenerator:
    """vLLM-based batch generator for structured reasoning GT traces."""

    def __init__(self, model_id: str, tensor_parallel_size: int = 4,
                 gpu_memory_utilization: float = 0.92, max_model_len: int = 4096,
                 enforce_eager: bool = False):
        print(f"Initializing vLLM engine: {model_id}...")
        import pynvml
        import wandb as _wandb_mod  # noqa: F401 — verify importable early
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        # enforce_eager=True skips CUDA graph capture — required for TP>=8 where
        # the shm_broadcast collective times out during compile_or_warm_up_model.
        # VLLM_FLASHINFER_PREFILL_BACKEND=triton must be set in the environment.
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
        )
        print("vLLM engine ready")

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            repetition_penalty=1.1,
        )

        # Tokenizer: chat template formatting only — NOT used for encode/decode.
        # apply_chat_template is string-only so calling it from a prefetch thread
        # is safe. Do NOT add encode/decode to _load_batch without adding a lock.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_id      = model_id
        self.max_model_len = max_model_len
        # enable_thinking is Qwen3-family-specific; skip for other model families
        self._chat_kwargs  = {"enable_thinking": False} if "qwen" in model_id.lower() else {}
        import pynvml as _pynvml
        _pynvml.nvmlInit()
        self._pynvml    = _pynvml
        self.gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)

    def _load_batch(self, batch_files: List[Path]) -> Optional[List[Dict]]:
        """Load JSON files and build prompt wrappers (prefetch thread).

        Returns None on any error so the main loop can skip gracefully.
        """
        try:
            batch_data = []
            for json_path in batch_files:
                split, sample_id = parse_filename(json_path)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                raw_prompt = format_attention_data(data)
                messages = [{"role": "user", "content": raw_prompt}]
                batch_data.append({
                    'data':      data,
                    'prompt':    self.tokenizer.apply_chat_template(
                                     messages, tokenize=False,
                                     add_generation_prompt=True,
                                     **self._chat_kwargs),
                    'split':     split,
                    'sample_id': sample_id,
                })
            return batch_data
        except Exception as e:
            print(f"  Prefetch load error: {e}")
            return None

    def _init_wandb(self, batch_size: int, output_dir: str,
                    dataset: str, wandb_project: str, wandb_tags: List[str]) -> bool:
        """Initialize W&B. Returns False on failure — generation continues offline."""
        import wandb
        try:
            wandb.init(
                project=wandb_project,
                group="reasoning-generation-structured",
                tags=wandb_tags,
                name=self.model_id,
                config={
                    "model":              self.model_id,
                    "inference_backend":  "vllm",
                    "batch_size":         batch_size,
                    "max_new_tokens":     1024,
                    "temperature":        0.7,
                    "top_p":              0.9,
                    "repetition_penalty": 1.1,
                    "cuda_graphs":        True,
                    "max_model_len":      self.max_model_len,
                    "dataset":            dataset,
                    "output_dir":         output_dir,
                    "template_format":    "structured_v1",
                }
            )
            wandb.define_metric("progress/samples_processed")
            for prefix in ["run_health/*", "gpu/*", "quality/*",
                            "per_class/*", "qualitative/*"]:
                wandb.define_metric(prefix, step_metric="progress/samples_processed")
            print(f"\n{'='*60}")
            print(f"W&B dashboard: {wandb.run.url}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"  W&B init failed ({e}). Continuing without logging.")
            return False

    def batch_generate(self, input_dir: str, output_dir: str,
                       dataset: str = "ucihar",
                       max_samples: Optional[int] = None,
                       batch_size: int = 128,
                       graph_save_every: int = 1000,
                       wandb_project: str = "trace-ts",
                       wandb_tags: Optional[List[str]] = None):
        """Run batch generation with prefetch pipeline, resume, and W&B logging."""
        # Re-init NVML in case a previous batch_generate call shut it down
        try:
            self._pynvml.nvmlInit()
        except Exception:
            pass
        import wandb
        if wandb_tags is None:
            wandb_tags = [dataset, "vllm", "structured-template", "gt-generation"]

        train_out_dir = os.path.join(output_dir, 'reasoning_json', 'train')
        test_out_dir  = os.path.join(output_dir, 'reasoning_json', 'test')
        os.makedirs(train_out_dir, exist_ok=True)
        os.makedirs(test_out_dir,  exist_ok=True)
        split_out_dirs = {
            'train':   train_out_dir,
            'test':    test_out_dir,
            'unknown': os.path.join(output_dir, 'reasoning_json', 'unknown'),
        }
        graph_save_dir = os.path.join(output_dir, 'reasoning_graphs')

        wandb_ok   = self._init_wandb(batch_size, output_dir, dataset,
                                       wandb_project, wandb_tags)
        qual_table = wandb.Table(
            columns=["sample_id", "split", "activity", "confidence",
                     "output_tokens", "reasoning_snippet"]
        ) if wandb_ok else None

        class_stats      = defaultdict(lambda: {"tokens": [], "confidence": [], "anomalies": 0})
        batch_latencies  = []
        rolling_tokens   = []
        rolling_chars    = []
        rolling_leaks    = []
        rolling_refusals = []
        rolling_parse_fails = []
        rolling_n_obs    = []
        rolling_n_inf    = []
        samples_done     = 0
        written_ids      = {'train': set(), 'test': set(), 'unknown': set()}
        training_data    = {'train': [], 'test': [], 'unknown': []}
        all_reasoning    = []
        global_sample_counter = 0

        try:
            all_files = sorted(Path(input_dir).glob('*.json'))

            # Resume: scan both train/ and test/ for already-processed (split, id) pairs.
            # UCI-HAR reuses the same numeric IDs across splits, so key on (split, id).
            processed_ids = set()
            for split_dir in [train_out_dir, test_out_dir]:
                split_name = Path(split_dir).name
                for f in Path(split_dir).glob('*.json'):
                    m = re.match(r'(\d+)', f.name)
                    if m:
                        processed_ids.add((split_name, str(int(m.group(1)))))

            # Pre-populate written_ids from existing JSONL for cross-run dedup
            for split_name in ('train', 'test', 'unknown'):
                jsonl_path = os.path.join(output_dir, f'{split_name}_dataset.jsonl')
                if os.path.exists(jsonl_path):
                    with open(jsonl_path, 'r') as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                sid = item.get('metadata', {}).get('sample_id')
                                if sid:
                                    written_ids[split_name].add(str(sid))
                            except json.JSONDecodeError:
                                pass

            files_to_process = []
            for f in all_files:
                split, sample_id = parse_filename(f)
                if sample_id == '0' and not re.search(r'_s(\d+)$', f.stem):
                    print(f"Warning: couldn't extract ID from {f.name}")
                    continue
                if (split, sample_id) not in processed_ids:
                    files_to_process.append(f)

            # Sort by file size before max_samples slice so a debug run with
            # MAX_SAMPLES=N gets a cross-class sample, not just the first N alphabetically.
            files_to_process.sort(key=lambda f: f.stat().st_size)

            if max_samples:
                files_to_process = files_to_process[:max_samples]

            if not files_to_process:
                print("All samples already processed. Skipping...")
                return []

            total_batches = (len(files_to_process) + batch_size - 1) // batch_size
            split_counts  = defaultdict(int)
            for f in files_to_process:
                split_counts[parse_filename(f)[0]] += 1
            print(f"Source files:      {len(all_files)}")
            print(f"Already processed: {len(processed_ids)}")
            print(f"Remaining:         {len(files_to_process)} "
                  f"(train={split_counts['train']}, test={split_counts['test']})")
            print(f"Total batches:     {total_batches}  (batch_size={batch_size})")
            print(f"Graph save every:  {graph_save_every} samples → {graph_save_dir}")

            batch_ranges = [
                files_to_process[i : i + batch_size]
                for i in range(0, len(files_to_process), batch_size)
            ]

            executor    = ThreadPoolExecutor(max_workers=2)
            next_future = executor.submit(self._load_batch, batch_ranges[0])

            for batch_idx, batch_files in enumerate(batch_ranges):
                try:
                    batch_data = next_future.result()
                except Exception as e:
                    print(f"  Batch {batch_idx+1}/{total_batches} prefetch crashed: {e}")
                    batch_data = None

                if batch_idx + 1 < len(batch_ranges):
                    next_future = executor.submit(self._load_batch,
                                                  batch_ranges[batch_idx + 1])

                if batch_data is None:
                    print(f"  Batch {batch_idx+1}/{total_batches} skipped (load failed)")
                    continue

                print(f"Generating batch {batch_idx+1}/{total_batches} "
                      f"({len(batch_data)} samples)")
                t0 = time.time()

                try:
                    prompts = [item['prompt'] for item in batch_data]

                    # Filter oversized prompts before calling generate
                    safe_indices   = []
                    oversized_log  = os.path.join(output_dir, 'oversized_prompts.txt')
                    for idx, (p, item) in enumerate(zip(prompts, batch_data)):
                        tok_len = len(self.tokenizer.encode(p))
                        if tok_len <= self.max_model_len:
                            safe_indices.append(idx)
                        else:
                            print(f"  Oversized prompt: sample {item['sample_id']} "
                                  f"({tok_len} tokens > {self.max_model_len}) — skipping")
                            with open(oversized_log, 'a') as olf:
                                olf.write(f"{item['split']}\t{item['sample_id']}\t{tok_len}\n")

                    if not safe_indices:
                        print(f"  Batch {batch_idx+1}/{total_batches}: all prompts oversized, skipping")
                        continue

                    safe_prompts    = [prompts[i]    for i in safe_indices]
                    safe_batch_data = [batch_data[i] for i in safe_indices]

                    outputs  = self.llm.generate(safe_prompts, self.sampling_params)
                    gen_time = max(time.time() - t0, 1e-6)
                    batch_output_tokens = 0

                    for j, vllm_output in enumerate(outputs):
                        text   = vllm_output.outputs[0].text
                        item   = safe_batch_data[j]
                        data   = item['data']
                        split  = item['split']
                        sid    = item['sample_id']
                        parsed = parse_response(text)

                        reasoning = ReasoningOutput(
                            sample_id=sid,
                            split=split,
                            predicted_activity=data['activity'],
                            confidence=data['confidence'],
                            overall_reasoning=parsed['overall_reasoning'],
                            reasoning_graph=parsed['reasoning_graph'],
                            generation_timestamp=datetime.now().isoformat(),
                            model_used=self.model_id,
                            mantis_embedding=data.get('mantis_embedding', {}),
                        )

                        out_dir = split_out_dirs.get(split, split_out_dirs['unknown'])
                        os.makedirs(out_dir, exist_ok=True)
                        reasoning.to_json(os.path.join(out_dir, f'{sid}_reasoning.json'))
                        training_data[split].append(reasoning.to_training_format())
                        all_reasoning.append(reasoning)

                        toks    = len(vllm_output.outputs[0].token_ids)
                        chars   = len(text)
                        leak    = bool(NUMERIC_LEAK_RE.search(text))
                        empty   = len(text.strip()) == 0
                        short   = toks < 50
                        refusal = bool(REFUSAL_RE.search(text))
                        batch_output_tokens += toks

                        act = data['activity']
                        class_stats[act]["tokens"].append(toks)
                        class_stats[act]["confidence"].append(data['confidence'])
                        if leak or empty or short or refusal:
                            class_stats[act]["anomalies"] += 1

                        for lst, val in [(rolling_tokens, toks), (rolling_chars, chars),
                                         (rolling_leaks, int(leak)), (rolling_refusals, int(refusal))]:
                            lst.append(val)
                            lst[:] = lst[-100:]

                        rg = parsed['reasoning_graph']
                        for lst, val in [(rolling_parse_fails, int(rg['parse_failed'])),
                                         (rolling_n_obs, rg['n_observations']),
                                         (rolling_n_inf, rg['n_inferences'])]:
                            lst.append(val)
                            lst[:] = lst[-100:]

                        global_sample_counter += 1
                        if global_sample_counter % graph_save_every == 0:
                            graph_path = save_reasoning_graph(
                                rg, sid, data['activity'], graph_save_dir)
                            if graph_path:
                                print(f"  Graph saved: {graph_path}")

                    batch_latency = time.time() - t0
                    batch_latencies.append(batch_latency)
                    samples_done += len(outputs)
                    remaining     = len(files_to_process) - samples_done
                    avg_lat       = sum(batch_latencies[-10:]) / len(batch_latencies[-10:])

                    if wandb_ok:
                        import wandb
                        mem  = self._pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        util = self._pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        wandb.log({
                            "progress/samples_processed":           samples_done,
                            "run_health/generation_time_sec":       round(gen_time, 2),
                            "run_health/batch_latency_sec":         round(batch_latency, 2),
                            "run_health/throughput_tokens_per_sec": round(batch_output_tokens / gen_time, 1),
                            "run_health/throughput_samples_per_hr": round(3600 * len(outputs) / gen_time, 1),
                            "run_health/pct_complete":              round(100.0 * samples_done / len(files_to_process), 2),
                            "run_health/eta_minutes":               round((remaining / max(batch_size, 1)) * avg_lat / 60, 1),
                            "gpu/memory_used_gb":                   round(mem.used / 1e9, 2),
                            "gpu/memory_free_gb":                   round((mem.total - mem.used) / 1e9, 2),
                            "gpu/utilization_pct":                  util.gpu,
                            "quality/rolling_avg_output_tokens":    round(sum(rolling_tokens)   / max(len(rolling_tokens), 1),   1),
                            "quality/rolling_avg_reasoning_chars":  round(sum(rolling_chars)    / max(len(rolling_chars), 1),    1),
                            "quality/rolling_numeric_leak_rate":    round(sum(rolling_leaks)    / max(len(rolling_leaks), 1),    4),
                            "quality/rolling_refusal_rate":         round(sum(rolling_refusals) / max(len(rolling_refusals), 1), 4),
                            "quality/batch_truncated_count":        sum(1 for o in outputs if len(o.outputs[0].token_ids) >= 1024),
                            "quality/rolling_parse_fail_rate":      round(sum(rolling_parse_fails) / max(len(rolling_parse_fails), 1), 4),
                            "quality/rolling_avg_observations":     round(sum(rolling_n_obs) / max(len(rolling_n_obs), 1), 2),
                            "quality/rolling_avg_inferences":       round(sum(rolling_n_inf) / max(len(rolling_n_inf), 1), 2),
                        })

                        if batch_idx % 50 == 0:
                            for vout, bitem in zip(outputs[:2], safe_batch_data[:2]):
                                d = bitem['data']
                                qual_table.add_data(
                                    bitem['sample_id'], bitem['split'],
                                    d['activity'], round(d['confidence'], 4),
                                    len(vout.outputs[0].token_ids),
                                    vout.outputs[0].text[:500],
                                )
                            import wandb
                            wandb.log({
                                "qualitative/reasoning_samples": qual_table,
                                "progress/samples_processed":    samples_done,
                            })

                except Exception as e:
                    print(f"  Batch {batch_idx+1}/{total_batches} failed: {e}")
                    continue

            executor.shutdown(wait=True)

            # Write JSONL files — append mode, cross-run deduped, no mantis_embedding
            for split_name, items in training_data.items():
                if not items:
                    continue
                jsonl_path = os.path.join(output_dir, f'{split_name}_dataset.jsonl')
                with open(jsonl_path, 'a') as f:
                    for item in items:
                        sid = item['metadata']['sample_id']
                        if sid not in written_ids[split_name]:
                            lean = {
                                'input':    item['input'],
                                'output':   item['output'],
                                'metadata': {k: v for k, v in item['metadata'].items()
                                             if k != 'mantis_embedding'},
                            }
                            f.write(json.dumps(lean) + '\n')
                            written_ids[split_name].add(sid)

            # Per-class end-of-run W&B summary
            if wandb_ok:
                import wandb
                for cls, s in class_stats.items():
                    if s["tokens"]:
                        wandb.log({
                            f"per_class/{cls}/avg_output_tokens": round(sum(s['tokens'])     / len(s['tokens']),     1),
                            f"per_class/{cls}/avg_confidence":    round(sum(s['confidence']) / len(s['confidence']), 4),
                            f"per_class/{cls}/anomaly_rate":      round(s['anomalies']       / len(s['tokens']),     4),
                            f"per_class/{cls}/sample_count":      len(s['tokens']),
                            "progress/samples_processed":         samples_done,
                        })

        finally:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            if wandb_ok:
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass

        return all_reasoning


def parse_gt_args() -> dict:
    """CLI argument parser for gt_generate.py."""
    p = argparse.ArgumentParser(
        description="Ground-truth reasoning trace generation via vLLM teacher model"
    )
    # Single-dataset mode
    p.add_argument("--input_dir",  default=None,
                   help="Directory of IG attribution JSONs (single-dataset mode)")
    p.add_argument("--output_dir", default=None,
                   help="Root output directory (single-dataset mode)")
    p.add_argument("--dataset",    default="ucihar",
                   help="Dataset name (single-dataset mode; also used for W&B tags)")
    # Multi-dataset mode (loads model once, iterates all datasets)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="List of dataset names to run sequentially with one model load")
    p.add_argument("--data_root", default="",
                   help="Root dir containing per-dataset IG JSONs (multi-dataset mode)")
    p.add_argument("--output_root", default="",
                   help="Root dir for output (multi-dataset mode)")
    p.add_argument("--output_suffix", default="gt_reasoning",
                   help="Subdirectory name under output_root/<dataset>/ (multi-dataset mode)")
    p.add_argument("--model_id",   default="Qwen/Qwen3.5-122B-A10B",
                   help="HuggingFace model ID for the teacher LLM")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of samples (debug runs; cross-class sampling guaranteed)")
    p.add_argument("--tensor_parallel_size", type=int, default=4,
                   help="Number of GPUs for vLLM tensor parallelism")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--enforce_eager", action="store_true", default=False,
                   help="Disable CUDA graph capture (required for TP>=8)")
    p.add_argument("--graph_save_every", type=int, default=1000,
                   help="Save a DAG visualization PNG every N samples (0 to disable)")
    p.add_argument("--wandb_project", default="trace-ts")
    p.add_argument("--wandb_tags", nargs="*", default=None,
                   help="Extra W&B tags (dataset and 'gt-generation' always added)")
    args = p.parse_args()
    return vars(args)
