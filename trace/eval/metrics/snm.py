# =============================================================================
# snm_metric.py — Semantic Node Match (SNM) Evaluation
# =============================================================================
#
# OVERVIEW:
#   Post-hoc evaluation script that reads an existing evaluate JSON file
#   (produced by cross_attention_updated.py or evaluate_structured.py) and
#   computes the SNM metric suite over all samples that have non-empty
#   ground_truth_reasoning. Outputs a new JSON file mirroring the input
#   structure but containing SNM metrics instead of re-running inference.
#
# WHAT SNM MEASURES:
#   For each sample, GT and generated texts are parsed into DAGs. Nodes are
#   matched across three layers using a greedy sensor-gated bipartite match,
#   with an LLM judge scoring semantic equivalence of pattern/inference text.
#
#   Layer 1 — OBSERVATION (SNM-O):
#     Match GT obs → pred obs. Cost matrix built as:
#       sensor mismatch  → cost 1.0  (hard gate, no judge call needed)
#       sensor match     → LLM judge: YES=0.0, NO=1.0
#     Hungarian algorithm finds min-cost assignment.
#     Precision = matched_pred / total_pred
#     Recall    = matched_gt   / total_gt
#     F1        = harmonic mean
#     Provenance bonus (SNM-O-Temporal): fraction of matched pairs where
#     temporal field also agrees (rule-based, no judge needed).
#
#   Layer 2 — INFERENCE (SNM-I):
#     Match GT inferences → pred inferences by semantic equivalence of the
#     inference text (LLM judge, YES/NO).
#     After matching, SNM-IProv checks: for each matched inference pair,
#     does the predicted inference cite the same (obs-mapped) observation IDs
#     as the GT inference? Jaccard of mapped ID sets.
#
#   Layer 3 — SYNTHESIS (SNM-S):
#     Single node per sample — no matching needed. LLM judge rates semantic
#     agreement on a 0/1 scale. Only computed when both GT and pred have a
#     synthesis node.
#
# SENSOR CANONICALIZATION (added in this version):
#   Raw sensor strings emitted by the model and the GT may differ in
#   formatting (case, separators, 'acceleration' vs 'acc', 'left ankle' vs
#   'lankle', 'gyroscope' vs 'gyr', etc.) without representing different
#   signals. Before matching, every sensor field is run through a two-stage
#   canonicalization (sensor_canon.canonicalize_sensor) keyed to the dataset
#   chosen via --dataset. Pedantic differences collapse to the same canonical
#   name, while cross-axis (y vs z) and cross-modality (acc vs gyro vs mag)
#   remain HARD mismatches by design.
#
# JUDGE MODEL:
#   Qwen/Qwen3.5-35B-A3B via vLLM in-process engine (no API server).
#   Requests are batched (--judge_batch_size, default 256) to maximise
#   throughput on multi-H100 setups. All judge calls are YES/NO binary.
#   Justification is requested for audit but only the YES/NO token is used
#   for scoring.
#
# INPUT:
#   --eval_json   Path to existing evaluate JSON file (one or more). Must have
#                 top-level "per_sample" list where each entry has:
#                   sample_id, gt_activity, predicted_activity,
#                   ground_truth_reasoning, generated
#                 Samples with empty ground_truth_reasoning are skipped.
#   --dataset     Which canonical sensor set to use for normalization.
#                 Choices: ucihar, pamap2, uschad, capture24, opportunity,
#                          shoaib, mhealth.
#
# OUTPUT:
#   <eval_json_stem>_snm_<timestamp>.json with structure:
#     {
#       "source_eval_json": "<path>",
#       "config": { ... run config including dataset ... },
#       "snm_aggregate":           { ... },   # all scored samples (schema below)
#       "snm_aggregate_correct":   { ... },   # classifier-correct subset
#       "snm_aggregate_incorrect": { ... },   # classifier-incorrect subset
#       # aggregate schema:
#       #   "n_total": int,          total per_sample entries
#       #   "n_scored": int,         samples with non-empty GT reasoning
#       #   "n_skipped": int,        samples with empty GT reasoning
#       #   "n_gt_parse_fail": int,  GT reasoning failed to parse
#       #   "n_pred_parse_fail": int,predicted reasoning failed to parse
#       #   "SNM-OF1": float,        Observation F1
#       #   "SNM-OP": float,         Observation Precision
#       #   "SNM-OR": float,         Observation Recall
#       #   "SNM-O-Temporal": float, Temporal agreement on matched obs pairs
#       #   "SNM-O-Cover": float,    Fraction of GT sensor channels with ≥1 hit
#       #   "SNM-IF1": float,        Inference F1
#       #   "SNM-IP": float,
#       #   "SNM-IR": float,
#       #   "SNM-IProv-Abs": float,  Provenance: Jaccard of evidence sets
#       #   "SNM-IProv-Clean": float,Provenance: 0 if any cited obs hallucinated
#       #   "SNM-IProv-Penalised": float,
#       #   "SNM-SA": float,         Synthesis Agreement (0/1)
#       #   "n_judge_calls": int,    total LLM judge API calls (only in snm_aggregate)
#       "unresolved_sensors": [     # diagnostic
#         {"dataset": str, "raw": str, "count": int},
#         ...
#       ],
#       "per_sample": [
#         {
#           "sample_id": str,
#           "gt_activity": str,
#           "predicted_activity": str,
#           "snm_skipped": bool,     # True if GT reasoning empty / parse fail
#           "snm_skip_reason": str,  # "no_gt_reasoning" | "no_pred_text" |
#                                    # "gt_parse_fail" | "pred_parse_fail"
#           # Only present when snm_skipped=False:
#           ... (same fields as original) ...
#         },
#         ...
#       ]
#     }
#
# USAGE:
#   python snm_metric.py \
#       --eval_json /path/to/crossattn_..._evaluate_....json \
#       --dataset mhealth \
#       --judge_model Qwen/Qwen3.5-35B-A3B \
#       --judge_batch_size 256
#
# KEY DESIGN DECISIONS:
#   - Sensor gate before judge: if canonical(GT obs sensor) !=
#     canonical(pred obs sensor), cost=1.0 immediately. No judge call wasted
#     on cross-sensor pairs.
#   - Hungarian matching via scipy.optimize.linear_sum_assignment. Threshold
#     is 0.5: pairs with cost >= 0.5 (NO verdict or sensor mismatch) are not
#     counted as hits even if min-cost assignment.
#   - Inference matching is sensor-agnostic (judge scores the full text).
#     We do not gate on based_on IDs before judging — text is the primary signal.
#   - IProv uses the obs_id mapping from Layer 1.
#   - Synthesis: both GT and pred must have a synthesis node. If either is
#     missing, SNM-SA=NaN for that sample and it's excluded from corpus mean.
#   - All judge prompts disable thinking mode via apply_chat_template.
#   - Output JSON uses float("nan") serialised as null via custom encoder.
# =============================================================================

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# scipy required for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("[SNM] ERROR: scipy not installed. Run: pip install scipy --break-system-packages")
    sys.exit(1)

# vLLM — requires vllm and transformers (same env as teacher generation)
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    print("[SNM] ERROR: vllm/transformers not found. See requirements.txt.")
    sys.exit(1)

# Sensor canonicalization (drop sensor_canon.py in same directory)
try:
    from trace.data.sensor_canon import canonicalize_sensor, get_unresolved_log, list_datasets, clear_unresolved_log
except ImportError:
    print("[SNM] ERROR: sensor_canon.py not found. Place it in the same directory as snm_metric.py.")
    sys.exit(1)


# ===========================================================================
# 0. CONSTANTS & REGEX (same as evaluate_structured.py)
# ===========================================================================

_OBS_RE = re.compile(
    r'\[OBSERVATION\s*\|\s*id:\s*(O\d+)\]\s*\n'
    r'sensor:\s*(.+?)\n'
    r'temporal:\s*(.+?)\n'
    r'pattern:\s*(.+?)\n'
    r'confidence:\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_INF_RE = re.compile(
    r'\[INFERENCE\s*\|\s*id:\s*(I\d+)\]\s*\n'
    r'based_on:\s*(.+?)\n'
    r'inference:\s*(.+?)\n'
    r'confidence:\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)
_SYN_RE = re.compile(
    r'\[SYNTHESIS\]\s*\n'
    r'based_on:\s*(.+?)\n'
    r'([\s\S]+?)(?=\[ACTIVITY\]|\Z)',
    re.IGNORECASE,
)
_ACT_RE = re.compile(
    r'\[ACTIVITY\]\s*:\s*(.+?)$',
    re.IGNORECASE | re.MULTILINE,
)

# Canonical temporal vocabulary (for temporal agreement check)
TEMPORAL_VOCAB = {
    "early", "mid", "late",
    "early_to_mid", "mid_to_late", "full_window",
}


# ===========================================================================
# 1. DAG PARSING
# ===========================================================================

def parse_dag(text: str, dataset: str) -> dict:
    """
    Parse structured reasoning text → DAG dict.
    The sensor field is canonicalized against the chosen dataset's sensor set
    so pedantic differences (case, separators, 'acceleration' vs 'acc',
    'left ankle' vs 'lankle', etc.) collapse to the same canonical name.
    Cross-axis and cross-modality remain hard mismatches by design.

    Returns:
        observations: list of {id, sensor, temporal, pattern, confidence}
        inferences:   list of {id, based_on (list[str]), inference, confidence}
        synthesis:    {based_on (list[str]), text} or None
        activity:     str or None
        n_observations, n_inferences: int
        parse_failed: bool (True if 0 observations extracted)
    """
    text = text.strip() if text else ""
    if not text:
        return _empty_dag(parse_failed=True)

    observations = []
    for m in _OBS_RE.finditer(text):
        canon_sensor, _resolved = canonicalize_sensor(m.group(2), dataset)
        observations.append({
            "id":         m.group(1).strip(),
            "sensor":     canon_sensor,
            "temporal":   m.group(3).strip().lower(),
            "pattern":    m.group(4).strip(),
            "confidence": m.group(5).strip().lower(),
        })

    inferences = []
    for m in _INF_RE.finditer(text):
        based_on_ids = [x.strip() for x in m.group(2).strip().split(",")]
        inferences.append({
            "id":         m.group(1).strip(),
            "based_on":   based_on_ids,
            "inference":  m.group(3).strip(),
            "confidence": m.group(4).strip().lower(),
        })

    synthesis = None
    syn_m = _SYN_RE.search(text)
    if syn_m:
        syn_based_on = [x.strip() for x in syn_m.group(1).strip().split(",")]
        syn_text = re.sub(r"\n\s*\n", "\n", syn_m.group(2)).strip()
        synthesis = {"based_on": syn_based_on, "text": syn_text}

    activity = None
    act_m = _ACT_RE.search(text)
    if act_m:
        activity = act_m.group(1).strip().lower().replace("_", " ")

    return {
        "observations":   observations,
        "inferences":     inferences,
        "synthesis":      synthesis,
        "activity":       activity,
        "n_observations": len(observations),
        "n_inferences":   len(inferences),
        "parse_failed":   len(observations) == 0,
    }


def _empty_dag(parse_failed: bool = True) -> dict:
    return {
        "observations": [], "inferences": [], "synthesis": None,
        "activity": None, "n_observations": 0, "n_inferences": 0,
        "parse_failed": parse_failed,
    }


# ===========================================================================
# 2. LLM JUDGE CLIENT
# ===========================================================================

# Temporal window ordered scale for adjacency checks.
# Steps: early=0, early_to_mid=1, mid=2, mid_to_late=3, late=4
# 'throughout' overlaps everything → distance 0 vs any bucket.
_TEMPORAL_ORDER: dict = {
    "early": 0, "early_to_mid": 1, "mid": 2, "mid_to_late": 3, "late": 4,
}

def _temporal_distance(t1: str, t2: str) -> int:
    """Steps between two temporal window labels. 'throughout' ≡ distance 0."""
    if t1 == t2:
        return 0
    if t1 == "throughout" or t2 == "throughout":
        return 0
    i1 = _TEMPORAL_ORDER.get(t1 or "", -1)
    i2 = _TEMPORAL_ORDER.get(t2 or "", -1)
    if i1 == -1 or i2 == -1:
        return 999  # unknown label → treat as far apart
    return abs(i1 - i2)


class JudgeClient:
    """
    LLM-as-judge using local vLLM engine directly (same pattern as teacher
    generation scripts). No API server required — model loaded in-process.

    Judge prompt design:
      - Prompt built via apply_chat_template with enable_thinking=False
        (prevents Qwen3.5 thinking-mode tokens bleeding into output).
      - Requests YES or NO on first line + one sentence justification.
      - Deterministic: temperature=0.0.
      - Batched: all pending pairs for a matching step are sent together
        as a list to llm.generate() for maximum GPU utilisation.
    """

    # ── Strict prompts (default, current behaviour) ────────────────────────
    _SYS_OBS = (
        "You are a sensor signal pattern equivalence judge. "
        "Decide if two descriptions refer to the same signal phenomenon. "
        "Output YES or NO on the first line, then one sentence of justification. "
        "Do not use chain-of-thought or internal monologue."
    )
    _SYS_INF = (
        "You are a biomechanical inference equivalence judge. "
        "Decide if two inferences express the same biomechanical interpretation. "
        "Output YES or NO on the first line, then one sentence of justification. "
        "Do not use chain-of-thought or internal monologue."
    )
    _SYS_SYN = (
        "You are a sensor activity synthesis equivalence judge. "
        "Decide if two synthesis paragraphs reach the same activity conclusion. "
        "Output YES or NO on the first line, then one sentence of justification. "
        "Do not use chain-of-thought or internal monologue."
    )

    # ── Relaxed prompts (opt-in via --judge_relaxed) ───────────────────────
    # Obs: temporal removed from prompt (handled by _temporal_distance guard);
    #      judge focuses on signal DIRECTION only (rising/falling/stable/peaked).
    # Inf: focuses on the biomechanical STATE conclusion, not exact mechanism.
    _SYS_OBS_RELAXED = (
        "You are a sensor signal direction judge. "
        "Decide if two pattern descriptions agree on the DIRECTION of signal change: "
        "both rising, both falling, both stable, or both peaked/oscillating. "
        "Ignore differences in phrasing, magnitude, or level of detail. "
        "Output YES if directions agree, NO if they clearly disagree. "
        "First line must be YES or NO only, then one justification sentence."
    )
    _SYS_INF_RELAXED = (
        "You are a biomechanical state judge. "
        "Decide if two inferences agree on the STATE of the body: "
        "same posture, same direction of movement, or same muscle engagement. "
        "Ignore differences in phrasing or mechanistic detail. "
        "Output YES if body states agree, NO if they clearly disagree. "
        "First line must be YES or NO only, then one justification sentence."
    )

    def __init__(self, model_path: str, tensor_parallel_size: int = 4,
                 gpu_memory_utilization: float = 0.85,
                 relaxed: bool = False, temporal_tolerance: int = 1):
        print(f"[Judge] Loading vLLM engine: {model_path}")
        # ROOT CAUSE OF PRIOR CRASHES (documented, all fixed):
        #
        #   CRASH 1 — VL model class loaded instead of MoE text-only:
        #     Qwen3.5-35B-A3B config.json has model_type="qwen3_5" which this vLLM
        #     version routes to Qwen3_5ForConditionalGeneration (VL class).
        #     That class instantiates Qwen3_VisionTransformer which imports
        #     flash_attn.ops.triton.rotary from the SYSTEM flash_attn package at
        #     /usr/local/lib/python3.12/dist-packages/ — ABI mismatch with venv
        #     PyTorch → ImportError: undefined symbol.
        #     FIX: VLLM_FLASHINFER_PREFILL_BACKEND=triton env var (set in sbatch,
        #     same as 122B working script) makes vLLM use its internal triton kernel
        #     instead of flash_attn, so the VisionTransformer rotary path is never
        #     reached even if the wrong model class is loaded.
        #
        #   CRASH 2 — gdn_prefill_backend kwarg not in LLM():
        #     This vLLM version does not accept gdn_prefill_backend as an LLM()
        #     kwarg. The correct way is via VLLM_FLASHINFER_PREFILL_BACKEND=triton
        #     env var (exactly as the 122B sbatch does it). Removed from LLM().
        #
        #   CRASH 3 — override_architecture kwarg not in LLM():
        #     EngineArgs.__init__() does not have this param in this vLLM version.
        #     Removed. The env var fix above makes it unnecessary anyway.
        #
        #   Config now exactly mirrors qwen3_5-122B_A10B script LLM() call.
        #   The VLLM_FLASHINFER_PREFILL_BACKEND=triton env var is set in sbatch.
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            # enforce_eager=True: disables CUDA graph capture AND custom all-reduce.
            # Root cause of custom_all_reduce.cuh:455 'invalid argument':
            #   vLLM's custom all-reduce kernel allocates workspace proportional
            #   to num_gpu_blocks. With TP=4 and 35B-A3B MoE (16.5GB weights total,
            #   ~4GB per GPU), weights + KV cache profiling run returns 0 free blocks,
            #   making workspace size = 0 → invalid CUDA kernel argument.
            # enforce_eager=True skips both CUDA graphs and custom all-reduce entirely.
            enforce_eager=True,
            # max_model_len=2048: synthesis prompts for verbose datasets (opportunity,
            # pamap2) can reach 600-800 tokens; 512 caused EngineDeadError mid-run.
            max_model_len=2048,
            # disable_custom_all_reduce=True: belt+suspenders alongside enforce_eager.
            disable_custom_all_reduce=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=80,
            repetition_penalty=1.1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._n_calls = 0
        self.relaxed = relaxed
        # temporal_tolerance: max steps apart that still gets sent to the judge.
        # 0 = exact match only (strict default). 1 = adjacent buckets allowed
        # (early↔early_to_mid, mid↔mid_to_late). Pairs beyond this → auto-NO.
        self.temporal_tolerance = temporal_tolerance
        if relaxed:
            print(f"[Judge] Relaxed mode ON — temporal_tolerance={temporal_tolerance}, "
                  f"pattern-direction obs prompt, body-state inf prompt")
        print(f"[Judge] vLLM engine ready")

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _parse_verdict(self, response: str) -> str:
        if not response:
            return "ERROR"
        first = response.split("\n")[0].strip().upper()
        if first.startswith("YES"):
            return "YES"
        if first.startswith("NO"):
            return "NO"
        if "YES" in first:
            return "YES"
        if "NO" in first:
            return "NO"
        return "ERROR"

    def _run_batch(self, prompts: list) -> list:
        if not prompts:
            return []
        outputs = self.llm.generate(prompts, self.sampling_params)
        self._n_calls += len(prompts)
        return [self._parse_verdict(o.outputs[0].text) for o in outputs]

    def judge_observation_batch(self, pairs: list) -> list:
        prompts = []
        for sensor, tgt, pgt, tpd, ppd in pairs:
            user = (
                f"Sensor channel: {sensor}\n\n"
                f"Ground truth observation:\n"
                f"  temporal: {tgt}\n"
                f"  pattern: {pgt}\n\n"
                f"Predicted observation:\n"
                f"  temporal: {tpd}\n"
                f"  pattern: {ppd}\n\n"
                f"Do these describe the same signal phenomenon on this sensor?"
            )
            prompts.append(self._build_prompt(self._SYS_OBS, user))
        return self._run_batch(prompts)

    def judge_inference_batch(self, pairs: list) -> list:
        prompts = []
        for igt, ipd in pairs:
            user = (
                f"Ground truth inference:\n  {igt}\n\n"
                f"Predicted inference:\n  {ipd}\n\n"
                f"Do these express the same biomechanical interpretation?"
            )
            prompts.append(self._build_prompt(self._SYS_INF, user))
        return self._run_batch(prompts)

    def judge_synthesis(self, synthesis_gt: str, synthesis_pred: str) -> str:
        user = (
            f"Ground truth synthesis:\n{synthesis_gt}\n\n"
            f"Predicted synthesis:\n{synthesis_pred}\n\n"
            f"Do these reach the same conclusion about the activity?"
        )
        prompt = self._build_prompt(self._SYS_SYN, user)
        results = self._run_batch([prompt])
        return results[0] if results else "ERROR"

    # ── Prompt builders (no generate call) — used by batched evaluation loop ──

    def build_obs_prompts(self, pairs: list) -> list:
        """Build one prompt per (sensor, gt_temporal, gt_pattern, pred_temporal, pred_pattern) pair.

        Relaxed mode changes:
          - Pairs whose temporal windows exceed self.temporal_tolerance steps apart
            return None (auto-NO, no judge call saved).
          - Remaining pairs use _SYS_OBS_RELAXED: temporal removed from user
            message, judge evaluates signal direction only.
        Strict mode (default): unchanged behaviour, all pairs sent to judge.
        """
        prompts = []
        for sensor, tgt, pgt, tpd, ppd in pairs:
            if self.relaxed:
                dist = _temporal_distance(tgt or "", tpd or "")
                if dist > self.temporal_tolerance:
                    prompts.append(None)  # auto-NO
                    continue
                user = (
                    f"Sensor channel: {sensor}\n\n"
                    f"Ground truth pattern: {pgt}\n\n"
                    f"Predicted pattern:    {ppd}\n\n"
                    f"Do these describe the same direction of signal change?"
                )
                prompts.append(self._build_prompt(self._SYS_OBS_RELAXED, user))
            else:
                user = (
                    f"Sensor channel: {sensor}\n\n"
                    f"Ground truth observation:\n"
                    f"  temporal: {tgt}\n"
                    f"  pattern: {pgt}\n\n"
                    f"Predicted observation:\n"
                    f"  temporal: {tpd}\n"
                    f"  pattern: {ppd}\n\n"
                    f"Do these describe the same signal phenomenon on this sensor?"
                )
                prompts.append(self._build_prompt(self._SYS_OBS, user))
        return prompts

    def build_inf_prompts(self, pairs: list) -> list:
        sys_prompt = self._SYS_INF_RELAXED if self.relaxed else self._SYS_INF
        question = (
            "Do these agree on the state of the body?"
            if self.relaxed else
            "Do these express the same biomechanical interpretation?"
        )
        prompts = []
        for igt, ipd in pairs:
            user = (
                f"Ground truth inference:\n  {igt}\n\n"
                f"Predicted inference:\n  {ipd}\n\n"
                f"{question}"
            )
            prompts.append(self._build_prompt(sys_prompt, user))
        return prompts

    def build_syn_prompt(self, synthesis_gt: str, synthesis_pred: str) -> str:
        user = (
            f"Ground truth synthesis:\n{synthesis_gt}\n\n"
            f"Predicted synthesis:\n{synthesis_pred}\n\n"
            f"Do these reach the same conclusion about the activity?"
        )
        return self._build_prompt(self._SYS_SYN, user)

    def generate(self, prompts: list) -> list:
        # None entries are auto-NO (temporal distance exceeded tolerance).
        # Filter them out, generate on the rest, then stitch back.
        real_idx = [i for i, p in enumerate(prompts) if p is not None]
        real_prompts = [prompts[i] for i in real_idx]
        raw = self._run_batch(real_prompts) if real_prompts else []
        verdicts = ["NO"] * len(prompts)
        for i, v in zip(real_idx, raw):
            verdicts[i] = v
        return verdicts


# ===========================================================================
# 3. HUNGARIAN MATCHING
# ===========================================================================

def match_observations(
    gt_obs: list, pred_obs: list, judge: JudgeClient
) -> dict:
    """
    Bipartite matching of GT observations → predicted observations,
    performed per sensor channel (mini-Hungarian per channel).

    Judge pairs are collected only for same-sensor pairs, then a single
    batched judge call is made.  Within each sensor channel a separate
    Hungarian assignment is solved so cross-channel confusion is impossible.

    A pair is a "hit" only if the assignment cost < 0.5 (i.e. judge said YES).

    Returns SNM-OP, SNM-OR, SNM-OF1, SNM-O-Temporal, and SNM-O-Cover.
    """
    nan = float("nan")

    n_gt   = len(gt_obs)
    n_pred = len(pred_obs)

    if n_gt == 0 and n_pred == 0:
        return _obs_result_empty()

    if n_gt == 0:
        return _obs_result_empty(pred_hallucinations=n_pred)

    if n_pred == 0:
        return _obs_result_empty(gt_misses=n_gt)

    # Group by sensor channel, preserving original indices
    gt_by_sensor: dict   = {}
    pred_by_sensor: dict = {}

    for i, go in enumerate(gt_obs):
        if go["sensor"]:
            gt_by_sensor.setdefault(go["sensor"], []).append((i, go))

    for j, po in enumerate(pred_obs):
        if po["sensor"]:
            pred_by_sensor.setdefault(po["sensor"], []).append((j, po))

    # Collect same-sensor judge pairs (same gate logic as before)
    judge_pairs   = []
    judge_indices = []

    for sensor, gt_entries in gt_by_sensor.items():
        pred_entries = pred_by_sensor.get(sensor, [])
        for orig_i, go in gt_entries:
            for orig_j, po in pred_entries:
                if go["sensor"] and po["sensor"]:
                    judge_pairs.append((
                        go["sensor"],
                        go["temporal"], go["pattern"],
                        po["temporal"], po["pattern"],
                    ))
                    judge_indices.append((orig_i, orig_j))

    verdicts = []
    if judge_pairs:
        verdicts = judge.judge_observation_batch(judge_pairs)

    verdict_lookup: dict = {}
    for (orig_i, orig_j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[(orig_i, orig_j)] = verdict

    # Per-channel mini-Hungarian
    matches             = []
    gt_matched          = set()
    pred_matched        = set()
    hits                = 0
    temporal_hits       = 0
    n_temporal_check    = 0
    gt_channels_covered = 0

    for sensor, gt_entries in gt_by_sensor.items():
        pred_entries = pred_by_sensor.get(sensor, [])

        if not pred_entries:
            for _orig_i, go in gt_entries:
                matches.append({
                    "gt_id":          go["id"],
                    "pred_id":        None,
                    "sensor":         sensor,
                    "temporal_match": False,
                    "judge_verdict":  "NO_PRED_COVERAGE",
                    "hit":            False,
                    "cost":           1.0,
                })
            continue

        n_c_gt   = len(gt_entries)
        n_c_pred = len(pred_entries)
        cost_c   = np.ones((n_c_gt, n_c_pred), dtype=float)

        for local_i, (orig_i, _go) in enumerate(gt_entries):
            for local_j, (orig_j, _po) in enumerate(pred_entries):
                verdict = verdict_lookup.get((orig_i, orig_j))
                if verdict == "YES":
                    cost_c[local_i, local_j] = 0.0
                elif verdict == "NO":
                    cost_c[local_i, local_j] = 1.0
                elif verdict == "ERROR":
                    cost_c[local_i, local_j] = 0.5

        row_ind, col_ind = linear_sum_assignment(cost_c)

        channel_has_hit   = False
        assigned_gt_local = set(row_ind)

        for local_i, local_j in zip(row_ind, col_ind):
            c      = cost_c[local_i, local_j]
            is_hit = c < 0.5
            orig_i, go = gt_entries[local_i]
            orig_j, po = pred_entries[local_j]
            temporal_match = (go["temporal"] == po["temporal"])
            verdict = verdict_lookup.get((orig_i, orig_j), "UNMATCHED")

            matches.append({
                "gt_id":          go["id"],
                "pred_id":        po["id"],
                "sensor":         sensor,
                "temporal_match": temporal_match,
                "judge_verdict":  verdict,
                "hit":            is_hit,
                "cost":           float(c),
            })

            if is_hit:
                hits += 1
                gt_matched.add(orig_i)
                pred_matched.add(orig_j)
                channel_has_hit = True
                if temporal_match:
                    temporal_hits += 1
                n_temporal_check += 1

        for local_i, (orig_i, go) in enumerate(gt_entries):
            if local_i not in assigned_gt_local:
                matches.append({
                    "gt_id":          go["id"],
                    "pred_id":        None,
                    "sensor":         sensor,
                    "temporal_match": False,
                    "judge_verdict":  "UNMATCHED",
                    "hit":            False,
                    "cost":           1.0,
                })

        if channel_has_hit:
            gt_channels_covered += 1

    gt_unmatched   = [gt_obs[i]["id"]   for i in range(n_gt)   if i not in gt_matched]
    pred_unmatched = [pred_obs[j]["id"] for j in range(n_pred) if j not in pred_matched]

    precision = hits / n_pred if n_pred > 0 else nan
    recall    = hits / n_gt   if n_gt   > 0 else nan
    f1        = _f1(precision, recall)
    temporal_agreement = temporal_hits / n_temporal_check if n_temporal_check > 0 else nan

    n_gt_channels = len(gt_by_sensor)
    snm_o_cover   = gt_channels_covered / n_gt_channels if n_gt_channels > 0 else nan

    return {
        "matches":             matches,
        "hits":                hits,
        "gt_misses":           len(gt_unmatched),
        "pred_hallucinations": len(pred_unmatched),
        "gt_unmatched":        gt_unmatched,
        "pred_unmatched":      pred_unmatched,
        "SNM-OP":              precision,
        "SNM-OR":              recall,
        "SNM-OF1":             f1,
        "SNM-O-Temporal":      temporal_agreement,
        "SNM-O-Cover":         snm_o_cover,
    }


def _obs_result_empty(gt_misses: int = 0, pred_hallucinations: int = 0) -> dict:
    nan = float("nan")
    return {
        "matches": [], "hits": 0,
        "gt_misses": gt_misses, "pred_hallucinations": pred_hallucinations,
        "gt_unmatched": [], "pred_unmatched": [],
        "SNM-OP": nan, "SNM-OR": nan, "SNM-OF1": nan,
        "SNM-O-Temporal": nan, "SNM-O-Cover": nan,
    }


# ===========================================================================
# 3b. PREPARE / FINALISE HELPERS FOR BATCHED EVALUATION
# ===========================================================================

def _prepare_obs(gt_obs: list, pred_obs: list) -> tuple:
    """
    Phase 1 of obs matching: build judge pairs and pre-compute cost structure.
    Returns (judge_pairs, state) where state is passed to _finalise_obs().
    Sensor gate operates on canonicalized strings (set in parse_dag).
    """
    n_gt   = len(gt_obs)
    n_pred = len(pred_obs)

    if n_gt == 0 or n_pred == 0:
        return [], {"empty": True, "n_gt": n_gt, "n_pred": n_pred,
                    "gt_obs": gt_obs, "pred_obs": pred_obs}

    gt_by_sensor = {}
    for i, go in enumerate(gt_obs):
        if go["sensor"]:
            gt_by_sensor.setdefault(go["sensor"], []).append((i, go))

    pred_by_sensor = {}
    for j, po in enumerate(pred_obs):
        if po["sensor"]:
            pred_by_sensor.setdefault(po["sensor"], []).append((j, po))

    judge_pairs   = []
    judge_indices = []
    for i, go in enumerate(gt_obs):
        for j, po in enumerate(pred_obs):
            if go["sensor"] and po["sensor"] and go["sensor"] == po["sensor"]:
                judge_pairs.append((
                    go["sensor"],
                    go["temporal"], go["pattern"],
                    po["temporal"], po["pattern"],
                ))
                judge_indices.append((i, j))

    return judge_pairs, {
        "empty":          False,
        "n_gt":           n_gt,
        "n_pred":         n_pred,
        "gt_obs":         gt_obs,
        "pred_obs":       pred_obs,
        "gt_by_sensor":   gt_by_sensor,
        "pred_by_sensor": pred_by_sensor,
        "judge_indices":  judge_indices,
    }


def _finalise_obs(state: dict, verdicts: list) -> dict:
    nan = float("nan")

    if state.get("empty"):
        n_gt, n_pred = state["n_gt"], state["n_pred"]
        if n_gt == 0 and n_pred == 0:
            return _obs_result_empty()
        if n_gt == 0:
            return _obs_result_empty(pred_hallucinations=n_pred)
        return _obs_result_empty(gt_misses=n_gt)

    n_gt           = state["n_gt"]
    n_pred         = state["n_pred"]
    gt_obs         = state["gt_obs"]
    pred_obs       = state["pred_obs"]
    gt_by_sensor   = state["gt_by_sensor"]
    pred_by_sensor = state["pred_by_sensor"]
    judge_indices  = state["judge_indices"]

    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[(i, j)] = verdict

    matches             = []
    gt_matched          = set()
    pred_matched        = set()
    hits                = 0
    temporal_hits       = 0
    n_temporal_check    = 0
    gt_channels_covered = 0

    for sensor, gt_entries in gt_by_sensor.items():
        pred_entries = pred_by_sensor.get(sensor, [])

        if not pred_entries:
            for _orig_i, go in gt_entries:
                matches.append({
                    "gt_id":          go["id"],
                    "pred_id":        None,
                    "sensor":         sensor,
                    "temporal_match": False,
                    "judge_verdict":  "NO_PRED_COVERAGE",
                    "hit":            False,
                    "cost":           1.0,
                })
            continue

        n_c_gt   = len(gt_entries)
        n_c_pred = len(pred_entries)
        cost_c   = np.ones((n_c_gt, n_c_pred), dtype=float)

        for local_i, (orig_i, _go) in enumerate(gt_entries):
            for local_j, (orig_j, _po) in enumerate(pred_entries):
                verdict = verdict_lookup.get((orig_i, orig_j))
                if verdict == "YES":
                    cost_c[local_i, local_j] = 0.0
                elif verdict == "NO":
                    cost_c[local_i, local_j] = 1.0
                elif verdict == "ERROR":
                    cost_c[local_i, local_j] = 0.5

        row_ind, col_ind = linear_sum_assignment(cost_c)

        channel_has_hit   = False
        assigned_gt_local = set(row_ind)

        for local_i, local_j in zip(row_ind, col_ind):
            c      = cost_c[local_i, local_j]
            is_hit = c < 0.5
            orig_i, go = gt_entries[local_i]
            orig_j, po = pred_entries[local_j]
            temporal_match = (go["temporal"] == po["temporal"])
            verdict = verdict_lookup.get((orig_i, orig_j), "UNMATCHED")

            matches.append({
                "gt_id":          go["id"],
                "pred_id":        po["id"],
                "sensor":         sensor,
                "temporal_match": temporal_match,
                "judge_verdict":  verdict,
                "hit":            is_hit,
                "cost":           float(c),
            })

            if is_hit:
                hits += 1
                gt_matched.add(orig_i)
                pred_matched.add(orig_j)
                channel_has_hit = True
                if temporal_match:
                    temporal_hits += 1
                n_temporal_check += 1

        for local_i, (orig_i, go) in enumerate(gt_entries):
            if local_i not in assigned_gt_local:
                matches.append({
                    "gt_id":          go["id"],
                    "pred_id":        None,
                    "sensor":         sensor,
                    "temporal_match": False,
                    "judge_verdict":  "UNMATCHED",
                    "hit":            False,
                    "cost":           1.0,
                })

        if channel_has_hit:
            gt_channels_covered += 1

    gt_unmatched   = [gt_obs[i]["id"]   for i in range(n_gt)   if i not in gt_matched]
    pred_unmatched = [pred_obs[j]["id"] for j in range(n_pred) if j not in pred_matched]

    precision = hits / n_pred if n_pred > 0 else nan
    recall    = hits / n_gt   if n_gt   > 0 else nan
    f1        = _f1(precision, recall)
    temporal_agreement = temporal_hits / n_temporal_check if n_temporal_check > 0 else nan

    n_gt_channels = len(gt_by_sensor)
    snm_o_cover   = gt_channels_covered / n_gt_channels if n_gt_channels > 0 else nan

    return {
        "matches":             matches,
        "hits":                hits,
        "gt_misses":           len(gt_unmatched),
        "pred_hallucinations": len(pred_unmatched),
        "gt_unmatched":        gt_unmatched,
        "pred_unmatched":      pred_unmatched,
        "SNM-OP":              precision,
        "SNM-OR":              recall,
        "SNM-OF1":             f1,
        "SNM-O-Temporal":      temporal_agreement,
        "SNM-O-Cover":         snm_o_cover,
    }


def _collect_pred_obs_ids(pred_obs: list) -> set:
    return {obs["id"] for obs in pred_obs}


def _prepare_inf(gt_infs: list, pred_infs: list) -> tuple:
    n_gt   = len(gt_infs)
    n_pred = len(pred_infs)

    if n_gt == 0 or n_pred == 0:
        return [], {"empty": True, "n_gt": n_gt, "n_pred": n_pred,
                    "gt_infs": gt_infs, "pred_infs": pred_infs}

    judge_pairs   = []
    judge_indices = []
    for i, gi in enumerate(gt_infs):
        for j, pi in enumerate(pred_infs):
            judge_pairs.append((gi["inference"], pi["inference"]))
            judge_indices.append((i, j))

    return judge_pairs, {
        "empty":         False,
        "n_gt":          n_gt,
        "n_pred":        n_pred,
        "gt_infs":       gt_infs,
        "pred_infs":     pred_infs,
        "judge_indices": judge_indices,
    }


def _finalise_inf(state: dict, verdicts: list,
                  obs_id_map: dict, valid_pred_obs: set) -> dict:
    nan = float("nan")

    if state.get("empty"):
        n_gt, n_pred = state["n_gt"], state["n_pred"]
        if n_gt == 0 and n_pred == 0:
            return _inf_result_empty()
        if n_gt == 0:
            return _inf_result_empty(pred_hallucinations=n_pred)
        return _inf_result_empty(gt_misses=n_gt)

    n_gt          = state["n_gt"]
    n_pred        = state["n_pred"]
    gt_infs       = state["gt_infs"]
    pred_infs     = state["pred_infs"]
    judge_indices = state["judge_indices"]

    cost           = np.ones((n_gt, n_pred), dtype=float)
    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[(i, j)] = verdict
        if verdict == "YES":
            cost[i, j] = 0.0
        elif verdict == "NO":
            cost[i, j] = 1.0
        else:
            cost[i, j] = 0.5

    row_ind, col_ind = linear_sum_assignment(cost)

    matches            = []
    gt_matched         = set()
    pred_matched       = set()
    hits               = 0
    prov_abs_scores    = []
    prov_clean_scores  = []
    prov_pen_scores    = []

    for i, j in zip(row_ind, col_ind):
        c      = cost[i, j]
        is_hit = c < 0.5
        gi = gt_infs[i]
        pi = pred_infs[j]

        gt_based   = set(gi["based_on"])
        pred_based = set(pi["based_on"])

        mapped_gt   = {obs_id_map[x] for x in gt_based if x in obs_id_map}
        union       = mapped_gt | pred_based
        intersect   = mapped_gt & pred_based
        jaccard_abs = len(intersect) / len(union) if union else nan

        hallucinated_basis = pred_based - valid_pred_obs
        n_hallucinated     = len(hallucinated_basis)
        jaccard_clean      = 0.0 if n_hallucinated > 0 else jaccard_abs

        valid_pred_cited  = pred_based & valid_pred_obs
        n_int_clean       = len(mapped_gt & valid_pred_cited)
        n_union_clean     = len(mapped_gt | valid_pred_cited)
        base_j            = n_int_clean / n_union_clean if n_union_clean > 0 else nan
        if not math.isnan(base_j):
            penalty     = n_hallucinated / len(pred_based) if pred_based else 0.0
            jaccard_pen = max(-1.0, base_j - penalty)
        else:
            jaccard_pen = nan

        matches.append({
            "gt_id":                gi["id"],
            "pred_id":              pi["id"],
            "judge_verdict":        verdict_lookup.get((i, j), "ERROR"),
            "hit":                  is_hit,
            "cost":                 float(c),
            "gt_based_on":          list(gt_based),
            "pred_based_on":        list(pred_based),
            "mapped_gt_obs":        list(mapped_gt),
            "hallucinated_basis":   list(hallucinated_basis),
            "n_hallucinated_basis": n_hallucinated,
            "iprov_abs":            jaccard_abs if not math.isnan(jaccard_abs) else None,
            "iprov_clean":          jaccard_clean if not math.isnan(jaccard_clean) else None,
            "iprov_penalised":      jaccard_pen if not math.isnan(jaccard_pen) else None,
        })

        if is_hit:
            hits += 1
            gt_matched.add(i)
            pred_matched.add(j)
            if not math.isnan(jaccard_abs):    prov_abs_scores.append(jaccard_abs)
            if not math.isnan(jaccard_clean):  prov_clean_scores.append(jaccard_clean)
            if not math.isnan(jaccard_pen):    prov_pen_scores.append(jaccard_pen)

    gt_unmatched   = [gt_infs[i]["id"]   for i in range(n_gt)   if i not in gt_matched]
    pred_unmatched = [pred_infs[j]["id"] for j in range(n_pred) if j not in pred_matched]

    precision   = hits / n_pred if n_pred > 0 else nan
    recall      = hits / n_gt   if n_gt   > 0 else nan
    f1          = _f1(precision, recall)
    iprov_abs   = float(np.mean(prov_abs_scores))   if prov_abs_scores   else nan
    iprov_clean = float(np.mean(prov_clean_scores)) if prov_clean_scores else nan
    iprov_pen   = float(np.mean(prov_pen_scores))   if prov_pen_scores   else nan

    return {
        "matches":              matches,
        "hits":                 hits,
        "gt_misses":            len(gt_unmatched),
        "pred_hallucinations":  len(pred_unmatched),
        "gt_unmatched":         gt_unmatched,
        "pred_unmatched":       pred_unmatched,
        "SNM-IP":               precision,
        "SNM-IR":               recall,
        "SNM-IF1":              f1,
        "SNM-IProv-Abs":        iprov_abs,
        "SNM-IProv-Clean":      iprov_clean,
        "SNM-IProv-Penalised":  iprov_pen,
    }


def match_inferences(
    gt_infs: list, pred_infs: list,
    obs_id_map: dict,
    judge: JudgeClient,
    all_pred_obs_ids: set,
) -> dict:
    """
    Bipartite matching of GT inferences → predicted inferences.
    See _finalise_inf for provenance variant computation details.
    """
    n_gt   = len(gt_infs)
    n_pred = len(pred_infs)
    nan    = float("nan")

    if n_gt == 0 and n_pred == 0:
        return _inf_result_empty()
    if n_gt == 0:
        return _inf_result_empty(pred_hallucinations=n_pred)
    if n_pred == 0:
        return _inf_result_empty(gt_misses=n_gt)

    valid_pred_obs = all_pred_obs_ids

    judge_pairs   = []
    judge_indices = []
    for i, gi in enumerate(gt_infs):
        for j, pi in enumerate(pred_infs):
            judge_pairs.append((gi["inference"], pi["inference"]))
            judge_indices.append((i, j))

    verdicts = judge.judge_inference_batch(judge_pairs)

    cost = np.ones((n_gt, n_pred), dtype=float)
    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[(i, j)] = verdict
        if verdict == "YES":
            cost[i, j] = 0.0
        elif verdict == "NO":
            cost[i, j] = 1.0
        else:
            cost[i, j] = 0.5

    row_ind, col_ind = linear_sum_assignment(cost)

    matches           = []
    gt_matched        = set()
    pred_matched      = set()
    hits              = 0
    prov_abs_scores   = []
    prov_clean_scores = []
    prov_pen_scores   = []

    for i, j in zip(row_ind, col_ind):
        c      = cost[i, j]
        is_hit = c < 0.5

        gi = gt_infs[i]
        pi = pred_infs[j]

        gt_based   = set(gi["based_on"])
        pred_based = set(pi["based_on"])

        mapped_gt  = {obs_id_map[x] for x in gt_based if x in obs_id_map}
        union      = mapped_gt | pred_based
        intersect  = mapped_gt & pred_based
        jaccard_abs = len(intersect) / len(union) if union else nan

        hallucinated_basis = pred_based - valid_pred_obs
        n_hallucinated     = len(hallucinated_basis)
        if n_hallucinated > 0:
            jaccard_clean = 0.0
        else:
            jaccard_clean = jaccard_abs

        valid_pred_cited = pred_based & valid_pred_obs
        n_intersect_clean = len(mapped_gt & valid_pred_cited)
        n_union_clean     = len(mapped_gt | valid_pred_cited)
        base_j            = n_intersect_clean / n_union_clean if n_union_clean > 0 else nan
        if not math.isnan(base_j):
            penalty        = n_hallucinated / len(pred_based) if pred_based else 0.0
            jaccard_pen    = max(-1.0, base_j - penalty)
        else:
            jaccard_pen    = nan

        matches.append({
            "gt_id":                gi["id"],
            "pred_id":              pi["id"],
            "judge_verdict":        verdict_lookup.get((i, j), "ERROR"),
            "hit":                  is_hit,
            "cost":                 float(c),
            "gt_based_on":          list(gt_based),
            "pred_based_on":        list(pred_based),
            "mapped_gt_obs":        list(mapped_gt),
            "hallucinated_basis":   list(hallucinated_basis),
            "n_hallucinated_basis": n_hallucinated,
            "iprov_abs":            jaccard_abs if not math.isnan(jaccard_abs) else None,
            "iprov_clean":          jaccard_clean if not math.isnan(jaccard_clean) else None,
            "iprov_penalised":      jaccard_pen if not math.isnan(jaccard_pen) else None,
        })

        if is_hit:
            hits += 1
            gt_matched.add(i)
            pred_matched.add(j)
            if not math.isnan(jaccard_abs):
                prov_abs_scores.append(jaccard_abs)
            if not math.isnan(jaccard_clean):
                prov_clean_scores.append(jaccard_clean)
            if not math.isnan(jaccard_pen):
                prov_pen_scores.append(jaccard_pen)

    gt_unmatched   = [gt_infs[i]["id"]   for i in range(n_gt)   if i not in gt_matched]
    pred_unmatched = [pred_infs[j]["id"] for j in range(n_pred) if j not in pred_matched]

    precision   = hits / n_pred if n_pred > 0 else nan
    recall      = hits / n_gt   if n_gt   > 0 else nan
    f1          = _f1(precision, recall)
    iprov_abs   = float(np.mean(prov_abs_scores))   if prov_abs_scores   else nan
    iprov_clean = float(np.mean(prov_clean_scores)) if prov_clean_scores else nan
    iprov_pen   = float(np.mean(prov_pen_scores))   if prov_pen_scores   else nan

    return {
        "matches":              matches,
        "hits":                 hits,
        "gt_misses":            len(gt_unmatched),
        "pred_hallucinations":  len(pred_unmatched),
        "gt_unmatched":         gt_unmatched,
        "pred_unmatched":       pred_unmatched,
        "SNM-IP":               precision,
        "SNM-IR":               recall,
        "SNM-IF1":              f1,
        "SNM-IProv-Abs":        iprov_abs,
        "SNM-IProv-Clean":      iprov_clean,
        "SNM-IProv-Penalised":  iprov_pen,
    }


def _inf_result_empty(gt_misses: int = 0, pred_hallucinations: int = 0) -> dict:
    nan = float("nan")
    return {
        "matches": [], "hits": 0,
        "gt_misses": gt_misses, "pred_hallucinations": pred_hallucinations,
        "gt_unmatched": [], "pred_unmatched": [],
        "SNM-IP": nan, "SNM-IR": nan, "SNM-IF1": nan,
        "SNM-IProv-Abs": nan, "SNM-IProv-Clean": nan, "SNM-IProv-Penalised": nan,
    }


# ===========================================================================
# 4. PER-SAMPLE SNM (legacy, kept for callers that import it)
# ===========================================================================

def compute_snm_sample(
    gt_text: str, pred_text: str, judge: JudgeClient, dataset: str
) -> dict:
    """
    Compute all SNM sub-metrics for a single sample. Legacy non-batched path —
    the batched path in run_snm_single bypasses this for performance, but this
    function is preserved for any external caller that imports it.
    """
    nan = float("nan")

    gt_dag   = parse_dag(gt_text, dataset)
    pred_dag = parse_dag(pred_text, dataset)

    if gt_dag["parse_failed"]:
        return {"snm_skipped": True, "snm_skip_reason": "gt_parse_fail"}
    if pred_dag["parse_failed"]:
        return {"snm_skipped": True, "snm_skip_reason": "pred_parse_fail"}

    obs_result = match_observations(
        gt_dag["observations"], pred_dag["observations"], judge
    )

    obs_id_map = {}
    for m in obs_result["matches"]:
        if m["hit"]:
            obs_id_map[m["gt_id"]] = m["pred_id"]

    all_pred_obs_ids = {obs["id"] for obs in pred_dag["observations"]}
    inf_result = match_inferences(
        gt_dag["inferences"], pred_dag["inferences"], obs_id_map, judge,
        all_pred_obs_ids,
    )

    gt_syn   = gt_dag["synthesis"]
    pred_syn = pred_dag["synthesis"]

    if gt_syn is not None and pred_syn is not None:
        syn_verdict = judge.judge_synthesis(gt_syn["text"], pred_syn["text"])
        snm_sa = 1.0 if syn_verdict == "YES" else (0.0 if syn_verdict == "NO" else nan)
    else:
        syn_verdict = "N/A"
        snm_sa = nan

    return {
        "snm_skipped":    False,
        "n_gt_obs":       gt_dag["n_observations"],
        "n_pred_obs":     pred_dag["n_observations"],
        "n_gt_inf":       gt_dag["n_inferences"],
        "n_pred_inf":     pred_dag["n_inferences"],
        "obs_matches":           obs_result["matches"],
        "obs_hits":              obs_result["hits"],
        "obs_misses":            obs_result["gt_misses"],
        "obs_hallucinations":    obs_result["pred_hallucinations"],
        "SNM-OP":                obs_result["SNM-OP"],
        "SNM-OR":                obs_result["SNM-OR"],
        "SNM-OF1":               obs_result["SNM-OF1"],
        "SNM-O-Temporal":        obs_result["SNM-O-Temporal"],
        "SNM-O-Cover":           obs_result["SNM-O-Cover"],
        "inf_matches":           inf_result["matches"],
        "inf_hits":              inf_result["hits"],
        "inf_misses":            inf_result["gt_misses"],
        "inf_hallucinations":    inf_result["pred_hallucinations"],
        "SNM-IP":                inf_result["SNM-IP"],
        "SNM-IR":                inf_result["SNM-IR"],
        "SNM-IF1":               inf_result["SNM-IF1"],
        "SNM-IProv-Abs":         inf_result["SNM-IProv-Abs"],
        "SNM-IProv-Clean":       inf_result["SNM-IProv-Clean"],
        "SNM-IProv-Penalised":   inf_result["SNM-IProv-Penalised"],
        "synthesis_judge":       syn_verdict,
        "SNM-SA":                snm_sa,
    }


# ===========================================================================
# 5. AGGREGATE
# ===========================================================================

def aggregate_snm(per_sample_results: list) -> dict:
    nan = float("nan")

    keys = [
        "SNM-OP", "SNM-OR", "SNM-OF1", "SNM-O-Temporal", "SNM-O-Cover",
        "SNM-IP", "SNM-IR", "SNM-IF1",
        "SNM-IProv-Abs", "SNM-IProv-Clean", "SNM-IProv-Penalised",
        "SNM-SA",
    ]

    accum = {k: [] for k in keys}

    n_total      = len(per_sample_results)
    n_scored     = 0
    n_skipped    = 0
    n_gt_pf      = 0
    n_pred_pf    = 0

    for r in per_sample_results:
        if r.get("snm_skipped", True):
            n_skipped += 1
            reason = r.get("snm_skip_reason", "")
            if reason == "gt_parse_fail":
                n_gt_pf += 1
            elif reason == "pred_parse_fail":
                n_pred_pf += 1
            continue

        n_scored += 1
        for k in keys:
            v = r.get(k, nan)
            if isinstance(v, float) and not math.isnan(v):
                accum[k].append(v)

    agg = {}
    for k in keys:
        vals = accum[k]
        agg[k] = float(np.mean(vals)) if vals else nan

    return {
        "n_total":          n_total,
        "n_scored":         n_scored,
        "n_skipped":        n_skipped,
        "n_gt_parse_fail":  n_gt_pf,
        "n_pred_parse_fail": n_pred_pf,
        **agg,
    }


# ===========================================================================
# 6. HELPERS
# ===========================================================================

def _f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


class _NaNEncoder(json.JSONEncoder):
    """Serialise float nan/inf as null in JSON by pre-processing the object tree."""

    def encode(self, obj):
        return super().encode(_replace_nan(obj))

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(_replace_nan(obj), _one_shot=_one_shot)


def _replace_nan(obj):
    """Recursively replace float nan/inf with None, and numpy scalars with
    native Python types, for JSON serialisation."""
    import numpy as np
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_replace_nan(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan(v) for v in obj]
    return obj


# ===========================================================================
# 7. MAIN EVALUATION LOOP
# ===========================================================================

def run_snm_single(judge: "JudgeClient", cfg: dict, eval_path: Path) -> dict:
    """Run SNM evaluation for a single eval JSON file using a pre-loaded judge."""
    if not eval_path.exists():
        print(f"[SNM] ERROR: eval_json not found: {eval_path}")
        return {}

    dataset = cfg["dataset"]
    # Per-file unresolved log: clear before each file so the diagnostic in the
    # output JSON reflects only that file's sensors, not accumulated history.
    clear_unresolved_log()

    print(f"[SNM] Loading eval file: {eval_path}")
    print(f"[SNM] Dataset for sensor canonicalization: {dataset}")
    with open(eval_path) as f:
        eval_data = json.load(f)

    if isinstance(eval_data, list):
        per_sample_raw = eval_data
        source_config  = {}
    elif isinstance(eval_data, dict):
        per_sample_raw = eval_data.get("per_sample", [])
        source_config  = eval_data.get("config", {})
    else:
        print("[SNM] ERROR: Unexpected eval JSON format (not list or dict).")
        return {}

    print(f"[SNM] Found {len(per_sample_raw)} samples in eval file.")

    # ── Pre-classification: only samples with non-empty GT AND pred text ──
    skipped_out       = []
    scored_candidates = []

    for k, raw in enumerate(per_sample_raw):
        sid           = raw.get("sample_id", str(k))
        gt_activity   = raw.get("gt_activity", "")
        pred_activity = raw.get("predicted_activity", "")
        gt_text       = raw.get("ground_truth_reasoning", "").strip()
        pred_text     = raw.get("generated", "").strip()

        entry = {
            "sample_id":          sid,
            "gt_activity":        gt_activity,
            "predicted_activity": pred_activity,
        }

        skip_reason = None
        gt_dag = pred_dag = None
        if not gt_text:
            skip_reason = "no_gt_reasoning"
        elif not pred_text:
            skip_reason = "no_pred_text"
        else:
            gt_dag   = parse_dag(gt_text, dataset)
            pred_dag = parse_dag(pred_text, dataset)
            if gt_dag["parse_failed"]:
                skip_reason = "gt_parse_fail"
            elif pred_dag["parse_failed"]:
                skip_reason = "pred_parse_fail"

        if skip_reason:
            entry["snm_skipped"]     = True
            entry["snm_skip_reason"] = skip_reason
            skipped_out.append((k, entry))
        else:
            scored_candidates.append((k, raw, gt_dag, pred_dag, entry))

    n_total      = len(per_sample_raw)
    n_with_gt    = sum(1 for r in per_sample_raw if r.get("ground_truth_reasoning", "").strip())
    n_scoreable  = len(scored_candidates)
    n_skipped    = len(skipped_out)
    print(f"[SNM] Samples with non-empty ground_truth_reasoning: {n_with_gt}")
    print(f"[SNM] Scoreable (GT + pred + parse OK): {n_scoreable} / {n_total}")
    print(f"[SNM] Pre-skipped (no GT / no pred / parse fail): {n_skipped}")

    if n_scoreable == 0:
        print("[SNM] WARNING: No scoreable samples. Nothing to judge.")

    batch_size = cfg.get("judge_batch_size", 256)
    print(f"[SNM] Batch size: {batch_size} scoreable samples per generate() call")

    # ── Batched 3-pass evaluation loop ────────────────────────────────────
    scored_results = [None] * n_scoreable
    t0 = time.time()

    # Checkpoint: resume partial runs without reprocessing completed batches.
    _ckpt_dir = eval_path.parent / "evaluate_snm_new"
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _ckpt_dir / f"{eval_path.stem}_snm_ckpt.json"
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as _f:
                _ckpt = json.load(_f)
            if _ckpt.get("n_scoreable") == n_scoreable:
                for _i, _r in enumerate(_ckpt.get("results", [])):
                    if _r is not None and _i < n_scoreable:
                        scored_results[_i] = _r
                _n_resumed = sum(1 for r in scored_results if r is not None)
                print(f"[SNM] Checkpoint loaded: {_n_resumed}/{n_scoreable} samples already done")
            else:
                print(f"[SNM] Checkpoint n_scoreable mismatch "
                      f"({_ckpt.get('n_scoreable')} vs {n_scoreable}) — ignoring")
        except Exception as _e:
            print(f"[SNM] Could not load checkpoint: {_e}")

    for batch_start in range(0, n_scoreable, batch_size):
        batch_cands = scored_candidates[batch_start : batch_start + batch_size]
        batch_end   = batch_start + len(batch_cands)

        if all(scored_results[i] is not None for i in range(batch_start, batch_end)):
            print(f"[SNM] Batch {batch_start}–{batch_end-1} already done (checkpoint). Skipping.")
            continue

        print(f"[SNM] Batch {batch_start}–{batch_end-1} of {n_scoreable} scoreable "
              f"| judge_calls={judge.n_calls} | elapsed={time.time()-t0:.0f}s")

        gt_dags   = [c[2] for c in batch_cands]
        pred_dags = [c[3] for c in batch_cands]
        entries   = [c[4] for c in batch_cands]
        n_scored_batch = len(batch_cands)
        scored_idx = list(range(n_scored_batch))

        if n_scored_batch == 0:
            continue

        # ── PASS 1: Observations ──────────────────────────────────────────
        obs_states         = []
        obs_prompt_offsets = []
        obs_prompts_flat   = []

        for s in range(n_scored_batch):
            pairs, state = _prepare_obs(
                gt_dags[s]["observations"], pred_dags[s]["observations"]
            )
            prompts = judge.build_obs_prompts(pairs)
            start   = len(obs_prompts_flat)
            obs_prompts_flat.extend(prompts)
            obs_prompt_offsets.append((start, start + len(prompts)))
            obs_states.append(state)

        obs_verdicts_flat = judge.generate(obs_prompts_flat) if obs_prompts_flat else []

        obs_results  = []
        obs_id_maps  = []

        for s in range(n_scored_batch):
            start, end = obs_prompt_offsets[s]
            verdicts_s  = obs_verdicts_flat[start:end]
            obs_r       = _finalise_obs(obs_states[s], verdicts_s)
            obs_results.append(obs_r)
            obs_id_map = {
                m["gt_id"]: m["pred_id"]
                for m in obs_r["matches"] if m["hit"]
            }
            obs_id_maps.append(obs_id_map)

        # ── PASS 2: Inferences ────────────────────────────────────────────
        inf_states          = []
        inf_prompt_offsets  = []
        inf_prompts_flat    = []
        valid_pred_obs_list = []

        for s in range(n_scored_batch):
            valid_pred_obs = {obs["id"] for obs in pred_dags[s]["observations"]}
            valid_pred_obs_list.append(valid_pred_obs)
            pairs, state = _prepare_inf(
                gt_dags[s]["inferences"], pred_dags[s]["inferences"]
            )
            prompts = judge.build_inf_prompts(pairs)
            start   = len(inf_prompts_flat)
            inf_prompts_flat.extend(prompts)
            inf_prompt_offsets.append((start, start + len(prompts)))
            inf_states.append(state)

        inf_verdicts_flat = judge.generate(inf_prompts_flat) if inf_prompts_flat else []

        inf_results = []
        for s in range(n_scored_batch):
            start, end = inf_prompt_offsets[s]
            verdicts_s  = inf_verdicts_flat[start:end]
            inf_r       = _finalise_inf(
                inf_states[s], verdicts_s,
                obs_id_maps[s], valid_pred_obs_list[s]
            )
            inf_results.append(inf_r)

        # ── PASS 3: Synthesis ─────────────────────────────────────────────
        syn_prompts_flat   = []
        syn_prompt_indices = []

        for s in range(n_scored_batch):
            gt_syn   = gt_dags[s]["synthesis"]
            pred_syn = pred_dags[s]["synthesis"]
            if gt_syn is not None and pred_syn is not None:
                prompt = judge.build_syn_prompt(gt_syn["text"], pred_syn["text"])
                syn_prompt_indices.append((s, len(syn_prompts_flat)))
                syn_prompts_flat.append(prompt)

        syn_verdicts_flat = judge.generate(syn_prompts_flat) if syn_prompts_flat else []

        syn_verdict_by_s = {}
        for s, idx in syn_prompt_indices:
            syn_verdict_by_s[s] = syn_verdicts_flat[idx]

        # ── Assemble per-sample results ───────────────────────────────────
        nan = float("nan")
        for s, ei in enumerate(scored_idx):
            obs_r     = obs_results[s]
            inf_r     = inf_results[s]
            syn_v     = syn_verdict_by_s.get(s, "N/A")
            snm_sa    = (1.0 if syn_v == "YES" else
                         0.0 if syn_v == "NO"  else nan)

            entries[ei].update({
                "snm_skipped":    False,
                "n_gt_obs":       gt_dags[s]["n_observations"],
                "n_pred_obs":     pred_dags[s]["n_observations"],
                "n_gt_inf":       gt_dags[s]["n_inferences"],
                "n_pred_inf":     pred_dags[s]["n_inferences"],
                "obs_matches":           obs_r["matches"],
                "obs_hits":              obs_r["hits"],
                "obs_misses":            obs_r["gt_misses"],
                "obs_hallucinations":    obs_r["pred_hallucinations"],
                "SNM-OP":                obs_r["SNM-OP"],
                "SNM-OR":                obs_r["SNM-OR"],
                "SNM-OF1":               obs_r["SNM-OF1"],
                "SNM-O-Temporal":        obs_r["SNM-O-Temporal"],
                "SNM-O-Cover":           obs_r["SNM-O-Cover"],
                "inf_matches":           inf_r["matches"],
                "inf_hits":              inf_r["hits"],
                "inf_misses":            inf_r["gt_misses"],
                "inf_hallucinations":    inf_r["pred_hallucinations"],
                "SNM-IP":                inf_r["SNM-IP"],
                "SNM-IR":                inf_r["SNM-IR"],
                "SNM-IF1":               inf_r["SNM-IF1"],
                "SNM-IProv-Abs":         inf_r["SNM-IProv-Abs"],
                "SNM-IProv-Clean":       inf_r["SNM-IProv-Clean"],
                "SNM-IProv-Penalised":   inf_r["SNM-IProv-Penalised"],
                "synthesis_judge":       syn_v,
                "SNM-SA":                snm_sa,
            })

        for s in range(n_scored_batch):
            scored_results[batch_start + s] = entries[s]

        # Save checkpoint after every batch so a crash can resume mid-dataset.
        with open(ckpt_path, "w") as _f:
            json.dump({"n_scoreable": n_scoreable, "results": scored_results},
                      _f, cls=_NaNEncoder)

    # ── Reassemble per_sample_out in original index order ────────────────
    skipped_map = {orig_k: entry for orig_k, entry in skipped_out}
    scored_map  = {scored_candidates[i][0]: scored_results[i]
                   for i in range(n_scoreable) if scored_results[i] is not None}
    per_sample_out = []
    for k in range(n_total):
        if k in skipped_map:
            per_sample_out.append(skipped_map[k])
        elif k in scored_map:
            per_sample_out.append(scored_map[k])

    # ── Aggregate: overall + correct/incorrect strata ────────────────────
    agg = aggregate_snm(per_sample_out)
    agg["n_judge_calls"] = judge.n_calls

    # Stratify by classifier correctness (gt_activity == predicted_activity).
    # Only non-skipped samples enter each stratum so the split mirrors what
    # aggregate_snm already scores.
    correct_slice   = [r for r in per_sample_out
                       if not r.get("snm_skipped", True)
                       and r.get("gt_activity") == r.get("predicted_activity")]
    incorrect_slice = [r for r in per_sample_out
                       if not r.get("snm_skipped", True)
                       and r.get("gt_activity") != r.get("predicted_activity")]
    agg_correct   = aggregate_snm(correct_slice)
    agg_incorrect = aggregate_snm(incorrect_slice)

    elapsed_total = time.time() - t0
    print(f"\n[SNM] Done. {judge.n_calls} judge calls in {elapsed_total:.1f}s")
    print(f"[SNM] Scored {agg['n_scored']} / {agg['n_total']} samples "
          f"(correct: {len(correct_slice)}, incorrect: {len(incorrect_slice)})")

    _SNM_KEYS = [
        "SNM-OF1", "SNM-OP", "SNM-OR", "SNM-O-Temporal", "SNM-O-Cover",
        "SNM-IF1", "SNM-IP", "SNM-IR",
        "SNM-IProv-Abs", "SNM-IProv-Clean", "SNM-IProv-Penalised",
        "SNM-SA",
    ]

    def _print_agg(label, a):
        print(f"\n{'='*55}")
        print(f"  {label}  (n={a.get('n_scored', 0)})")
        print(f"{'='*55}")
        for k in _SNM_KEYS:
            v = a.get(k, float("nan"))
            vs = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "N/A"
            print(f"  {k:<20s}: {vs}")
        print(f"{'='*55}")

    _print_agg("SNM AGGREGATE — ALL",       agg)
    _print_agg("SNM AGGREGATE — CORRECT",   agg_correct)
    _print_agg("SNM AGGREGATE — INCORRECT", agg_incorrect)
    print()

    # ── Unresolved sensor diagnostic ─────────────────────────────────────
    unresolved = get_unresolved_log()
    if unresolved:
        print(f"[SNM] Unresolved sensor strings (top 30 of {len(unresolved)}):")
        for (ds, raw), n in sorted(unresolved.items(), key=lambda x: -x[1])[:30]:
            print(f"  [{ds}] {n:5d}  {raw!r}")
    else:
        print(f"[SNM] All sensor strings resolved to canonical names.")

    unresolved_list = [
        {"dataset": ds, "raw": raw, "count": n}
        for (ds, raw), n in sorted(unresolved.items(), key=lambda x: -x[1])
    ]

    # ── Save output ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir  = eval_path.parent / "evaluate_snm_new"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{eval_path.stem}_snm_{ts}.json"
    out_path = out_dir / out_name

    output = {
        "source_eval_json": str(eval_path),
        "config": {
            "dataset":                cfg["dataset"],
            "judge_model":            cfg["judge_model"],
            "tensor_parallel_size":   cfg.get("tensor_parallel_size", 4),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.85),
            "judge_batch_size":       cfg.get("judge_batch_size", 256),
            "timestamp":              ts,
            "source_config":          source_config,
        },
        "snm_aggregate":           agg,
        "snm_aggregate_correct":   agg_correct,
        "snm_aggregate_incorrect": agg_incorrect,
        "unresolved_sensors":      unresolved_list,
        "per_sample":              per_sample_out,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NaNEncoder)

    print(f"[SNM] Saved → {out_path}")
    ckpt_path.unlink(missing_ok=True)
    return output


def run_snm(cfg: dict):
    """Legacy single-file entry point. Loads judge, runs one file, exits."""
    judge = JudgeClient(
        model_path=cfg["judge_model"],
        tensor_parallel_size=cfg.get("tensor_parallel_size", 4),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
    )
    print(f"[SNM] Judge loaded: {cfg['judge_model']}")
    eval_path = Path(cfg["eval_json"])
    run_snm_single(judge, cfg, eval_path)


# ===========================================================================
# 8. CLI
# ===========================================================================

def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="TRACE-TS — Semantic Node Match (SNM) Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--eval_json", required=True, nargs="+", dest="eval_jsons",
        help="Path(s) to evaluate JSON file(s). Pass multiple times or space-separated. "
             "The judge (vLLM engine) is loaded ONCE and reused across all files."
    )
    p.add_argument(
        "--dataset", required=True, nargs="+",
        help="Dataset name(s) for sensor canonicalization. Pass one value to "
             "apply to all --eval_json files, or one value per file (must match "
             "count). Valid values: " + ", ".join(list_datasets())
    )
    p.add_argument(
        "--judge_model", default="Qwen/Qwen3.5-35B-A3B",
        help="HF model ID or local path of judge model. "
             "Loaded in-process via vLLM (no server needed). "
             "Must be accessible from HF_HOME cache or as a local path."
    )
    p.add_argument(
        "--tensor_parallel_size", type=int, default=4,
        help="Number of GPUs for tensor parallelism in vLLM judge engine."
    )
    p.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85,
        help="GPU memory fraction for vLLM judge engine."
    )
    p.add_argument(
        "--judge_batch_size", type=int, default=256,
        help="Number of samples processed per generate() call."
    )
    p.add_argument(
        "--judge_relaxed", action="store_true", default=False,
        help="Enable relaxed judge mode: obs prompt focuses on signal direction "
             "only (temporal removed); inf prompt focuses on body-state agreement. "
             "Pairs whose temporal windows exceed --temporal_tolerance steps are "
             "auto-NO without a judge call. Omit to use strict mode (default)."
    )
    p.add_argument(
        "--temporal_tolerance", type=int, default=1,
        help="Only used when --judge_relaxed is set. Max temporal step-distance "
             "that still gets sent to the judge. 0=exact-match-only, 1=adjacent "
             "buckets OK (early↔early_to_mid, mid↔mid_to_late), 2=two steps. "
             "Pairs farther apart are auto-NO. Default: 1."
    )

    args = p.parse_args()
    return vars(args)


def _check_gpus_free(n_required: int) -> None:
    """Abort early if fewer than n_required GPUs are free."""
    import subprocess, sys
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as e:
        print(f"[GPU-check] WARNING: could not run nvidia-smi ({e}). Proceeding anyway.")
        return

    occupied = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            occupied.setdefault(parts[0], []).append(parts[1])

    try:
        n_total = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().count("\n") + 1)
    except Exception:
        n_total = "?"

    if occupied:
        print(f"[GPU-check] FAIL — {len(occupied)} GPU(s) occupied "
              f"(need {n_required} free out of {n_total} total):")
        for uuid, pids in occupied.items():
            print(f"  {uuid}: PIDs {pids}")
        print("[GPU-check] Wait for those jobs to finish or scancel them, then resubmit.")
        sys.exit(1)

    print(f"[GPU-check] OK — all {n_total} GPUs free, proceeding.")


def main():
    cfg = parse_args()
    _check_gpus_free(cfg.get("tensor_parallel_size", 4))

    eval_jsons = cfg.get("eval_jsons") or ([cfg.get("eval_json")] if cfg.get("eval_json") else [])
    eval_jsons = [Path(p) for p in eval_jsons if p]

    if not eval_jsons:
        print("[SNM] ERROR: no --eval_json provided.")
        sys.exit(1)

    datasets = cfg["dataset"]
    valid = list_datasets()
    for ds in datasets:
        if ds not in valid:
            print(f"[SNM] ERROR: unknown dataset '{ds}'. Valid: {valid}")
            sys.exit(1)
    if len(datasets) == 1:
        datasets = datasets * len(eval_jsons)
    elif len(datasets) != len(eval_jsons):
        print(f"[SNM] ERROR: --dataset count ({len(datasets)}) must be 1 or match "
              f"--eval_json count ({len(eval_jsons)}).")
        sys.exit(1)

    print(f"[SNM] Processing {len(eval_jsons)} eval file(s) with one judge load.")
    for ef, ds in zip(eval_jsons, datasets):
        print(f"[SNM]   {ds}: {ef.name}")
    if cfg.get("judge_relaxed"):
        print(f"[SNM] Judge mode: RELAXED (temporal_tolerance={cfg.get('temporal_tolerance',1)})")
    else:
        print(f"[SNM] Judge mode: STRICT (default)")

    judge = JudgeClient(
        model_path=cfg["judge_model"],
        tensor_parallel_size=cfg.get("tensor_parallel_size", 4),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        relaxed=cfg.get("judge_relaxed", False),
        temporal_tolerance=cfg.get("temporal_tolerance", 1),
    )
    print(f"[SNM] Judge loaded: {cfg['judge_model']}")

    n_ok, n_fail = 0, 0
    for eval_path, dataset in zip(eval_jsons, datasets):
        cfg["dataset"] = dataset
        print(f"\n{'='*60}")
        print(f"[SNM] File {n_ok+n_fail+1}/{len(eval_jsons)}: {eval_path.name}  [{dataset}]")
        print(f"{'='*60}")
        try:
            result = run_snm_single(judge, cfg, eval_path)
            if result:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"[SNM] ERROR processing {eval_path}: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1

    print(f"\n[SNM] All done. {n_ok} succeeded, {n_fail} failed.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()