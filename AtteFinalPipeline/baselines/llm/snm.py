import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from scipy.optimize import linear_sum_assignment

_OBS_RE = re.compile(
    "\\[OBSERVATION\\s*\\|\\s*id:\\s*(O\\d+)\\]\\s*\\nsensor:\\s*(.+?)\\ntemporal:\\s*(.+?)\\npattern:\\s*(.+?)\\nconfidence:\\s*(.+?)(?:\\n|$)",
    re.IGNORECASE,
)
_INF_RE = re.compile(
    "\\[INFERENCE\\s*\\|\\s*id:\\s*(I\\d+)\\]\\s*\\nbased_on:\\s*(.+?)\\ninference:\\s*(.+?)\\nconfidence:\\s*(.+?)(?:\\n|$)",
    re.IGNORECASE,
)
_SYN_RE = re.compile(
    "\\[SYNTHESIS\\]\\s*\\nbased_on:\\s*(.+?)\\n([\\s\\S]+?)(?=\\[ACTIVITY\\]|\\Z)",
    re.IGNORECASE,
)
_ACT_RE = re.compile("\\[ACTIVITY\\]\\s*:\\s*(.+?)$", re.IGNORECASE | re.MULTILINE)
TEMPORAL_VOCAB = {"early", "mid", "late", "early_to_mid", "mid_to_late", "full_window"}


def parse_dag(text: str) -> dict:
    text = text.strip() if text else ""
    if not text:
        return _empty_dag(parse_failed=True)
    observations = []
    for m in _OBS_RE.finditer(text):
        observations.append(
            {
                "id": m.group(1).strip(),
                "sensor": m.group(2).strip().lower(),
                "temporal": m.group(3).strip().lower(),
                "pattern": m.group(4).strip(),
                "confidence": m.group(5).strip().lower(),
            }
        )
    inferences = []
    for m in _INF_RE.finditer(text):
        based_on_ids = [x.strip() for x in m.group(2).strip().split(",")]
        inferences.append(
            {
                "id": m.group(1).strip(),
                "based_on": based_on_ids,
                "inference": m.group(3).strip(),
                "confidence": m.group(4).strip().lower(),
            }
        )
    synthesis = None
    syn_m = _SYN_RE.search(text)
    if syn_m:
        syn_based_on = [x.strip() for x in syn_m.group(1).strip().split(",")]
        syn_text = re.sub("\\n\\s*\\n", "\n", syn_m.group(2)).strip()
        synthesis = {"based_on": syn_based_on, "text": syn_text}
    activity = None
    act_m = _ACT_RE.search(text)
    if act_m:
        activity = act_m.group(1).strip().lower().replace("_", " ")
    return {
        "observations": observations,
        "inferences": inferences,
        "synthesis": synthesis,
        "activity": activity,
        "n_observations": len(observations),
        "n_inferences": len(inferences),
        "parse_failed": len(observations) == 0,
    }


def _empty_dag(parse_failed: bool = True) -> dict:
    return {
        "observations": [],
        "inferences": [],
        "synthesis": None,
        "activity": None,
        "n_observations": 0,
        "n_inferences": 0,
        "parse_failed": parse_failed,
    }


class JudgeClient:
    _SYS_OBS = "You are a sensor signal pattern equivalence judge. Decide if two descriptions refer to the same signal phenomenon. Output YES or NO on the first line, then one sentence of justification. Do not use chain-of-thought or internal monologue."
    _SYS_INF = "You are a biomechanical inference equivalence judge. Decide if two inferences express the same biomechanical interpretation. Output YES or NO on the first line, then one sentence of justification. Do not use chain-of-thought or internal monologue."
    _SYS_SYN = "You are a sensor activity synthesis equivalence judge. Decide if two synthesis paragraphs reach the same activity conclusion. Output YES or NO on the first line, then one sentence of justification. Do not use chain-of-thought or internal monologue."

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.85,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[Judge] Loading vLLM engine: {model_path}")
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=512,
            disable_custom_all_reduce=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=80, repetition_penalty=1.1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._n_calls = 0
        print(f"[Judge] vLLM engine ready")

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
            user = f"Sensor channel: {sensor}\n\nGround truth observation:\n  temporal: {tgt}\n  pattern: {pgt}\n\nPredicted observation:\n  temporal: {tpd}\n  pattern: {ppd}\n\nDo these describe the same signal phenomenon on this sensor?"
            prompts.append(self._build_prompt(self._SYS_OBS, user))
        return self._run_batch(prompts)

    def judge_inference_batch(self, pairs: list) -> list:
        prompts = []
        for igt, ipd in pairs:
            user = f"Ground truth inference:\n  {igt}\n\nPredicted inference:\n  {ipd}\n\nDo these express the same biomechanical interpretation?"
            prompts.append(self._build_prompt(self._SYS_INF, user))
        return self._run_batch(prompts)

    def judge_synthesis(self, synthesis_gt: str, synthesis_pred: str) -> str:
        user = f"Ground truth synthesis:\n{synthesis_gt}\n\nPredicted synthesis:\n{synthesis_pred}\n\nDo these reach the same conclusion about the activity?"
        prompt = self._build_prompt(self._SYS_SYN, user)
        results = self._run_batch([prompt])
        return results[0] if results else "ERROR"

    def build_obs_prompts(self, pairs: list) -> list:
        prompts = []
        for sensor, tgt, pgt, tpd, ppd in pairs:
            user = f"Sensor channel: {sensor}\n\nGround truth observation:\n  temporal: {tgt}\n  pattern: {pgt}\n\nPredicted observation:\n  temporal: {tpd}\n  pattern: {ppd}\n\nDo these describe the same signal phenomenon on this sensor?"
            prompts.append(self._build_prompt(self._SYS_OBS, user))
        return prompts

    def build_inf_prompts(self, pairs: list) -> list:
        prompts = []
        for igt, ipd in pairs:
            user = f"Ground truth inference:\n  {igt}\n\nPredicted inference:\n  {ipd}\n\nDo these express the same biomechanical interpretation?"
            prompts.append(self._build_prompt(self._SYS_INF, user))
        return prompts

    def build_syn_prompt(self, synthesis_gt: str, synthesis_pred: str) -> str:
        user = f"Ground truth synthesis:\n{synthesis_gt}\n\nPredicted synthesis:\n{synthesis_pred}\n\nDo these reach the same conclusion about the activity?"
        return self._build_prompt(self._SYS_SYN, user)

    def generate(self, prompts: list) -> list:
        return self._run_batch(prompts)


def match_observations(gt_obs: list, pred_obs: list, judge: JudgeClient) -> dict:
    n_gt = len(gt_obs)
    n_pred = len(pred_obs)
    if n_gt == 0 and n_pred == 0:
        return _obs_result_empty()
    if n_gt == 0:
        return _obs_result_empty(pred_hallucinations=n_pred)
    if n_pred == 0:
        return _obs_result_empty(gt_misses=n_gt)
    cost = np.ones((n_gt, n_pred), dtype=float)
    judge_pairs = []
    judge_indices = []
    for i, go in enumerate(gt_obs):
        for j, po in enumerate(pred_obs):
            if go["sensor"] == po["sensor"]:
                judge_pairs.append(
                    (
                        go["sensor"],
                        go["temporal"],
                        go["pattern"],
                        po["temporal"],
                        po["pattern"],
                    )
                )
                judge_indices.append((i, j))
    verdicts = []
    if judge_pairs:
        verdicts = judge.judge_observation_batch(judge_pairs)
        for (i, j), verdict in zip(judge_indices, verdicts):
            if verdict == "YES":
                cost[i, j] = 0.0
            elif verdict == "NO":
                cost[i, j] = 1.0
            else:
                cost[i, j] = 0.5
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    gt_matched = set()
    pred_matched = set()
    hits = 0
    temporal_hits = 0
    n_temporal_check = 0
    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[i, j] = verdict
    for i, j in zip(row_ind, col_ind):
        c = cost[i, j]
        is_hit = c < 0.5
        go = gt_obs[i]
        po = pred_obs[j]
        temporal_match = go["temporal"] == po["temporal"]
        judge_verdict = verdict_lookup.get((i, j), "SENSOR_MISMATCH")
        matches.append(
            {
                "gt_id": go["id"],
                "pred_id": po["id"],
                "sensor": go["sensor"],
                "temporal_match": temporal_match,
                "judge_verdict": judge_verdict,
                "hit": is_hit,
                "cost": float(c),
            }
        )
        if is_hit:
            hits += 1
            gt_matched.add(i)
            pred_matched.add(j)
            if temporal_match:
                temporal_hits += 1
            n_temporal_check += 1
    gt_unmatched = [gt_obs[i]["id"] for i in range(n_gt) if i not in gt_matched]
    pred_unmatched = [pred_obs[j]["id"] for j in range(n_pred) if j not in pred_matched]
    precision = hits / n_pred if n_pred > 0 else float("nan")
    recall = hits / n_gt if n_gt > 0 else float("nan")
    f1 = _f1(precision, recall)
    temporal_agreement = (
        temporal_hits / n_temporal_check if n_temporal_check > 0 else float("nan")
    )
    return {
        "matches": matches,
        "hits": hits,
        "gt_misses": len(gt_unmatched),
        "pred_hallucinations": len(pred_unmatched),
        "gt_unmatched": gt_unmatched,
        "pred_unmatched": pred_unmatched,
        "SNM-OP": precision,
        "SNM-OR": recall,
        "SNM-OF1": f1,
        "SNM-O-Temporal": temporal_agreement,
    }


def _obs_result_empty(gt_misses: int = 0, pred_hallucinations: int = 0) -> dict:
    nan = float("nan")
    return {
        "matches": [],
        "hits": 0,
        "gt_misses": gt_misses,
        "pred_hallucinations": pred_hallucinations,
        "gt_unmatched": [],
        "pred_unmatched": [],
        "SNM-OP": nan,
        "SNM-OR": nan,
        "SNM-OF1": nan,
        "SNM-O-Temporal": nan,
    }


def _prepare_obs(gt_obs: list, pred_obs: list) -> tuple:
    n_gt = len(gt_obs)
    n_pred = len(pred_obs)
    if n_gt == 0 or n_pred == 0:
        return (
            [],
            {
                "empty": True,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "gt_obs": gt_obs,
                "pred_obs": pred_obs,
            },
        )
    judge_pairs = []
    judge_indices = []
    for i, go in enumerate(gt_obs):
        for j, po in enumerate(pred_obs):
            if go["sensor"] == po["sensor"]:
                judge_pairs.append(
                    (
                        go["sensor"],
                        go["temporal"],
                        go["pattern"],
                        po["temporal"],
                        po["pattern"],
                    )
                )
                judge_indices.append((i, j))
    return (
        judge_pairs,
        {
            "empty": False,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_obs": gt_obs,
            "pred_obs": pred_obs,
            "judge_indices": judge_indices,
        },
    )


def _finalise_obs(state: dict, verdicts: list) -> dict:
    nan = float("nan")
    if state.get("empty"):
        n_gt, n_pred = (state["n_gt"], state["n_pred"])
        if n_gt == 0 and n_pred == 0:
            return _obs_result_empty()
        if n_gt == 0:
            return _obs_result_empty(pred_hallucinations=n_pred)
        return _obs_result_empty(gt_misses=n_gt)
    n_gt = state["n_gt"]
    n_pred = state["n_pred"]
    gt_obs = state["gt_obs"]
    pred_obs = state["pred_obs"]
    judge_indices = state["judge_indices"]
    cost = np.ones((n_gt, n_pred), dtype=float)
    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[i, j] = verdict
        if verdict == "YES":
            cost[i, j] = 0.0
        elif verdict == "NO":
            cost[i, j] = 1.0
        else:
            cost[i, j] = 0.5
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    gt_matched = set()
    pred_matched = set()
    hits = 0
    temporal_hits = 0
    n_temporal_check = 0
    for i, j in zip(row_ind, col_ind):
        c = cost[i, j]
        is_hit = c < 0.5
        go = gt_obs[i]
        po = pred_obs[j]
        temporal_match = go["temporal"] == po["temporal"]
        judge_verdict = verdict_lookup.get((i, j), "SENSOR_MISMATCH")
        matches.append(
            {
                "gt_id": go["id"],
                "pred_id": po["id"],
                "sensor": go["sensor"],
                "temporal_match": temporal_match,
                "judge_verdict": judge_verdict,
                "hit": is_hit,
                "cost": float(c),
            }
        )
        if is_hit:
            hits += 1
            gt_matched.add(i)
            pred_matched.add(j)
            if temporal_match:
                temporal_hits += 1
            n_temporal_check += 1
    gt_unmatched = [gt_obs[i]["id"] for i in range(n_gt) if i not in gt_matched]
    pred_unmatched = [pred_obs[j]["id"] for j in range(n_pred) if j not in pred_matched]
    precision = hits / n_pred if n_pred > 0 else nan
    recall = hits / n_gt if n_gt > 0 else nan
    f1 = _f1(precision, recall)
    temporal_agreement = (
        temporal_hits / n_temporal_check if n_temporal_check > 0 else nan
    )
    return {
        "matches": matches,
        "hits": hits,
        "gt_misses": len(gt_unmatched),
        "pred_hallucinations": len(pred_unmatched),
        "gt_unmatched": gt_unmatched,
        "pred_unmatched": pred_unmatched,
        "SNM-OP": precision,
        "SNM-OR": recall,
        "SNM-OF1": f1,
        "SNM-O-Temporal": temporal_agreement,
    }


def _prepare_inf(gt_infs: list, pred_infs: list) -> tuple:
    n_gt = len(gt_infs)
    n_pred = len(pred_infs)
    if n_gt == 0 or n_pred == 0:
        return (
            [],
            {
                "empty": True,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "gt_infs": gt_infs,
                "pred_infs": pred_infs,
            },
        )
    judge_pairs = []
    judge_indices = []
    for i, gi in enumerate(gt_infs):
        for j, pi in enumerate(pred_infs):
            judge_pairs.append((gi["inference"], pi["inference"]))
            judge_indices.append((i, j))
    return (
        judge_pairs,
        {
            "empty": False,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_infs": gt_infs,
            "pred_infs": pred_infs,
            "judge_indices": judge_indices,
        },
    )


def _finalise_inf(
    state: dict, verdicts: list, obs_id_map: dict, valid_pred_obs: set
) -> dict:
    nan = float("nan")
    if state.get("empty"):
        n_gt, n_pred = (state["n_gt"], state["n_pred"])
        if n_gt == 0 and n_pred == 0:
            return _inf_result_empty()
        if n_gt == 0:
            return _inf_result_empty(pred_hallucinations=n_pred)
        return _inf_result_empty(gt_misses=n_gt)
    n_gt = state["n_gt"]
    n_pred = state["n_pred"]
    gt_infs = state["gt_infs"]
    pred_infs = state["pred_infs"]
    judge_indices = state["judge_indices"]
    cost = np.ones((n_gt, n_pred), dtype=float)
    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[i, j] = verdict
        if verdict == "YES":
            cost[i, j] = 0.0
        elif verdict == "NO":
            cost[i, j] = 1.0
        else:
            cost[i, j] = 0.5
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    gt_matched = set()
    pred_matched = set()
    hits = 0
    prov_abs_scores = []
    prov_clean_scores = []
    prov_pen_scores = []
    for i, j in zip(row_ind, col_ind):
        c = cost[i, j]
        is_hit = c < 0.5
        gi = gt_infs[i]
        pi = pred_infs[j]
        gt_based = set(gi["based_on"])
        pred_based = set(pi["based_on"])
        mapped_gt = {obs_id_map[x] for x in gt_based if x in obs_id_map}
        union = mapped_gt | pred_based
        intersect = mapped_gt & pred_based
        jaccard_abs = len(intersect) / len(union) if union else nan
        hallucinated_basis = pred_based - valid_pred_obs
        n_hallucinated = len(hallucinated_basis)
        jaccard_clean = 0.0 if n_hallucinated > 0 else jaccard_abs
        valid_pred_cited = pred_based & valid_pred_obs
        n_int_clean = len(mapped_gt & valid_pred_cited)
        n_union_clean = len(mapped_gt | valid_pred_cited)
        base_j = n_int_clean / n_union_clean if n_union_clean > 0 else nan
        if not math.isnan(base_j):
            penalty = n_hallucinated / len(pred_based) if pred_based else 0.0
            jaccard_pen = max(-1.0, base_j - penalty)
        else:
            jaccard_pen = nan
        matches.append(
            {
                "gt_id": gi["id"],
                "pred_id": pi["id"],
                "judge_verdict": verdict_lookup.get((i, j), "ERROR"),
                "hit": is_hit,
                "cost": float(c),
                "gt_based_on": list(gt_based),
                "pred_based_on": list(pred_based),
                "mapped_gt_obs": list(mapped_gt),
                "hallucinated_basis": list(hallucinated_basis),
                "n_hallucinated_basis": n_hallucinated,
                "iprov_abs": jaccard_abs if not math.isnan(jaccard_abs) else None,
                "iprov_clean": jaccard_clean if not math.isnan(jaccard_clean) else None,
                "iprov_penalised": jaccard_pen if not math.isnan(jaccard_pen) else None,
            }
        )
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
    gt_unmatched = [gt_infs[i]["id"] for i in range(n_gt) if i not in gt_matched]
    pred_unmatched = [
        pred_infs[j]["id"] for j in range(n_pred) if j not in pred_matched
    ]
    precision = hits / n_pred if n_pred > 0 else nan
    recall = hits / n_gt if n_gt > 0 else nan
    f1 = _f1(precision, recall)
    iprov_abs = float(np.mean(prov_abs_scores)) if prov_abs_scores else nan
    iprov_clean = float(np.mean(prov_clean_scores)) if prov_clean_scores else nan
    iprov_pen = float(np.mean(prov_pen_scores)) if prov_pen_scores else nan
    return {
        "matches": matches,
        "hits": hits,
        "gt_misses": len(gt_unmatched),
        "pred_hallucinations": len(pred_unmatched),
        "gt_unmatched": gt_unmatched,
        "pred_unmatched": pred_unmatched,
        "SNM-IP": precision,
        "SNM-IR": recall,
        "SNM-IF1": f1,
        "SNM-IProv-Abs": iprov_abs,
        "SNM-IProv-Clean": iprov_clean,
        "SNM-IProv-Penalised": iprov_pen,
    }


def match_inferences(
    gt_infs: list, pred_infs: list, obs_id_map: dict, judge: JudgeClient
) -> dict:
    n_gt = len(gt_infs)
    n_pred = len(pred_infs)
    nan = float("nan")
    if n_gt == 0 and n_pred == 0:
        return _inf_result_empty()
    if n_gt == 0:
        return _inf_result_empty(pred_hallucinations=n_pred)
    if n_pred == 0:
        return _inf_result_empty(gt_misses=n_gt)
    valid_pred_obs = set(obs_id_map.values())
    judge_pairs = []
    judge_indices = []
    for i, gi in enumerate(gt_infs):
        for j, pi in enumerate(pred_infs):
            judge_pairs.append((gi["inference"], pi["inference"]))
            judge_indices.append((i, j))
    verdicts = judge.judge_inference_batch(judge_pairs)
    cost = np.ones((n_gt, n_pred), dtype=float)
    verdict_lookup = {}
    for (i, j), verdict in zip(judge_indices, verdicts):
        verdict_lookup[i, j] = verdict
        if verdict == "YES":
            cost[i, j] = 0.0
        elif verdict == "NO":
            cost[i, j] = 1.0
        else:
            cost[i, j] = 0.5
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    gt_matched = set()
    pred_matched = set()
    hits = 0
    prov_abs_scores = []
    prov_clean_scores = []
    prov_pen_scores = []
    for i, j in zip(row_ind, col_ind):
        c = cost[i, j]
        is_hit = c < 0.5
        gi = gt_infs[i]
        pi = pred_infs[j]
        gt_based = set(gi["based_on"])
        pred_based = set(pi["based_on"])
        mapped_gt = {obs_id_map[x] for x in gt_based if x in obs_id_map}
        union = mapped_gt | pred_based
        intersect = mapped_gt & pred_based
        jaccard_abs = len(intersect) / len(union) if union else nan
        hallucinated_basis = pred_based - valid_pred_obs
        n_hallucinated = len(hallucinated_basis)
        if n_hallucinated > 0:
            jaccard_clean = 0.0
        else:
            jaccard_clean = jaccard_abs
        valid_pred_cited = pred_based & valid_pred_obs
        n_intersect_clean = len(mapped_gt & valid_pred_cited)
        n_union_clean = len(mapped_gt | valid_pred_cited)
        base_j = n_intersect_clean / n_union_clean if n_union_clean > 0 else nan
        if not math.isnan(base_j):
            penalty = n_hallucinated / len(pred_based) if pred_based else 0.0
            jaccard_pen = max(-1.0, base_j - penalty)
        else:
            jaccard_pen = nan
        matches.append(
            {
                "gt_id": gi["id"],
                "pred_id": pi["id"],
                "judge_verdict": verdict_lookup.get((i, j), "ERROR"),
                "hit": is_hit,
                "cost": float(c),
                "gt_based_on": list(gt_based),
                "pred_based_on": list(pred_based),
                "mapped_gt_obs": list(mapped_gt),
                "hallucinated_basis": list(hallucinated_basis),
                "n_hallucinated_basis": n_hallucinated,
                "iprov_abs": jaccard_abs if not math.isnan(jaccard_abs) else None,
                "iprov_clean": jaccard_clean if not math.isnan(jaccard_clean) else None,
                "iprov_penalised": jaccard_pen if not math.isnan(jaccard_pen) else None,
            }
        )
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
    gt_unmatched = [gt_infs[i]["id"] for i in range(n_gt) if i not in gt_matched]
    pred_unmatched = [
        pred_infs[j]["id"] for j in range(n_pred) if j not in pred_matched
    ]
    precision = hits / n_pred if n_pred > 0 else nan
    recall = hits / n_gt if n_gt > 0 else nan
    f1 = _f1(precision, recall)
    iprov_abs = float(np.mean(prov_abs_scores)) if prov_abs_scores else nan
    iprov_clean = float(np.mean(prov_clean_scores)) if prov_clean_scores else nan
    iprov_pen = float(np.mean(prov_pen_scores)) if prov_pen_scores else nan
    return {
        "matches": matches,
        "hits": hits,
        "gt_misses": len(gt_unmatched),
        "pred_hallucinations": len(pred_unmatched),
        "gt_unmatched": gt_unmatched,
        "pred_unmatched": pred_unmatched,
        "SNM-IP": precision,
        "SNM-IR": recall,
        "SNM-IF1": f1,
        "SNM-IProv-Abs": iprov_abs,
        "SNM-IProv-Clean": iprov_clean,
        "SNM-IProv-Penalised": iprov_pen,
    }


def _inf_result_empty(gt_misses: int = 0, pred_hallucinations: int = 0) -> dict:
    nan = float("nan")
    return {
        "matches": [],
        "hits": 0,
        "gt_misses": gt_misses,
        "pred_hallucinations": pred_hallucinations,
        "gt_unmatched": [],
        "pred_unmatched": [],
        "SNM-IP": nan,
        "SNM-IR": nan,
        "SNM-IF1": nan,
        "SNM-IProv-Abs": nan,
        "SNM-IProv-Clean": nan,
        "SNM-IProv-Penalised": nan,
    }


def compute_snm_sample(gt_text: str, pred_text: str, judge: JudgeClient) -> dict:
    nan = float("nan")
    gt_dag = parse_dag(gt_text)
    pred_dag = parse_dag(pred_text)
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
    inf_result = match_inferences(
        gt_dag["inferences"], pred_dag["inferences"], obs_id_map, judge
    )
    gt_syn = gt_dag["synthesis"]
    pred_syn = pred_dag["synthesis"]
    if gt_syn is not None and pred_syn is not None:
        syn_verdict = judge.judge_synthesis(gt_syn["text"], pred_syn["text"])
        snm_sa = 1.0 if syn_verdict == "YES" else 0.0 if syn_verdict == "NO" else nan
    else:
        syn_verdict = "N/A"
        snm_sa = nan
    return {
        "snm_skipped": False,
        "n_gt_obs": gt_dag["n_observations"],
        "n_pred_obs": pred_dag["n_observations"],
        "n_gt_inf": gt_dag["n_inferences"],
        "n_pred_inf": pred_dag["n_inferences"],
        "obs_matches": obs_result["matches"],
        "obs_hits": obs_result["hits"],
        "obs_misses": obs_result["gt_misses"],
        "obs_hallucinations": obs_result["pred_hallucinations"],
        "SNM-OP": obs_result["SNM-OP"],
        "SNM-OR": obs_result["SNM-OR"],
        "SNM-OF1": obs_result["SNM-OF1"],
        "SNM-O-Temporal": obs_result["SNM-O-Temporal"],
        "inf_matches": inf_result["matches"],
        "inf_hits": inf_result["hits"],
        "inf_misses": inf_result["gt_misses"],
        "inf_hallucinations": inf_result["pred_hallucinations"],
        "SNM-IP": inf_result["SNM-IP"],
        "SNM-IR": inf_result["SNM-IR"],
        "SNM-IF1": inf_result["SNM-IF1"],
        "SNM-IProv-Abs": inf_result["SNM-IProv-Abs"],
        "SNM-IProv-Clean": inf_result["SNM-IProv-Clean"],
        "SNM-IProv-Penalised": inf_result["SNM-IProv-Penalised"],
        "synthesis_judge": syn_verdict,
        "SNM-SA": snm_sa,
    }


def aggregate_snm(per_sample_results: list) -> dict:
    nan = float("nan")
    keys = [
        "SNM-OP",
        "SNM-OR",
        "SNM-OF1",
        "SNM-O-Temporal",
        "SNM-IP",
        "SNM-IR",
        "SNM-IF1",
        "SNM-IProv-Abs",
        "SNM-IProv-Clean",
        "SNM-IProv-Penalised",
        "SNM-SA",
    ]
    accum = {k: [] for k in keys}
    n_total = len(per_sample_results)
    n_scored = 0
    n_skipped = 0
    n_gt_pf = 0
    n_pred_pf = 0
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
            if isinstance(v, float) and (not math.isnan(v)):
                accum[k].append(v)
    agg = {}
    for k in keys:
        vals = accum[k]
        agg[k] = float(np.mean(vals)) if vals else nan
    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_skipped": n_skipped,
        "n_gt_parse_fail": n_gt_pf,
        "n_pred_parse_fail": n_pred_pf,
        **agg,
    }


def _f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


class _NaNEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(_replace_nan(obj))

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(_replace_nan(obj), _one_shot=_one_shot)


def _replace_nan(obj):
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


def run_snm_single(judge: "JudgeClient", cfg: dict, eval_path: Path) -> dict:
    if not eval_path.exists():
        print(f"[SNM] ERROR: eval_json not found: {eval_path}")
        return {}
    print(f"[SNM] Loading eval file: {eval_path}")
    with open(eval_path) as f:
        eval_data = json.load(f)
    if isinstance(eval_data, list):
        per_sample_raw = eval_data
        source_config = {}
    elif isinstance(eval_data, dict):
        per_sample_raw = eval_data.get("per_sample", [])
        source_config = eval_data.get("config", {})
    else:
        print("[SNM] ERROR: Unexpected eval JSON format (not list or dict).")
        return {}
    print(f"[SNM] Found {len(per_sample_raw)} samples in eval file.")
    skipped_out = []
    scored_candidates = []
    for k, raw in enumerate(per_sample_raw):
        sid = raw.get("sample_id", str(k))
        gt_activity = raw.get("gt_activity", "")
        pred_activity = raw.get("predicted_activity", "")
        gt_text = raw.get("ground_truth_reasoning", "").strip()
        pred_text = raw.get("generated", "").strip()
        entry = {
            "sample_id": sid,
            "gt_activity": gt_activity,
            "predicted_activity": pred_activity,
        }
        skip_reason = None
        if not gt_text:
            skip_reason = "no_gt_reasoning"
        elif not pred_text:
            skip_reason = "no_pred_text"
        else:
            gt_dag = parse_dag(gt_text)
            pred_dag = parse_dag(pred_text)
            if gt_dag["parse_failed"]:
                skip_reason = "gt_parse_fail"
            elif pred_dag["parse_failed"]:
                skip_reason = "pred_parse_fail"
        if skip_reason:
            entry["snm_skipped"] = True
            entry["snm_skip_reason"] = skip_reason
            skipped_out.append((k, entry))
        else:
            scored_candidates.append((k, raw, gt_dag, pred_dag, entry))
    n_total = len(per_sample_raw)
    n_with_gt = sum(
        (1 for r in per_sample_raw if r.get("ground_truth_reasoning", "").strip())
    )
    n_scoreable = len(scored_candidates)
    n_skipped = len(skipped_out)
    print(f"[SNM] Samples with non-empty ground_truth_reasoning: {n_with_gt}")
    print(f"[SNM] Scoreable (GT + pred + parse OK): {n_scoreable} / {n_total}")
    print(f"[SNM] Pre-skipped (no GT / no pred / parse fail): {n_skipped}")
    if n_scoreable == 0:
        print("[SNM] WARNING: No scoreable samples. Nothing to judge.")
    batch_size = cfg.get("judge_batch_size", 256)
    print(f"[SNM] Batch size: {batch_size} scoreable samples per generate() call")
    scored_results = [None] * n_scoreable
    t0 = time.time()
    for batch_start in range(0, n_scoreable, batch_size):
        batch_cands = scored_candidates[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch_cands)
        print(
            f"[SNM] Batch {batch_start}–{batch_end - 1} of {n_scoreable} scoreable | judge_calls={judge.n_calls} | elapsed={time.time() - t0:.0f}s"
        )
        gt_dags = [c[2] for c in batch_cands]
        pred_dags = [c[3] for c in batch_cands]
        entries = [c[4] for c in batch_cands]
        n_scored_batch = len(batch_cands)
        scored_idx = list(range(n_scored_batch))
        if n_scored_batch == 0:
            continue
        obs_states = []
        obs_prompt_offsets = []
        obs_prompts_flat = []
        for s in range(n_scored_batch):
            pairs, state = _prepare_obs(
                gt_dags[s]["observations"], pred_dags[s]["observations"]
            )
            prompts = judge.build_obs_prompts(pairs)
            start = len(obs_prompts_flat)
            obs_prompts_flat.extend(prompts)
            obs_prompt_offsets.append((start, start + len(prompts)))
            obs_states.append(state)
        obs_verdicts_flat = judge.generate(obs_prompts_flat) if obs_prompts_flat else []
        obs_results = []
        obs_id_maps = []
        for s in range(n_scored_batch):
            start, end = obs_prompt_offsets[s]
            verdicts_s = obs_verdicts_flat[start:end]
            obs_r = _finalise_obs(obs_states[s], verdicts_s)
            obs_results.append(obs_r)
            obs_id_map = {
                m["gt_id"]: m["pred_id"] for m in obs_r["matches"] if m["hit"]
            }
            obs_id_maps.append(obs_id_map)
        inf_states = []
        inf_prompt_offsets = []
        inf_prompts_flat = []
        valid_pred_obs_list = []
        for s in range(n_scored_batch):
            valid_pred_obs = set(obs_id_maps[s].values())
            valid_pred_obs_list.append(valid_pred_obs)
            pairs, state = _prepare_inf(
                gt_dags[s]["inferences"], pred_dags[s]["inferences"]
            )
            prompts = judge.build_inf_prompts(pairs)
            start = len(inf_prompts_flat)
            inf_prompts_flat.extend(prompts)
            inf_prompt_offsets.append((start, start + len(prompts)))
            inf_states.append(state)
        inf_verdicts_flat = judge.generate(inf_prompts_flat) if inf_prompts_flat else []
        inf_results = []
        for s in range(n_scored_batch):
            start, end = inf_prompt_offsets[s]
            verdicts_s = inf_verdicts_flat[start:end]
            inf_r = _finalise_inf(
                inf_states[s], verdicts_s, obs_id_maps[s], valid_pred_obs_list[s]
            )
            inf_results.append(inf_r)
        syn_prompts_flat = []
        syn_prompt_indices = []
        for s in range(n_scored_batch):
            gt_syn = gt_dags[s]["synthesis"]
            pred_syn = pred_dags[s]["synthesis"]
            if gt_syn is not None and pred_syn is not None:
                prompt = judge.build_syn_prompt(gt_syn["text"], pred_syn["text"])
                syn_prompt_indices.append((s, len(syn_prompts_flat)))
                syn_prompts_flat.append(prompt)
        syn_verdicts_flat = judge.generate(syn_prompts_flat) if syn_prompts_flat else []
        syn_verdict_by_s = {}
        for s, idx in syn_prompt_indices:
            syn_verdict_by_s[s] = syn_verdicts_flat[idx]
        nan = float("nan")
        for s, ei in enumerate(scored_idx):
            obs_r = obs_results[s]
            inf_r = inf_results[s]
            syn_v = syn_verdict_by_s.get(s, "N/A")
            snm_sa = 1.0 if syn_v == "YES" else 0.0 if syn_v == "NO" else nan
            entries[ei].update(
                {
                    "snm_skipped": False,
                    "n_gt_obs": gt_dags[s]["n_observations"],
                    "n_pred_obs": pred_dags[s]["n_observations"],
                    "n_gt_inf": gt_dags[s]["n_inferences"],
                    "n_pred_inf": pred_dags[s]["n_inferences"],
                    "obs_matches": obs_r["matches"],
                    "obs_hits": obs_r["hits"],
                    "obs_misses": obs_r["gt_misses"],
                    "obs_hallucinations": obs_r["pred_hallucinations"],
                    "SNM-OP": obs_r["SNM-OP"],
                    "SNM-OR": obs_r["SNM-OR"],
                    "SNM-OF1": obs_r["SNM-OF1"],
                    "SNM-O-Temporal": obs_r["SNM-O-Temporal"],
                    "inf_matches": inf_r["matches"],
                    "inf_hits": inf_r["hits"],
                    "inf_misses": inf_r["gt_misses"],
                    "inf_hallucinations": inf_r["pred_hallucinations"],
                    "SNM-IP": inf_r["SNM-IP"],
                    "SNM-IR": inf_r["SNM-IR"],
                    "SNM-IF1": inf_r["SNM-IF1"],
                    "SNM-IProv-Abs": inf_r["SNM-IProv-Abs"],
                    "SNM-IProv-Clean": inf_r["SNM-IProv-Clean"],
                    "SNM-IProv-Penalised": inf_r["SNM-IProv-Penalised"],
                    "synthesis_judge": syn_v,
                    "SNM-SA": snm_sa,
                }
            )
        for s in range(n_scored_batch):
            scored_results[batch_start + s] = entries[s]
    skipped_map = {orig_k: entry for orig_k, entry in skipped_out}
    scored_map = {
        scored_candidates[i][0]: scored_results[i]
        for i in range(n_scoreable)
        if scored_results[i] is not None
    }
    per_sample_out = []
    for k in range(n_total):
        if k in skipped_map:
            per_sample_out.append(skipped_map[k])
        elif k in scored_map:
            per_sample_out.append(scored_map[k])
    agg = aggregate_snm(per_sample_out)
    agg["n_judge_calls"] = judge.n_calls
    elapsed_total = time.time() - t0
    print(f"\n[SNM] Done. {judge.n_calls} judge calls in {elapsed_total:.1f}s")
    print(f"[SNM] Scored {agg['n_scored']} / {agg['n_total']} samples")
    print(f"\n{'=' * 55}")
    print(f"  SNM AGGREGATE RESULTS")
    print(f"{'=' * 55}")
    for k in [
        "SNM-OF1",
        "SNM-OP",
        "SNM-OR",
        "SNM-O-Temporal",
        "SNM-IF1",
        "SNM-IP",
        "SNM-IR",
        "SNM-IProv-Abs",
        "SNM-IProv-Clean",
        "SNM-IProv-Penalised",
        "SNM-SA",
    ]:
        v = agg.get(k, float("nan"))
        vs = f"{v:.4f}" if isinstance(v, float) and (not math.isnan(v)) else "N/A"
        print(f"  {k:<20s}: {vs}")
    print(f"{'=' * 55}\n")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = eval_path.parent / "evaluate_snm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{eval_path.stem}_snm_{ts}.json"
    out_path = out_dir / out_name
    output = {
        "source_eval_json": str(eval_path),
        "config": {
            "judge_model": cfg["judge_model"],
            "tensor_parallel_size": cfg.get("tensor_parallel_size", 4),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.85),
            "judge_batch_size": cfg.get("judge_batch_size", 256),
            "timestamp": ts,
            "source_config": source_config,
        },
        "snm_aggregate": agg,
        "per_sample": per_sample_out,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NaNEncoder)
    print(f"[SNM] Saved → {out_path}")
    return output


def run_snm(cfg: dict):
    judge = JudgeClient(
        model_path=cfg["judge_model"],
        tensor_parallel_size=cfg.get("tensor_parallel_size", 4),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
    )
    print(f"[SNM] Judge loaded: {cfg['judge_model']}")
    eval_path = Path(cfg["eval_json"])
    run_snm_single(judge, cfg, eval_path)


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="SensorLLM — Semantic Node Match (SNM) Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--eval_json",
        required=True,
        nargs="+",
        dest="eval_jsons",
        help="Path(s) to evaluate JSON file(s). Pass multiple times or space-separated. The judge (vLLM engine) is loaded ONCE and reused across all files — no reload penalty between files. Each entry must have: sample_id, gt_activity, predicted_activity, ground_truth_reasoning, generated.",
    )
    p.add_argument(
        "--judge_model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="HF model ID or local path of judge model. Loaded in-process via vLLM (no server needed). Must be accessible from HF_HOME cache or as a local path.",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism in vLLM judge engine.",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory fraction for vLLM judge engine (default 0.92).",
    )
    p.add_argument(
        "--judge_batch_size",
        type=int,
        default=256,
        help="Number of samples processed per generate() call. All obs/inf/syn judge prompts from this many samples are packed into a single llm.generate() call per layer. Higher = better GPU utilisation on multi-H100 setups. Default 256 is good for 4×H100 with Qwen3.5-35B-A3B (TP=4).",
    )
    args = p.parse_args()
    return vars(args)


def _check_gpus_free(n_required: int) -> None:
    import subprocess, sys

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as e:
        print(
            f"[GPU-check] WARNING: could not run nvidia-smi ({e}). Proceeding anyway."
        )
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
        n_total = int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .count("\n")
            + 1
        )
    except Exception:
        n_total = "?"
    if occupied:
        print(
            f"[GPU-check] FAIL — {len(occupied)} GPU(s) occupied (need {n_required} free out of {n_total} total):"
        )
        for uuid, pids in occupied.items():
            print(f"  {uuid}: PIDs {pids}")
        print(
            "[GPU-check] Wait for those jobs to finish or scancel them, then resubmit."
        )
        sys.exit(1)
    print(f"[GPU-check] OK — all {n_total} GPUs free, proceeding.")


def main():
    cfg = parse_args()
    _check_gpus_free(cfg.get("tensor_parallel_size", 4))
    eval_jsons = cfg.get("eval_jsons", [cfg.get("eval_json")])
    eval_jsons = [Path(p) for p in eval_jsons if p]
    if not eval_jsons:
        print("[SNM] ERROR: no --eval_json provided.")
        sys.exit(1)
    print(f"[SNM] Processing {len(eval_jsons)} eval file(s) with one judge load.")
    judge = JudgeClient(
        model_path=cfg["judge_model"],
        tensor_parallel_size=cfg.get("tensor_parallel_size", 4),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
    )
    print(f"[SNM] Judge loaded: {cfg['judge_model']}")
    n_ok, n_fail = (0, 0)
    for eval_path in eval_jsons:
        print(f"\n{'=' * 60}")
        print(f"[SNM] File {n_ok + n_fail + 1}/{len(eval_jsons)}: {eval_path.name}")
        print(f"{'=' * 60}")
        try:
            result = run_snm_single(judge, cfg, eval_path)
            if result:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"[SNM] ERROR processing {eval_path}: {e}")
            import traceback

            traceback.print_exc()
            n_fail += 1
    print(f"\n[SNM] All done. {n_ok} succeeded, {n_fail} failed.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
