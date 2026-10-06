from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoTokenizer
from AtteFinalPipeline.data.labels import USCHAD_CLASS_MAP
from AtteFinalPipeline.paths import PIPELINE_ROOT
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import re
import json
import time
import random
import logging
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(PIPELINE_ROOT)
INPUT_DIRS: Dict[str, Path] = {
    "ucihar": PROJECT_ROOT / "data" / "ucihar",
    "uschad": PROJECT_ROOT / "data" / "uschad",
    "pamap2": PROJECT_ROOT / "data" / "pamap2",
    "capture24": PROJECT_ROOT / "data" / "capture24",
    "shoaib": PROJECT_ROOT / "data" / "shoaib",
    "mhealth": PROJECT_ROOT / "data" / "mhealth",
    "opportunity": PROJECT_ROOT / "data" / "opportunity",
}
OUTPUT_ROOT = PROJECT_ROOT / "cot"
DATASET_LABEL_CONFIGS: Dict[str, List[str]] = {
    "ucihar": [
        "walking",
        "walking upstairs",
        "walking downstairs",
        "sitting",
        "standing",
        "laying",
    ],
    "uschad": [label.lower() for label in USCHAD_CLASS_MAP],
    "pamap2": [
        "ascending stairs",
        "cycling",
        "descending stairs",
        "ironing",
        "lying",
        "nordic walking",
        "rope jumping",
        "running",
        "sitting",
        "standing",
        "vacuum cleaning",
        "walking",
    ],
    "capture24": ["bicycling", "mixed", "sit-stand", "sleep", "vehicle", "walking"],
    "shoaib": [
        "biking",
        "jogging",
        "sitting",
        "standing",
        "walking",
        "walking downstairs",
        "walking upstairs",
    ],
    "mhealth": [
        "climbing stairs",
        "cycling",
        "frontal elevation of arms",
        "jogging",
        "jump front & back",
        "knees bending",
        "lying down",
        "running",
        "sitting and relaxing",
        "standing still",
        "waist bends forward",
        "walking",
    ],
    "opportunity": [
        "clean table",
        "close dishwasher",
        "close door 1",
        "close door 2",
        "close drawer 1",
        "close drawer 2",
        "close drawer 3",
        "close fridge",
        "drink from cup",
        "open dishwasher",
        "open door 1",
        "open door 2",
        "open drawer 1",
        "open drawer 2",
        "open drawer 3",
        "open fridge",
        "toggle switch",
    ],
}
DATASET_SENSOR_TERMS: Dict[str, List[str]] = {
    "ucihar": [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accelerometer",
        "gyroscope",
        "x-axis",
        "y-axis",
        "z-axis",
    ],
    "uschad": [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accelerometer",
        "gyroscope",
        "x-axis",
        "y-axis",
        "z-axis",
    ],
    "pamap2": [
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "temperature",
        "heart rate",
        "hand",
        "chest",
        "ankle",
        "x-axis",
        "y-axis",
        "z-axis",
    ],
    "capture24": [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "accelerometer",
        "x-axis",
        "y-axis",
        "z-axis",
    ],
    "shoaib": [
        "wrist",
        "belt",
        "pocket",
        "upper arm",
        "accelerometer",
        "gyroscope",
        "linear acceleration",
        "x-axis",
        "y-axis",
        "z-axis",
    ],
    "mhealth": [
        "chest",
        "ankle",
        "wrist",
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "ecg",
        "x-axis",
        "y-axis",
        "z-axis",
    ],
    "opportunity": [
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "back",
        "arm",
        "shoe",
        "hip",
    ],
}
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
TENSOR_PARALLEL_SIZE = 4
BATCH_SIZE = 128
SAMPLES_PER_DATASET: Optional[int] = None
SAMPLING_STRATEGY: str = "balanced"
SAMPLING_SEED: int = 42
ACTIVE_DATASETS: Optional[List[str]] = None
TARGET_SENSOR_BLOCK_TOKENS = 5500
MAX_MODEL_LEN = 10240
MAX_PROMPT_TOKENS = 7000
MAX_NEW_TOKENS = 2048
GS_TOP_K = 3
N_RUNS = 3
WANDB_PROJECT = "sensorllm"
WANDB_ENABLED = True
NULL_LABEL_TOKENS = {"null", "none", "background", "other", "unknown", "0", ""}
NON_SAMPLE_FILES = {
    "prediction_summary.json",
    "sample_summary.json",
    "summary.json",
    "progress.json",
    "perf_log.jsonl",
    "config.json",
    "class_map.json",
    "run_summary.json",
    "test_dataset.jsonl",
    "train_dataset.jsonl",
    "oversized_prompts.txt",
}
REFUSAL_RE = re.compile(
    "(?i)(i cannot|i can't|as an ai|i am unable|i'm unable|i apologize|i'm sorry, but|sorry, i cannot|not able to (provide|generate|analyze))"
)


def _is_sample_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    if path.name in NON_SAMPLE_FILES:
        return False
    if path.name.startswith("tracking_worker"):
        return False
    return True


def _discover_test_sample_jsons(input_dir: Path, logger: logging.Logger) -> List[Path]:
    SKIP_DIRS = {"reasoning_json", "cot_baseline", "reasoning"}
    direct = [p for p in sorted(input_dir.glob("test_*.json")) if _is_sample_json(p)]
    if direct:
        logger.info(f"  Discovered {len(direct)} test_*.json (non-recursive)")
        return direct
    logger.warning(f"  No test_*.json at top level; trying recursive")
    found: List[Path] = []
    for p in sorted(input_dir.rglob("test_*.json")):
        if any((part in SKIP_DIRS for part in p.parts)):
            continue
        if _is_sample_json(p):
            found.append(p)
    logger.info(f"  Discovered {len(found)} test_*.json (recursive)")
    return found


def parse_filename(path: Path) -> Tuple[str, str]:
    name = path.stem
    split_m = re.match("^(train|test|val|valid)_", name)
    sample_m = re.search("_s(\\d+)$", name)
    split = split_m.group(1) if split_m else "unknown"
    sample_id = str(int(sample_m.group(1))) if sample_m else "0"
    if split in ("val", "valid"):
        split = "val"
    return (split, sample_id)


def setup_logger(output_dir: Path, name: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(output_dir / "run.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def is_null_label(label: Optional[str]) -> bool:
    if not label:
        return True
    return label.strip().lower() in NULL_LABEL_TOKENS


def _match_label(activity: str, valid_labels: List[str]) -> Optional[str]:
    act_lower = activity.strip().lower()
    for lbl in valid_labels:
        if lbl.lower() == act_lower:
            return lbl
    for lbl in sorted(valid_labels, key=len, reverse=True):
        if lbl.lower() in act_lower or act_lower in lbl.lower():
            return lbl
    return None


def parse_predicted_activity(
    text: str, class_labels: List[str], logger: logging.Logger, sample_id: str
) -> str:
    _PRED_ACT_RE = re.compile(
        "(?i)predicted\\s*activity\\s*[:\\-]\\s*\\*{0,2}([A-Za-z0-9 \\-&/()_]+?)\\*{0,2}\\s*(?:\\.|\\n|$)"
    )
    m = _PRED_ACT_RE.search(text)
    if m:
        raw = m.group(1).strip().lower().replace("_", " ")
        for cls in sorted(class_labels, key=len, reverse=True):
            if cls == raw or cls in raw:
                return cls
        return raw if raw else "unknown"
    text_normalised = text.lower().replace("_", " ")
    for cls in sorted(class_labels, key=len, reverse=True):
        if cls in text_normalised:
            return cls
    logger.warning(f"  Parse failed sample {sample_id}: {repr(text[-200:])}")
    return "unknown"


@dataclass
class DatasetConfig:
    name: str
    channel_order: List[str]
    class_names: List[str]
    valid_class_labels: List[str]
    sensor_terms: List[str]
    num_timesteps: int
    downsample_factor: int


def _format_sensor_block_for_probe(raw: Dict, channels: List[str], ds: int) -> str:
    lines = []
    for ch in channels:
        vals = raw.get(ch, [0.0] * 100)[::ds]
        lines.append(f"{ch}: [{', '.join((f'{v:.2f}' for v in vals))}]")
    return "\n".join(lines)


def discover_dataset_config(
    dataset_key: str, input_dir: Path, tokenizer: AutoTokenizer, logger: logging.Logger
) -> Optional[DatasetConfig]:
    CHANNEL_PROBE_N = 50
    candidates = _discover_test_sample_jsons(input_dir, logger)
    if not candidates:
        return None
    try:
        with open(candidates[0], "r") as f:
            data0 = json.load(f)
    except Exception as e:
        logger.error(f"  Cannot read probe file: {e}")
        return None
    meta = data0.get("analysis_metadata", {})
    declared_channels = meta.get("sensor_channels", [])
    num_t = meta.get("num_timesteps")
    if not num_t:
        raw0 = data0.get("raw_sensor_data", {})
        first_present = next(iter(raw0), None)
        if first_present:
            num_t = len(raw0[first_present])
            logger.warning(
                f"  num_timesteps inferred from raw_sensor_data['{first_present}']: {num_t}"
            )
        else:
            logger.error(f"  Cannot determine num_timesteps — raw_sensor_data is empty")
            return None
    CHANNEL_PROBE_N = 50
    CHANNEL_MAJORITY_PCT = 0.8
    import random as _rnd

    probe_candidates = candidates.copy()
    _rnd.seed(42)
    _rnd.shuffle(probe_candidates)
    probe_files = probe_candidates[:CHANNEL_PROBE_N]
    channel_counts: Dict[str, int] = defaultdict(int)
    n_valid = 0
    for fpath in probe_files:
        try:
            with open(fpath, "r") as f:
                d = json.load(f)
            present = set(d.get("raw_sensor_data", {}).keys())
            if not present:
                continue
            for ch in present:
                channel_counts[ch] += 1
            n_valid += 1
        except Exception:
            continue
    if n_valid == 0:
        logger.warning(
            f"  Could not read raw_sensor_data from any probe file; using declared channels (may cause skips)"
        )
        channels = list(declared_channels)
    else:
        threshold = CHANNEL_MAJORITY_PCT * n_valid
        common = {ch for ch, cnt in channel_counts.items() if cnt >= threshold}
        channels = sorted(common)
        common_lower = {c.lower() for c in common}
        declared_missing = [
            c for c in declared_channels if c.lower() not in common_lower
        ]
        if declared_missing:
            logger.warning(
                f"  {len(declared_missing)} declared channels not found in raw_sensor_data (naming mismatch or truly absent): {declared_missing[:5]}{('...' if len(declared_missing) > 5 else '')}"
            )
        logger.info(
            f"  Channel majority-vote: {n_valid} valid probe files, threshold={CHANNEL_MAJORITY_PCT * 100:.0f}% → {len(channels)} channels retained from raw_sensor_data keys"
        )
    if not channels:
        logger.error(f"  No common channels found across probe files")
        return None
    if dataset_key in DATASET_LABEL_CONFIGS:
        valid_labels = DATASET_LABEL_CONFIGS[dataset_key]
    else:
        logger.warning(
            f"  '{dataset_key}' not in DATASET_LABEL_CONFIGS; using analysis_metadata class_names"
        )
        valid_labels = [l for l in meta.get("class_names", []) if not is_null_label(l)]
    if not valid_labels:
        logger.error(f"  No valid labels for '{dataset_key}'")
        return None
    sensor_terms = DATASET_SENSOR_TERMS.get(dataset_key, [])
    raw = data0.get("raw_sensor_data", {})
    downsample = 16
    for ds in range(1, 17):
        block = _format_sensor_block_for_probe(raw, channels, ds)
        if len(tokenizer.encode(block)) <= TARGET_SENSOR_BLOCK_TOKENS:
            downsample = ds
            break
    else:
        logger.warning(f"  Even 16x downsample exceeds token budget; proceeding.")
    eff_t = (int(num_t) + downsample - 1) // downsample
    logger.info(
        f"  '{dataset_key}': ch={len(channels)} t={num_t}→{eff_t} ds={downsample}x labels={len(valid_labels)}"
    )
    return DatasetConfig(
        name=dataset_key,
        channel_order=list(channels),
        class_names=list(meta.get("class_names", [])),
        valid_class_labels=valid_labels,
        sensor_terms=sensor_terms,
        num_timesteps=int(num_t),
        downsample_factor=downsample,
    )


def _peek_activity(path: Path) -> Optional[str]:
    try:
        with open(path, "r") as f:
            d = json.load(f)
        act = d.get("activity")
        return act if act and (not is_null_label(act)) else None
    except Exception:
        return None


def _select_samples(
    files: List[Path],
    n: Optional[int],
    strategy: str,
    seed: int,
    logger: logging.Logger,
) -> List[Path]:
    if n is None or n <= 0 or n >= len(files):
        return files
    rng = random.Random(seed)
    if strategy == "first":
        return files[:n]
    if strategy == "random":
        sampled = rng.sample(files, n)
        logger.info(f"  Sampling: random → {len(sampled)}/{len(files)}")
        return sampled
    by_class: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        act = _peek_activity(f)
        if act:
            by_class[act.lower()].append(f)
    if not by_class:
        return files[:n]
    for cls in by_class:
        rng.shuffle(by_class[cls])
    classes = sorted(by_class.keys())
    base = n // len(classes)
    remainder = n - base * len(classes)
    quotas = {c: base for c in classes}
    for c in sorted(classes, key=lambda x: len(by_class[x]), reverse=True)[:remainder]:
        quotas[c] += 1
    selected, leftover = ([], [])
    for cls in classes:
        avail = by_class[cls]
        q = quotas[cls]
        selected.extend(avail[:q] if len(avail) >= q else avail)
        if len(avail) > q:
            leftover.extend(avail[q:])
    deficit = n - len(selected)
    if deficit > 0 and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:deficit])
    rng.shuffle(selected)
    dist = defaultdict(int)
    for f in selected:
        act = _peek_activity(f)
        if act:
            dist[act.lower()] += 1
    logger.info(
        f"  Selected {len(selected)}: "
        + ", ".join((f"{c}={dist[c]}" for c in sorted(dist)))
    )
    return selected


@dataclass
class ReasoningOutput:
    sample_id: str
    split: str
    dataset: str
    true_activity: str
    predicted_activity: str
    classifier_confidence: float
    correct: bool
    reasoning: str
    raw_model_output: str
    generation_timestamp: str
    model_used: str
    downsample_factor: int
    ig_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ground_truth_reasoning"] = ""
        return d

    def to_json(self, filepath: Path):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_training_format(self) -> dict:
        return {
            "input": f"Activity Recognition Task:\nAnalyze the following raw inertial sensor window and predict the activity.\n\nDataset: {self.dataset}\nSample ID: {self.sample_id}\n",
            "output": f"Reasoning:\n{self.reasoning}\n\nPredicted Activity: {self.predicted_activity}\n",
            "metadata": self.to_dict(),
        }


def compute_classification_metrics(
    results: List[ReasoningOutput], valid_labels: List[str]
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    all_classes = [c.lower() for c in valid_labels] + ["unknown"]
    trues = [r.true_activity.lower() for r in results]
    preds = [r.predicted_activity.lower() for r in results]
    acc = accuracy_score(trues, preds)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        trues, preds, labels=all_classes, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        trues, preds, labels=all_classes, average=None, zero_division=0
    )
    cm = confusion_matrix(trues, preds, labels=all_classes).tolist()
    n_correct = int(sum((t == p for t, p in zip(trues, preds))))
    return {
        "n_total": len(results),
        "n_correct": n_correct,
        "n_out_of_vocab": int(sum((p == "unknown" for p in preds))),
        "accuracy": float(acc),
        "macro_f1": float(f1_mac),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "per_class": {
            cls: {
                "precision": float(p_cls[i]),
                "recall": float(r_cls[i]),
                "f1": float(f1_cls[i]),
                "support": int(sup_cls[i]),
            }
            for i, cls in enumerate(all_classes)
        },
        "confusion_matrix": {"labels": all_classes, "matrix": cm},
    }


def compute_text_metrics(results: List[ReasoningOutput]) -> dict:
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from rouge_score import rouge_scorer as rouge_lib
    from nltk.translate.meteor_score import meteor_score as nltk_meteor

    rouge_scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    all_results = results
    n_all = len(all_results)
    if n_all == 0:
        nan = float("nan")
        return {
            "rouge_l": nan,
            "bertscore_f1": nan,
            "meteor": nan,
            "sensor_term_recall": nan,
            "conditioned": {},
        }
    generated = [r.reasoning for r in all_results]
    references = [r.true_activity for r in all_results]
    pred_acts = [r.predicted_activity.lower() for r in all_results]
    gt_acts = [r.true_activity.lower() for r in all_results]
    rouge_scores = [
        rouge_scorer.score(ref, gen)["rougeL"].fmeasure
        for gen, ref in zip(generated, references)
    ]
    rouge_l = float(np.mean(rouge_scores))
    meteor_scores = [
        nltk_meteor([ref.split()], gen.split())
        for gen, ref in zip(generated, references)
    ]
    meteor = float(np.mean(meteor_scores))
    bertscore = float("nan")
    bert_f1_list = [float("nan")] * n_all
    try:
        from bert_score import score as bert_score_fn

        _, _, bert_f1_t = bert_score_fn(generated, references, lang="en", verbose=False)
        bert_f1_list = bert_f1_t.tolist()
        bertscore = float(bert_f1_t.mean().item())
    except Exception as e:
        print(f"[metrics] WARNING: BERTScore failed ({e})")
    correct_mask = [p == g for p, g in zip(pred_acts, gt_acts)]
    incorrect_mask = [not m for m in correct_mask]

    def _mean_if_any(vals, mask):
        sub = [
            v
            for v, m in zip(vals, mask)
            if m and (not (isinstance(v, float) and np.isnan(v)))
        ]
        return float(np.mean(sub)) if sub else float("nan")

    conditioned = {
        "n_total_cls": n_all,
        "n_total_text": n_all,
        "n_correct_activity": int(sum(correct_mask)),
        "n_incorrect_activity": int(sum(incorrect_mask)),
        "rouge_l_correct": _mean_if_any(rouge_scores, correct_mask),
        "rouge_l_incorrect": _mean_if_any(rouge_scores, incorrect_mask),
        "bertscore_correct": _mean_if_any(bert_f1_list, correct_mask),
        "bertscore_incorrect": _mean_if_any(bert_f1_list, incorrect_mask),
        "meteor_correct": _mean_if_any(meteor_scores, correct_mask),
        "meteor_incorrect": _mean_if_any(meteor_scores, incorrect_mask),
    }
    return {
        "rouge_l": rouge_l,
        "bertscore_f1": bertscore,
        "meteor": meteor,
        "_rouge_scores": rouge_scores,
        "_bert_f1_list": bert_f1_list,
        "_meteor_scores": meteor_scores,
        "_correct_mask": correct_mask,
        "conditioned": conditioned,
    }


def compute_sensor_term_recall(
    results: List[ReasoningOutput],
    sensor_terms: List[str],
    text_metrics_out: Optional[dict] = None,
) -> dict:
    if not sensor_terms:
        nan = float("nan")
        result = {"sensor_term_recall": nan, "n_terms": 0, "n_scored": 0}
        if text_metrics_out is not None:
            text_metrics_out["conditioned"]["sensor_term_recall_correct"] = nan
            text_metrics_out["conditioned"]["sensor_term_recall_incorrect"] = nan
        return result
    terms_lower = [t.lower() for t in sensor_terms]

    def _str_sample(reasoning: str) -> float:
        text_lower = reasoning.lower()
        return sum((1 for t in terms_lower if t in text_lower)) / len(terms_lower)

    str_vals = [_str_sample(r.reasoning) for r in results]
    str_mean = float(np.mean(str_vals)) if str_vals else float("nan")
    if text_metrics_out is not None:
        correct_mask = text_metrics_out.get("_correct_mask", [True] * len(results))

        def _mean_if_any(vals, mask):
            sub = [v for v, m in zip(vals, mask) if m]
            return float(np.mean(sub)) if sub else float("nan")

        text_metrics_out["conditioned"]["sensor_term_recall_correct"] = _mean_if_any(
            str_vals, correct_mask
        )
        text_metrics_out["conditioned"]["sensor_term_recall_incorrect"] = _mean_if_any(
            str_vals, [not m for m in correct_mask]
        )
    return {
        "sensor_term_recall": str_mean,
        "n_terms": len(sensor_terms),
        "n_scored": len(str_vals),
    }


def build_ig_index(input_dir: Path, logger: logging.Logger) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    pattern = re.compile("_s(\\d+)\\.json$", re.IGNORECASE)
    for fpath in input_dir.glob("*.json"):
        if fpath.name in NON_SAMPLE_FILES:
            continue
        m = pattern.search(fpath.name)
        if m:
            index[str(int(m.group(1)))] = fpath
    logger.info(f"  IG index: {len(index)} files in {input_dir.name}")
    return index


def compute_gs_prose(
    results: List[ReasoningOutput], ig_index: Dict[str, Path], top_k: int = 3
) -> dict:
    nan_result = {
        "gs_recall": float("nan"),
        "gs_precision": float("nan"),
        "gs_f1": float("nan"),
        "n_scored": 0,
        "n_skipped": 0,
    }
    if not ig_index:
        return {
            "overall": nan_result,
            "correct": nan_result,
            "incorrect": nan_result,
            "note": "ig_index empty — GS-Prose skipped",
            "top_k": top_k,
        }

    def _get_important(ig_data: dict) -> set:
        regions = ig_data.get("high_attribution_regions", [])
        if regions:
            sorted_r = sorted(
                regions, key=lambda r: r.get("mean_importance", 0), reverse=True
            )
            return {
                r.get("sensor", "").lower().strip()
                for r in sorted_r[:top_k]
                if r.get("sensor", "").strip()
            }
        sr = ig_data.get("sensor_importance_ranking", [])
        if isinstance(sr, dict):
            items = sorted(sr.items(), key=lambda x: x[1], reverse=True)
            return {k.lower().strip() for k, _ in items[:top_k] if k.strip()}
        elif isinstance(sr, list):
            items = sorted(sr, key=lambda x: x.get("importance", 0), reverse=True)
            return {
                item.get("sensor", "").lower().strip()
                for item in items[:top_k]
                if item.get("sensor", "").strip()
            }
        return set()

    def _get_mentioned(reasoning: str, channels: List[str]) -> set:
        text_lower = reasoning.lower()
        return {ch.lower() for ch in channels if ch.lower() in text_lower}

    def _score_group(group: List[ReasoningOutput]) -> dict:
        gs_r, gs_p, gs_f = ([], [], [])
        n_skipped = 0
        for r in group:
            ig_path = ig_index.get(r.sample_id)
            if ig_path is None:
                n_skipped += 1
                continue
            try:
                with open(ig_path) as f:
                    ig_data = json.load(f)
            except Exception:
                n_skipped += 1
                continue
            important = _get_important(ig_data)
            important.discard("")
            if not important:
                n_skipped += 1
                continue
            channels = ig_data.get("analysis_metadata", {}).get("sensor_channels", [])
            mentioned = _get_mentioned(r.reasoning, channels)
            mentioned.discard("")
            if not mentioned:
                n_skipped += 1
                continue
            overlap = important & mentioned
            r_val = len(overlap) / len(important)
            p_val = len(overlap) / len(mentioned)
            f_val = 2 * p_val * r_val / (p_val + r_val) if p_val + r_val > 0 else 0.0
            gs_r.append(r_val)
            gs_p.append(p_val)
            gs_f.append(f_val)
        if not gs_f:
            return {
                "gs_recall": float("nan"),
                "gs_precision": float("nan"),
                "gs_f1": float("nan"),
                "n_scored": 0,
                "n_skipped": n_skipped,
            }
        return {
            "gs_recall": float(np.mean(gs_r)),
            "gs_precision": float(np.mean(gs_p)),
            "gs_f1": float(np.mean(gs_f)),
            "n_scored": len(gs_f),
            "n_skipped": n_skipped,
        }

    valid = results
    correct = [r for r in valid if r.correct]
    incorrect = [r for r in valid if not r.correct]
    return {
        "overall": _score_group(valid),
        "correct": _score_group(correct),
        "incorrect": _score_group(incorrect),
        "top_k": top_k,
    }


class CoTBaselineGenerator:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        logger: logging.Logger,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = MAX_MODEL_LEN,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.logger = logger
        logger.info(
            f"Loading vLLM: {model_name}  (tensor_parallel_size={tensor_parallel_size}  gpu_mem={gpu_memory_utilization}  max_model_len={max_model_len})"
        )
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_new_tokens,
            repetition_penalty=1.1,
        )
        _PROBE_FALLBACK_TOK = "Qwen/Qwen2.5-7B"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._chat_template_tok = self.tokenizer
        except Exception as _tok_err:
            logger.warning(
                f"Failed to load {model_name} tokenizer ({_tok_err}). Falling back to {_PROBE_FALLBACK_TOK} for length checking. Chat template will use raw user/assistant formatting."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                _PROBE_FALLBACK_TOK, trust_remote_code=True
            )
            self._chat_template_tok = self.tokenizer
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}<|im_start|>assistant\n"
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        logger.info("vLLM ready")

    def _format_sensor_block(self, raw: Dict, ds_cfg: DatasetConfig) -> str:
        lines = []
        for ch in ds_cfg.channel_order:
            if ch not in raw:
                raise KeyError(f"Channel '{ch}' missing (dataset={ds_cfg.name})")
            vals = raw[ch][:: ds_cfg.downsample_factor]
            lines.append(f"{ch}: [{', '.join((f'{v:.2f}' for v in vals))}]")
        return "\n".join(lines)

    def _build_prompt(self, sample_id: str, raw: Dict, ds_cfg: DatasetConfig) -> str:
        sensor_block = self._format_sensor_block(raw, ds_cfg)
        labels_str = ", ".join((f'"{l}"' for l in ds_cfg.valid_class_labels))
        n_ch = len(ds_cfg.channel_order)
        n_t = (
            ds_cfg.num_timesteps + ds_cfg.downsample_factor - 1
        ) // ds_cfg.downsample_factor
        ds_note = (
            f" Note: every {ds_cfg.downsample_factor}th sample is kept; temporal ordering is preserved."
            if ds_cfg.downsample_factor > 1
            else ""
        )
        return f'You are an expert biomechanist analyzing wearable sensor data. You will be shown the raw signal from a body-worn sensor system across {n_ch} channels, each with {n_t} timesteps in temporal order.{ds_note}\n\nEach channel name encodes sensor type (Acc = accelerometer, Gyro = gyroscope, Mag = magnetometer, ECG = electrocardiogram, LinAcc = linear acceleration with gravity removed, Total_Acc = raw acceleration including gravity, Quat/Orient = orientation, etc.) and where applicable the body location (Chest, Wrist, Ankle, Hand, Belt, Pocket, Hip, Back, Shoe, Knee, etc.) and axis (X, Y, Z).\n\n## Raw Sensor Window (Sample ID: {sample_id})\n\n{sensor_block}\n\n## Your Task\n\nThis sample is from exactly one of {len(ds_cfg.valid_class_labels)} activities: {labels_str}.\n\nYour job:\n1. Reason step by step: which channels show strong oscillation vs quiet; what gravity-aligned axes reveal about posture; whether periodic gait-like patterns or sustained static values appear; what gyroscope/rotational channels indicate; differences between body locations; how early / middle / late portions evolve.\n2. Decide which listed activity best explains the signal.\n3. Write a 100-150 word biomechanical explanation in flowing prose. Cover sensor-specific patterns (name the channels), sensor/location contrasts, phase-by-phase progression (early / middle / late), and biomechanical grounding (heel strikes, push-off, postural sway, weight transfer, trunk orientation, arm swing — whichever apply).\n4. End with EXACTLY this line on its own, no extra punctuation:\n\nPredicted Activity: <one of the listed labels, spelled exactly as listed>\n\n### Rules\nDo NOT include raw numeric values in your explanation. No bullet points, headers, or lists — one flowing prose paragraph then the "Predicted Activity:" line.\n\nBegin:\n'

    def _clean_reasoning(self, text: str) -> str:
        text = text.strip()
        text = re.sub("^#+\\s.*$", "", text, flags=re.MULTILINE).strip()
        text = re.sub(
            "(?i)\\n*\\s*predicted\\s*activity\\s*[:\\-].*$", "", text
        ).strip()
        text = re.sub("\\n+", " ", text).strip()
        return text

    def _load_batch(
        self, batch_files: List[Path], ds_cfg: DatasetConfig
    ) -> Optional[List[Dict]]:
        try:
            batch_data = []
            for json_path in batch_files:
                _, sample_id = parse_filename(json_path)
                with open(json_path, "r") as f:
                    data = json.load(f)
                raw = data.get("raw_sensor_data", {})
                if not raw:
                    self.logger.warning(
                        f"  {sample_id}: missing raw_sensor_data — skip"
                    )
                    continue
                true_activity = data.get("activity", "")
                if not true_activity or is_null_label(true_activity):
                    self.logger.info(f"  {sample_id}: null/missing activity — skip")
                    continue
                matched = _match_label(true_activity, ds_cfg.valid_class_labels)
                if matched is None:
                    self.logger.warning(
                        f"  {sample_id}: '{true_activity}' not in label set — skip"
                    )
                    continue
                missing = [c for c in ds_cfg.channel_order if c not in raw]
                if missing:
                    self.logger.warning(
                        f"  {sample_id}: missing channels {missing} — skip"
                    )
                    continue
                try:
                    raw_prompt = self._build_prompt(sample_id, raw, ds_cfg)
                except KeyError as e:
                    self.logger.warning(f"  {sample_id}: prompt build error {e} — skip")
                    continue
                chat_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_data.append(
                    {
                        "prompt": chat_prompt,
                        "sample_id": sample_id,
                        "true_activity": matched,
                        "classifier_confidence": float(data.get("confidence", 0.0)),
                        "ig_path": str(json_path),
                    }
                )
            return batch_data
        except Exception as e:
            self.logger.error(f"  Batch load error: {e}")
            return None

    def run_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        ds_cfg: DatasetConfig,
        samples_per_dataset: Optional[int] = None,
        sampling_strategy: str = "balanced",
        sampling_seed: int = 42,
        batch_size: int = 64,
    ) -> List[ReasoningOutput]:
        test_out_dir = output_dir / "reasoning_json" / "test"
        test_out_dir.mkdir(parents=True, exist_ok=True)
        ig_index = build_ig_index(input_dir, self.logger)
        _wb_run_name = (
            f"{self.model_name.split('/')[-1]}__{ds_cfg.name}__{output_dir.name}"
        )
        _wb = _init_wandb(
            _wb_run_name,
            {
                "model": self.model_name,
                "dataset": ds_cfg.name,
                "run_dir": str(output_dir),
                "n_channels": len(ds_cfg.channel_order),
                "num_timesteps": ds_cfg.num_timesteps,
                "downsample_factor": ds_cfg.downsample_factor,
                "n_labels": len(ds_cfg.valid_class_labels),
                "max_new_tokens": self.sampling_params.max_tokens,
                "batch_size": batch_size,
                "gs_top_k": GS_TOP_K,
            },
        )
        class_stats: Dict = defaultdict(
            lambda: {"tokens": [], "correct": 0, "total": 0, "anomalies": 0}
        )
        batch_latencies: List[float] = []
        samples_done = total_correct = total_parsed = total_parse_fails = 0
        written_ids: set = set()
        training_data: List[dict] = []
        all_results: List[ReasoningOutput] = []
        all_files = _discover_test_sample_jsons(input_dir, self.logger)
        self.logger.info(f"Discovered {len(all_files)} test JSONs in {input_dir}")
        processed_ids: set = set()
        for f in test_out_dir.glob("*.json"):
            m = re.match("(\\d+)", f.name)
            if m:
                processed_ids.add(str(int(m.group(1))))
        jsonl_path = output_dir / "test_dataset.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r") as f:
                for line in f:
                    try:
                        sid = json.loads(line).get("metadata", {}).get("sample_id")
                        if sid:
                            written_ids.add(str(sid))
                    except json.JSONDecodeError:
                        pass
        files_to_process, skipped_id = ([], 0)
        for f in all_files:
            _, sample_id = parse_filename(f)
            if sample_id == "0" and (not re.search("_s(\\d+)$", f.stem)):
                skipped_id += 1
                continue
            if sample_id not in processed_ids:
                files_to_process.append(f)
        if skipped_id:
            self.logger.warning(f"Skipped {skipped_id} files with unparseable IDs")
        files_to_process.sort(key=lambda f: f.stat().st_size)
        if samples_per_dataset is not None:
            files_to_process = _select_samples(
                files_to_process,
                samples_per_dataset,
                sampling_strategy,
                sampling_seed,
                self.logger,
            )
        if not files_to_process:
            self.logger.info("All test samples already processed.")
            return []
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size
        self.logger.info(
            f"To process: {len(files_to_process)}  already done: {len(processed_ids)}  batches: {total_batches}  batch_size: {batch_size}"
        )
        batch_ranges = [
            files_to_process[i : i + batch_size]
            for i in range(0, len(files_to_process), batch_size)
        ]
        executor = ThreadPoolExecutor(max_workers=2)
        next_future = executor.submit(self._load_batch, batch_ranges[0], ds_cfg)
        try:
            for batch_idx, batch_files in enumerate(batch_ranges):
                try:
                    batch_data = next_future.result()
                except Exception as e:
                    self.logger.error(f"  Batch {batch_idx + 1} prefetch crashed: {e}")
                    batch_data = None
                if batch_idx + 1 < len(batch_ranges):
                    next_future = executor.submit(
                        self._load_batch, batch_ranges[batch_idx + 1], ds_cfg
                    )
                if not batch_data:
                    self.logger.error(
                        f"  Batch {batch_idx + 1}/{total_batches} empty — skip"
                    )
                    continue
                self.logger.info(
                    f"Generating batch {batch_idx + 1}/{total_batches} ({len(batch_data)} samples)"
                )
                t0 = time.time()
                try:
                    prompts = [item["prompt"] for item in batch_data]
                    oversized_log = output_dir / "oversized_prompts.txt"
                    safe_idx = []
                    for i, (p, item) in enumerate(zip(prompts, batch_data)):
                        tok_len = len(self.tokenizer.encode(p))
                        if tok_len <= MAX_PROMPT_TOKENS:
                            safe_idx.append(i)
                        else:
                            self.logger.warning(
                                f"  Oversized: {item['sample_id']} ({tok_len} toks > {MAX_PROMPT_TOKENS}) — skip"
                            )
                            with open(oversized_log, "a") as olf:
                                olf.write(f"test\t{item['sample_id']}\t{tok_len}\n")
                    if not safe_idx:
                        self.logger.error(
                            f"  Batch {batch_idx + 1}: all oversized — skip"
                        )
                        continue
                    safe_prompts = [prompts[i] for i in safe_idx]
                    safe_batch = [batch_data[i] for i in safe_idx]
                    outputs = self.llm.generate(safe_prompts, self.sampling_params)
                    gen_time = time.time() - t0
                    batch_correct = batch_tokens = 0
                    for j, vllm_out in enumerate(outputs):
                        text = vllm_out.outputs[0].text
                        item = safe_batch[j]
                        sid = item["sample_id"]
                        true_act = item["true_activity"]
                        predicted = parse_predicted_activity(
                            text, ds_cfg.valid_class_labels, self.logger, sid
                        )
                        cleaned = self._clean_reasoning(text)
                        correct = predicted == true_act
                        result = ReasoningOutput(
                            sample_id=sid,
                            split="test",
                            dataset=ds_cfg.name,
                            true_activity=true_act,
                            predicted_activity=predicted,
                            classifier_confidence=item["classifier_confidence"],
                            correct=correct,
                            reasoning=cleaned,
                            raw_model_output=text,
                            generation_timestamp=datetime.now().isoformat(),
                            model_used=self.model_name,
                            downsample_factor=ds_cfg.downsample_factor,
                            ig_path=item["ig_path"],
                        )
                        result.to_json(test_out_dir / f"{sid}_reasoning.json")
                        training_data.append(result.to_training_format())
                        all_results.append(result)
                        toks = len(vllm_out.outputs[0].token_ids)
                        batch_tokens += toks
                        parsed = predicted != "unknown"
                        if not parsed:
                            total_parse_fails += 1
                        else:
                            total_parsed += 1
                            if correct:
                                total_correct += 1
                                batch_correct += 1
                        class_stats[true_act]["tokens"].append(toks)
                        class_stats[true_act]["total"] += 1
                        if correct:
                            class_stats[true_act]["correct"] += 1
                        if (
                            len(text.strip()) == 0
                            or toks < 50
                            or bool(REFUSAL_RE.search(text))
                            or (not parsed)
                        ):
                            class_stats[true_act]["anomalies"] += 1
                    batch_lat = time.time() - t0
                    batch_latencies.append(batch_lat)
                    samples_done += len(outputs)
                    remaining = len(files_to_process) - samples_done
                    avg_lat = sum(batch_latencies[-10:]) / len(batch_latencies[-10:])
                    eta = remaining / max(batch_size, 1) * avg_lat / 60
                    roll_acc = total_correct / total_parsed if total_parsed else 0.0
                    self.logger.info(
                        f"  [{batch_idx + 1}/{total_batches}] gen={gen_time:.1f}s wall={batch_lat:.1f}s tok={batch_tokens} batch_acc={batch_correct}/{len(outputs)} run_acc={roll_acc:.3f} pf={total_parse_fails} ETA={eta:.1f}min"
                    )
                    if _wb is not None:
                        try:
                            _wb.log(
                                {
                                    "batch/accuracy": batch_correct
                                    / max(len(outputs), 1),
                                    "batch/running_acc": roll_acc,
                                    "batch/tokens": batch_tokens,
                                    "batch/parse_fails": total_parse_fails,
                                    "batch/gen_time_s": gen_time,
                                    "batch_idx": batch_idx,
                                }
                            )
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.exception(f"  Batch {batch_idx + 1} crashed: {e}")
                    continue
        finally:
            executor.shutdown(wait=True)
            if training_data:
                with open(jsonl_path, "a") as f:
                    for item in training_data:
                        sid = item["metadata"]["sample_id"]
                        if sid not in written_ids:
                            f.write(json.dumps(item) + "\n")
                            written_ids.add(sid)
                self.logger.info(f"Wrote {len(training_data)} → {jsonl_path}")
            self.logger.info("Computing metrics ...")
            cls_m = compute_classification_metrics(
                all_results, ds_cfg.valid_class_labels
            )
            txt_m = compute_text_metrics(all_results)
            str_m = compute_sensor_term_recall(
                all_results, ds_cfg.sensor_terms, text_metrics_out=txt_m
            )
            gs_m = compute_gs_prose(all_results, ig_index, top_k=GS_TOP_K)
            txt_m_clean = {k: v for k, v in txt_m.items() if not k.startswith("_")}
            per_class_gen = {
                cls: {
                    "correct": s["correct"],
                    "total": s["total"],
                    "accuracy": s["correct"] / s["total"],
                    "avg_tokens": float(np.mean(s["tokens"]))
                    if s["tokens"]
                    else float("nan"),
                    "anomalies": s["anomalies"],
                }
                for cls, s in class_stats.items()
                if s["total"]
            }
            summary = {
                "dataset": ds_cfg.name,
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "downsample_factor": ds_cfg.downsample_factor,
                "num_channels": len(ds_cfg.channel_order),
                "num_timesteps": ds_cfg.num_timesteps,
                "num_valid_labels": len(ds_cfg.valid_class_labels),
                "valid_labels": ds_cfg.valid_class_labels,
                "samples_processed": samples_done,
                "n_cls_samples": cls_m["n_total"],
                "n_correct": cls_m["n_correct"],
                "n_out_of_vocab": cls_m["n_out_of_vocab"],
                "accuracy": cls_m["accuracy"],
                "macro_f1": cls_m["macro_f1"],
                "macro_precision": cls_m["macro_precision"],
                "macro_recall": cls_m["macro_recall"],
                "per_class": cls_m["per_class"],
                "confusion_matrix": cls_m["confusion_matrix"],
                "n_text_samples": len(all_results),
                "rouge_l": txt_m_clean["rouge_l"],
                "bertscore_f1": txt_m_clean["bertscore_f1"],
                "meteor": txt_m_clean["meteor"],
                "sensor_term_recall": str_m["sensor_term_recall"],
                "conditioned": txt_m_clean["conditioned"],
                "gs_prose": gs_m,
                "per_class_generation": per_class_gen,
            }
            with open(output_dir / "run_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info("=" * 70)
            self.logger.info(f"RESULTS: {ds_cfg.name}")
            self.logger.info(f"  Samples: {samples_done}")
            self.logger.info(
                f"  [Tier-1 CLS]  n={cls_m['n_total']}  acc={cls_m['accuracy']:.4f}  f1={cls_m['macro_f1']:.4f}  P={cls_m['macro_precision']:.4f}  R={cls_m['macro_recall']:.4f}  OOV={cls_m['n_out_of_vocab']}"
            )
            self.logger.info(
                f"  [Tier-1 TXT]  n={len(all_results)}  rougeL={txt_m_clean['rouge_l']:.4f}  bert={txt_m_clean['bertscore_f1']:.4f}  meteor={txt_m_clean['meteor']:.4f}  STR={str_m['sensor_term_recall']:.4f}"
            )
            cond = txt_m_clean.get("conditioned", {})
            self.logger.info(
                f"  [Conditioned] rougeL: corr={cond.get('rouge_l_correct', float('nan')):.4f}  incorr={cond.get('rouge_l_incorrect', float('nan')):.4f}  |  STR: corr={cond.get('sensor_term_recall_correct', float('nan')):.4f}  incorr={cond.get('sensor_term_recall_incorrect', float('nan')):.4f}"
            )
            g = gs_m
            for tag, grp in [
                ("overall", g.get("overall", {})),
                ("correct", g.get("correct", {})),
                ("incorrect", g.get("incorrect", {})),
            ]:
                self.logger.info(
                    f"  [GS-Prose  {tag:>9s}]  n={grp.get('n_scored', 0)}  R={grp.get('gs_recall', float('nan')):.4f}  P={grp.get('gs_precision', float('nan')):.4f}  F1={grp.get('gs_f1', float('nan')):.4f}  skip={grp.get('n_skipped', 0)}"
                )
            self.logger.info("=" * 70)
            if _wb is not None:
                try:
                    cond = txt_m_clean.get("conditioned", {})
                    _wb.log(
                        {
                            "cls/accuracy": cls_m["accuracy"],
                            "cls/macro_f1": cls_m["macro_f1"],
                            "cls/macro_precision": cls_m["macro_precision"],
                            "cls/macro_recall": cls_m["macro_recall"],
                            "cls/n_out_of_vocab": cls_m["n_out_of_vocab"],
                            "cls/n_total": cls_m["n_total"],
                            "text/rouge_l": txt_m_clean["rouge_l"],
                            "text/bertscore_f1": txt_m_clean["bertscore_f1"],
                            "text/meteor": txt_m_clean["meteor"],
                            "text/str": str_m["sensor_term_recall"],
                            "text/rouge_l_correct": cond.get("rouge_l_correct"),
                            "text/rouge_l_incorrect": cond.get("rouge_l_incorrect"),
                            "text/bert_correct": cond.get("bertscore_correct"),
                            "text/bert_incorrect": cond.get("bertscore_incorrect"),
                            "text/meteor_correct": cond.get("meteor_correct"),
                            "text/meteor_incorrect": cond.get("meteor_incorrect"),
                            "text/str_correct": cond.get("sensor_term_recall_correct"),
                            "text/str_incorrect": cond.get(
                                "sensor_term_recall_incorrect"
                            ),
                            "gs/overall_f1": gs_m.get("overall", {}).get("gs_f1"),
                            "gs/overall_recall": gs_m.get("overall", {}).get(
                                "gs_recall"
                            ),
                            "gs/overall_precision": gs_m.get("overall", {}).get(
                                "gs_precision"
                            ),
                            "gs/correct_f1": gs_m.get("correct", {}).get("gs_f1"),
                            "gs/incorrect_f1": gs_m.get("incorrect", {}).get("gs_f1"),
                            "gen/samples_processed": samples_done,
                        }
                    )
                    _finish_wandb(_wb)
                except Exception as e:
                    self.logger.warning(f"W&B final log failed: {e}")
                    _finish_wandb(_wb)
        return all_results


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _collect_scalar(run_summaries: List[dict], *key_path) -> List[float]:
    vals = []
    for s in run_summaries:
        node = s
        try:
            for k in key_path:
                node = node[k]
            v = _safe_float(node)
            if v is not None:
                vals.append(v)
        except (KeyError, TypeError):
            pass
    return vals


def _mean_std(vals: List[float]) -> dict:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        "n": len(vals),
    }


def compute_aggregate_summary(
    dataset_key: str, run_summaries: List[dict], logger: logging.Logger
) -> dict:

    def ms(*path):
        return _mean_std(_collect_scalar(run_summaries, *path))

    agg = {
        "dataset": dataset_key,
        "n_runs": len(run_summaries),
        "timestamp": datetime.now().isoformat(),
        "accuracy": ms("accuracy"),
        "macro_f1": ms("macro_f1"),
        "macro_precision": ms("macro_precision"),
        "macro_recall": ms("macro_recall"),
        "rouge_l": ms("rouge_l"),
        "bertscore_f1": ms("bertscore_f1"),
        "meteor": ms("meteor"),
        "sensor_term_recall": ms("sensor_term_recall"),
        "conditioned": {
            "rouge_l_correct": ms("conditioned", "rouge_l_correct"),
            "rouge_l_incorrect": ms("conditioned", "rouge_l_incorrect"),
            "bertscore_correct": ms("conditioned", "bertscore_correct"),
            "bertscore_incorrect": ms("conditioned", "bertscore_incorrect"),
            "meteor_correct": ms("conditioned", "meteor_correct"),
            "meteor_incorrect": ms("conditioned", "meteor_incorrect"),
            "sensor_term_recall_correct": ms(
                "conditioned", "sensor_term_recall_correct"
            ),
            "sensor_term_recall_incorrect": ms(
                "conditioned", "sensor_term_recall_incorrect"
            ),
        },
        "gs_prose": {
            "overall": {
                "gs_recall": ms("gs_prose", "overall", "gs_recall"),
                "gs_precision": ms("gs_prose", "overall", "gs_precision"),
                "gs_f1": ms("gs_prose", "overall", "gs_f1"),
            },
            "correct": {
                "gs_recall": ms("gs_prose", "correct", "gs_recall"),
                "gs_precision": ms("gs_prose", "correct", "gs_precision"),
                "gs_f1": ms("gs_prose", "correct", "gs_f1"),
            },
            "incorrect": {
                "gs_recall": ms("gs_prose", "incorrect", "gs_recall"),
                "gs_precision": ms("gs_prose", "incorrect", "gs_precision"),
                "gs_f1": ms("gs_prose", "incorrect", "gs_f1"),
            },
        },
    }
    all_classes = set()
    for s in run_summaries:
        all_classes.update(s.get("per_class", {}).keys())
    agg["per_class_f1"] = {
        cls: ms("per_class", cls, "f1") for cls in sorted(all_classes)
    }
    logger.info(f"  AGGREGATE [{dataset_key}]  runs={len(run_summaries)}")
    logger.info(
        f"    acc={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f}  f1={agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f}"
    )
    logger.info(
        f"    rougeL={agg['rouge_l']['mean']:.4f}±{agg['rouge_l']['std']:.4f}  bert={agg['bertscore_f1']['mean']:.4f}±{agg['bertscore_f1']['std']:.4f}  meteor={agg['meteor']['mean']:.4f}±{agg['meteor']['std']:.4f}  STR={agg['sensor_term_recall']['mean']:.4f}±{agg['sensor_term_recall']['std']:.4f}"
    )
    gs_o = agg["gs_prose"]["overall"]
    logger.info(
        f"    GS-Prose overall: R={gs_o['gs_recall']['mean']:.4f}±{gs_o['gs_recall']['std']:.4f}  P={gs_o['gs_precision']['mean']:.4f}±{gs_o['gs_precision']['std']:.4f}  F1={gs_o['gs_f1']['mean']:.4f}±{gs_o['gs_f1']['std']:.4f}"
    )
    if WANDB_ENABLED:
        try:
            import wandb

            agg_run = wandb.init(
                project=WANDB_PROJECT,
                name=f"{dataset_key}__aggregate__{len(run_summaries)}runs",
                job_type="aggregate",
                resume="allow",
            )
            flat: dict = {}
            for metric in [
                "accuracy",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "rouge_l",
                "bertscore_f1",
                "meteor",
                "sensor_term_recall",
            ]:
                v = agg.get(metric, {})
                if isinstance(v, dict):
                    flat[f"agg/{metric}_mean"] = v.get("mean")
                    flat[f"agg/{metric}_std"] = v.get("std")
            gs_o = agg.get("gs_prose", {}).get("overall", {})
            for k in ["gs_f1", "gs_recall", "gs_precision"]:
                v = gs_o.get(k, {})
                if isinstance(v, dict):
                    flat[f"agg/gs_{k}_mean"] = v.get("mean")
                    flat[f"agg/gs_{k}_std"] = v.get("std")
            for ckey in [
                "rouge_l_correct",
                "rouge_l_incorrect",
                "bertscore_correct",
                "bertscore_incorrect",
                "sensor_term_recall_correct",
                "sensor_term_recall_incorrect",
            ]:
                v = agg.get("conditioned", {}).get(ckey, {})
                if isinstance(v, dict):
                    flat[f"agg/cond_{ckey}_mean"] = v.get("mean")
                    flat[f"agg/cond_{ckey}_std"] = v.get("std")
            agg_run.log(flat)
            agg_run.finish()
            logger.info(f"  W&B aggregate run logged: {agg_run.url}")
        except Exception as e:
            logger.warning(f"  W&B aggregate log failed: {e}")
    return agg


def _init_wandb(run_name: str, config: dict) -> Optional[object]:
    if not WANDB_ENABLED:
        return None
    try:
        import wandb

        run = wandb.init(
            project=WANDB_PROJECT, name=run_name, config=config, resume="allow"
        )
        print(f"[wandb] ┌──────────────────────────────────────────────────────┐")
        print(f"[wandb] │  Run : {run_name:<46s}│")
        print(f"[wandb] │  URL : {run.url:<46s}│")
        print(f"[wandb] └──────────────────────────────────────────────────────┘")
        return run
    except Exception as e:
        print(f"[wandb] WARNING: failed to init ({e}) — continuing without W&B")
        return None


def _finish_wandb(run: Optional[object]):
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CoT baseline reasoning generation")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--tensor_parallel", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_model_len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--gpu_mem",
        type=float,
        default=0.92,
        help="vLLM gpu_memory_utilization (default 0.92; use 0.95 for 70B)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Override output root. Default: <project>/cot/<model_short>",
    )
    parser.add_argument("--n_runs", type=int, default=N_RUNS)
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLES_PER_DATASET,
        help="Samples per dataset per run. None = full test split.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset keys to run. Default: all 7.",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable W&B logging entirely."
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    global WANDB_ENABLED
    if args.no_wandb:
        WANDB_ENABLED = False
    if args.output_root:
        out_root = Path(args.output_root)
    else:
        model_short = (
            args.model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        )
        out_root = PROJECT_ROOT / "cot" / model_short
    out_root.mkdir(parents=True, exist_ok=True)
    root_log = setup_logger(out_root, f"cot_baseline_{out_root.name}_root")
    root_log.info(f"Model:           {args.model_id}")
    root_log.info(
        f"Tensor parallel: {args.tensor_parallel}  batch_size: {args.batch_size}"
    )
    root_log.info(
        f"max_model_len:   {args.max_model_len}  max_new_tokens:  {args.max_new_tokens}"
    )
    root_log.info(f"gpu_mem_util:    {args.gpu_mem}")
    root_log.info(f"Output root:     {out_root}")
    root_log.info(f"N_RUNS:          {args.n_runs}")
    root_log.info(f"Samples/ds:      {args.samples or 'ALL'}")
    root_log.info(f"GS top-k:        {GS_TOP_K}")
    if args.datasets:
        active = [d.strip() for d in args.datasets.split(",")]
        unknown = [k for k in active if k not in INPUT_DIRS]
        if unknown:
            root_log.error(f"Unknown dataset keys: {unknown}")
            return
        selected = [(k, INPUT_DIRS[k]) for k in active]
    elif ACTIVE_DATASETS:
        selected = [(k, INPUT_DIRS[k]) for k in ACTIVE_DATASETS]
    else:
        selected = list(INPUT_DIRS.items())
    root_log.info(f"Datasets ({len(selected)}): {[k for k, _ in selected]}")
    valid_inputs: List[Tuple[str, Path]] = []
    for key, p in selected:
        if not p.exists() or not p.is_dir():
            root_log.error(f"[{key}] Missing or not a dir: {p} — SKIPPING")
            continue
        valid_inputs.append((key, p))
    if not valid_inputs:
        root_log.error("No valid input dirs. Aborting.")
        return
    root_log.info("=" * 70)
    root_log.info("PHASE 1: probing dataset configs")
    _PROBE_FALLBACK_TOK = "Qwen/Qwen2.5-7B"
    try:
        probe_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception as _tok_err:
        print(
            f"[tokenizer] WARNING: failed to load {args.model_id} tokenizer ({_tok_err}). Falling back to {_PROBE_FALLBACK_TOK} for probe."
        )
        probe_tok = AutoTokenizer.from_pretrained(
            _PROBE_FALLBACK_TOK, trust_remote_code=True
        )
    dataset_plan: List[Tuple[str, Path, DatasetConfig]] = []
    for key, input_dir in valid_inputs:
        (out_root / key).mkdir(parents=True, exist_ok=True)
        ds_log = setup_logger(out_root / key, f"cot_baseline_{key}")
        ds_cfg = discover_dataset_config(key, input_dir, probe_tok, ds_log)
        if ds_cfg is None:
            root_log.error(f"[{key}] Config probe failed — SKIPPING")
            continue
        dataset_plan.append((key, input_dir, ds_cfg))
        root_log.info(
            f"  {key}: ch={len(ds_cfg.channel_order)} t={ds_cfg.num_timesteps} ds={ds_cfg.downsample_factor}x labels={len(ds_cfg.valid_class_labels)}"
        )
    del probe_tok
    if not dataset_plan:
        root_log.error("No usable datasets. Aborting.")
        return
    root_log.info("=" * 70)
    root_log.info(f"PHASE 2: {args.n_runs} runs × {len(dataset_plan)} datasets")
    generator = CoTBaselineGenerator(
        model_name=args.model_id,
        tensor_parallel_size=args.tensor_parallel,
        logger=root_log,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
    )
    all_run_summaries: Dict[str, List[dict]] = defaultdict(list)
    for run_idx in range(args.n_runs):
        run_seed = SAMPLING_SEED + run_idx
        root_log.info("=" * 70)
        root_log.info(f"RUN {run_idx + 1}/{args.n_runs}  (seed={run_seed})")
        root_log.info("=" * 70)
        for key, input_dir, ds_cfg in dataset_plan:
            run_out_dir = out_root / key / f"run_{run_idx}"
            run_out_dir.mkdir(parents=True, exist_ok=True)
            ds_log = setup_logger(run_out_dir, f"cot_baseline_{key}_run{run_idx}")
            orig_log = generator.logger
            generator.logger = ds_log
            ds_log.info(f"RUN {run_idx + 1}/{args.n_runs} — {key}  seed={run_seed}")
            root_log.info(f"==> run {run_idx + 1}/{args.n_runs}: {key}")
            try:
                generator.run_dataset(
                    input_dir=input_dir,
                    output_dir=run_out_dir,
                    ds_cfg=ds_cfg,
                    samples_per_dataset=args.samples,
                    sampling_strategy=SAMPLING_STRATEGY,
                    sampling_seed=run_seed,
                    batch_size=args.batch_size,
                )
                summary_path = run_out_dir / "run_summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        all_run_summaries[key].append(json.load(f))
                else:
                    root_log.warning(
                        f"  run_summary.json missing for {key} run {run_idx}"
                    )
            except Exception as e:
                ds_log.exception(f"FATAL on {key} run {run_idx}: {e}")
                root_log.exception(f"FATAL on {key} run {run_idx}: {e}")
            finally:
                generator.logger = orig_log
            root_log.info(f"<== done run {run_idx + 1}/{args.n_runs}: {key}")
    root_log.info("=" * 70)
    root_log.info("PHASE 3: aggregating mean ± std across runs")
    all_agg: Dict[str, dict] = {}
    for key, run_summaries in all_run_summaries.items():
        if not run_summaries:
            root_log.warning(f"  {key}: no summaries to aggregate")
            continue
        ds_log = logging.getLogger(f"cot_baseline_{key}")
        agg = compute_aggregate_summary(key, run_summaries, ds_log)
        all_agg[key] = agg
        agg_path = out_root / key / "aggregate_summary.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        root_log.info(f"  Saved aggregate → {agg_path}")
    combined_path = out_root / "all_datasets_aggregate.json"
    with open(combined_path, "w") as f:
        json.dump(all_agg, f, indent=2)
    root_log.info("=" * 70)
    root_log.info("ALL RUNS COMPLETE")
    root_log.info(
        f"Per-dataset aggregates : {out_root}/<dataset>/aggregate_summary.json"
    )
    root_log.info(f"Combined aggregate     : {combined_path}")
    root_log.info("=" * 70)


if __name__ == "__main__":
    main()
