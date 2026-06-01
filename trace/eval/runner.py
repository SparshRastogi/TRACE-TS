"""Evaluation orchestrator: compute_all_metrics, CSV export, W&B logging, run_evaluate."""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from trace.eval.generate import (
    run_generation, extract_predicted_activity,
    _evaluate_json_dir, _csvs_dir,
)
from trace.eval.metrics.graph_parser import parse_reasoning_graph
from trace.eval.metrics.graph_metrics import (
    compute_psr, compute_gs_corpus, compute_rcv_corpus, compute_gc,
    compute_node_error_rates, _load_ig_json_index,
)
from trace.eval.metrics.classification import compute_classification_metrics
from trace.eval.metrics.text_quality import compute_text_quality_metrics
from trace.training.logging_utils import _get_or_create_wandb_run
from trace.utils.naming import _run_prefix


def compute_all_metrics(results: list, cfg: dict, tokenizer=None, llm=None,
                        device: str = "cpu") -> dict:
    """Compute all evaluation metrics.

    results_all (ALL samples)   → classification metrics (accuracy, F1, confusion matrix).
    results_gt  (GT-only subset) → text quality + Tier 2 graph metrics.
    Samples without GT reasoning are excluded from text/graph metrics.
    """
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    activity_classes = [c.lower().strip() for c in cfg["activity_classes"]]
    sensor_terms     = cfg["sensor_terms"]
    skip_model       = cfg.get("skip_model_metrics", False)
    all_classes      = activity_classes + ["unknown"]

    generated_all = [r["generated"] for r in results]
    gt_acts_all   = [r["gt_activity"] for r in results]
    pred_acts_all = [extract_predicted_activity(g, activity_classes) for g in generated_all]
    for r, pa in zip(results, pred_acts_all):
        r["predicted_activity"] = pa

    has_gt = [bool(r.get("ground_truth_reasoning", "").strip()) for r in results]
    results_gt = [r for r, h in zip(results, has_gt) if h]
    n_all = len(results)
    n_gt  = len(results_gt)
    print(f"[eval] Classification metrics: {n_all} samples | "
          f"Text/Graph metrics: {n_gt} samples (excl {n_all - n_gt} without reasoning GT)")

    cls_metrics = compute_classification_metrics(gt_acts_all, pred_acts_all, all_classes)

    if n_gt == 0:
        print("[eval] WARNING: No samples have reasoning GT — text/graph metrics all NaN.")
        tq = {
            "rouge_l": float("nan"), "meteor": float("nan"),
            "sensor_term_recall": float("nan"), "bertscore_f1": float("nan"),
            "_rouge_scores": [], "_meteor_scores": [], "_str_vals": [], "_bert_f1_list": [],
        }
        pred_acts_gt = gt_acts_gt = []
    else:
        generated_gt  = [r["generated"] for r in results_gt]
        references_gt = [r["ground_truth_reasoning"] for r in results_gt]
        pred_acts_gt  = [r["predicted_activity"] for r in results_gt]
        gt_acts_gt    = [r["gt_activity"] for r in results_gt]
        tq = compute_text_quality_metrics(
            generated_gt, references_gt, sensor_terms, device=device, skip_model=skip_model)

    def _mean_if_any(vals, mask):
        sub = [v for v, m in zip(vals, mask) if m and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(sub)) if sub else float("nan")

    if n_gt > 0:
        correct_mask_gt   = [p == g for p, g in zip(pred_acts_gt, gt_acts_gt)]
        incorrect_mask_gt = [not m for m in correct_mask_gt]
        conditioned = {
            "n_total_cls":            n_all,
            "n_total_text_gt":        n_gt,
            "n_correct_activity":     int(sum(p == g for p, g in zip(pred_acts_all, gt_acts_all))),
            "n_incorrect_activity":   int(sum(p != g for p, g in zip(pred_acts_all, gt_acts_all))),
            "n_correct_activity_gt":  int(sum(correct_mask_gt)),
            "n_incorrect_activity_gt": int(sum(incorrect_mask_gt)),
            "rouge_l_correct":   _mean_if_any(tq["_rouge_scores"], correct_mask_gt),
            "rouge_l_incorrect": _mean_if_any(tq["_rouge_scores"], incorrect_mask_gt),
            "bertscore_correct":   _mean_if_any(tq["_bert_f1_list"], correct_mask_gt),
            "bertscore_incorrect": _mean_if_any(tq["_bert_f1_list"], incorrect_mask_gt),
            "meteor_correct":   _mean_if_any(tq["_meteor_scores"], correct_mask_gt),
            "meteor_incorrect": _mean_if_any(tq["_meteor_scores"], incorrect_mask_gt),
            "sensor_term_recall_correct":   _mean_if_any(tq["_str_vals"], correct_mask_gt),
            "sensor_term_recall_incorrect": _mean_if_any(tq["_str_vals"], incorrect_mask_gt),
        }
    else:
        conditioned = {
            "n_total_cls": n_all, "n_total_text_gt": 0,
            "n_correct_activity": int(sum(p == g for p, g in zip(pred_acts_all, gt_acts_all))),
            "n_incorrect_activity": int(sum(p != g for p, g in zip(pred_acts_all, gt_acts_all))),
            "n_correct_activity_gt": 0, "n_incorrect_activity_gt": 0,
        }

    print("[eval] Parsing reasoning graphs for Tier 2 metrics...")
    graphs_gen_gt = [parse_reasoning_graph(r.get("generated", "")) for r in results_gt]
    graphs_ref_gt = [parse_reasoning_graph(r.get("ground_truth_reasoning", "")) for r in results_gt]
    for r in results:
        r["_gen_graph"] = parse_reasoning_graph(r.get("generated", ""))
    for r, gg in zip(results_gt, graphs_gen_gt):
        r["_gen_graph"] = gg

    psr     = compute_psr(graphs_gen_gt)
    ref_psr = compute_psr(graphs_ref_gt)

    ig_index = _load_ig_json_index(cfg.get("ig_json_dir"))
    gs  = compute_gs_corpus(results_gt, ig_index, top_k=cfg.get("gs_top_k", 3))
    rcv = compute_rcv_corpus(graphs_gen_gt)
    gc  = compute_gc(graphs_gen_gt, graphs_ref_gt,
                     [r["gt_activity"] for r in results_gt])
    node_errors = compute_node_error_rates(
        graphs_gen_gt, cfg["activity_classes"], cfg["valid_sensors"])

    return {
        "n_cls_samples":  n_all,
        "n_text_samples": n_gt,
        **{k: v for k, v in cls_metrics.items()},
        "rouge_l": tq["rouge_l"], "bertscore_f1": tq["bertscore_f1"],
        "meteor": tq["meteor"], "sensor_term_recall": tq["sensor_term_recall"],
        "conditioned": conditioned,
        "psr": psr, "ref_psr": ref_psr,
        "gs": gs, "rcv": rcv, "gc": gc,
        "node_errors": node_errors,
    }


def log_eval_to_wandb(metrics: dict, results: list, cfg: dict):
    wb_r = _get_or_create_wandb_run(cfg)
    if wb_r is None:
        return
    try:
        import wandb as _wandb

        def _safe(v, d=4):
            return round(v, d) if isinstance(v, (int, float)) and v == v else None

        for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall",
                  "rouge_l", "bertscore_f1", "meteor", "sensor_term_recall"):
            v = _safe(metrics.get(k))
            if v is not None:
                wb_r.summary[f"eval/{k}"] = v
        wb_r.summary["eval/psr"] = _safe(metrics.get("psr"))
        for sub_k in ("gs_f1", "gs_recall", "gs_precision"):
            wb_r.summary[f"eval/{sub_k}"] = _safe(metrics.get("gs", {}).get(sub_k))
        for sub_k in ("rcv_structural", "rcv_semantic", "rcv_combined"):
            wb_r.summary[f"eval/{sub_k}"] = _safe(metrics.get("rcv", {}).get(sub_k))
        wb_r.summary["eval/gc_mean"] = _safe(metrics.get("gc", {}).get("gc_mean"))
        ne = metrics.get("node_errors", {})
        for ne_k in ("obs_parse_rate", "obs_sensor_valid_rate", "obs_temporal_rate",
                     "obs_confidence_rate", "obs_coverage_rate",
                     "inf_present_rate", "inf_structural_rate", "inf_semantic_rate",
                     "inf_obs_coverage", "syn_present_rate", "syn_ref_rate",
                     "syn_activity_valid_rate"):
            wb_r.summary[f"eval/{ne_k}"] = _safe(ne.get(ne_k))
        wb_r.summary["eval/n_samples"] = len(results)
        wb_r.summary["eval/n_cls_samples"]  = metrics.get("n_cls_samples", len(results))
        wb_r.summary["eval/n_text_samples"] = metrics.get("n_text_samples", len(results))
        pc = metrics["per_class"]
        pc_table = _wandb.Table(
            columns=["Activity", "Precision", "Recall", "F1", "Support"],
            data=[[cls, _safe(pc[cls]["precision"]), _safe(pc[cls]["recall"]),
                   _safe(pc[cls]["f1"]), pc[cls]["support"]] for cls in pc])
        wb_r.log({"eval/per_class_table": pc_table})
        wb_r.log({"eval/per_class_f1_chart": _wandb.plot.bar(
            pc_table, "Activity", "F1", title="Per-Class F1")})
        cm_labels = metrics["confusion_matrix"]["labels"]
        cm_matrix = metrics["confusion_matrix"]["matrix"]
        cm_labels_norm = [l.lower().strip() for l in cm_labels]
        try:
            cm_table = _wandb.Table(columns=["true \\ pred"] + cm_labels_norm)
            for gi, row in enumerate(cm_matrix):
                cm_table.add_data(cm_labels_norm[gi], *[int(v) for v in row])
            wb_r.log({"eval/confusion_matrix": cm_table})
        except Exception as e:
            print(f"[wandb] WARNING: confusion matrix logging failed ({e})")
        cond = metrics["conditioned"]
        cond_table = _wandb.Table(
            columns=["Condition", "N (text GT)", "ROUGE-L", "BERTScore", "METEOR", "STR"],
            data=[
                ["Correct", cond.get("n_correct_activity_gt", 0),
                 _safe(cond.get("rouge_l_correct")), _safe(cond.get("bertscore_correct")),
                 _safe(cond.get("meteor_correct")), _safe(cond.get("sensor_term_recall_correct"))],
                ["Incorrect", cond.get("n_incorrect_activity_gt", 0),
                 _safe(cond.get("rouge_l_incorrect")), _safe(cond.get("bertscore_incorrect")),
                 _safe(cond.get("meteor_incorrect")), _safe(cond.get("sensor_term_recall_incorrect"))],
            ])
        wb_r.log({"eval/conditioned_table": cond_table})
        sampled = [r for i, r in enumerate(results) if i % 50 == 0 or i == len(results) - 1]
        sample_data = []
        for r in sampled:
            gg = r.get("_gen_graph", parse_reasoning_graph(r.get("generated", "")))
            sample_data.append([r.get("sample_id", ""), r["gt_activity"], r["predicted_activity"],
                "Y" if r["gt_activity"] == r["predicted_activity"] else "N",
                gg["n_observations"], gg["n_inferences"],
                "Y" if not gg["parse_failed"] else "N", r.get("generated", "")[:400]])
        wb_r.log({"eval/sample_table": _wandb.Table(
            columns=["ID", "GT", "Pred", "OK", "N_Obs", "N_Inf", "Parse", "Generated"],
            data=sample_data)})
        gc_data = metrics.get("gc", {})
        if gc_data.get("per_class_gc"):
            gc_rows = [[cls, gc_val, gc_data["class_ref_obs_means"].get(cls, 0.0)]
                       for cls, gc_val in gc_data["per_class_gc"].items()]
            gc_table = _wandb.Table(columns=["Activity", "GC Mean", "Ref Obs Mean"], data=gc_rows)
            wb_r.log({"eval/per_class_gc_table": gc_table})
        print(f"[logging] W&B eval logged: {len(results)} samples")
    except Exception as e:
        import traceback
        print(f"[logging] WARNING: W&B eval failed ({e})")
        traceback.print_exc()


def _save_metrics_csv(metrics: dict, cfg: dict, csv_path: str):
    def _f(v, d=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.{d}f}" if isinstance(v, float) else str(v)

    rows = []
    rows.append(["## RUN CONFIG", ""])
    for k in ["dataset", "model_id", "data_version", "n_tokens", "adapter_rank",
               "adapter_layers", "n_cls_samples", "n_text_samples"]:
        v = metrics.get(k) if k in ("n_cls_samples", "n_text_samples") else cfg.get(k, "")
        rows.append([k, str(v) if v is not None else ""])
    rows.append(["", ""])

    rows.append([f"## TIER 1A — CLASSIFICATION  (n={metrics.get('n_cls_samples','?')})", ""])
    rows.append(["metric", "value"])
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        rows.append([k, _f(metrics.get(k))])
    rows.append(["", ""])

    rows.append([f"## TIER 1B — TEXT QUALITY  (n={metrics.get('n_text_samples','?')})", ""])
    rows.append(["metric", "value"])
    for k in ("rouge_l", "bertscore_f1", "meteor", "sensor_term_recall"):
        rows.append([k, _f(metrics.get(k))])
    rows.append(["", ""])

    c = metrics.get("conditioned", {})
    rows.append(["## CONDITIONED ON CORRECT/INCORRECT CLASSIFICATION", ""])
    rows.append(["condition", "n", "rouge_l", "bertscore", "meteor", "sensor_term_recall"])
    rows.append(["correct",
                 str(c.get("n_correct_activity_gt", "?")),
                 _f(c.get("rouge_l_correct")), _f(c.get("bertscore_correct")),
                 _f(c.get("meteor_correct")), _f(c.get("sensor_term_recall_correct"))])
    rows.append(["incorrect",
                 str(c.get("n_incorrect_activity_gt", "?")),
                 _f(c.get("rouge_l_incorrect")), _f(c.get("bertscore_incorrect")),
                 _f(c.get("meteor_incorrect")), _f(c.get("sensor_term_recall_incorrect"))])
    rows.append(["", ""])

    ne = metrics.get("node_errors", {})
    n_parsed = ne.get("_n_parsed", "?")
    n_ne_total = ne.get("_n_total", "?")
    rows.append([f"## TIER 2 — GRAPH-LEVEL METRICS  (n={metrics.get('n_text_samples','?')})", ""])
    rows.append(["metric", "value"])
    rows.append(["ref_psr", _f(metrics.get("ref_psr"))])
    gs = metrics.get("gs", {})
    rows.append(["gs_f1",        _f(gs.get("gs_f1"))])
    rows.append(["gs_precision", _f(gs.get("gs_precision"))])
    rows.append(["gs_recall",    _f(gs.get("gs_recall"))])
    rows.append(["gs_n_scored",  str(gs.get("n_scored", "N/A"))])
    rcv = metrics.get("rcv", {})
    rows.append(["rcv_structural", _f(rcv.get("rcv_structural"))])
    rows.append(["rcv_semantic",   _f(rcv.get("rcv_semantic"))])
    rows.append(["rcv_combined",   _f(rcv.get("rcv_combined"))])
    gc = metrics.get("gc", {})
    rows.append(["gc_mean", _f(gc.get("gc_mean"))])
    rows.append(["", ""])

    rows.append(["## PER-CLASS METRICS  (classification, all samples)", ""])
    rows.append(["class", "precision", "recall", "f1", "support"])
    per_class_data = metrics.get("per_class", {})
    for cls, vals in per_class_data.items():
        if cls == "unknown":
            continue
        rows.append([cls, _f(vals["precision"]), _f(vals["recall"]),
                     _f(vals["f1"]), str(vals["support"])])
    if "unknown" in per_class_data:
        vals = per_class_data["unknown"]
        rows.append(["unknown", _f(vals.get("precision")), _f(vals.get("recall")),
                     _f(vals.get("f1")), str(vals.get("support", 0))])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[evaluate] Saved CSV  -> {csv_path}")


def _save_combined_metrics_csv(all_metrics: list, cfg: dict, csv_path: str):
    def _f(v, d=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.{d}f}" if isinstance(v, float) else str(v)

    def _flatten(m: dict, prefix="") -> dict:
        out = {}
        skip_keys = {"per_class", "confusion_matrix", "conditioned",
                     "gs", "rcv", "gc", "node_errors"}
        for k, v in m.items():
            if k.startswith("_") or k in skip_keys:
                continue
            full_k = f"{prefix}{k}" if prefix else k
            if isinstance(v, (int, float)):
                out[full_k] = v
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if not sk.startswith("_") and isinstance(sv, (int, float)):
                        out[f"{full_k}.{sk}"] = sv
        return out

    flat_runs = [_flatten(m) for m in all_metrics]
    all_keys = list(flat_runs[0].keys())
    n_runs = len(flat_runs)
    rows = []
    rows.append(["## COMBINED METRICS — mean ± std across runs", ""])
    rows.append([f"n_runs = {n_runs}", f"dataset = {cfg.get('dataset','')}",
                 f"model = {cfg.get('model_id','')}"])
    rows.append([""])
    header = ["metric"] + [f"run_{i+1}" for i in range(n_runs)] + ["mean", "std"]
    rows.append(header)
    for key in all_keys:
        vals = [r.get(key, float("nan")) for r in flat_runs]
        valid = [v for v in vals if isinstance(v, float) and not math.isnan(v)]
        mean_v = float(np.mean(valid)) if valid else float("nan")
        std_v  = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        rows.append([key] + [_f(v) for v in vals] + [_f(mean_v), _f(std_v)])
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[evaluate] Saved combined CSV -> {csv_path}")


def run_evaluate(cfg: dict):
    import os
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    n_runs = cfg.get("n_eval_runs", 1)
    if local_rank == 0:
        print(f"[evaluate] Device: {device}  |  n_tokens={cfg['n_tokens']}  "
              f"adapter_rank={cfg['adapter_rank']}  n_eval_runs={n_runs}"
              + (f"  world_size={world_size}" if is_ddp else ""))

    # In DDP mode each rank loads its own model inside run_generation.
    # In single-process mode, pre-load once and reuse across n_runs.
    tokenizer, llm = None, None
    if not is_ddp and not cfg.get("skip_model_metrics", False):
        from trace.model.backbone import load_model_and_tokenizer
        tokenizer, llm = load_model_and_tokenizer(cfg["model_id"], device_map="auto")
        llm.eval()

    all_metrics = []

    for run_idx in range(1, n_runs + 1):
        if local_rank == 0:
            print(f"\n{'='*65}")
            print(f"[evaluate] RUN {run_idx}/{n_runs}")
            print(f"{'='*65}")
            print(f"[evaluate] Run {run_idx}: running generation...")
        run_cfg = dict(cfg)
        run_cfg["_run_idx"] = run_idx
        inf_path = run_generation(run_cfg, tokenizer=tokenizer, llm=llm)

        # Non-rank-0 DDP processes: generation shard is done; skip metrics entirely
        if inf_path is None:
            continue

        with open(inf_path) as f:
            results = json.load(f)

        limit = cfg["num_eval_samples"]
        if limit > 0:
            results = results[:limit]

        print(f"[evaluate] Run {run_idx}: computing metrics on {len(results)} samples...")
        metrics = compute_all_metrics(results, cfg, tokenizer, llm, device)
        all_metrics.append(metrics)

        print(f"\n  --- Run {run_idx} Results ---")
        for k in ("accuracy", "macro_f1", "rouge_l", "bertscore_f1"):
            v = metrics.get(k, float("nan"))
            print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) and v == v else f"  {k:25s}: N/A")
        gs = metrics.get("gs", {})
        print(f"  {'GS-F1':25s}: {gs.get('gs_f1', float('nan')):.4f}")
        print(f"  {'PSR':25s}: {metrics.get('psr', float('nan')):.4f}")

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = _evaluate_json_dir(cfg) / f"{_run_prefix(cfg)}_evaluate_run{run_idx}_{ts}.json"
        per_sample = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        output = {
            "run": run_idx, "n_runs": n_runs,
            "config": {k: str(v) for k, v in cfg.items()
                       if k not in ("activity_classes", "sensor_terms", "valid_sensors")},
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
            "per_sample": per_sample,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"[evaluate] Saved run {run_idx} JSON -> {out_path}")

        csv_path = str(_csvs_dir(cfg) / f"{_run_prefix(cfg)}_metrics_run{run_idx}_{ts}.csv")
        _save_metrics_csv(metrics, cfg, csv_path)

        if run_idx == 1 and cfg.get("ig_json_dir") is not None:
            try:
                from trace.viz.fig3 import generate_attention_attribution_figure
                print(f"[evaluate] Run 1: generating Figure 3 attention-attribution plot...")
                generate_attention_attribution_figure(cfg)
            except Exception as e:
                print(f"[evaluate] WARNING: Figure 3 failed ({e})")

        log_eval_to_wandb(metrics, results, cfg)

    # Non-rank-0 DDP processes have no metrics — skip aggregate summary
    if local_rank != 0:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()
        return []

    print(f"\n{'='*65}")
    print(f"[evaluate] AGGREGATE RESULTS ({n_runs} runs)")
    print(f"{'='*65}")

    def _agg(key, nested=None):
        vals = []
        for m in all_metrics:
            v = m.get(nested, m).get(key) if nested else m.get(key)
            if isinstance(v, float) and not np.isnan(v):
                vals.append(v)
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    print(f"\n  {'Metric':25s}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*45}")
    for k in ("accuracy", "macro_f1", "rouge_l", "bertscore_f1"):
        mean_v, std_v = _agg(k)
        fmt = f"{mean_v:.4f} ± {std_v:.4f}" if not np.isnan(mean_v) else "N/A"
        print(f"  {k:25s}: {fmt}")
    for sk in ("gs_f1", "gs_recall", "gs_precision"):
        mean_v, std_v = _agg(sk, nested="gs")
        fmt = f"{mean_v:.4f} ± {std_v:.4f}" if not np.isnan(mean_v) else "N/A"
        print(f"  {'GS-'+sk.split('_')[1]:25s}: {fmt}")
    psr_mean, psr_std = _agg("psr")
    print(f"  {'PSR':25s}: {psr_mean:.4f} ± {psr_std:.4f}" if not np.isnan(psr_mean) else f"  {'PSR':25s}: N/A")
    print(f"{'='*65}\n")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    combined_csv = str(_csvs_dir(cfg) / f"{_run_prefix(cfg)}_combined_{n_runs}runs_{ts}.csv")
    _save_combined_metrics_csv(all_metrics, cfg, combined_csv)

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

    return all_metrics
