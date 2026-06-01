#!/usr/bin/env python3
"""Evaluation and figure entry point for TRACE cross-attention model."""

from trace.config import parse_args
from trace.utils.naming import _run_prefix
from trace.eval.runner import run_evaluate
from trace.viz.fig3 import generate_attention_attribution_figure


def main():
    cfg = parse_args()
    if cfg["adapter_layers"] == "none":
        cfg["_n_adapter_layers"] = 0
    elif cfg["adapter_layers"] not in ("all", "every2", "every4"):
        cfg["_n_adapter_layers"] = len(cfg["adapter_layers"].split(","))
    else:
        cfg["_n_adapter_layers"] = "?"
    print(f"\n{'='*60}")
    print(f"TRACE Cross-Attention — {cfg['mode'].upper()}")
    print(f"{'='*60}")
    print(f"  model:         {cfg['model_id']}")
    print(f"  dataset:       {cfg['dataset']}")
    print(f"  data_version:  {cfg['data_version']}")
    print(f"  n_tokens:      {cfg['n_tokens']}")
    print(f"  adapter_rank:  {cfg['adapter_rank']}")
    print(f"  adapter_layers:{cfg['adapter_layers']} ({cfg['_n_adapter_layers']} adapters)")
    print(f"  run_prefix:    {_run_prefix(cfg)}")
    print(f"  activity_classes ({len(cfg['activity_classes'])}): {cfg['activity_classes']}")
    print(f"  raw_embed_dir: {cfg.get('raw_embed_dir', 'not set (use test_json_dir only)')}")
    print(f"{'='*60}\n")
    if cfg["mode"] == "evaluate":
        run_evaluate(cfg)
    elif cfg["mode"] == "figure3":
        generate_attention_attribution_figure(cfg)
    else:
        raise ValueError(f"evaluate.py only handles mode=evaluate or mode=figure3, got: {cfg['mode']}")


if __name__ == "__main__":
    main()
