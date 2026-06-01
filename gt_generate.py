#!/usr/bin/env python3
"""Ground-truth reasoning trace generation entry point.

Uses a large teacher LLM via vLLM to generate structured
[OBSERVATION]→[INFERENCE]→[SYNTHESIS]→[ACTIVITY] traces from IG attribution
JSONs. Outputs go to reasoning_json/{train,test}/ with JSONL datasets.

Single-dataset mode (original):
    python gt_generate.py --input_dir /path/to/ig_jsons/<dataset> \\
                          --output_dir /path/to/outputs/<dataset> \\
                          --dataset ucihar

Multi-dataset mode (loads model once, loops over all datasets):
    python gt_generate.py --datasets ucihar uschad pamap2 capture24 opportunity shoaib mhealth \\
                          --data_root /path/to/data \\
                          --output_root /path/to/TRACE/outputs \\
                          --output_suffix gt_reasoning_llama70b
"""

import sys
import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from trace.gt_gen.generator import ReasoningGenerator, parse_gt_args


def main():
    cfg = parse_gt_args()

    print(f"\n{'='*60}")
    print(f"TRACE Ground-Truth Generation")
    print(f"{'='*60}")
    print(f"  model:        {cfg['model_id']}")
    print(f"  tensor_parallel_size: {cfg['tensor_parallel_size']}")

    generator = ReasoningGenerator(
        model_id=cfg['model_id'],
        tensor_parallel_size=cfg['tensor_parallel_size'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],
        max_model_len=cfg['max_model_len'],
        enforce_eager=cfg['enforce_eager'],
    )

    # Multi-dataset mode: loop over all datasets with model loaded once
    if cfg.get('datasets'):
        datasets = cfg['datasets']
        data_root   = cfg['data_root']
        output_root = cfg['output_root']
        suffix      = cfg['output_suffix']
        print(f"  datasets:     {datasets}")
        print(f"  output_suffix:{suffix}")
        print(f"{'='*60}\n")
        for ds in datasets:
            input_dir  = os.path.join(data_root,   ds)
            output_dir = os.path.join(output_root, ds, suffix)
            print(f"\n{'─'*60}")
            print(f"  Dataset: {ds}  →  {output_dir}")
            print(f"{'─'*60}")
            generator.batch_generate(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset=ds,
                max_samples=cfg['max_samples'],
                batch_size=cfg['batch_size'],
                graph_save_every=cfg['graph_save_every'],
                wandb_project=cfg['wandb_project'],
                wandb_tags=cfg.get('wandb_tags'),
            )
            print(f"  Done: {ds}")
        print(f"\n{'='*60}")
        print(f"All {len(datasets)} datasets complete.")
    else:
        # Single-dataset mode (original behaviour)
        print(f"  dataset:      {cfg['dataset']}")
        print(f"  input_dir:    {cfg['input_dir']}")
        print(f"  output_dir:   {cfg['output_dir']}")
        print(f"  batch_size:   {cfg['batch_size']}")
        print(f"  max_samples:  {cfg['max_samples'] or 'all'}")
        print(f"{'='*60}\n")
        generator.batch_generate(
            input_dir=cfg['input_dir'],
            output_dir=cfg['output_dir'],
            dataset=cfg['dataset'],
            max_samples=cfg['max_samples'],
            batch_size=cfg['batch_size'],
            graph_save_every=cfg['graph_save_every'],
            wandb_project=cfg['wandb_project'],
            wandb_tags=cfg.get('wandb_tags'),
        )
        print(f"\nFinished! Results in: {cfg['output_dir']}")

    sys.exit(0)


if __name__ == '__main__':
    main()
