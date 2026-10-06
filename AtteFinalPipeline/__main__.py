import argparse
import runpy
import sys

COMMANDS = {
    "train": "expert.train",
    "preprocess": "data.preprocess",
    "prepare-uschad": "data.prepare_uschad",
    "prepare-shoaib": "data.prepare_shoaib",
    "prepare-mhealth": "data.prepare_mhealth",
    "prepare-capture24": "data.prepare_capture24",
    "prepare-capture24-subset": "data.prepare_capture24_subset",
    "prepare-opportunity-pamap": "data.prepare_opportunity_pamap",
    "attribute": "attribution.combined",
    "attribute-ig": "attribution.ig_only",
    "attribute-shap": "attribution.shap_only",
    "attribute-random": "attribution.random",
    "attribute-bottom10pct": "attribution.bottom10pct",
    "attribute-moment": "attribution.moment",
    "attribute-deepconvlstm": "attribution.deepconvlstm",
    "attribute-sampled": "attribution.sampled",
    "attribute-none": "attribution.no_xai",
    "embed-mantis": "embeddings.mantis",
    "embed-moment": "embeddings.moment",
    "embed-raw": "embeddings.with_raw_data",
    "train-chronos": "baselines.chronos.train",
    "train-deepconvlstm": "baselines.deepconvlstm.train",
    "train-mantis": "baselines.mantis",
    "train-nst": "baselines.nst.train",
    "train-patchtst": "baselines.patchtst.train",
    "train-unimts": "baselines.unimts.train",
    "cot": "baselines.llm.cot",
    "cot-structured": "baselines.llm.cot_structured",
    "benchmark-llm": "baselines.llm.benchmark",
    "sample": "tools.sample",
    "filter-subset": "tools.filter_subset",
    "extract-test": "tools.extract_test",
    "proportions": "tools.proportions",
}


def main():
    parser = argparse.ArgumentParser(prog="python -m AtteFinalPipeline")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"{parser.prog} {args.command}", *sys.argv[2:]]
    runpy.run_module(f"AtteFinalPipeline.{COMMANDS[args.command]}", run_name="__main__")


if __name__ == "__main__":
    main()
