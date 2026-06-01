"""Naming and path helpers for run names, checkpoint dirs, and sort keys."""

import re
from datetime import datetime
from pathlib import Path


def _model_short(model_id: str) -> str:
    """Extract compact model name, e.g. 'Qwen/Qwen3.5-4B' → 'qwen4b'."""
    name = model_id.split("/")[-1].lower()
    for suffix in ("-instruct", "-chat", "-it", "-hf", "-base", "-preview"):
        name = name.replace(suffix, "")
    family_match = re.match(r"([a-z]+)", name)
    family = family_match.group(1) if family_match else "model"
    moe_match = re.search(r"(\d+x\d+\.?\d*b)", name)
    if moe_match:
        size = moe_match.group(1)
    else:
        size_match = re.search(r"(\d+\.?\d*b)", name)
        if size_match:
            size = size_match.group(1)
        else:
            version_match = re.search(r"[^\d](\d+\.?\d*)(?:[^\d]|$)", name)
            size = version_match.group(1) if version_match else ""
    return f"{family}{size}"


def _resolve_adapter_layer_indices(adapter_layers_str: str, n_llm_layers: int) -> list[int]:
    s = adapter_layers_str.strip().lower()
    if s == "none":
        return []
    if s == "all":
        return list(range(n_llm_layers))
    elif s == "every2":
        return list(range(0, n_llm_layers, 2))
    elif s == "every4":
        return list(range(0, n_llm_layers, 4))
    else:
        indices = [int(x.strip()) for x in s.split(",")]
        for idx in indices:
            if idx < 0 or idx >= n_llm_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {n_llm_layers-1}]")
        return sorted(set(indices))


def _run_name(cfg: dict, include_timestamp: bool = True) -> str:
    n_adapter_layers = cfg.get("_n_adapter_layers", "?")
    parts = [
        "crossattn", f"{cfg['n_tokens']}tok", f"L{n_adapter_layers}",
        f"R{cfg['adapter_rank']}", cfg["dataset"],
        _model_short(cfg["model_id"]), cfg.get("data_version", "structured"),
    ]
    parts.extend([f"lr{cfg['lr']}", f"bs{cfg['batch_size']}"])
    if include_timestamp:
        parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


def _run_prefix(cfg: dict) -> str:
    return _run_name(cfg, include_timestamp=False)


def _dataset_dir(base: Path, dataset: str) -> Path:
    d = base / dataset.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _numeric_sort_key(path: str) -> int:
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else 0
