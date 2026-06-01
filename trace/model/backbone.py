"""LLM backbone loading and generic layer/dim accessors."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_id: str, local_rank: int = 0, device_map=None):
    print(f"[model] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _device_map = device_map if device_map is not None else {"": local_rank}
    print(f"[model] Loading LLM: {model_id} (device_map={_device_map})")
    # Gemma-4 has global_head_dim=512 which exceeds FA2 cap → force eager
    is_gemma4 = "gemma-4" in model_id.lower() or "gemma4" in model_id.lower()
    if is_gemma4:
        print("[model] Gemma 4 detected: forcing eager attention (global_head_dim=512 > FA2 cap)")
        llm = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=_device_map, trust_remote_code=True,
        )
        print("[model] Eager attention (BF16)")
    else:
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=_device_map, trust_remote_code=True,
            )
            print("[model] Flash Attention 2 enabled (BF16)")
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"[model] WARNING: Flash Attention 2 not available ({e}). Falling back.")
            llm = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                device_map=_device_map, trust_remote_code=True,
            )
            print("[model] Standard attention (BF16)")
    return tokenizer, llm


def _get_llm_layers(llm):
    """Generic layer list accessor — works for Qwen, Llama, Gemma, etc."""
    for fn in [
        lambda: llm.model.layers,
        lambda: llm.model.language_model.layers,
        lambda: llm.layers,
    ]:
        try:
            layers = fn()
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot find transformer layers in {type(llm).__name__}. "
        "Tried: llm.model.layers, llm.model.language_model.layers, llm.layers"
    )


def _get_llm_dim(llm) -> int:
    """Generic hidden_size accessor."""
    if hasattr(llm.config, "hidden_size"):
        return llm.config.hidden_size
    try:
        return llm.model.language_model.config.hidden_size
    except AttributeError:
        pass
    raise AttributeError(
        f"Cannot find hidden_size for {type(llm).__name__}. "
        f"Top config keys: {list(llm.config.to_dict().keys())[:20]}"
    )
