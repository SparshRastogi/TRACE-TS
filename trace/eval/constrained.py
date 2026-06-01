"""
Trie-based constrained activity decoding.

After [ACTIVITY]: appears in the generated sequence, masks logits so the
model can only produce tokens that continue toward one of the valid
activity class strings in cfg["activity_classes"].

ROLLBACK: set --constrained_decoding to False (the default). The rest of
the codebase is unchanged — generate.py passes an empty LogitsProcessorList
when the flag is off, which is a no-op for llm.generate().
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList


# ---------------------------------------------------------------------------
# Trie builder
# ---------------------------------------------------------------------------

def _build_trie(tokenizer, activity_classes: list[str]) -> dict:
    """
    Builds a token-id prefix trie for all valid activity class strings.

    Each class is inserted both with and without a leading space so the
    trie works regardless of how the tokenizer encodes post-colon context.
    Leaf nodes hold {terminator_id: {}} so the processor knows when a
    complete class has been emitted and only allows EOS/newline next.
    """
    eos_id = tokenizer.eos_token_id
    terminators = {eos_id}
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    if nl_ids:
        terminators.add(nl_ids[0])

    trie = {}
    for cls_name in activity_classes:
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + cls_name, add_special_tokens=False)
            node = trie
            for tid in ids:
                node = node.setdefault(tid, {})
            for t in terminators:
                node.setdefault(t, {})
    return trie


# ---------------------------------------------------------------------------
# Marker detection
# ---------------------------------------------------------------------------

def _marker_candidates(tokenizer) -> list[list[int]]:
    """Token-id sequences for the [ACTIVITY]: marker in likely surface forms."""
    seen, out = set(), []
    for text in ["[ACTIVITY]:", "\n[ACTIVITY]:", "[ACTIVITY] :", " [ACTIVITY]:"]:
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        if ids and ids not in seen:
            seen.add(ids)
            out.append(list(ids))
    return out


def _find_marker_end(ids: list[int], markers: list[list[int]]) -> int:
    """
    Returns the index just after the last occurrence of any marker sequence,
    or -1 if no marker has appeared yet.
    """
    best = -1
    for marker in markers:
        m = len(marker)
        for i in range(len(ids) - m, -1, -1):
            if ids[i : i + m] == marker:
                best = max(best, i + m)
                break
    return best


# ---------------------------------------------------------------------------
# LogitsProcessor
# ---------------------------------------------------------------------------

class ActivityConstrainedLogitsProcessor(LogitsProcessor):
    """
    Constrains generation after [ACTIVITY]: to valid activity class tokens.
    Each sequence in the batch is handled independently (safe with do_sample).
    """

    def __init__(self, tokenizer, activity_classes: list[str]):
        self._markers = _marker_candidates(tokenizer)
        self._trie    = _build_trie(tokenizer, activity_classes)
        self._vocab   = len(tokenizer)
        print(f"[constrained] Trie root branches: {len(self._trie)} "
              f"covering {len(activity_classes)} classes")
        print(f"[constrained] Marker forms: {len(self._markers)}")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores:    torch.FloatTensor,
    ) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            ids = input_ids[i].tolist()
            marker_end = _find_marker_end(ids, self._markers)
            if marker_end < 0:
                continue  # [ACTIVITY]: not yet in sequence

            # Walk the trie along tokens generated after the marker
            node = self._trie
            for tid in ids[marker_end:]:
                node = node.get(tid)
                if node is None:
                    break

            if node is None or not node:
                # Off-trie (bad state) or leaf already consumed — don't constrain
                continue

            # Greedy argmax among valid trie continuations.
            # Sampling at temperature=0.3 between e.g. "1"/"2"/"3" adds noise
            # without benefit — pick the highest-scoring valid token deterministically.
            valid = torch.tensor(
                [v for v in node.keys() if v < self._vocab],
                dtype=torch.long,
                device=scores.device,
            )
            if valid.numel() == 0:
                continue
            best = valid[scores[i][valid].argmax()]
            mask = torch.full_like(scores[i], float("-inf"))
            mask[best] = 0.0
            scores[i] = mask

        return scores


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_constrained_processor(tokenizer, cfg: dict) -> LogitsProcessorList:
    """
    Returns a LogitsProcessorList for use in llm.generate(logits_processor=...).
    Returns an empty list (no-op) when constrained_decoding is False.
    """
    if not cfg.get("constrained_decoding", False):
        return LogitsProcessorList()
    proc = ActivityConstrainedLogitsProcessor(tokenizer, cfg["activity_classes"])
    return LogitsProcessorList([proc])
