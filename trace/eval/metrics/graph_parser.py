"""Parse structured reasoning text into a DAG representation."""

import re

from trace.config import _OBS_RE, _INF_RE, _SYN_RE, _ACT_RE


def parse_reasoning_graph(text: str) -> dict:
    text = text.strip() if text else ""
    observations = [
        {"id": m.group(1).strip(), "sensor": m.group(2).strip().lower(),
         "temporal": m.group(3).strip().lower(), "pattern": m.group(4).strip(),
         "confidence": m.group(5).strip().lower()}
        for m in _OBS_RE.finditer(text)
    ]
    inferences = [
        {"id": m.group(1).strip(),
         "based_on": [x.strip() for x in m.group(2).strip().split(",")],
         "inference": m.group(3).strip(), "confidence": m.group(4).strip().lower()}
        for m in _INF_RE.finditer(text)
    ]
    synthesis = None
    syn_m = _SYN_RE.search(text)
    if syn_m:
        synthesis = {"based_on": [x.strip() for x in syn_m.group(1).strip().split(",")],
                     "text": re.sub(r"\n\s*\n", "\n", syn_m.group(2)).strip()}
    activity = None
    act_m = _ACT_RE.search(text)
    if act_m:
        activity = act_m.group(1).strip().lower().replace("_", " ")
    return {"observations": observations, "inferences": inferences,
            "synthesis": synthesis, "activity": activity,
            "n_observations": len(observations), "n_inferences": len(inferences),
            "parse_failed": len(observations) == 0}
