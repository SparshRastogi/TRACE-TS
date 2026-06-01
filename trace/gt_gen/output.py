"""ReasoningOutput dataclass, filename parsing, and response parsing."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_filename(path: Path) -> Tuple[str, str]:
    """Extract (split, sample_id) from filenames like test_class0_Walking_s0081.json.

    split     = 'train' or 'test'
    sample_id = '81'  (leading zeros stripped)
    Returns ('unknown', '0') if pattern doesn't match — never raises.
    """
    name = path.stem
    split_match  = re.match(r'^(train|test)_', name)
    sample_match = re.search(r'_s(\d+)$', name)
    split     = split_match.group(1)  if split_match  else 'unknown'
    sample_id = str(int(sample_match.group(1))) if sample_match else '0'
    return split, sample_id


@dataclass
class ReasoningOutput:
    sample_id:            str
    split:                str
    predicted_activity:   str
    confidence:           float
    overall_reasoning:    str
    reasoning_graph:      dict
    generation_timestamp: str
    model_used:           str
    mantis_embedding:     dict

    def to_dict(self) -> Dict:
        return dict(self.__dict__)

    def to_json(self, filepath: str):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_training_format(self) -> Dict:
        return {
            'input': (
                f"Activity Recognition Task:\nAnalyze the following sensor data "
                f"and predict the activity.\n\nSample ID: {self.sample_id}\n"
                f"Model Confidence: {self.confidence:.4f}\n"
            ),
            'output': (
                f"Conclusion:\n{self.overall_reasoning}\n\n"
                f"Predicted Activity: {self.predicted_activity}\n"
            ),
            'metadata': dict(self.__dict__)
        }


def parse_response(response_text: str) -> Dict:
    """Parse structured reasoning trace into components.

    Returns:
      overall_reasoning  — full structured text (the 7B training target)
      reasoning_graph    — parsed DAG: {observations, inferences, synthesis, activity, ...}
      + backward-compat empty fields: temporal_analysis, sensor_analysis,
        detailed_reasoning, key_evidence
    """
    # Strip Qwen3 <think>...</think> blocks before parsing
    text = re.sub(r'<think>[\s\S]*?</think>', '', response_text, flags=re.IGNORECASE).strip()

    obs_pattern = re.compile(
        r'\[OBSERVATION\s*\|\s*id:\s*(O\d+)\]\s*\n'
        r'sensor:\s*(.+?)\n'
        r'temporal:\s*(.+?)\n'
        r'pattern:\s*(.+?)\n'
        r'confidence:\s*(.+?)(?:\n|$)',
        re.IGNORECASE
    )
    observations = [
        {
            'id':         m.group(1).strip(),
            'sensor':     m.group(2).strip().lower(),
            'temporal':   m.group(3).strip().lower(),
            'pattern':    m.group(4).strip(),
            'confidence': m.group(5).strip().lower(),
        }
        for m in obs_pattern.finditer(text)
    ]

    inf_pattern = re.compile(
        r'\[INFERENCE\s*\|\s*id:\s*(I\d+)\]\s*\n'
        r'based_on:\s*(.+?)\n'
        r'inference:\s*(.+?)\n'
        r'confidence:\s*(.+?)(?:\n|$)',
        re.IGNORECASE
    )
    inferences = [
        {
            'id':         m.group(1).strip(),
            'based_on':   [x.strip() for x in m.group(2).strip().split(',')],
            'inference':  m.group(3).strip(),
            'confidence': m.group(4).strip().lower(),
        }
        for m in inf_pattern.finditer(text)
    ]

    synthesis = None
    syn_pattern = re.compile(
        r'\[SYNTHESIS\]\s*\nbased_on:\s*(.+?)\n([\s\S]+?)(?=\[ACTIVITY\]|\Z)',
        re.IGNORECASE
    )
    syn_m = syn_pattern.search(text)
    if syn_m:
        syn_text = re.sub(r'\n\s*\n', '\n', syn_m.group(2)).strip()
        synthesis = {
            'based_on': [x.strip() for x in syn_m.group(1).strip().split(',')],
            'text':     syn_text,
        }

    activity = None
    act_m = re.search(r'\[ACTIVITY\]\s*:\s*(.+?)$', text, re.IGNORECASE | re.MULTILINE)
    if act_m:
        activity = act_m.group(1).strip()

    reasoning_graph = {
        'observations':   observations,
        'inferences':     inferences,
        'synthesis':      synthesis,
        'activity':       activity,
        'n_observations': len(observations),
        'n_inferences':   len(inferences),
        'parse_failed':   len(observations) == 0,
    }

    overall_reasoning = text.strip()
    overall_reasoning = re.sub(r'^```\w*\n?', '', overall_reasoning)
    overall_reasoning = re.sub(r'\n?```$', '', overall_reasoning)
    overall_reasoning = overall_reasoning.strip()

    return {
        'overall_reasoning':  overall_reasoning,
        'reasoning_graph':    reasoning_graph,
        'temporal_analysis':  '',
        'sensor_analysis':    '',
        'detailed_reasoning': '',
        'key_evidence':       [],
    }
