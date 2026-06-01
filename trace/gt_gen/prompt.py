"""Prompt building for structured GT reasoning generation.

Converts raw IG attribution JSON into a structured prompt that asks
Qwen3.5-122B to produce [OBSERVATION]→[INFERENCE]→[SYNTHESIS]→[ACTIVITY].
"""

import re
from typing import Dict

# Normalize IG sensor names → model vocabulary names.
# IG outputs: "Acc_X", "Total_Acc_Z", "Gyro_Y" etc.
# Model vocabulary: total_acc_x/y/z, body_acc_x/y/z, gyro_x/y/z
SENSOR_NAME_MAP = {
    "Acc_X":       "body_acc_x",
    "Acc_Y":       "body_acc_y",
    "Acc_Z":       "body_acc_z",
    "Total_Acc_X": "total_acc_x",
    "Total_Acc_Y": "total_acc_y",
    "Total_Acc_Z": "total_acc_z",
    "Gyro_X":      "gyro_x",
    "Gyro_Y":      "gyro_y",
    "Gyro_Z":      "gyro_z",
}

VALID_SENSORS = set(SENSOR_NAME_MAP.values())

WINDOW_LENGTH = 128  # UCI-HAR fixed window (128 timesteps = 2.56s @ 50Hz)

# Outputs matching these are flagged as refusals in W&B
REFUSAL_RE = re.compile(
    r'(?i)(i cannot|i can\'t|as an ai|i am unable|i\'m unable|i apologize|'
    r'i\'m sorry, but|sorry, i cannot|not able to (provide|generate|analyze))'
)

# Flags numeric leaks in output text (2+ digit numbers or any decimal).
# Intentionally excludes single digits like "1." from list markers.
NUMERIC_LEAK_RE = re.compile(r'\b\d{2,}\.?\d*\b|\b\d+\.\d+\b')


def timesteps_to_temporal(start_t: int, end_t: int) -> str:
    """Map timestep range → fixed 6-term temporal vocabulary.

    Uses the center of the region within the WINDOW_LENGTH window.
    Spans covering >75% of the window → 'full_window'.
    Returns one of: early | early_to_mid | mid | mid_to_late | late | full_window
    """
    start_frac = start_t / WINDOW_LENGTH
    end_frac = min(end_t / WINDOW_LENGTH, 1.0)
    span = end_frac - start_frac

    if span >= 0.75:
        return "full_window"

    center = (start_frac + end_frac) / 2.0

    if center < 0.25:
        return "early"
    elif center < 0.40:
        return "early_to_mid"
    elif center < 0.60:
        return "mid"
    elif center < 0.75:
        return "mid_to_late"
    else:
        return "late"


def importance_to_confidence(rank: int) -> str:
    """Map 0-indexed IG importance rank → categorical confidence tier.

    Top-3 → high, next 3 → moderate, rest → low.
    """
    if rank < 3:
        return "high"
    elif rank < 6:
        return "moderate"
    else:
        return "low"


def format_attention_data(attention_result: Dict) -> str:
    """Build the structured-template prompt for the 122B teacher model.

    Produces [OBSERVATION]→[INFERENCE]→[SYNTHESIS]→[ACTIVITY] format
    with explicit node IDs and based_on citations.
    Number of observations equals number of IG regions (up to 10).
    Sensor names normalized via SENSOR_NAME_MAP, temporal terms constrained
    to 6 fixed terms, confidence levels categorical.
    """
    sample_id  = attention_result['sample_idx']
    activity   = attention_result['activity']
    confidence = attention_result['confidence']
    regions    = attention_result['high_attribution_regions']
    phases     = attention_result['temporal_phase_importance']
    threshold  = attention_result.get('attribution_threshold_p90', 0.0)

    top_regions = regions[:10]

    evidence_lines = []
    for i, reg in enumerate(top_regions):
        raw_sensor  = reg['sensor']
        norm_sensor = SENSOR_NAME_MAP.get(raw_sensor, raw_sensor.lower())
        temporal    = timesteps_to_temporal(reg['start_t'], reg['end_t'])
        conf        = importance_to_confidence(i)
        evidence_lines.append(
            f"{i+1}. Sensor: {norm_sensor} (original: {raw_sensor}), "
            f"Timesteps {reg['start_t']}–{reg['end_t']} "
            f"(length {reg['length']})\n"
            f"   - Mean Importance: {reg['mean_importance']:.4f}\n"
            f"   - Max Importance:  {reg['max_importance']:.4f}\n"
            f"   - Peak Timestep:   {reg['peak_timestep']}\n"
            f"   - Temporal Region: {temporal}\n"
            f"   - Confidence Tier: {conf}"
        )

    if not evidence_lines:
        evidence_lines.append("(No regions exceeded the 90th percentile threshold.)")

    phase_lines = [
        f"Phase {p['phase']}: Average Importance {p['importance']:.4f}"
        for p in phases
    ]

    n_obs = len(top_regions)
    all_sensor_names = sorted(VALID_SENSORS)

    return f"""You are an expert biomechanist analyzing wearable sensor data from a wearable sensor dataset. You will produce a STRUCTURED reasoning trace — a chain of observations, inferences, and a synthesis that explains why this data corresponds to a specific activity.

## Raw Numeric Evidence (use this to ground your observations — but paraphrase ALL numbers)

Sample ID: {sample_id}
Predicted Activity: {activity}
Confidence: {confidence:.4f}
Attribution threshold (global p90): {threshold:.4f}

### High-Importance Regions (sorted by importance):
{chr(10).join(evidence_lines)}

### Temporal Phase Analysis:
{chr(10).join(phase_lines)}

## YOUR TASK: Generate a Structured Reasoning Trace

You must output EXACTLY this format. Every section header must appear EXACTLY as shown (including the brackets and pipe characters). Do NOT add any other headers, bullet points, or markdown.

### FORMAT SPECIFICATION:

**STEP 1 — OBSERVATIONS** (one per important sensor region)
Generate exactly {n_obs} observations, one for each high-importance region listed above. Each observation describes what a specific sensor is doing in a specific temporal window.

[OBSERVATION | id: O1]
sensor: <sensor_name from the evidence, using normalized lowercase format>
temporal: <EXACTLY one of: early | early_to_mid | mid | mid_to_late | late | full_window>
pattern: <faithfully describe only what the evidence above shows this sensor doing — peaks, oscillations, stability, sustained elevation, sudden transitions, etc. Do NOT invent patterns not present in the evidence. Use natural language, NO numbers.>
confidence: <high, moderate, or low — use the confidence tier from the evidence>

**STEP 2 — INFERENCES** (combine observations into biomechanical interpretations)
Generate 2-4 inferences. Each inference MUST cite which observations it builds on using their IDs (O1, O2, etc.).

[INFERENCE | id: I1]
based_on: O1, O3
inference: <what the cited observations actually imply biomechanically — heel strikes, push-off phases, postural stability, arm swing, weight transfer, etc. Must follow directly from the observations cited in based_on. Do not introduce evidence not present in those observations.>
confidence: <high, moderate, or low>

**STEP 3 — SYNTHESIS** (combine inferences into final explanation)
One synthesis block that ties everything together. Must cite which inferences it builds on.

[SYNTHESIS]
based_on: I1, I2
<A coherent 50-80 word paragraph explaining why the overall sensor profile indicates this activity. Reference the key biomechanical mechanisms. NO numbers — speak in terms of body mechanics.>

**STEP 4 — ACTIVITY**

[ACTIVITY]: <the activity name, exactly as given: {activity}>

### CRITICAL RULES:
1. sensor names MUST use one of these exact values: {', '.join(all_sensor_names)}
2. temporal MUST be EXACTLY one of: early | early_to_mid | mid | mid_to_late | late | full_window
   Replace any free-form phrase (e.g. "beginning", "around the middle", "toward the end") with the closest canonical label.
3. pattern (OBSERVATION) MUST faithfully describe only what the evidence shows. DO NOT INVENT SENSOR BEHAVIOURS WHICH ARE NOT PRESENT.
4. inference MUST follow directly from the observations cited in its based_on field — do not introduce biomechanical claims that are not grounded in those specific observations.
5. [SYNTHESIS] MUST have a based_on field citing inference IDs (I1, I2, etc.)
6. Do NOT use any numbers, percentages, or timestep values in any text
7. Do NOT add any markdown formatting, code fences, headers, or bullet points outside the specified format
8. Observation IDs must be sequential: O1, O2, O3, ... O{n_obs}
9. Inference IDs must be sequential starting from I1
10. Output ONLY the structured trace — no preamble, no explanation, no commentary
12. All output shall be in English only — no Chinese or any other language

### PARAPHRASING RULES:
Instead of numbers, use natural language:
- Timesteps → "early portion", "midway through", "toward the end"
- Sensor values → "sharp peak", "rapid oscillation", "sustained elevation"
- Importance → "strong evidence", "a secondary signal"

### LANGUAGE RULES — READ CAREFULLY:
Your pattern descriptions MUST be plain, direct, and mechanically grounded.
FORBIDDEN language styles — outputs containing these will be rejected:

- Dramatic and abstract language. this is not a language exam, you need to write everything in a manner that a layman can understand.
- Padding phrases: "occurring within the initial segment", "distributed across the central portion of the timeline" are FORBIDDEN
- Any phrase that sounds like it is trying to sound impressive rather than understandable, KEEP IT SIMPLE,

REQUIRED language style:
- Describe what the signal IS DOING in the simplest possible terms
- Maximum 12 words per pattern description
- If nothing significant is happening, SAY THAT PLAINLY:
  BAD:  "subtle undulations indicating gentle sway rather than violent motion"
  GOOD: "signal holds near baseline with no significant change"
  BAD:  "isolated single point deviation indicating a negligible side-to-side shift"
  GOOD: "near-flat signal with one small spike, otherwise no movement"


PHYSIOLOGICAL SENSORS (HR, EDA, temperature only)
Before treating an anomaly as real, judge whether it's noise or artifact. For example, wrist HR, temperature sensors are prone to noise during movement due to loss of contact with body.

Keep noisy sections if a reasonable cause exists, like heart rate elevated by orders of magnitude in walking. This is CLEARLY due to sensor losing contact with body intermittently.
IF YOU KEEP THE NOISE IN THE REASON, MENTION THAT IT IS NOISE, AND THE REASON (loss of contact etc) ALONG WITH IT.
Discard if no plausible cause. Never surface noise as a physiological event.

### STATIC ACTIVITY RULE (sitting, standing, lying down):
If the predicted activity is a static or low-motion activity (Standing, Sitting, Sitting Down,
Standing Up, Laying), your observations MUST reflect that directly.
- Do NOT dress up flat signals with dramatic vocabulary
- If a sensor holds near a constant value, write: "holds near [high/low/baseline] throughout"
- If there are only tiny fluctuations, write: "small fluctuations around a stable value, no clear movement"

### BAD vs GOOD PATTERN EXAMPLES (study these carefully):

BAD: "distinct sharp fluctuations with extreme intensity peaks occurring within the initial segment"
GOOD: "sharp peak then drops back to baseline"

BAD: "intermittent bursts of energy signaling brief shifts in vertical load distribution"
GOOD: "small oscillations around a stable baseline"

BAD: "localized surges in horizontal force corresponding to minor adjustments"
GOOD: "minor variations, no clear directional movement"


### SELF-CHECK before writing each pattern:
Ask yourself: "Is this phrase something that the layperson can understand? or does it sound like creative writing, or something with a lot of fancy words for no reason?" If creative writing → rewrite it.

### EXAMPLE OUTPUT:

[OBSERVATION | id: O1]
sensor: total_acc_y
temporal: early
pattern: sharp peak then stabilizes near a constant value
confidence: high

[OBSERVATION | id: O2]
sensor: total_acc_y
temporal: late
pattern: small fluctuations around a stable value, no clear movement
confidence: high

[OBSERVATION | id: O3]
sensor: total_acc_y
temporal: mid
pattern: holds near constant, minimal variance throughout
confidence: high

[OBSERVATION | id: O4]
sensor: body_acc_x
temporal: mid
pattern: stays near zero, no meaningful lateral movement
confidence: moderate

[OBSERVATION | id: O5]
sensor: gyro_z
temporal: full_window
pattern: near-zero throughout, confirming no rotation
confidence: moderate

[INFERENCE | id: I1]
based_on: O1, O2, O3
inference: vertical signal holds steady after initial settling, consistent with remaining upright and still
confidence: high

[INFERENCE | id: I2]
based_on: O4, O5
inference: no lateral movement or rotation detected, ruling out walking or any locomotion
confidence: high

[SYNTHESIS]
based_on: I1, I2
The vertical axis holds near a constant value with no rhythmic pattern. Lateral and rotational signals stay near zero throughout. There is no stepping, turning, or directional movement detectable. This is consistent with the subject standing still.

[ACTIVITY]: Standing
Now generate the structured reasoning trace for **{activity}**:"""
