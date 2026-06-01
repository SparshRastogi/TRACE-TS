"""Tier 2 structured graph metrics: PSR, GS, RCV, GC, node error rates."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from trace.eval.metrics.graph_parser import parse_reasoning_graph


_SENSOR_NAME_MAP = {
    # UCI-HAR
    "Acc_X": "body_acc_x", "Acc_Y": "body_acc_y", "Acc_Z": "body_acc_z",
    "Total_Acc_X": "total_acc_x", "Total_Acc_Y": "total_acc_y", "Total_Acc_Z": "total_acc_z",
    "Gyro_X": "gyro_x", "Gyro_Y": "gyro_y", "Gyro_Z": "gyro_z",
    # USC-HAD / Capture24 variants
    "acc_X": "body_acc_x", "acc_Y": "body_acc_y", "acc_Z": "body_acc_z",
    "AccX": "body_acc_x", "AccY": "body_acc_y", "AccZ": "body_acc_z",
    "Acc_X_uschad": "body_acc_x",
    # Shoaib
    "Wrist_Acc_X": "wrist_acc_x", "Wrist_Acc_Y": "wrist_acc_y", "Wrist_Acc_Z": "wrist_acc_z",
    "Wrist_Gyro_X": "wrist_gyro_x", "Wrist_Gyro_Y": "wrist_gyro_y", "Wrist_Gyro_Z": "wrist_gyro_z",
    "Belt_Acc_X": "belt_acc_x", "Belt_Acc_Y": "belt_acc_y", "Belt_Acc_Z": "belt_acc_z",
    "Belt_Gyro_X": "belt_gyro_x", "Belt_Gyro_Y": "belt_gyro_y", "Belt_Gyro_Z": "belt_gyro_z",
    "UpperArm_Acc_X": "upperarm_acc_x", "UpperArm_Acc_Y": "upperarm_acc_y", "UpperArm_Acc_Z": "upperarm_acc_z",
    "LeftPocket_Acc_X": "leftpocket_acc_x", "LeftPocket_Acc_Y": "leftpocket_acc_y", "LeftPocket_Acc_Z": "leftpocket_acc_z",
    "RightPocket_Acc_X": "rightpocket_acc_x", "RightPocket_Acc_Y": "rightpocket_acc_y", "RightPocket_Acc_Z": "rightpocket_acc_z",
    # MHealth
    "LANkle_Acc_X": "lankle_acc_x", "LANkle_Acc_Y": "lankle_acc_y", "LANkle_Acc_Z": "lankle_acc_z",
    "LANkle_Gyro_X": "lankle_gyro_x", "LANkle_Gyro_Y": "lankle_gyro_y", "LANkle_Gyro_Z": "lankle_gyro_z",
    "LANkle_Mag_X": "lankle_mag_x", "LANkle_Mag_Y": "lankle_mag_y", "LANkle_Mag_Z": "lankle_mag_z",
    "Chest_Acc_X": "chest_acc_x", "Chest_Acc_Y": "chest_acc_y", "Chest_Acc_Z": "chest_acc_z",
    "Chest_ECG_I": "chest_ecg_i", "Chest_ECG_II": "chest_ecg_ii", "Chest_ECG_III": "chest_ecg_iii",
    "RWrist_Acc_X": "rwrist_acc_x", "RWrist_Acc_Y": "rwrist_acc_y", "RWrist_Acc_Z": "rwrist_acc_z",
    "RWrist_Gyro_X": "rwrist_gyro_x", "RWrist_Gyro_Y": "rwrist_gyro_y", "RWrist_Gyro_Z": "rwrist_gyro_z",
    "ECG_I": "ecg_i", "ECG_II": "ecg_ii", "ECG_III": "ecg_iii",
    # PAMAP2
    "Heart_Rate": "heart_rate", "Hand_Temp": "hand_temp",
    "Hand_Acc16g_X": "hand_acc16g_x", "Hand_Acc16g_Y": "hand_acc16g_y", "Hand_Acc16g_Z": "hand_acc16g_z",
    "Chest_Acc16g_X": "chest_acc16g_x", "Chest_Acc16g_Y": "chest_acc16g_y", "Chest_Acc16g_Z": "chest_acc16g_z",
    "Ankle_Acc16g_X": "ankle_acc16g_x", "Ankle_Acc16g_Y": "ankle_acc16g_y", "Ankle_Acc16g_Z": "ankle_acc16g_z",
    "Hand_Gyro_X": "hand_gyro_x", "Hand_Gyro_Y": "hand_gyro_y", "Hand_Gyro_Z": "hand_gyro_z",
    "Chest_Gyro_X": "chest_gyro_x", "Chest_Gyro_Y": "chest_gyro_y", "Chest_Gyro_Z": "chest_gyro_z",
    "Ankle_Gyro_X": "ankle_gyro_x", "Ankle_Gyro_Y": "ankle_gyro_y", "Ankle_Gyro_Z": "ankle_gyro_z",
}


def _load_ig_json_index(ig_json_dir) -> dict:
    if ig_json_dir is None:
        return {}
    ig_json_dir = Path(ig_json_dir)
    if not ig_json_dir.exists():
        print(f"[GS] WARNING: ig_json_dir not found: {ig_json_dir}. GS skipped.")
        return {}
    pattern = re.compile(r"_s(\d+)\.json$", re.IGNORECASE)
    index = {}
    for fpath in ig_json_dir.glob("*.json"):
        m = pattern.search(fpath.name)
        if m:
            index[str(int(m.group(1)))] = fpath
    print(f"[GS] Indexed {len(index)} IG JSONs from {ig_json_dir.name}")
    return index


def compute_psr(graphs: list) -> float:
    if not graphs:
        return float("nan")
    return sum(1 for g in graphs if not g["parse_failed"]) / len(graphs)


def compute_gs_sample(graph_gen: dict, ig_data: dict, top_k: int = 3,
                      full_text: str = "") -> dict | None:
    """Grounding Score for one sample.

    important: top-k IG sensors via _SENSOR_NAME_MAP → lowercase vocab names.
    mentioned: obs node sensor: fields → lowercased + stripped.
    Both are lowercased to prevent silent case-mismatch false negatives.
    """
    if graph_gen["parse_failed"]:
        return None
    regions = ig_data.get("high_attribution_regions", [])
    if not regions:
        return None
    sorted_regions = sorted(regions, key=lambda r: r.get("mean_importance", 0), reverse=True)
    important = {_SENSOR_NAME_MAP.get(r.get("sensor", ""), r.get("sensor", "").lower())
                 for r in sorted_regions[:top_k]}
    important.discard("")
    if not important:
        return None
    mentioned = {obs.get("sensor", "").lower().strip()
                 for obs in graph_gen["observations"]
                 if obs.get("sensor", "").strip()}
    mentioned.discard("")
    if not mentioned:
        return None
    overlap = important & mentioned
    gs_r = len(overlap) / len(important) if important else 0.0
    gs_p = len(overlap) / len(mentioned) if mentioned else 0.0
    gs_f1 = 2 * gs_p * gs_r / (gs_p + gs_r) if (gs_p + gs_r) > 0 else 0.0
    return {"gs_recall": gs_r, "gs_precision": gs_p, "gs_f1": gs_f1}


def compute_gs_corpus(results: list, ig_index: dict, top_k: int = 3) -> dict:
    gs_r, gs_p, gs_f1 = [], [], []
    n_skipped, n_parse_fail = 0, 0
    for r in results:
        gen_graph = r.get("_gen_graph", parse_reasoning_graph(r.get("generated", "")))
        if gen_graph["parse_failed"]:
            n_parse_fail += 1
            continue
        ig_path = ig_index.get(str(r.get("sample_id", "")))
        if ig_path is None:
            n_skipped += 1
            continue
        try:
            with open(ig_path) as f:
                ig_data = json.load(f)
        except Exception:
            n_skipped += 1
            continue
        sample_gs = compute_gs_sample(gen_graph, ig_data, top_k,
                                       full_text=r.get("generated", ""))
        if sample_gs is None:
            n_skipped += 1
            continue
        gs_r.append(sample_gs["gs_recall"])
        gs_p.append(sample_gs["gs_precision"])
        gs_f1.append(sample_gs["gs_f1"])
    if not gs_f1:
        return {"gs_recall": float("nan"), "gs_precision": float("nan"), "gs_f1": float("nan"),
                "n_scored": 0, "n_skipped": n_skipped, "n_parse_fail": n_parse_fail}
    return {"gs_recall": float(np.mean(gs_r)), "gs_precision": float(np.mean(gs_p)),
            "gs_f1": float(np.mean(gs_f1)),
            "n_scored": len(gs_f1), "n_skipped": n_skipped, "n_parse_fail": n_parse_fail}


def compute_rcv_sample(graph: dict) -> dict | None:
    if graph["parse_failed"] or not graph["inferences"]:
        return None
    obs_map = {obs["id"]: obs["sensor"] for obs in graph["observations"]}
    obs_ids = set(obs_map.keys())
    n_struct, n_sem, n_both = 0, 0, 0
    for inf in graph["inferences"]:
        based_on = inf.get("based_on", [])
        inf_text = inf.get("inference", "").lower()
        dangling = [b for b in based_on if b not in obs_ids]
        pass_struct = len(dangling) == 0
        pass_sem = False
        if pass_struct:
            cited_sensors = {obs_map[b] for b in based_on if b in obs_map}
            for sensor in cited_sensors:
                if sensor in inf_text or sensor.replace("_", " ") in inf_text:
                    pass_sem = True
                    break
            if not pass_sem:
                for sensor in cited_sensors:
                    axis = sensor.split("_")[-1]
                    if axis in ("x", "y", "z") and axis in inf_text:
                        pass_sem = True
                        break
        n_struct += int(pass_struct)
        n_sem += int(pass_sem)
        n_both += int(pass_struct and pass_sem)
    n = len(graph["inferences"])
    return {"rcv_structural": n_struct / n, "rcv_semantic": n_sem / n, "rcv_combined": n_both / n}


def compute_rcv_corpus(graphs_gen: list) -> dict:
    struct_v, sem_v, comb_v = [], [], []
    for g in graphs_gen:
        if g["parse_failed"]:
            continue
        r = compute_rcv_sample(g)
        if r is None:
            continue
        struct_v.append(r["rcv_structural"])
        sem_v.append(r["rcv_semantic"])
        comb_v.append(r["rcv_combined"])
    if not comb_v:
        return {"rcv_structural": float("nan"), "rcv_semantic": float("nan"), "rcv_combined": float("nan")}
    return {"rcv_structural": float(np.mean(struct_v)), "rcv_semantic": float(np.mean(sem_v)),
            "rcv_combined": float(np.mean(comb_v))}


def compute_gc(graphs_gen: list, graphs_ref: list, gt_activities: list) -> dict:
    class_ref_obs = defaultdict(list)
    for g, act in zip(graphs_ref, gt_activities):
        if not g["parse_failed"]:
            class_ref_obs[act].append(g["n_observations"])
    class_ref_means = {cls: float(np.mean(v)) for cls, v in class_ref_obs.items() if v}
    gc_vals, per_class_gc = [], defaultdict(list)
    for g_gen, g_ref, act in zip(graphs_gen, graphs_ref, gt_activities):
        if g_gen["parse_failed"]:
            continue
        ref_mean = class_ref_means.get(act, 0.0)
        if ref_mean <= 0:
            continue
        gc = min(g_gen["n_observations"] / ref_mean, 1.0)
        gc_vals.append(gc)
        per_class_gc[act].append(gc)
    return {"gc_mean": float(np.mean(gc_vals)) if gc_vals else float("nan"),
            "per_class_gc": {cls: float(np.mean(v)) for cls, v in per_class_gc.items()},
            "class_ref_obs_means": class_ref_means}


def compute_node_error_rates(graphs_gen: list, activity_classes: list,
                              valid_sensors: set) -> dict:
    """Node-level error rate breakdown across observation / inference / synthesis layers."""
    _VALID_CONFIDENCE = {"high", "medium", "low"}
    activity_classes_norm = {c.lower().strip() for c in activity_classes}

    n_total = len(graphs_gen)
    n_parsed = 0
    obs_sensor_hits, obs_sensor_total = 0, 0
    obs_temporal_present, obs_temporal_total = 0, 0
    obs_conf_valid, obs_conf_total = 0, 0
    n_has_both_obs_and_inf = 0

    n_has_inf = 0
    inf_struct_ok, inf_struct_total = 0, 0
    inf_sem_ok, inf_sem_total = 0, 0
    obs_node_referenced, obs_node_total = 0, 0

    n_has_syn = 0
    syn_ref_ok, syn_ref_total = 0, 0
    act_vocab_ok = 0

    for g in graphs_gen:
        act_text = g.get("activity")
        if act_text is not None and act_text.strip():
            if act_text.strip() in activity_classes_norm:
                act_vocab_ok += 1

        if g["parse_failed"]:
            continue
        n_parsed += 1

        observations = g["observations"]
        inferences   = g["inferences"]
        synthesis    = g["synthesis"]

        for obs in observations:
            obs_sensor_total += 1
            sensor = obs.get("sensor", "").strip()
            if sensor in valid_sensors:
                obs_sensor_hits += 1

            obs_temporal_total += 1
            temporal = obs.get("temporal", "").strip()
            if temporal:
                obs_temporal_present += 1

            obs_conf_total += 1
            conf = obs.get("confidence", "").strip().lower()
            if conf in _VALID_CONFIDENCE:
                obs_conf_valid += 1

        if inferences:
            n_has_both_obs_and_inf += 1

        obs_map = {obs["id"]: obs["sensor"] for obs in observations}
        obs_ids = set(obs_map.keys())

        if inferences:
            n_has_inf += 1
            referenced_obs_ids = set()
            for inf in inferences:
                based_on  = inf.get("based_on", [])
                inf_text  = inf.get("inference", "").lower()
                dangling  = [b for b in based_on if b not in obs_ids]

                inf_struct_total += 1
                if not dangling:
                    inf_struct_ok += 1

                inf_sem_total += 1
                cited_sensors = {obs_map[b] for b in based_on if b in obs_map}
                sem_ok = False
                for sensor in cited_sensors:
                    if sensor in inf_text or sensor.replace("_", " ") in inf_text:
                        sem_ok = True
                        break
                    axis = sensor.split("_")[-1]
                    if axis in ("x", "y", "z") and axis in inf_text:
                        sem_ok = True
                        break
                if sem_ok:
                    inf_sem_ok += 1
                referenced_obs_ids.update(b for b in based_on if b in obs_ids)

            obs_node_total      += len(obs_ids)
            obs_node_referenced += len(referenced_obs_ids & obs_ids)

        if synthesis is not None:
            n_has_syn += 1
            syn_based_on = synthesis.get("based_on", [])
            inf_ids = {inf["id"] for inf in inferences}
            syn_ref_total += 1
            if any(b in inf_ids for b in syn_based_on):
                syn_ref_ok += 1

    def _rate(num, den):
        return float(num / den) if den > 0 else float("nan")

    return {
        "obs_parse_rate":        _rate(n_parsed, n_total),
        "obs_sensor_valid_rate": _rate(obs_sensor_hits, obs_sensor_total),
        "obs_temporal_rate":     _rate(obs_temporal_present, obs_temporal_total),
        "obs_confidence_rate":   _rate(obs_conf_valid, obs_conf_total),
        "obs_coverage_rate":     _rate(n_has_both_obs_and_inf, n_parsed) if n_parsed else float("nan"),
        "inf_present_rate":      _rate(n_has_inf, n_parsed) if n_parsed else float("nan"),
        "inf_structural_rate":   _rate(inf_struct_ok, inf_struct_total),
        "inf_semantic_rate":     _rate(inf_sem_ok, inf_sem_total),
        "inf_obs_coverage":      _rate(obs_node_referenced, obs_node_total),
        "syn_present_rate":      _rate(n_has_syn, n_parsed) if n_parsed else float("nan"),
        "syn_ref_rate":          _rate(syn_ref_ok, syn_ref_total),
        "syn_activity_valid_rate": _rate(act_vocab_ok, n_total),
        "_n_total": n_total,
        "_n_parsed": n_parsed,
    }
