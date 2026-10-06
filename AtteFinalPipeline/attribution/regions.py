import numpy as np


def extract_high_attribution_regions(combined, sensor_names, percentile=90):
    threshold = float(np.percentile(combined, percentile))
    mask = combined >= threshold
    T, D = combined.shape
    regions = []
    for ch in range(D):
        col = mask[:, ch]
        if not col.any():
            continue
        padded = np.concatenate(([False], col, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] - 1
        for s, e in zip(starts, ends):
            vals = combined[s : e + 1, ch]
            peak_local = int(np.argmax(vals))
            regions.append(
                {
                    "sensor": sensor_names[ch],
                    "sensor_idx": int(ch),
                    "start_t": int(s),
                    "end_t": int(e),
                    "length": int(e - s + 1),
                    "mean_importance": float(vals.mean()),
                    "max_importance": float(vals.max()),
                    "peak_timestep": int(s + peak_local),
                    "importance_values": np.round(vals, 6).tolist(),
                }
            )
    regions.sort(key=lambda r: r["mean_importance"], reverse=True)
    return (regions, threshold)


def build_result_dict(
    data_np,
    combined,
    pred_class,
    probs,
    sample_idx,
    true_label,
    sensor_names,
    split,
    class_map,
):
    T, D = data_np.shape
    confidence = float(probs[pred_class])
    sensor_importance = combined.mean(axis=0)
    n_phases = 5
    phase_size = T // n_phases
    phase_importance = []
    for p in range(n_phases):
        s = p * phase_size
        e = s + phase_size if p < n_phases - 1 else T
        phase_importance.append(float(combined[s:e].mean()))
    regions, attr_threshold = extract_high_attribution_regions(combined, sensor_names)
    label_name = (
        class_map[true_label] if true_label < len(class_map) else f"Class_{true_label}"
    )
    return {
        "sample_idx": int(sample_idx),
        "true_label": int(true_label),
        "predicted_label": pred_class,
        "label_name": label_name,
        "confidence": confidence,
        "correct": bool(int(pred_class) == int(true_label)),
        "split": split,
        "class_probabilities": {
            class_map[i]: float(probs[i]) for i in range(len(probs))
        },
        "sensor_importance_ranking": {
            sensor_names[i]: float(sensor_importance[i])
            for i in np.argsort(sensor_importance)[::-1]
        },
        "temporal_phase_importance": [
            {
                "phase": f"Phase {p} ({p * phase_size}-{min((p + 1) * phase_size, T)})",
                "importance": phase_importance[p],
            }
            for p in range(n_phases)
        ],
        "attribution_threshold_p90": attr_threshold,
        "high_attribution_regions": regions,
        "_data": data_np,
        "_combined": combined,
        "_sensor_importance": sensor_importance,
        "_phase_importance": phase_importance,
    }
