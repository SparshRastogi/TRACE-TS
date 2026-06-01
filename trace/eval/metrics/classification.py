"""Classification metric wrappers (accuracy, F1, per-class, confusion matrix)."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(gt_acts: list, pred_acts: list, all_classes: list) -> dict:
    acc = accuracy_score(gt_acts, pred_acts)
    # Macro averages over real classes only — "unknown" has support=0 (no GT sample is truly
    # unknown) so including it in the macro average artificially deflates F1 vs baselines.
    real_classes = [c for c in all_classes if c != "unknown"]
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        gt_acts, pred_acts, labels=real_classes, average="macro", zero_division=0)
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        gt_acts, pred_acts, labels=all_classes, average=None, zero_division=0)
    cm = confusion_matrix(gt_acts, pred_acts, labels=all_classes).tolist()
    per_class = {cls: {"precision": float(p_cls[i]), "recall": float(r_cls[i]),
                       "f1": float(f1_cls[i]), "support": int(sup_cls[i])}
                 for i, cls in enumerate(all_classes)}
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f1_mac),
        "per_class": per_class,
        "confusion_matrix": {"labels": all_classes, "matrix": cm},
    }
