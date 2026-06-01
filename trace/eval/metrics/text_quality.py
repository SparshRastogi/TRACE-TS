"""Text quality metrics: ROUGE-L, BERTScore, METEOR, sensor term recall.

Note: perplexity is excluded per project design decision.
"""

import numpy as np


def compute_sensor_term_recall(generated: str, reference: str, sensor_terms: list) -> float:
    ref_lower = reference.lower()
    gen_lower = generated.lower()
    present = [t for t in sensor_terms if t in ref_lower]
    if not present:
        return float("nan")
    return sum(1 for t in present if t in gen_lower) / len(present)


def compute_text_quality_metrics(generated_list: list, references_list: list,
                                  sensor_terms: list, device: str = "cpu",
                                  skip_model: bool = False) -> dict:
    """Compute ROUGE-L, BERTScore, METEOR, sensor term recall for a list of samples.

    Returns per-sample lists and corpus means.
    """
    from rouge_score import rouge_scorer as rouge_lib
    from nltk.translate.meteor_score import meteor_score as nltk_meteor

    rouge_scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge_scorer.score(ref, gen)["rougeL"].fmeasure
                    for gen, ref in zip(generated_list, references_list)]

    meteor_scores = [nltk_meteor([ref.split()], gen.split())
                     for gen, ref in zip(generated_list, references_list)]

    str_vals = [compute_sensor_term_recall(gen, ref, sensor_terms)
                for gen, ref in zip(generated_list, references_list)]
    str_clean = [v for v in str_vals if not np.isnan(v)]

    bertscore_mean = float("nan")
    bert_f1_list = [float("nan")] * len(generated_list)
    if not skip_model:
        try:
            from bert_score import score as bert_score_fn
            _, _, bert_f1_t = bert_score_fn(generated_list, references_list, lang="en",
                                            verbose=False, device=device)
            bert_f1_list = bert_f1_t.tolist()
            bertscore_mean = float(bert_f1_t.mean().item())
        except Exception as e:
            print(f"[eval] WARNING: BERTScore failed ({e})")

    return {
        "rouge_l": float(np.mean(rouge_scores)),
        "meteor": float(np.mean(meteor_scores)),
        "sensor_term_recall": float(np.mean(str_clean)) if str_clean else float("nan"),
        "bertscore_f1": bertscore_mean,
        # per-sample lists for conditioned breakdown
        "_rouge_scores": rouge_scores,
        "_meteor_scores": meteor_scores,
        "_str_vals": str_vals,
        "_bert_f1_list": bert_f1_list,
    }
