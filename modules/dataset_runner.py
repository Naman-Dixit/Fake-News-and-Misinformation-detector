"""
dataset_runner.py
Runs the full three-stage AI pipeline over every row of an uploaded CSV
dataset and computes aggregate classification metrics against ground_truth
(when present).

Expected CSV columns:
    article_id, title, text, source, topic, ground_truth (optional)

Produces, per row:
    article_id, title, topic, source, credibility_score, ensemble_vote,
    claim_count, model_disagreement, ground_truth, match
"""

from .claim_extractor import extract_claims
from .corroboration import verify_claims
from .semantic_analysis import analyze_semantics
from .ensemble_engine import run_ensemble

LABELS = ["True", "Suspect", "False"]


def run_pipeline_for_row(title, text, source, topic, api_mode="offline", api_key=None):
    claims = extract_claims(title, text, source, topic, api_mode=api_mode, api_key=api_key)
    verifications = verify_claims(claims, source, topic, api_mode=api_mode, api_key=api_key)
    semantics = analyze_semantics(title, text, api_mode=api_mode, api_key=api_key)
    result = run_ensemble(claims, verifications, semantics, source)
    return result


def _normalize_label(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("true", "real", "1", "yes"):
        return "True"
    if v in ("false", "fake", "0", "no"):
        return "False"
    if v in ("suspect", "suspicious", "unclear", "unverified"):
        return "Suspect"
    return None


def run_bulk_dataset(rows, api_mode="offline", api_key=None):
    """
    rows: list of dicts parsed from the uploaded CSV (csv.DictReader output)
    Returns: (results list, metrics dict or None)
    """
    results = []
    has_ground_truth = False

    for row in rows:
        article_id = row.get("article_id", "").strip()
        title = row.get("title", "").strip()
        text = row.get("text", "").strip()
        source = row.get("source", "").strip()
        topic = row.get("topic", "").strip()
        ground_truth_raw = row.get("ground_truth", "")
        ground_truth = _normalize_label(ground_truth_raw)
        if ground_truth:
            has_ground_truth = True

        pipeline_result = run_pipeline_for_row(title, text, source, topic, api_mode=api_mode, api_key=api_key)

        match = None
        if ground_truth:
            match = (ground_truth == pipeline_result["verdict"])

        results.append({
            "article_id": article_id,
            "title": title,
            "topic": topic,
            "source": source,
            "credibility_score": pipeline_result["credibility_score"],
            "ensemble_vote": pipeline_result["verdict"],
            "claim_count": pipeline_result["claim_count"],
            "model_disagreement": pipeline_result["model_disagreement"]["label"],
            "model_disagreement_value": pipeline_result["model_disagreement"]["value"],
            "ground_truth": ground_truth_raw,
            "match": match,
            "full_result": pipeline_result,
        })

    metrics = compute_metrics(results) if has_ground_truth else None
    return results, metrics


def compute_metrics(results):
    """
    Computes accuracy, macro precision/recall/f1, and a confusion matrix
    over the 3-class label space (True / Suspect / False), using only rows
    that have a usable ground_truth label.
    """
    labeled = [r for r in results if _normalize_label(r["ground_truth"])]
    if not labeled:
        return None

    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}

    correct = 0
    for r in labeled:
        actual = _normalize_label(r["ground_truth"])
        predicted = r["ensemble_vote"]
        confusion[actual][predicted] += 1
        if actual == predicted:
            correct += 1

    total = len(labeled)
    accuracy = round(correct / total, 4) if total else 0.0

    precisions, recalls, f1s = [], [], []
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Only classes that actually appear in the ground truth (support > 0)
        # count toward the macro average, so absent classes don't artificially
        # deflate the reported precision/recall/f1.
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_precision = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    macro_recall = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    macro_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": LABELS,
        "total_labeled": total,
    }
