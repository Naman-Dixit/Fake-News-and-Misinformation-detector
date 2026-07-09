"""
ensemble_engine.py
Combines Stage A (claims), Stage B (corroboration), and Stage C (semantics)
into a final credibility score and verdict.

Per-claim score:
    S_c = (W * 0.7) + (L / 5 * 0.3)
    where W: Corroborated = 1.0, No Tracked Records = 0.5, Contradicted = 0
          L: consistency_score (1-5) from Stage C (shared across all claims
             in the article, since semantic analysis operates on the whole
             article rather than per-claim)

Final score:
    CS_A = mean(S_c) * AuthorityMultiplier * 100

Voting:
    CS_A >= 70            -> True
    40 <= CS_A < 70        -> Suspect
    CS_A < 40              -> False

Model disagreement is reported as the population standard deviation of the
per-claim S_c values (scaled to 0-100), reflecting how much the individual
claims disagree with one another once corroboration is factored in.
"""

import statistics
from .utils import classify_source_tier, authority_multiplier, tier_label, clamp

W_MAP = {
    "Corroborated": 1.0,
    "No Tracked Records": 0.5,
    "Contradicted": 0.0,
}


def _claim_score(status: str, consistency_score: int) -> float:
    w = W_MAP.get(status, 0.5)
    l = consistency_score
    return (w * 0.7) + ((l / 5) * 0.3)


def run_ensemble(claims, verification_results, semantics, article_source):
    """
    claims: list of Stage A claim dicts (claim_id, statement, ...)
    verification_results: list of Stage B dicts (claim_id, status, confidence, domain_authority, ...)
    semantics: Stage C dict (includes consistency_score)
    article_source: str, the article's declared source name
    """
    verification_by_id = {v["claim_id"]: v for v in verification_results}
    consistency_score = semantics.get("consistency_score", 3)

    per_claim_scores = []
    claim_breakdown = []

    for claim in claims:
        cid = claim["claim_id"]
        verification = verification_by_id.get(cid, {"status": "No Tracked Records", "confidence": 0.5, "domain_authority": 3})
        status = verification.get("status", "No Tracked Records")
        s_c = _claim_score(status, consistency_score)
        per_claim_scores.append(s_c)

        claim_breakdown.append({
            "claim_id": cid,
            "statement": claim["statement"],
            "alleged_source": claim.get("alleged_source", "Unattributed"),
            "verification_category": claim.get("verification_category", "General Factual Claim"),
            "status": status,
            "evidence_summary": verification.get("evidence_summary", ""),
            "domain_authority": verification.get("domain_authority", 3),
            "confidence": verification.get("confidence", 0.5),
            "claim_score": round(s_c * 100, 1),
        })

    if per_claim_scores:
        mean_s_c = sum(per_claim_scores) / len(per_claim_scores)
    else:
        mean_s_c = 0.5

    source_tier = classify_source_tier(article_source)
    multiplier = authority_multiplier(source_tier)

    cs_a = mean_s_c * multiplier * 100
    cs_a = round(clamp(cs_a, 0, 100), 1)

    if cs_a >= 70:
        verdict = "True"
    elif cs_a >= 40:
        verdict = "Suspect"
    else:
        verdict = "False"

    if len(per_claim_scores) > 1:
        disagreement_raw = statistics.pstdev(per_claim_scores) * 100
    else:
        disagreement_raw = 0.0
    disagreement_raw = round(disagreement_raw, 1)

    if disagreement_raw < 10:
        disagreement_label = "Low"
    elif disagreement_raw < 25:
        disagreement_label = "Medium"
    else:
        disagreement_label = "High"

    return {
        "credibility_score": cs_a,
        "verdict": verdict,
        "source_tier": source_tier,
        "source_tier_label": tier_label(source_tier),
        "authority_multiplier": multiplier,
        "model_disagreement": {
            "value": disagreement_raw,
            "label": disagreement_label,
        },
        "claim_count": len(claims),
        "claims": claim_breakdown,
        "semantic_analysis": semantics,
    }
