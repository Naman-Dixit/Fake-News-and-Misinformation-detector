"""
corroboration.py
Stage B of the AI pipeline: Cross Verification.

For each extracted claim, determines:
    - status: "Corroborated" | "Contradicted" | "No Tracked Records"
    - evidence_summary
    - domain_authority (tier of the alleged source, 1-3)
    - confidence (0-1)

Offline mode uses a deterministic rule-based heuristic driven by linguistic
hedging cues, source authority, and specificity of the claim (numbers, named
entities, direct attribution). API modes ask the selected LLM to reason over
its own knowledge and phrase a verification judgement in the same schema.
"""

import json
import re
import requests

from .utils import classify_source_tier, stable_unit_float, clamp

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "openai/gpt-oss-20b"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

HEDGE_WORDS = re.compile(
    r"\b(alleged|allegedly|reportedly|some say|rumor|rumour|unconfirmed|"
    r"claims that|sources say|it is said|supposedly|conspiracy|secretly)\b",
    re.I,
)

STRONG_MARKERS = re.compile(
    r"\b(confirmed|official|verified|announced|published|according to|study found|"
    r"data shows|reported by)\b",
    re.I,
)

SPECIFICITY_PATTERN = re.compile(r"\d")


def _evidence_summary(status: str, claim_statement: str, source_tier: int) -> str:
    tier_desc = {1: "a Tier 1 authoritative source", 2: "a Tier 2 mainstream outlet", 3: "an unverified Tier 3 source"}[source_tier]
    if status == "Corroborated":
        return f"Consistent language patterns and attribution markers align with reporting typically originating from {tier_desc}. No contradicting signals detected in the claim's phrasing."
    if status == "Contradicted":
        return f"The claim contains hedging or unverifiable language inconsistent with confirmed reporting, and is not typically associated with {tier_desc}."
    return f"No independent corroborating or contradicting markers were found for this claim relative to {tier_desc}; treated as an unverified data point."


def _score_claim_offline(claim_statement: str, alleged_source: str, article_source: str, source_tier: int):
    hedges = len(HEDGE_WORDS.findall(claim_statement))
    strong = len(STRONG_MARKERS.findall(claim_statement))
    specific = 1 if SPECIFICITY_PATTERN.search(claim_statement) else 0

    base = stable_unit_float(claim_statement + "|" + alleged_source)

    raw_score = (strong * 1.2) + (specific * 0.6) - (hedges * 1.5)
    # Blend in source authority: higher tier nudges toward corroboration
    tier_bonus = {1: 0.8, 2: 0.3, 3: -0.4}[source_tier]
    raw_score += tier_bonus
    # Small deterministic jitter so identical claims across different
    # sources/articles don't all collapse to the exact same bucket
    raw_score += (base - 0.5) * 0.6

    if raw_score >= 0.9:
        status = "Corroborated"
    elif raw_score <= -0.9:
        status = "Contradicted"
    else:
        status = "No Tracked Records"

    confidence = clamp(0.5 + abs(raw_score) * 0.12, 0.35, 0.97)
    return status, round(confidence, 2)


def verify_claims_offline(claims, article_source, topic):
    source_tier = classify_source_tier(article_source)
    results = []
    for claim in claims:
        status, confidence = _score_claim_offline(
            claim["statement"], claim.get("alleged_source", ""), article_source, source_tier
        )
        results.append({
            "claim_id": claim["claim_id"],
            "status": status,
            "evidence_summary": _evidence_summary(status, claim["statement"], source_tier),
            "domain_authority": source_tier,
            "confidence": confidence,
        })
    return results


def _build_prompt(claims, article_source, topic):
    claims_json = json.dumps(claims, indent=2)
    return f"""You are a cross-verification fact-checking engine. For each claim below, assess whether it is Corroborated, Contradicted, or has No Tracked Records, based on your knowledge and general plausibility. The article's source is "{article_source}" and topic is "{topic}".

Claims:
{claims_json}

Return ONLY a JSON array (no prose, no markdown fences) where each element has exactly these fields:
- claim_id (string, matching the input claim_id)
- status (string, one of: "Corroborated", "Contradicted", "No Tracked Records")
- evidence_summary (string, 1-2 sentences explaining the judgement)
- domain_authority (integer 1-3, your estimate of the alleged source's authority tier)
- confidence (float between 0 and 1)
"""


def _parse_json_array(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.S)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def verify_claims_nvidia(claims, article_source, topic, api_key):
    prompt = _build_prompt(claims, article_source, topic)
    resp = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": NVIDIA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_json_array(content)


def verify_claims_claude(claims, article_source, topic, api_key):
    prompt = _build_prompt(claims, article_source, topic)
    resp = requests.post(
        CLAUDE_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return _parse_json_array("\n".join(text_blocks))


def verify_claims(claims, article_source, topic, api_mode="offline", api_key=None):
    try:
        if api_mode == "nvidia" and api_key:
            results = verify_claims_nvidia(claims, article_source, topic, api_key)
        elif api_mode == "claude" and api_key:
            results = verify_claims_claude(claims, article_source, topic, api_key)
        else:
            results = verify_claims_offline(claims, article_source, topic)

        if not results or not isinstance(results, list):
            raise ValueError("Empty or malformed verification list")
        return results
    except Exception:
        return verify_claims_offline(claims, article_source, topic)
