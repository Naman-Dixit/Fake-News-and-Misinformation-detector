"""
claim_extractor.py
Stage A of the AI pipeline: Claim Extraction.

Extracts 3-10 discrete factual claims from an article and returns them as
structured JSON:
    claim_id, statement, alleged_source, verification_category

Supports three API modes:
    1. offline  -> rule-based, no external API calls
    2. nvidia   -> NVIDIA NIM (integrate.api.nvidia.com), model openai/gpt-oss-20b
    3. claude   -> Anthropic Claude API
"""

import json
import re
import requests

from .utils import split_sentences, stable_hash, clamp

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "openai/gpt-oss-20b"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

MAX_CLAIMS = 10
MIN_CLAIMS = 3

STAT_PATTERN = re.compile(r"\b\d+(\.\d+)?\s?(%|percent|million|billion|thousand)\b", re.I)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
QUOTE_PATTERN = re.compile(r"[\"\u201c].{5,}[\"\u201d]")
ATTRIBUTION_PATTERN = re.compile(
    r"\baccording to ((?:the )?[A-Z][\w&.'-]*(?:\s+(?:of|for|and)?\s?[A-Z][\w&.'-]*){0,4})"
)
SCIENCE_KEYWORDS = re.compile(
    r"\b(study|research|scientists|researchers|trial|data shows|survey|report shows)\b",
    re.I,
)


def _verification_category(sentence: str) -> str:
    if STAT_PATTERN.search(sentence):
        return "Statistical Claim"
    if QUOTE_PATTERN.search(sentence):
        return "Direct Quote / Attribution"
    if SCIENCE_KEYWORDS.search(sentence):
        return "Scientific / Research Claim"
    if YEAR_PATTERN.search(sentence):
        return "Historical / Event Claim"
    return "General Factual Claim"


def _alleged_source(sentence: str, fallback_source: str) -> str:
    match = ATTRIBUTION_PATTERN.search(sentence)
    if match:
        return match.group(1).strip().rstrip(".,")
    return fallback_source or "Unattributed"


def _score_sentence(sentence: str) -> float:
    """Heuristic 'claim-worthiness' score used to rank candidate sentences."""
    score = 0.0
    if STAT_PATTERN.search(sentence):
        score += 3
    if YEAR_PATTERN.search(sentence):
        score += 2
    if QUOTE_PATTERN.search(sentence):
        score += 2
    if ATTRIBUTION_PATTERN.search(sentence):
        score += 2
    if SCIENCE_KEYWORDS.search(sentence):
        score += 2
    # Capitalized proper-noun density as a weak signal of specific/checkable content
    caps = len(re.findall(r"\b[A-Z][a-z]{2,}\b", sentence))
    score += min(caps * 0.3, 2)
    # Length sanity: prefer medium-length declarative sentences
    length = len(sentence.split())
    if 8 <= length <= 40:
        score += 1
    return score


def extract_claims_offline(title: str, text: str, source: str, topic: str):
    full_text = f"{title.strip()}. {text.strip()}" if title else text.strip()
    sentences = split_sentences(full_text)

    if not sentences:
        return []

    ranked = sorted(sentences, key=lambda s: (-_score_sentence(s), stable_hash(s)))

    n_claims = clamp(len(sentences), MIN_CLAIMS, MAX_CLAIMS)
    selected = ranked[:n_claims]

    # Preserve original article order for readability
    selected_sorted = sorted(selected, key=lambda s: sentences.index(s))

    claims = []
    for i, sentence in enumerate(selected_sorted, start=1):
        claims.append({
            "claim_id": f"C{i:02d}",
            "statement": sentence.strip(),
            "alleged_source": _alleged_source(sentence, source),
            "verification_category": _verification_category(sentence),
        })
    return claims


def _build_prompt(title, text, source, topic):
    return f"""You are a fact-checking claim extraction engine. Extract between 3 and 10 discrete, checkable factual claims from the article below.

Title: {title}
Source: {source}
Topic: {topic}
Article Body:
{text}

Return ONLY a JSON array (no prose, no markdown fences) where each element has exactly these fields:
- claim_id (string, e.g. "C01")
- statement (string, the factual claim as a standalone sentence)
- alleged_source (string, who/what the article attributes this claim to, or "Unattributed")
- verification_category (string, one of: "Statistical Claim", "Direct Quote / Attribution", "Scientific / Research Claim", "Historical / Event Claim", "General Factual Claim")
"""


def _parse_json_array(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.S)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def extract_claims_nvidia(title, text, source, topic, api_key):
    prompt = _build_prompt(title, text, source, topic)
    resp = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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


def extract_claims_claude(title, text, source, topic, api_key):
    prompt = _build_prompt(title, text, source, topic)
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


def extract_claims(title, text, source, topic, api_mode="offline", api_key=None):
    """
    Main entry point for Stage A.
    api_mode: "offline" | "nvidia" | "claude"
    Falls back to the offline heuristic if an API call fails, so the
    pipeline never breaks the user's request.
    """
    try:
        if api_mode == "nvidia" and api_key:
            claims = extract_claims_nvidia(title, text, source, topic, api_key)
        elif api_mode == "claude" and api_key:
            claims = extract_claims_claude(title, text, source, topic, api_key)
        else:
            claims = extract_claims_offline(title, text, source, topic)

        if not claims or not isinstance(claims, list):
            raise ValueError("Empty or malformed claim list")
        return claims[:MAX_CLAIMS]
    except Exception:
        return extract_claims_offline(title, text, source, topic)
