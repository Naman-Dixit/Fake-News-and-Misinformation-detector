"""
semantic_analysis.py
Stage C of the AI pipeline: Semantic Analysis.

Detects, for the article as a whole:
    - bias
    - propaganda
    - logical fallacies
    - emotional manipulation
    - context stripping
    - consistency_score (1-5, integer; used as "L" in the ensemble formula)
"""

import json
import re
import requests

from .utils import clamp

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "openai/gpt-oss-20b"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

BIAS_WORDS = re.compile(
    r"\b(radical|extremist|regime|puppet|elite|mainstream media|deep state|"
    r"so-called|globalist|agenda|sheeple|woke|libtard|snowflake)\b", re.I
)

PROPAGANDA_PATTERNS = re.compile(
    r"\b(wake up|they don't want you to know|mainstream media won't tell you|"
    r"do your own research|share before it'?s deleted|banned information|"
    r"the truth they hide|patriots must|share this now)\b", re.I
)

FALLACY_PATTERNS = re.compile(
    r"\b(everyone knows|obviously|clearly this proves|it's common sense|"
    r"if .* then .* must|either .* or .*|all .* are|no true|slippery slope|"
    r"this always leads to)\b", re.I
)

EMOTIONAL_WORDS = re.compile(
    r"\b(shocking|terrifying|outrageous|horrifying|devastating|catastrophic|"
    r"unbelievable|you won't believe|heartbreaking|explosive|bombshell|"
    r"scandal|nightmare|panic|chaos|disaster)\b", re.I
)

CONTEXT_STRIPPING_PATTERNS = re.compile(
    r"\b(taken out of context|edited clip|partial quote|clip shows|"
    r"in a since-deleted|a resurfaced)\b", re.I
)

EXCLAMATION_PATTERN = re.compile(r"!")
ALLCAPS_WORD_PATTERN = re.compile(r"\b[A-Z]{4,}\b")


def analyze_semantics_offline(title: str, text: str):
    full_text = f"{title} {text}"

    bias_hits = len(BIAS_WORDS.findall(full_text))
    propaganda_hits = len(PROPAGANDA_PATTERNS.findall(full_text))
    fallacy_hits = len(FALLACY_PATTERNS.findall(full_text))
    emotional_hits = len(EMOTIONAL_WORDS.findall(full_text)) + len(EXCLAMATION_PATTERN.findall(full_text)) // 2
    context_hits = len(CONTEXT_STRIPPING_PATTERNS.findall(full_text))
    caps_hits = len(ALLCAPS_WORD_PATTERN.findall(full_text))

    bias_detected = bias_hits > 0
    propaganda_detected = propaganda_hits > 0
    fallacy_detected = fallacy_hits > 0
    emotional_detected = (emotional_hits + caps_hits) > 0
    context_stripping_detected = context_hits > 0

    total_flags = (
        bias_hits + propaganda_hits * 1.5 + fallacy_hits + emotional_hits * 0.7
        + context_hits * 1.5 + caps_hits * 0.5
    )

    # Consistency score: starts at 5 (fully consistent / neutral) and is
    # penalized by manipulation signals found in the text.
    consistency_score = 5 - clamp(round(total_flags / 2), 0, 4)
    consistency_score = int(clamp(consistency_score, 1, 5))

    return {
        "bias": {
            "detected": bias_detected,
            "detail": f"{bias_hits} loaded/partisan term(s) detected." if bias_detected else "No strongly loaded partisan language detected.",
        },
        "propaganda": {
            "detected": propaganda_detected,
            "detail": f"{propaganda_hits} propaganda-style phrase(s) detected (e.g. calls to share, distrust framing)." if propaganda_detected else "No propaganda-style rhetorical patterns detected.",
        },
        "logical_fallacies": {
            "detected": fallacy_detected,
            "detail": f"{fallacy_hits} potential logical fallacy pattern(s) detected." if fallacy_detected else "No obvious logical fallacy patterns detected.",
        },
        "emotional_manipulation": {
            "detected": emotional_detected,
            "detail": f"{emotional_hits} sensationalized/emotionally-charged term(s) and {caps_hits} all-caps emphasis word(s) detected." if emotional_detected else "Tone reads as measured rather than sensationalized.",
        },
        "context_stripping": {
            "detected": context_stripping_detected,
            "detail": f"{context_hits} marker(s) suggesting content may be stripped of context." if context_stripping_detected else "No indicators of selectively stripped context detected.",
        },
        "consistency_score": consistency_score,
    }


def _build_prompt(title, text):
    return f"""You are a semantic analysis engine for misinformation detection. Analyze the article below for bias, propaganda techniques, logical fallacies, emotional manipulation, and context stripping.

Title: {title}
Body: {text}

Return ONLY a JSON object (no prose, no markdown fences) with exactly this shape:
{{
  "bias": {{"detected": true/false, "detail": "short explanation"}},
  "propaganda": {{"detected": true/false, "detail": "short explanation"}},
  "logical_fallacies": {{"detected": true/false, "detail": "short explanation"}},
  "emotional_manipulation": {{"detected": true/false, "detail": "short explanation"}},
  "context_stripping": {{"detected": true/false, "detail": "short explanation"}},
  "consistency_score": integer from 1 to 5 (5 = highly consistent/neutral writing, 1 = highly manipulative/inconsistent)
}}
"""


def _parse_json_object(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def analyze_semantics_nvidia(title, text, api_key):
    prompt = _build_prompt(title, text)
    resp = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": NVIDIA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_json_object(content)


def analyze_semantics_claude(title, text, api_key):
    prompt = _build_prompt(title, text)
    resp = requests.post(
        CLAUDE_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return _parse_json_object("\n".join(text_blocks))


def analyze_semantics(title, text, api_mode="offline", api_key=None):
    try:
        if api_mode == "nvidia" and api_key:
            result = analyze_semantics_nvidia(title, text, api_key)
        elif api_mode == "claude" and api_key:
            result = analyze_semantics_claude(title, text, api_key)
        else:
            result = analyze_semantics_offline(title, text)

        required = {"bias", "propaganda", "logical_fallacies", "emotional_manipulation", "context_stripping", "consistency_score"}
        if not required.issubset(result.keys()):
            raise ValueError("Malformed semantic analysis result")
        result["consistency_score"] = int(clamp(result["consistency_score"], 1, 5))
        return result
    except Exception:
        return analyze_semantics_offline(title, text)
