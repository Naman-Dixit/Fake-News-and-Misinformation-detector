"""
utils.py
Shared helper utilities for the Fake News & Misinformation Detector:
- Source authority tier classification
- Text cleaning / sentence splitting
- Deterministic hashing helpers (used by the offline heuristics so that
  repeated runs on the same input are stable and reproducible)
"""

import re
import hashlib


# ---------------------------------------------------------------------------
# Source Authority Classification
# ---------------------------------------------------------------------------

TIER_1_SOURCES = [
    "reuters", "associated press", "ap news", "nasa", "who",
    "world health organization", "nature", "science", "the lancet",
    "nih", "cdc", "fda", "united nations", "un.org", "europa.eu",
    "gov.uk", ".gov", "government", "ministry", "university",
    "harvard", "mit", "stanford", "oxford", "cambridge", "npr",
    "bbc", "pnas", "jama", "nejm", "un ", "noaa", "eu commission",
]

TIER_2_SOURCES = [
    "cnn", "the new york times", "nytimes", "washington post",
    "the guardian", "al jazeera", "bloomberg", "wall street journal",
    "wsj", "the economist", "time", "forbes", "abc news", "cbs news",
    "nbc news", "fox news", "usa today", "politico", "axios",
    "the hill", "financial times", "sky news", "cbc", "abc.net",
]

TIER_3_HINTS = [
    "blog", "wordpress", "blogspot", "infowars", "naturalnews",
    "beforeitsnews", "anonymous", "conspiracy", "truth", "patriot",
    ".ru", ".xyz", "wakeup", "exposed", "real news network",
]

AUTHORITY_MULTIPLIER = {
    1: 1.00,
    2: 0.85,
    3: 0.50,
}


def classify_source_tier(source_name: str) -> int:
    """
    Classify a source name into an authority tier (1, 2, or 3).

    Tier 1: Government bodies, major wire services, scientific institutions,
            universities, and international bodies.
    Tier 2: Recognized mainstream media outlets.
    Tier 3: Unknown blogs, unverified sites, or conspiracy-associated domains.

    Defaults to Tier 3 when the source cannot be confidently classified,
    since unverified sources should not receive the benefit of the doubt.
    """
    if not source_name:
        return 3

    s = source_name.strip().lower()

    for hint in TIER_1_SOURCES:
        if hint in s:
            return 1

    for hint in TIER_2_SOURCES:
        if hint in s:
            return 2

    for hint in TIER_3_HINTS:
        if hint in s:
            return 3

    # Unknown / unrecognized source -> Tier 3 by default (conservative)
    return 3


def tier_label(tier: int) -> str:
    return {1: "Tier 1 — Authoritative", 2: "Tier 2 — Mainstream Media", 3: "Tier 3 — Unverified / Low Authority"}.get(tier, "Tier 3 — Unverified / Low Authority")


def authority_multiplier(tier: int) -> float:
    return AUTHORITY_MULTIPLIER.get(tier, 0.50)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    # Filter out very short fragments
    return [p.strip() for p in parts if len(p.strip()) > 15]


def stable_hash(value: str) -> int:
    """Return a deterministic integer hash for a string (stable across runs,
    unlike Python's built-in hash() which is randomized per-process)."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def stable_unit_float(value: str) -> float:
    """Deterministic pseudo-random float in [0, 1) derived from a string."""
    return (stable_hash(value) % 100000) / 100000.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))
