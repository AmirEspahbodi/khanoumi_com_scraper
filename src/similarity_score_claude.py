"""
cosmetic_similarity.py
======================
Production-ready cosmetic product name similarity scorer.

Matches Persian/English skincare product names scraped from e-commerce
sites against a verified Excel database. Implements a zero-false-positive
"Gated Lexical Scoring" architecture:

  Phase 1 → Aggressive, deterministic text normalization
  Phase 2 → Hard Gates: zero-tolerance attribute extraction + clash detection
  Phase 3 → Permutation-invariant fuzzy similarity (thefuzz.token_set_ratio)

A single digit or volume difference (NC41 ↔ NC42, 50ml ↔ 100ml) is treated
as a completely different product and returns 0.0 immediately.

Dependencies:
    pip install thefuzz python-Levenshtein

Usage:
    from cosmetic_similarity import similarity_score
    score = similarity_score("فاندیشن ان سی ۴۱", "فاندیشن NC41")  # → 1.0
"""

import re
from typing import FrozenSet

from thefuzz import fuzz

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL COMPILED PATTERNS
#  Compiled once at import time → zero per-call regex compilation overhead.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Phase 1 · Character-level translation tables ──────────────────────────────

# Arabic ي (U+064A) and ك (U+0643) → Persian ی (U+06CC) and ک (U+06A9)
_TBL_ARABIC_TO_PERSIAN = str.maketrans("\u064a\u0643", "\u06cc\u06a9")

# Persian-Indic digits ۰-۹ (U+06F0–U+06F9) → ASCII 0-9
_TBL_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# ── Phase 1 · Unit normalization ──────────────────────────────────────────────

# Volume → "XXml"
_RE_VOLUME_ML = re.compile(
    r"(\d+)\s*(?:"
    r"میلی\s*لیتر"
    r"|میل(?!ی)"
    r"|milliliter"
    r"|ml"
    r"|م(?!\w)"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Weight → "XXg"
_RE_WEIGHT_G = re.compile(
    r"(\d+)\s*(?:گرم|grams?|gr(?!\w)|g(?!\w))",
    re.IGNORECASE | re.UNICODE,
)

# ── Phase 1 · Phonetic Persian → ASCII code transliteration ──────────────────
_PHONETIC_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bاس\s*پی\s*اف\b", re.UNICODE), "spf"),
    (re.compile(r"\bان\s*سی\b", re.UNICODE), "nc"),
    (re.compile(r"\bاف\s*پی\b", re.UNICODE), "fp"),
    (re.compile(r"\bاس\s*پی\b", re.UNICODE), "sp"),
    (re.compile(r"\bبی\s*بی\b", re.UNICODE), "bb"),
    (re.compile(r"\bسی\s*سی\b", re.UNICODE), "cc"),
    (re.compile(r"\bای\s*دی\b", re.UNICODE), "id"),
    (re.compile(r"\bای\s*پی\b", re.UNICODE), "ip"),
    (re.compile(r"\bام\b", re.UNICODE), "m"),
]

_RE_DESP_LETTERS_DIGITS = re.compile(r"([a-z]+)\s+(\d+)", re.IGNORECASE)

# ── Phase 1 · Persian e-commerce stop-words ───────────────────────────────────
_RE_STOP_WORDS = re.compile(
    r"\b(?:مدل|شماره|حجم|زنانه|مردانه|رنگ|حاوی|مناسب|سری|ست)\b",
    re.UNICODE,
)

# ── Phase 1 · Whitespace collapse ─────────────────────────────────────────────
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# ── Phase 2 · Post-normalization extraction ───────────────────────────────────

# Measurement tokens: e.g. "100ml", "50g"
_RE_EXTRACT_MEASURE = re.compile(r"\b(\d+(?:ml|g))\b", re.IGNORECASE)

# Alphanumeric code tokens (shade codes, SPF values, model numbers):
# [FIXED EDGE-CASE 1]: Added \d+ to also capture pure numbers like "01" or "010"
_RE_EXTRACT_CODE = re.compile(r"\b([a-z]+\d+|\d+[a-z]+|\d+)\b", re.IGNORECASE)

# Pure ASCII word tokens (no digits), minimum 3 chars
_RE_EXTRACT_ENGLISH_WORD = re.compile(r"\b([a-z]{3,})\b", re.IGNORECASE)

_ENGLISH_GATE_EXCLUSIONS: FrozenSet[str] = frozenset(
    {
        "ml",
        "g",
        "gr",
        "the",
        "and",
        "for",
        "with",
        "new",
        "pro",
        "plus",
        "max",
        "mini",
        "de",
        "la",
        "le",
        "du",
        "von",
        "van",
        "eau",
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_text(text: str) -> str:
    text = text.translate(_TBL_ARABIC_TO_PERSIAN)
    text = text.replace("\u200c", " ")
    text = text.translate(_TBL_PERSIAN_DIGITS)
    text = _RE_VOLUME_ML.sub(lambda m: f"{m.group(1)}ml", text)
    text = _RE_WEIGHT_G.sub(lambda m: f"{m.group(1)}g", text)

    for pattern, replacement in _PHONETIC_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    text = _RE_DESP_LETTERS_DIGITS.sub(r"\1\2", text)
    text = _RE_DESP_LETTERS_DIGITS.sub(r"\1\2", text)
    text = _RE_STOP_WORDS.sub(" ", text)
    text = text.lower()
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — ATTRIBUTE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_measures(norm_text: str) -> FrozenSet[str]:
    return frozenset(tok.lower() for tok in _RE_EXTRACT_MEASURE.findall(norm_text))


def _extract_codes(norm_text: str) -> FrozenSet[str]:
    candidates = frozenset(c.lower() for c in _RE_EXTRACT_CODE.findall(norm_text))
    # Exclude measurement tokens — they are alphanumeric but belong to Gate 1
    return candidates - _extract_measures(norm_text)


def _extract_english_content_words(norm_text: str) -> FrozenSet[str]:
    all_words = frozenset(
        w.lower() for w in _RE_EXTRACT_ENGLISH_WORD.findall(norm_text)
    )
    return all_words - _ENGLISH_GATE_EXCLUSIONS


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — HARD-GATE CLASH DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════


def _measures_clash(
    excel_measures: FrozenSet[str],
    site_measures: FrozenSet[str],
) -> bool:
    if excel_measures or site_measures:
        return excel_measures != site_measures
    return False


def _codes_clash(
    excel_codes: FrozenSet[str],
    site_codes: FrozenSet[str],
) -> bool:
    if excel_codes or site_codes:
        return excel_codes != site_codes
    return False


def _english_words_clash(
    excel_words: FrozenSet[str],
    site_norm: str,
) -> bool:
    if not excel_words:
        return False

    for word in excel_words:
        if word in site_norm:
            continue

        if len(word) < 5:
            return True

        word_len = len(word)
        found = any(
            fuzz.ratio(word, site_norm[i : i + word_len]) >= 88
            for i in range(max(1, len(site_norm) - word_len + 1))
        )
        if not found:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════


def similarity_score(site_product_name: str, excel_product_name: str) -> float:
    if not site_product_name or not excel_product_name:
        return 0.0

    norm_site = _normalize_text(site_product_name)
    norm_excel = _normalize_text(excel_product_name)

    if norm_site == norm_excel:
        return 1.0

    excel_measures = _extract_measures(norm_excel)
    site_measures = _extract_measures(norm_site)
    if _measures_clash(excel_measures, site_measures):
        return 0.0

    excel_codes = _extract_codes(norm_excel)
    site_codes = _extract_codes(norm_site)
    if _codes_clash(excel_codes, site_codes):
        return 0.0

    excel_words = _extract_english_content_words(norm_excel)
    if _english_words_clash(excel_words, norm_site):
        return 0.0

    set_ratio = fuzz.token_set_ratio(norm_site, norm_excel)
    sort_ratio = fuzz.token_sort_ratio(norm_site, norm_excel)

    raw = (set_ratio * 0.4) + (sort_ratio * 0.6)

    return round(raw / 100.0, 4)
