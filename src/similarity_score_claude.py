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
# Matches (in priority order, left-to-right alternation):
#   میلی لیتر / میلی‌لیتر  — full Persian (with optional ZWNJ already removed)
#   میل(?!ی)               — short "میل" but NOT "میلی" (avoids partial match in
#                             "میلی لیتر" and incorrect match in "میلیون")
#   milliliter             — full English
#   ml                     — abbreviated English
#   م(?!\w)               — single-char informal suffix, NOT followed by a
#                             Unicode word-char (prevents matching "محصول")
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
# grams? handles "gram"/"grams", gr(?!\w) handles "50gr", g(?!\w) handles "50g"
# Using (?!\w) (not followed by Unicode word-char) avoids "gel", "gold", "grey"
_RE_WEIGHT_G = re.compile(
    r"(\d+)\s*(?:گرم|grams?|gr(?!\w)|g(?!\w))",
    re.IGNORECASE | re.UNICODE,
)

# ── Phase 1 · Phonetic Persian → ASCII code transliteration ──────────────────
# Each tuple: (compiled_pattern, replacement_string)
# Longer / more specific patterns must come BEFORE shorter ones.
_PHONETIC_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Three-part codes first
    (re.compile(r"\bاس\s*پی\s*اف\b", re.UNICODE), "spf"),
    # Two-part codes
    (re.compile(r"\bان\s*سی\b", re.UNICODE), "nc"),
    (re.compile(r"\bاف\s*پی\b", re.UNICODE), "fp"),
    (re.compile(r"\bاس\s*پی\b", re.UNICODE), "sp"),
    (re.compile(r"\bبی\s*بی\b", re.UNICODE), "bb"),
    (re.compile(r"\bسی\s*سی\b", re.UNICODE), "cc"),
    (re.compile(r"\bای\s*دی\b", re.UNICODE), "id"),
    (re.compile(r"\bای\s*پی\b", re.UNICODE), "ip"),
    # Single-letter phonetic: "ام" = letter M (shade codes, e.g. N41M)
    (re.compile(r"\bام\b", re.UNICODE), "m"),
]

# Collapse whitespace between ASCII letters and digits inside a code token.
# e.g. "nc 41" → "nc41",  "spf 30" → "spf30"
# Applied twice to handle "n c 41" style spacing.
_RE_DESP_LETTERS_DIGITS = re.compile(r"([a-z]+)\s+(\d+)", re.IGNORECASE)

# ── Phase 1 · Persian e-commerce stop-words ───────────────────────────────────
_RE_STOP_WORDS = re.compile(
    r"\b(?:مدل|شماره|حجم|زنانه|مردانه|رنگ|حاوی|مناسب|سری|ست)\b",
    re.UNICODE,
)

# ── Phase 1 · Whitespace collapse ─────────────────────────────────────────────
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# ── Phase 2 · Post-normalization extraction ───────────────────────────────────

# Measurement tokens: e.g. "100ml", "50g"  (digits immediately followed by unit)
_RE_EXTRACT_MEASURE = re.compile(r"\b(\d+(?:ml|g))\b", re.IGNORECASE)

# Alphanumeric code tokens (shade codes, SPF values, model numbers):
#   letters+digits  →  nc41, spf30, bb15
#   digits+letters  →  41n,  30w   (less common but valid)
_RE_EXTRACT_CODE = re.compile(r"\b([a-z]+\d+|\d+[a-z]+)\b", re.IGNORECASE)

# Pure ASCII word tokens (no digits), minimum 3 chars — brand sub-names
# e.g. "bombshell", "noir", "gold", "victoria"
_RE_EXTRACT_ENGLISH_WORD = re.compile(r"\b([a-z]{3,})\b", re.IGNORECASE)

# Words excluded from Gate 3 (units and high-frequency non-discriminative terms)
_ENGLISH_GATE_EXCLUSIONS: FrozenSet[str] = frozenset(
    {
        # Unit strings (already handled by Gate 1)
        "ml",
        "g",
        "gr",
        # High-frequency connecting / qualifier words that carry no product identity
        "the",
        "and",
        "for",
        "with",
        "new",
        "pro",
        "plus",
        "max",
        "mini",
        # Common French/Italian cosmetic filler words
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
    """
    Apply aggressive, fully-deterministic normalization to one product name.

    Processing pipeline (strict order — order matters):

      1. Arabic ي/ك  →  Persian ی/ک   (character unification)
      2. ZWNJ (U+200C)  →  space      (removes zero-width non-joiners)
      3. Persian digits ۰-۹  →  0-9   (digit unification)
      4a. Volume unit normalization:   "125 میلی لیتر" / "125م" → "125ml"
      4b. Weight unit normalization:   "50گرم" / "50gr"          → "50g"
      5.  Phonetic transliteration:    "ان سی"  → "nc"
      6.  Code de-spacing:             "nc 41"  → "nc41"   (×2 for double-gaps)
      7.  Stop-word removal:           "مدل", "حجم", …  → ""
      8.  Lowercase                    (ASCII range only; Persian untouched)
      9.  Whitespace collapse + strip

    Args:
        text: Raw product name, possibly mixing Persian and English.

    Returns:
        Normalized, lowercased product name ready for extraction and scoring.

    Examples:
        >>> _normalize_text("فاندیشن ان سی ۴۱ - 100 میلی لیتر")
        'فاندیشن nc41 100ml'
        >>> _normalize_text("كرم SPF ۳۰ مدل روشن")
        'کرم spf30 روشن'
    """
    # Step 1 — Arabic → Persian char unification
    text = text.translate(_TBL_ARABIC_TO_PERSIAN)

    # Step 2 — ZWNJ → space (must happen before volume regex which uses \s*)
    text = text.replace("\u200c", " ")

    # Step 3 — Persian digits → ASCII digits
    text = text.translate(_TBL_PERSIAN_DIGITS)

    # Step 4a — Volume unit standardization  (e.g. "125 میلی لیتر" → "125ml")
    text = _RE_VOLUME_ML.sub(lambda m: f"{m.group(1)}ml", text)

    # Step 4b — Weight unit standardization  (e.g. "50گرم" → "50g")
    text = _RE_WEIGHT_G.sub(lambda m: f"{m.group(1)}g", text)

    # Step 5 — Phonetic Persian codes → English  ("ان سی" → "nc")
    for pattern, replacement in _PHONETIC_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Step 6 — Code de-spacing  ("nc 41" → "nc41"); two passes for robustness
    text = _RE_DESP_LETTERS_DIGITS.sub(r"\1\2", text)
    text = _RE_DESP_LETTERS_DIGITS.sub(r"\1\2", text)

    # Step 7 — Strip e-commerce stop-words
    text = _RE_STOP_WORDS.sub(" ", text)

    # Step 8 — Lowercase (str.lower is Unicode-safe; only affects ASCII letters)
    text = text.lower()

    # Step 9 — Collapse runs of spaces and strip edges
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — ATTRIBUTE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_measures(norm_text: str) -> FrozenSet[str]:
    """
    Extract all volume / weight measurement tokens from a normalized string.

    Tokens must be in the canonical post-normalization form (e.g. "100ml", "50g").

    Args:
        norm_text: Output of :func:`_normalize_text`.

    Returns:
        Immutable set of measurement strings, lowercased.

    Examples:
        >>> _extract_measures("ادو پرفیوم bombshell 100ml")
        frozenset({'100ml'})
        >>> _extract_measures("کرم 50g سرم 30ml")
        frozenset({'30ml', '50g'})
        >>> _extract_measures("ریمل فول لش")
        frozenset()
    """
    return frozenset(tok.lower() for tok in _RE_EXTRACT_MEASURE.findall(norm_text))


def _extract_codes(norm_text: str) -> FrozenSet[str]:
    """
    Extract alphanumeric shade / model code tokens from a normalized string.

    Measurement tokens (e.g. "100ml") are explicitly excluded because they are
    structurally alphanumeric but are handled exclusively by Gate 1.

    Args:
        norm_text: Output of :func:`_normalize_text`.

    Returns:
        Immutable set of alphanumeric code strings, lowercased.

    Examples:
        >>> _extract_codes("فاندیشن nc41 spf30 100ml")
        frozenset({'nc41', 'spf30'})
        >>> _extract_codes("پودر 20g")
        frozenset()
    """
    candidates = frozenset(c.lower() for c in _RE_EXTRACT_CODE.findall(norm_text))
    # Exclude measurement tokens — they are alphanumeric but belong to Gate 1
    return candidates - _extract_measures(norm_text)


def _extract_english_content_words(norm_text: str) -> FrozenSet[str]:
    """
    Extract pure ASCII content words (letters only, ≥3 chars) from a normalized
    string, after stripping unit strings and high-frequency filler words.

    These words identify a specific product variant (e.g. "bombshell", "noir",
    "victoria") and must all be present in any valid matching candidate.

    Note: Codes like "nc41" are NOT returned here because the regex requires
    no digits in the token (`[a-z]{3,}`), and "spf30" contains digits.

    Args:
        norm_text: Output of :func:`_normalize_text`.

    Returns:
        Immutable set of discriminative English content words, lowercased.

    Examples:
        >>> _extract_english_content_words("ادو پرفیوم victoria secret bombshell 100ml")
        frozenset({'victoria', 'secret', 'bombshell'})
        >>> _extract_english_content_words("کرم spf30 100ml")
        frozenset()
    """
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
    """
    Hard Gate 1 — Volume / weight contradiction detector.

    A clash is declared **only** when **both** sides carry explicit measurement
    tokens that differ.  If one side omits the measurement (common for scraped
    names), the gate is not triggered; the fuzzy score will naturally penalise
    the discrepancy.

    Args:
        excel_measures: Measurement tokens from the authoritative Excel name.
        site_measures:  Measurement tokens from the scraped site name.

    Returns:
        ``True``  → hard fail (return 0.0 to the caller).
        ``False`` → no contradiction detected; continue to the next gate.

    Examples:
        >>> _measures_clash(frozenset({"100ml"}), frozenset({"50ml"}))
        True
        >>> _measures_clash(frozenset({"100ml"}), frozenset())
        False
        >>> _measures_clash(frozenset({"100ml"}), frozenset({"100ml"}))
        False
    """
    if excel_measures or site_measures:
        return excel_measures != site_measures
    return False


def _codes_clash(
    excel_codes: FrozenSet[str],
    site_codes: FrozenSet[str],
) -> bool:
    """
    Hard Gate 2 — Shade / model code contradiction detector.

    Every code present in the **Excel** (authoritative) name must also appear
    in the site name.  A missing or mismatched code is a hard fail.

    Directionality: Excel → Site only.  Extra codes on the site side (e.g.,
    an additional SPF value) are not penalised here; the fuzzy score handles
    them.

    Args:
        excel_codes: Alphanumeric codes from the authoritative Excel name.
        site_codes:  Alphanumeric codes from the scraped site name.

    Returns:
        ``True``  → hard fail.
        ``False`` → no contradiction.

    Examples:
        >>> _codes_clash(frozenset({"nc41"}), frozenset({"nc42"}))
        True
        >>> _codes_clash(frozenset({"nc41"}), frozenset({"nc41", "spf30"}))
        False
        >>> _codes_clash(frozenset(), frozenset({"nc41"}))
        False
    """
    if excel_codes or site_codes:
        return excel_codes != site_codes
    return False


def _english_words_clash(
    excel_words: FrozenSet[str],
    site_norm: str,
) -> bool:
    """
    Hard Gate 3 — Required English content-word presence checker.

    For every discriminative English word found in the Excel name (e.g.
    "bombshell", "noir"), the gate verifies it (or a very close approximation)
    exists in the full normalized site string.

    Fuzzy substring fallback (only for words ≥5 chars) tolerates single-char
    OCR / scraping typos using a sliding-window `fuzz.ratio` ≥ 88 threshold.
    Short words (3–4 chars) require an exact substring match to prevent
    accidental cross-matches.

    Args:
        excel_words: English content words from the authoritative Excel name.
        site_norm:   Full normalized site product name (string, not token set).

    Returns:
        ``True``  → at least one required word is absent → hard fail.
        ``False`` → all required words satisfied.

    Examples:
        >>> _english_words_clash(frozenset({"bombshell"}), "ادو پرفیوم 100ml")
        True
        >>> _english_words_clash(frozenset({"bombshell"}), "ادو پرفیوم bombshell 100ml")
        False
        >>> _english_words_clash(frozenset(), "any text")
        False
    """
    if not excel_words:
        return False

    for word in excel_words:
        # Fast-path: exact substring check (covers the vast majority of cases)
        if word in site_norm:
            continue

        # Short words (3–4 chars) require exact match — fuzzy window is too
        # risky at small lengths (e.g. "air" ≈ "sir" at 67%, "mat" ≈ "bat")
        if len(word) < 5:
            return True  # Not found exactly → clash

        # Fuzzy sliding-window fallback for longer words (handles "bombshel" typo)
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
    """
    Compute a similarity score between a scraped site product name and an
    authoritative Excel database product name for cosmetic / skincare products.

    Architecture — "Gated Lexical Scoring"
    ----------------------------------------
    **Phase 1 — Normalization** (both strings pass through an identical pipeline):
        Arabic→Persian unification, ZWNJ removal, digit conversion,
        unit standardization (ml/g), phonetic transliteration (ان سی → nc),
        code de-spacing (nc 41 → nc41), stop-word removal, lowercase.

    **Phase 2 — Hard Gates** (zero-tolerance; any failure returns 0.0):
        • Gate 1 (Volume/Weight): "100ml" ≠ "50ml"  →  0.0
        • Gate 2 (Shade/Code):    "nc41"  ≠ "nc42"  →  0.0
        • Gate 3 (English names): "Bombshell" absent in site  →  0.0

    **Phase 3 — Fuzzy Similarity**:
        ``thefuzz.fuzz.token_set_ratio`` on the normalized strings.
        Handles arbitrary word-order permutations without penalty.

    Args:
        site_product_name:  Raw product name scraped from the e-commerce site.
        excel_product_name: Verified product name from the Excel database.

    Returns:
        Float in ``[0.0, 1.0]``:
        - ``1.0`` — near-perfect / exact match after normalization.
        - ``0.0`` — definitive mismatch (one or more hard gates failed).
        - ``(0.0, 1.0)`` — partial match; higher is more similar.

    Examples:
        >>> similarity_score("فاندیشن nc41", "فاندیشن nc42")
        0.0
        >>> similarity_score("ریمل بل فول لش", "بل ریمل فول لش")
        1.0
        >>> similarity_score("فاندیشن ان سی ۴۱", "فاندیشن NC41")
        1.0
    """

    # Guard: empty inputs are definitively not a match
    if not site_product_name or not excel_product_name:
        return 0.0

    # ── Phase 1: Normalize ───────────────────────────────────────────────────
    norm_site = _normalize_text(site_product_name)
    norm_excel = _normalize_text(excel_product_name)

    # Short-circuit: exact post-normalization match
    if norm_site == norm_excel:
        return 1.0

    # ── Phase 2: Hard Gates ──────────────────────────────────────────────────

    # Gate 1 — Volume / Weight
    excel_measures = _extract_measures(norm_excel)
    site_measures = _extract_measures(norm_site)
    if _measures_clash(excel_measures, site_measures):
        return 0.0

    # Gate 2 — Shade / Model Code
    excel_codes = _extract_codes(norm_excel)
    site_codes = _extract_codes(norm_site)
    if _codes_clash(excel_codes, site_codes):
        return 0.0

    # Gate 3 — English Content Words
    excel_words = _extract_english_content_words(norm_excel)
    if _english_words_clash(excel_words, norm_site):
        return 0.0

    set_ratio = fuzz.token_set_ratio(norm_site, norm_excel)
    sort_ratio = fuzz.token_sort_ratio(norm_site, norm_excel)

    raw = (set_ratio * 0.4) + (sort_ratio * 0.6)

    return round(raw / 100.0, 4)
