import re
from typing import Set, Tuple

from thefuzz import fuzz

# ==========================================
# PHASE 1: PRE-COMPILED REGEX & CONSTANTS
# ==========================================
# Compiling at module-level ensures high-performance for real-time execution.

# Character Translation Maps
ARABIC_TO_PERSIAN = {ord("ي"): "ی", ord("ك"): "ک"}
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ENGLISH_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(PERSIAN_DIGITS, ENGLISH_DIGITS)

# Stop-words to strip
STOP_WORDS = {
    "مدل",
    "شماره",
    "حجم",
    "رنگ",
    "کد",
    "وزن",
    "سایز",
    "زنانه",
    "مردانه",
    "زن",
    "مرد",
    "حاوی",
    "دارای",
    "مناسب",
    "مخصوص",
    "عصاره",
    "با",
    "انواع",
    "فاقد",
    "بدون",
    "تیوپی",
    "شیشه",
    "شیشه ای",
    "کاسه",
    "کاسه ای",
    "پمپی",
    "استیکی",
    "ماژیکی",
    "اورجینال",
    "اصل",
    "ساعته",
}
# Unit Standardization Regex
# Uses lookbehind/lookahead to handle safe boundaries in mixed Perso-Arabic/English contexts
RE_ML = re.compile(
    r"(?<!\w)(\d+)\s*(میلی\s*لیتر|میل|م|ml|milliliter)(?!\w)", re.IGNORECASE
)
RE_G = re.compile(r"(?<!\w)(\d+)\s*(گرم|g|gram)(?!\w)", re.IGNORECASE)

# Phonetic & Code Fix Regex
RE_NC = re.compile(r"(?<!\w)ان\s*سی(?!\w)")
RE_M = re.compile(r"(?<!\w)ام(?!\w)")
RE_CODE_SPACE = re.compile(r"(?<!\w)([a-z]+)\s+(\d+)(?!\w)")

# ==========================================
# PHASE 2: EXTRACTOR REGEX (GATES)
# ==========================================
RE_VOL_WEIGHT = re.compile(r"(?<!\w)(\d+(?:ml|g))(?!\w)")
RE_ALPHANUM = re.compile(r"(?<!\w)([a-z]+\d+)(?!\w)")
RE_LONE_NUMBER = re.compile(r"(?<!\w)(\d+)(?!\w)")
RE_ENGLISH_WORD = re.compile(r"(?<!\w)([a-z]+)(?!\w)")


def _normalize_text(text: str) -> str:
    """
    Applies aggressive normalization on raw e-commerce strings.
    Handles character unification, standardizes units, and drops stop-words.
    """
    # 1. Character Unification & ZWNJ Removal
    text = text.translate(ARABIC_TO_PERSIAN)
    text = text.replace("\u200c", " ")

    # 2. Digit Conversion (Persian to English)
    text = text.translate(DIGIT_TRANS)

    # 3. Lowercasing for English token alignment
    text = text.lower()

    # 4. Phonetic Code Standardization
    text = RE_NC.sub("nc", text)
    text = RE_M.sub("m", text)

    # 5. Unit Standardization (e.g., '100 میلی لیتر' -> '100ml')
    text = RE_ML.sub(r"\1ml", text)
    text = RE_G.sub(r"\1g", text)

    # 6. Alphanumeric Space Removal (e.g., 'nc 41' -> 'nc41')
    text = RE_CODE_SPACE.sub(r"\1\2", text)

    # 7. E-commerce Stop-Word Removal
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)


def _extract_attributes(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extracts strictly typed cosmetic attributes for hard-gating.
    Returns: (Set of Volumes/Weights, Set of Codes/Shades, Set of English Tokens)
    """
    # Extract standardized volumes/weights
    volumes = set(RE_VOL_WEIGHT.findall(text))

    # Strip volumes out to avoid re-extracting their numbers as shades
    text_no_vol = RE_VOL_WEIGHT.sub(" ", text)

    # Extract alphanumeric model codes and lone numeric shades
    codes = set(RE_ALPHANUM.findall(text_no_vol))
    codes.update(RE_LONE_NUMBER.findall(text_no_vol))

    # Extract any remaining pure English tokens
    english_words = set(RE_ENGLISH_WORD.findall(text_no_vol))

    return volumes, codes, english_words


def similarity_score(site_product_name: str, excel_product_name: str) -> float:
    """
    Calculates the exact similarity between two cosmetic product names using Gated Lexical Scoring.
    Returns 0.0 immediately on hard-gate conflicts (mismatched volumes/codes).
    Returns a normalized float (0.0 - 1.0) based on permutation-invariant matching.
    """
    # Phase 1: Normalize
    norm_site = _normalize_text(site_product_name)
    norm_excel = _normalize_text(excel_product_name)

    # Phase 2: Extract Critical Attributes
    site_vol, site_codes, site_eng = _extract_attributes(norm_site)
    excel_vol, excel_codes, excel_eng = _extract_attributes(norm_excel)

    # Hard Gate 1: Volume / Weight Clash
    # Fails if both mention a volume, but they share NO common volume
    if excel_vol and site_vol:
        if not excel_vol.intersection(site_vol):
            return 0.0

    # Hard Gate 2: Code / Shade Clash
    # Fails if both mention a shade/code, but they share NO common code
    if excel_codes and site_codes:
        if not excel_codes.intersection(site_codes):
            return 0.0

    # Hard Gate 3: English Token / Model Mismatch
    # Fails if Excel mentions specific English words (like 'Bombshell') that are totally missing in Site
    if excel_eng:
        for e_word in excel_eng:
            match_found = False
            for s_word in site_eng:
                # Require an exact match, or a substantive substring match (len >= 3 to prevent single-char falses)
                if (
                    e_word == s_word
                    or (len(s_word) >= 3 and s_word in e_word)
                    or (len(e_word) >= 3 and e_word in s_word)
                ):
                    match_found = True
                    break

            if not match_found:
                return 0.0

    # Phase 3: Permutation-Invariant Similarity
    # fuzz.token_set_ratio naturally handles word order shifts ("ریمل بل" vs "بل ریمل")
    score = fuzz.token_set_ratio(norm_site, norm_excel)

    return float(score) / 100.0
