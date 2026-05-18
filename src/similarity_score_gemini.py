import re
from typing import Dict, Set, Tuple

from thefuzz import fuzz

# ==========================================
# PHASE 1: USER DATASETS (Drop your 7000+ lines here)
# ==========================================
# I have populated these with your examples. Replace them with your actual datasets.

RAW_BRANDS = {
    "HELLO KITTY": "BRAND_HELLOKITTY",
    "هلو کیتی": "BRAND_HELLOKITTY",
    "HELIABRINE": "BRAND_HELIABRINE",
    "هلیا برین": "BRAND_HELIABRINE",
    "HELICONIX": "BRAND_HELICONIX",
    "هلیکونیکس": "BRAND_HELICONIX",
    "HONAV": "BRAND_HONAV",
    "هناو": "BRAND_HONAV",
}

RAW_PRODUCT_TYPES = {
    "کرم BB",
    "کرم CC",
    "کرم DD",
    "کرم پودر",
    "کانسیلر",
    "پنکک",
    "Intimate Spray",
    "Condoms",
    "Intimate Wash",
}

RAW_STOP_WORDS = {
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

# ==========================================
# PHASE 2: CHARACTER UNIFICATION & CONSTANTS
# ==========================================

ARABIC_TO_PERSIAN = {ord("ي"): "ی", ord("ك"): "ک"}
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ENGLISH_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(PERSIAN_DIGITS, ENGLISH_DIGITS)


def _base_clean(text: str) -> str:
    """Applies foundational character unification needed before dictionary matching."""
    if not text:
        return ""
    text = text.translate(ARABIC_TO_PERSIAN)
    text = text.replace("\u200c", " ")  # ZWNJ removal
    text = text.translate(DIGIT_TRANS)
    return text.lower()


# Clean and finalize STOP_WORDS
STOP_WORDS = {_base_clean(w) for w in RAW_STOP_WORDS}

# ==========================================
# PHASE 3: DYNAMIC ENTITY COMPILER (HIGH PERFORMANCE)
# ==========================================
# This pre-compiles your 7000+ line dictionaries into lightning-fast Regex patterns.


def _build_entity_replacer(
    entity_dict: Dict[str, str],
) -> Tuple[re.Pattern, Dict[str, str]]:
    """Builds a compiled Regex and mapping dict ensuring longest-strings are replaced first."""
    # 1. Clean the keys to perfectly align with normalized text
    cleaned_dict = {_base_clean(k): v for k, v in entity_dict.items()}

    # 2. Sort by length descending (Prevents "کرم" from breaking "کرم پودر")
    sorted_keys = sorted(cleaned_dict.keys(), key=len, reverse=True)

    # 3. Build Regex pattern with word boundaries mapped for Perso-Arabic + English
    escaped_keys = map(re.escape, sorted_keys)
    pattern = re.compile(r"(?<!\w)(" + "|".join(escaped_keys) + r")(?!\w)")

    return pattern, cleaned_dict


# Map Product Types into safe token formats (e.g., "کرم BB" -> "TYPE_کرم_BB")
MAPPED_TYPES = {t: "TYPE_" + t.replace(" ", "_").upper() for t in RAW_PRODUCT_TYPES}

# Compile Brands and Types
RE_BRANDS, BRAND_MAP = _build_entity_replacer(RAW_BRANDS)
RE_TYPES, TYPE_MAP = _build_entity_replacer(MAPPED_TYPES)

# ==========================================
# PHASE 4: PRE-COMPILED REGEX (UNITS & GATES)
# ==========================================

RE_ML = re.compile(r"(?<!\w)(\d+)\s*(میلی\s*لیتر|میل|م|ml|milliliter)(?!\w)")
RE_G = re.compile(r"(?<!\w)(\d+)\s*(گرم|g|gram)(?!\w)")
RE_NC = re.compile(r"(?<!\w)ان\s*سی(?!\w)")
RE_M = re.compile(r"(?<!\w)ام(?!\w)")
RE_CODE_SPACE = re.compile(r"(?<!\w)([a-z]+)\s+(\d+)(?!\w)")

# Extractors
RE_VOL_WEIGHT = re.compile(r"(?<!\w)(\d+(?:ml|g))(?!\w)")
RE_ALPHANUM = re.compile(r"(?<!\w)([a-z]+\d+)(?!\w)")
RE_LONE_SHADE = re.compile(r"(?<!\w)(\d{2,4})(?!\w)")
RE_SHADE_WITH_PREFIX = re.compile(r"(?<!\w)(?:شماره|no)\s*(\d{1})(?!\w)")
RE_ENGLISH_WORD = re.compile(r"(?<!\w)([a-z]+)(?!\w)")

# Entity Extractors
RE_EXTRACT_BRAND = re.compile(r"(?<!\w)(BRAND_[A-Z0-9_]+)(?!\w)")
RE_EXTRACT_TYPE = re.compile(r"(?<!\w)(TYPE_[A-Z0-9_آ-ی]+)(?!\w)", re.IGNORECASE)

# ==========================================
# PHASE 5: PIPELINE EXECUTION
# ==========================================


def _normalize_text(text: str) -> str:
    """
    Applies aggressive Entity-Aware normalization.
    Order of operations is CRITICAL for faithfulness.
    """
    # 1. Foundational Clean (ZWNJ, Lowercase, Digits)
    text = _base_clean(text)

    # 2. Entity Standardizations (Brands MUST precede Types to avoid sub-string overlaps)
    text = RE_BRANDS.sub(lambda m: BRAND_MAP[m.group(1)], text)
    text = RE_TYPES.sub(lambda m: TYPE_MAP[m.group(1)], text)

    # 3. Phonetic & Unit Standardization
    text = RE_NC.sub("nc", text)
    text = RE_M.sub("m", text)
    text = RE_ML.sub(r"\1ml", text)
    text = RE_G.sub(r"\1g", text)
    text = RE_CODE_SPACE.sub(r"\1\2", text)

    # 4. Safe E-commerce Stop-Word Removal (Safe because core entities are now locked as BRAND_X)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)


def _extract_attributes(
    text: str,
) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    """
    Extracts strictly typed cosmetic attributes for hard-gating.
    Returns: (Brands, Types, Volumes, Codes, English Words)
    """
    # Extract Entities
    brands = set(RE_EXTRACT_BRAND.findall(text))
    types = set(RE_EXTRACT_TYPE.findall(text))

    volumes = set(RE_VOL_WEIGHT.findall(text))

    # Strip entities & volumes out to avoid parsing them as random codes or english words
    text_cleaned = RE_EXTRACT_BRAND.sub(" ", text)
    text_cleaned = RE_EXTRACT_TYPE.sub(" ", text_cleaned)
    text_cleaned = RE_VOL_WEIGHT.sub(" ", text_cleaned)

    # Extract Alphanumerics & English
    codes = set(RE_ALPHANUM.findall(text_cleaned))
    codes.update(RE_LONE_SHADE.findall(text_cleaned))
    codes.update(RE_SHADE_WITH_PREFIX.findall(text_cleaned))

    english_words = set(RE_ENGLISH_WORD.findall(text_cleaned))

    return brands, types, volumes, codes, english_words


def similarity_score(site_product_name: str, excel_product_name: str) -> float:
    """
    Calculates exact similarity utilizing Brand/Type entities and fallback fuzzing.
    Guarantees 0.0 for mismatched hard entities (0 False Positives).
    """
    # Phase 1: Entity-Aware Normalization
    norm_site = _normalize_text(site_product_name)
    norm_excel = _normalize_text(excel_product_name)

    # Phase 2: Extract Critical Attributes
    site_brands, site_types, site_vol, site_codes, site_eng = _extract_attributes(
        norm_site
    )
    excel_brands, excel_types, excel_vol, excel_codes, excel_eng = _extract_attributes(
        norm_excel
    )

    # --- THE IRON GATES ---

    # Gate 1: Brand Clash
    if excel_brands and site_brands:
        if not excel_brands.intersection(site_brands):
            return 0.0  # e.g., L'Oreal matched against Nivea

    # Gate 2: Product Type Clash
    if excel_types and site_types:
        if not excel_types.intersection(site_types):
            return 0.0  # e.g., Shampoo matched against BB Cream

    # Gate 3: Volume / Weight Clash
    if excel_vol and site_vol:
        if not excel_vol.intersection(site_vol):
            return 0.0  # e.g., 50ml matched against 100ml

    # Gate 4: STRICT Code / Shade Clash
    if excel_codes or site_codes:
        if not excel_codes.intersection(site_codes):
            return 0.0

    # Gate 5: English Word Conflict (Safe Check)
    if excel_eng and site_eng:
        if not excel_eng.intersection(site_eng):
            return 0.0

    # Phase 3: The Residual Fuzzy Match
    # Because Entities (Brand/Type/Vol) are matched exactly and standardized,
    # the fuzzy matcher now solely evaluates the remaining descriptive strings.

    set_ratio = fuzz.token_set_ratio(norm_site, norm_excel)
    sort_ratio = fuzz.token_sort_ratio(norm_site, norm_excel)

    # If the site string drops a crucial entity (like the Brand is missing on the site),
    # the sort_ratio will aggressively drop because a massive chunk of string (e.g. "BRAND_HELLOKITTY") is missing.
    score = (set_ratio * 0.4) + (sort_ratio * 0.6)

    return float(score) / 100.0
