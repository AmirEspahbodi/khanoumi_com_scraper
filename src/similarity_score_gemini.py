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
RE_LONE_SHADE = re.compile(r"(?<!\w)(\d{2,4})(?!\w)")
RE_SHADE_WITH_PREFIX = re.compile(r"(?<!\w)(?:شماره|no)\s*(\d{1})(?!\w)")
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

    # Extract alphanumeric model codes
    codes = set(RE_ALPHANUM.findall(text_no_vol))
    codes.update(RE_LONE_SHADE.findall(text_no_vol))
    codes.update(RE_SHADE_WITH_PREFIX.findall(text_no_vol))
    # ----------------------------------------------

    # Extract any remaining pure English tokens
    english_words = set(RE_ENGLISH_WORD.findall(text_no_vol))

    return volumes, codes, english_words


def similarity_score(site_product_name: str, excel_product_name: str) -> float:
    """
    Calculates the exact similarity between two cosmetic product names.
    Fixed to handle Subset Traps, Asymmetric Codes, and Cross-lingual false negatives.
    """
    # Phase 1: Normalize
    norm_site = _normalize_text(site_product_name)
    norm_excel = _normalize_text(excel_product_name)

    # Phase 2: Extract Critical Attributes
    site_vol, site_codes, site_eng = _extract_attributes(norm_site)
    excel_vol, excel_codes, excel_eng = _extract_attributes(norm_excel)

    # Hard Gate 1: Volume / Weight Clash
    # اگر هر دو حجم دارند، باید حتما اشتراک داشته باشند
    if excel_vol and site_vol:
        if not excel_vol.intersection(site_vol):
            return 0.0

    # Hard Gate 2: STRICT Code / Shade Clash
    # اصلاح شد: اگر یکی از آن‌ها کد رنگ/مدل دارد، دیگری هم حتما باید همان کد را داشته باشد
    # این کار جلوی مچ شدن محصول عمومی (بدون کد) با محصول خاص (کد دار) را می‌گیرد
    if excel_codes or site_codes:
        if not excel_codes.intersection(site_codes):
            return 0.0

    # Hard Gate 3: English Word Conflict (Safe Check)
    # اصلاح شد: بررسی سخت‌گیرانه فقط زمانی انجام می‌شود که "هر دو" کلمه انگلیسی داشته باشند.
    # این کار از رد شدن اشتباه Bourjois (در اکسل) با بورژوا (در سایت) جلوگیری می‌کند.
    if excel_eng and site_eng:
        if not excel_eng.intersection(site_eng):
            return 0.0

    # Phase 3: Permutation-Invariant & Subset-Penalized Similarity
    # رفع باگِ نام‌های طولانی و کوتاه (تله زیرمجموعه)

    # set_ratio: ترتیب را نادیده می‌گیرد اما به زیرمجموعه‌ها نمره ۱۰۰ می‌دهد
    set_ratio = fuzz.token_set_ratio(norm_site, norm_excel)

    # sort_ratio: ترتیب را نادیده می‌گیرد اما اگر طول رشته‌ها فرق کند یا کلمه‌ای جا بیفتد، نمره را به شدت کم می‌کند
    sort_ratio = fuzz.token_sort_ratio(norm_site, norm_excel)

    # وزن‌دهی ترکیبی (Hybrid Scoring):
    # با این فرمول، اگر سایت کلمات "Victoria Secret Bombshell" را نداشته باشد،
    # sort_ratio به شدت افت می‌کند و نمره نهایی را از حد نصاب (مثلا ۸۵٪) پایین می‌کشد.
    score = (set_ratio * 0.4) + (sort_ratio * 0.6)

    return float(score) / 100.0
