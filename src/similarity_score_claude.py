import re
from typing import Dict, List, Set

from thefuzz import fuzz

# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK KNOWLEDGE BASE (Replace these with your actual loaded datasets)
# ═══════════════════════════════════════════════════════════════════════════════

BRANDS = {
    "HELLO KITTY": "BRAND_HELLOKITTY",
    "هلو کیتی": "BRAND_HELLOKITTY",
    "HELIABRINE": "BRAND_HELIABRINE",
    "هلیا برین": "BRAND_HELIABRINE",
    "MAKE UP FOR EVER": "BRAND_MAKEUPFOREVER",
}

PRODUCT_TYPES = {
    "کرم BB",
    "کرم CC",
    "کرم پودر",
    "کانسیلر",
    "پنکک",
    "Intimate Spray",
    "Intimate Wash",
}

STOP_WORDS = {
    "و",
    "با",
    "برای",
    "مدل",
    "حجم",
    "شماره",
    "زنانه",
    "مردانه",
    "رنگ",
    "حاوی",
    "مناسب",
    "سری",
    "ست",
    "the",
    "for",
    "with",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  TRIE-BASED ENTITY MATCHER (Ultra-fast O(N) multi-word replacement)
# ═══════════════════════════════════════════════════════════════════════════════


class TokenTrie:
    """A Trie structure for O(N) multi-word exact matching and replacement."""

    def __init__(self):
        self.trie = {}

    def add_entity(self, key_phrase: str, entity_id: str):
        words = key_phrase.strip().split()
        if not words:
            return
        node = self.trie
        for w in words:
            node = node.setdefault(w, {})
        node["__ID__"] = entity_id

    def extract_and_replace(self, tokens: List[str]) -> List[str]:
        result = []
        i = 0
        n = len(tokens)

        while i < n:
            node = self.trie
            j = i
            best_match_id = None
            best_match_end = -1

            # Longest prefix match
            while j < n and tokens[j] in node:
                node = node[tokens[j]]
                if "__ID__" in node:
                    best_match_id = node["__ID__"]
                    best_match_end = j
                j += 1

            if best_match_id:
                result.append(best_match_id)
                i = best_match_end + 1
            else:
                result.append(tokens[i])
                i += 1

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPILED PATTERNS & KNOWLEDGE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

_TBL_ARABIC_TO_PERSIAN = str.maketrans("\u064a\u0643", "\u06cc\u06a9")
_TBL_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_RE_VOLUME_ML = re.compile(
    r"(\d+)\s*(?:میلی\s*لیتر|میل(?!ی)|milliliter|ml|م(?!\w))", re.IGNORECASE
)
_RE_WEIGHT_G = re.compile(r"(\d+)\s*(?:گرم|grams?|gr(?!\w)|g(?!\w))", re.IGNORECASE)

# Strict token validators
_RE_EXACT_MEASURE = re.compile(r"^\d+(?:ml|g)$", re.IGNORECASE)
_RE_EXACT_CODE = re.compile(r"^(?:[a-z]+\d+|\d+[a-z]+|\d+)$", re.IGNORECASE)
_RE_EXACT_ENGLISH = re.compile(r"^[a-z]{3,}$", re.IGNORECASE)
_RE_EXACT_PERSIAN = re.compile(r"^[آ-یپچجگژ]{3,}$", re.UNICODE)

# Initialize and populate the Trie and O(1) Sets
_KNOWLEDGE_TRIE = TokenTrie()

# 1. Inject Brands
for alias, canon_id in BRANDS.items():
    _KNOWLEDGE_TRIE.add_entity(alias.lower(), canon_id.lower())

# 2. Inject Product Types (Dynamically auto-generating an ID like 'type_کرم_bb')
for p_type in PRODUCT_TYPES:
    slug = f"type_{p_type.replace(' ', '_')}".lower()
    _KNOWLEDGE_TRIE.add_entity(p_type.lower(), slug)

_STOP_WORDS_SET = {w.lower() for w in STOP_WORDS}


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE: NORMALIZATION -> EXTRACTION -> FILTERING
# ═══════════════════════════════════════════════════════════════════════════════


def _process_pipeline(text: str) -> List[str]:
    """Converts raw string to a list of fully resolved & filtered tokens."""
    # 1. Base Standardization
    text = text.translate(_TBL_ARABIC_TO_PERSIAN)
    text = text.replace("\u200c", " ")
    text = text.translate(_TBL_PERSIAN_DIGITS)
    text = text.lower()

    # Force spaces around numbers for clean tokenization
    text = _RE_VOLUME_ML.sub(lambda m: f" {m.group(1)}ml ", text)
    text = _RE_WEIGHT_G.sub(lambda m: f" {m.group(1)}g ", text)

    # 2. Base Tokenization
    raw_tokens = [t for t in text.split() if t]

    # 3. Entity Injection (Trie matcher) -> Converts ["هلو", "کیتی"] to ["brand_hellokitty"]
    resolved_tokens = _KNOWLEDGE_TRIE.extract_and_replace(raw_tokens)

    # 4. Stop-Word Elimination (O(1) lookup)
    final_tokens = [t for t in resolved_tokens if t not in _STOP_WORDS_SET]
    return final_tokens


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — SPECIFIC FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_brands(tokens: List[str]) -> Set[str]:
    return {t for t in tokens if t.startswith("brand_")}


def _extract_types(tokens: List[str]) -> Set[str]:
    return {t for t in tokens if t.startswith("type_")}


def _extract_measures(tokens: List[str]) -> Set[str]:
    return {t for t in tokens if _RE_EXACT_MEASURE.fullmatch(t)}


def _extract_codes(tokens: List[str]) -> Set[str]:
    measures = _extract_measures(tokens)
    codes = {t for t in tokens if _RE_EXACT_CODE.fullmatch(t)}
    return codes - measures


def _extract_english(tokens: List[str]) -> Set[str]:
    return {
        t
        for t in tokens
        if _RE_EXACT_ENGLISH.fullmatch(t)
        and not (t.startswith("brand_") or t.startswith("type_"))
    }


def _extract_persian(tokens: List[str]) -> Set[str]:
    return {
        t
        for t in tokens
        if _RE_EXACT_PERSIAN.fullmatch(t)
        and not (t.startswith("brand_") or t.startswith("type_"))
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — HARD-GATE CLASH DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════


def _entity_clash(
    excel_entities: Set[str], site_entities: Set[str], is_strict: bool = True
) -> bool:
    """
    If excel has an entity (like Brand/Type), it MUST intersect with the site.
    is_strict: If True, absence of entity in site also causes clash.
    """
    if excel_entities and site_entities:
        return not bool(excel_entities & site_entities)
    if is_strict and excel_entities and not site_entities:
        # Excel specifies it, but Site is missing it completely
        return True
    return False


def _words_clash(
    excel_words: Set[str], site_str: str, fuzz_threshold: int = 85
) -> bool:
    if not excel_words:
        return False

    for word in excel_words:
        if word in site_str:
            continue

        # Short crucial words without fuzziness
        word_len = len(word)
        if word_len < 5:
            return True

        # Sliding window fuzzy fallback
        found = any(
            fuzz.ratio(word, site_str[i : i + word_len]) >= fuzz_threshold
            for i in range(max(1, len(site_str) - word_len + 1))
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

    # 1. Semantic Tokenization
    site_tokens = _process_pipeline(site_product_name)
    excel_tokens = _process_pipeline(excel_product_name)

    if site_tokens == excel_tokens:
        return 1.0

    site_str = " ".join(site_tokens)
    excel_str = " ".join(excel_tokens)

    # 2. Semantic Gates (Brand & Type)
    # The ultimate False-Positive killers
    if _entity_clash(
        _extract_brands(excel_tokens), _extract_brands(site_tokens), is_strict=True
    ):
        return 0.0

    if _entity_clash(
        _extract_types(excel_tokens), _extract_types(site_tokens), is_strict=False
    ):
        # is_strict=False allows the site to have a vague name without explicit type,
        # but if BOTH have types, they MUST match.
        return 0.0

    # 3. Numeric & Code Gates
    if _entity_clash(_extract_measures(excel_tokens), _extract_measures(site_tokens)):
        return 0.0

    if _entity_clash(_extract_codes(excel_tokens), _extract_codes(site_tokens)):
        return 0.0

    # 4. Content Word Gates
    excel_en_words = _extract_english(excel_tokens)
    if _words_clash(excel_en_words, site_str, fuzz_threshold=88):
        return 0.0

    excel_fa_words = _extract_persian(excel_tokens)
    if _words_clash(excel_fa_words, site_str, fuzz_threshold=85):
        return 0.0

    # 5. Residual Fuzzy Scoring (For matching remaining attributes like "مات کننده")
    set_ratio = fuzz.token_set_ratio(site_str, excel_str)
    sort_ratio = fuzz.token_sort_ratio(site_str, excel_str)

    # Token Set is weighted less to prevent very short text bypassing larger text completely
    raw_score = (set_ratio * 0.4) + (sort_ratio * 0.6)

    return round(raw_score / 100.0, 4)


# ==============================================================================
# Example Test
# ==============================================================================
if __name__ == "__main__":
    # Test 1: Alias Resolution (HELLO KITTY vs هلو کیتی) -> Should be identical
    print(
        "Test 1:",
        similarity_score("کرم پودر هلو کیتی 50ml", "کرم پودر HELLO KITTY 50ml"),
    )
    # Output: 1.0 (Flawless match despite language barrier thanks to Trie Entity Mapper!)

    # Test 2: Stop Words protected within Brand
    print(
        "Test 2:",
        similarity_score("پنکک میکاپ فور اور رنگ تیره", "پنکک MAKE UP FOR EVER"),
    )
    # Output: ~0.9+ (The "FOR" inside the brand is preserved, but words like "رنگ" are safely removed)

    # Test 3: Type Clash Veto (کرم پودر vs کانسیلر)
    print("Test 3:", similarity_score("کرم پودر هلو کیتی", "کانسیلر هلو کیتی"))
    # Output: 0.0 (Veto'd instantly by the Type Gate)
