"""
Persian-English product-name similarity pipeline.

The module implements a deterministic mixed-script product-name similarity pipeline
for Persian and English e-commerce names. It has no runtime dependency on a model,
network service, corpus statistics, or trained tokenizer.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import os
import re
import sys
import threading
from typing import Iterable, NamedTuple
import unicodedata


try:
    from doublemetaphone import doublemetaphone as _doublemetaphone_impl
except ImportError:
    _doublemetaphone_impl = None


class LexiconEntry(NamedTuple):
    """A static lexical mapping from a surface form to a canonical token."""
    canonical: str
    kind: str


class Token(NamedTuple):
    """A canonical, classified token emitted by Stage A preprocessing."""
    surface: str
    canonical: str
    token_class: int
    weight: float
    source: str
    kind: str
    original: str


class CompoundEntry(NamedTuple):
    """A multi-token lexical mapping used before per-token resolution."""
    canonical: str
    kind: str


# Token class constants from the specification.
CLASS_SKU = 1
CLASS_QUANTITY = 2
CLASS_QUALIFIER = 3
CLASS_BRAND = 4
CLASS_MODEL = 5
CLASS_CATEGORY = 6
CLASS_RESIDUAL = 7

WEIGHTS: dict[int, float] = {
    CLASS_SKU: 1.00,
    CLASS_QUANTITY: 0.90,
    CLASS_QUALIFIER: 0.85,
    CLASS_BRAND: 0.70,
    CLASS_MODEL: 0.55,
    CLASS_CATEGORY: 0.25,
    CLASS_RESIDUAL: 0.10,
}

SOURCE_DICT = "DICT"
SOURCE_PHONETIC = "PHONETIC"
SOURCE_LATIN = "LATIN"
SOURCE_STRUCTURAL = "STRUCTURAL"
SOURCE_RESIDUAL = "RESIDUAL"

KIND_BRAND = "brand"
KIND_MODEL = "model"
KIND_CATEGORY = "category"
KIND_QUALIFIER = "qualifier"
KIND_QUANTITY = "quantity"
KIND_SKU = "sku"
KIND_RESIDUAL = "residual"

PREFILTER_THRESHOLD = 0.15
PHONETIC_CONFIDENCE_DISCOUNT = 0.85
WEAK_QUERY_WEIGHT_THRESHOLD = 1.5
WEAK_QUERY_SCORE_CAP = 0.85
SINGLE_TOKEN_PRODUCT_SCORE_CAP = 0.70


# Persian and Arabic digit normalization.
_DIGIT_TRANSLATION = str.maketrans(
    {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)

# ALA-LC-inspired Persian romanization map. The consonant inventory and the three
# long vowel forms are covered; short vowels are absent in ordinary Persian script.
ROMANIZATION_MAP: dict[str, str] = {
    "ا": "a",
    "آ": "a",
    "أ": "a",
    "إ": "a",
    "ٱ": "a",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "z",
    "ر": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "و": "u",
    "ه": "h",
    "ۀ": "h",
    "ة": "h",
    "ی": "i",
    "ء": "",
    "ئ": "y",
    "ؤ": "v",
}

# Unit normalization table. Values are canonical SI or conventional e-commerce
# abbreviations. "cc" is normalized to ml because fragrance and cosmetics listings
# use it as a volume synonym.
UNIT_NORMALIZATION: dict[str, str] = {
    "ml": "ml",
    "m l": "ml",
    "milliliter": "ml",
    "millilitre": "ml",
    "milliliters": "ml",
    "millilitres": "ml",
    "cc": "ml",
    "سیسی": "ml",
    "سی سی": "ml",
    "میلیلیتر": "ml",
    "میلی لیتر": "ml",
    "میل": "ml",
    "l": "l",
    "lt": "l",
    "liter": "l",
    "litre": "l",
    "liters": "l",
    "litres": "l",
    "لیتر": "l",
    "g": "g",
    "gr": "g",
    "gram": "g",
    "grams": "g",
    "گرم": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "کیلوگرم": "kg",
    "کیلو گرم": "kg",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "میلیگرم": "mg",
    "میلی گرم": "mg",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "اونس": "oz",
    "iu": "iu",
    "عدد": "count",
    "عددی": "count",
    "pcs": "count",
    "pc": "count",
    "piece": "count",
    "pieces": "count",
    "pack": "count",
}

PERSIAN_STOPWORDS: frozenset[str] = frozenset(
    {
        "و",
        "یا",
        "با",
        "در",
        "برای",
        "از",
        "به",
        "تا",
        "که",
        "این",
        "آن",
        "یک",
        "های",
        "ها",
        "ی",
        "را",
        "بر",
        "روی",
        "داخل",
        "مدل",
        "سری",
        "اصل",
        "اورجینال",
        "اصلآ",
        "بهترین",
        "جدید",
        "new",
        "original",
        "model",
        "series",
        "for",
        "with",
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
    }
)

# Persian and English lexical mappings. The dictionary intentionally contains
# Persian variants and Latin canonical spellings so Stage A can apply dictionary
# resolution uniformly before Latin fallback.
TRANSLITERATION_DICTIONARY: dict[str, LexiconEntry] = {
    # Electronics brands.
    "سامسونگ": LexiconEntry("samsung", KIND_BRAND),
    "samsung": LexiconEntry("samsung", KIND_BRAND),
    "اپل": LexiconEntry("apple", KIND_BRAND),
    "اَپل": LexiconEntry("apple", KIND_BRAND),
    "apple": LexiconEntry("apple", KIND_BRAND),
    "شیائومی": LexiconEntry("xiaomi", KIND_BRAND),
    "شیاومی": LexiconEntry("xiaomi", KIND_BRAND),
    "xiaomi": LexiconEntry("xiaomi", KIND_BRAND),
    "هواوی": LexiconEntry("huawei", KIND_BRAND),
    "هوآوی": LexiconEntry("huawei", KIND_BRAND),
    "huawei": LexiconEntry("huawei", KIND_BRAND),
    "آنر": LexiconEntry("honor", KIND_BRAND),
    "honor": LexiconEntry("honor", KIND_BRAND),
    "نوکیا": LexiconEntry("nokia", KIND_BRAND),
    "nokia": LexiconEntry("nokia", KIND_BRAND),
    "سونی": LexiconEntry("sony", KIND_BRAND),
    "sony": LexiconEntry("sony", KIND_BRAND),
    "الجی": LexiconEntry("lg", KIND_BRAND),
    "ال جی": LexiconEntry("lg", KIND_BRAND),
    "lg": LexiconEntry("lg", KIND_BRAND),
    "لنوو": LexiconEntry("lenovo", KIND_BRAND),
    "lenovo": LexiconEntry("lenovo", KIND_BRAND),
    "ایسوس": LexiconEntry("asus", KIND_BRAND),
    "asus": LexiconEntry("asus", KIND_BRAND),
    "اچپی": LexiconEntry("hp", KIND_BRAND),
    "اچ پی": LexiconEntry("hp", KIND_BRAND),
    "hp": LexiconEntry("hp", KIND_BRAND),
    "دل": LexiconEntry("dell", KIND_BRAND),
    "dell": LexiconEntry("dell", KIND_BRAND),
    "ایسر": LexiconEntry("acer", KIND_BRAND),
    "acer": LexiconEntry("acer", KIND_BRAND),
    "کانن": LexiconEntry("canon", KIND_BRAND),
    "canon": LexiconEntry("canon", KIND_BRAND),
    "نیکون": LexiconEntry("nikon", KIND_BRAND),
    "nikon": LexiconEntry("nikon", KIND_BRAND),
    "فیلیپس": LexiconEntry("philips", KIND_BRAND),
    "philips": LexiconEntry("philips", KIND_BRAND),
    "بوش": LexiconEntry("bosch", KIND_BRAND),
    "bosch": LexiconEntry("bosch", KIND_BRAND),
    # Cosmetics and care brands.
    "مک": LexiconEntry("mac", KIND_BRAND),
    "mac": LexiconEntry("mac", KIND_BRAND),
    "لورال": LexiconEntry("loreal", KIND_BRAND),
    "لورل": LexiconEntry("loreal", KIND_BRAND),
    "لوریال": LexiconEntry("loreal", KIND_BRAND),
    "loreal": LexiconEntry("loreal", KIND_BRAND),
    "l'oréal": LexiconEntry("loreal", KIND_BRAND),
    "maybelline": LexiconEntry("maybelline", KIND_BRAND),
    "میبلین": LexiconEntry("maybelline", KIND_BRAND),
    "میبلاین": LexiconEntry("maybelline", KIND_BRAND),
    "بورژوا": LexiconEntry("bourjois", KIND_BRAND),
    "bourjois": LexiconEntry("bourjois", KIND_BRAND),
    "فلورمار": LexiconEntry("flormar", KIND_BRAND),
    "flormar": LexiconEntry("flormar", KIND_BRAND),
    "پیپا": LexiconEntry("pupa", KIND_BRAND),
    "پوپا": LexiconEntry("pupa", KIND_BRAND),
    "pupa": LexiconEntry("pupa", KIND_BRAND),
    "اسنس": LexiconEntry("essence", KIND_BRAND),
    "essence": LexiconEntry("essence", KIND_BRAND),
    "کاتریس": LexiconEntry("catrice", KIND_BRAND),
    "catrice": LexiconEntry("catrice", KIND_BRAND),
    "نارس": LexiconEntry("nars", KIND_BRAND),
    "nars": LexiconEntry("nars", KIND_BRAND),
    "دیور": LexiconEntry("dior", KIND_BRAND),
    "dior": LexiconEntry("dior", KIND_BRAND),
    "شنل": LexiconEntry("chanel", KIND_BRAND),
    "chanel": LexiconEntry("chanel", KIND_BRAND),
    "ایو سن لوران": LexiconEntry("ysl", KIND_BRAND),
    "ysl": LexiconEntry("ysl", KIND_BRAND),
    "لانکوم": LexiconEntry("lancome", KIND_BRAND),
    "lancome": LexiconEntry("lancome", KIND_BRAND),
    "کلینیک": LexiconEntry("clinique", KIND_BRAND),
    "clinique": LexiconEntry("clinique", KIND_BRAND),
    "استی لادر": LexiconEntry("estee_lauder", KIND_BRAND),
    "estee": LexiconEntry("estee_lauder", KIND_BRAND),
    "lauder": LexiconEntry("estee_lauder", KIND_BRAND),
    "estee_lauder": LexiconEntry("estee_lauder", KIND_BRAND),
    "سفورا": LexiconEntry("sephora", KIND_BRAND),
    "sephora": LexiconEntry("sephora", KIND_BRAND),
    "هودا": LexiconEntry("huda_beauty", KIND_BRAND),
    "huda": LexiconEntry("huda_beauty", KIND_BRAND),
    "nyx": LexiconEntry("nyx", KIND_BRAND),
    "نیکس": LexiconEntry("nyx", KIND_BRAND),
    "بنفت": LexiconEntry("benefit", KIND_BRAND),
    "benefit": LexiconEntry("benefit", KIND_BRAND),
    "توفیسد": LexiconEntry("too_faced", KIND_BRAND),
    "too_faced": LexiconEntry("too_faced", KIND_BRAND),
    "بابی براون": LexiconEntry("bobbi_brown", KIND_BRAND),
    "bobbi": LexiconEntry("bobbi_brown", KIND_BRAND),
    "brown": LexiconEntry("bobbi_brown", KIND_BRAND),
    "آناستازیا": LexiconEntry("anastasia", KIND_BRAND),
    "anastasia": LexiconEntry("anastasia", KIND_BRAND),
    "رولون": LexiconEntry("revlon", KIND_BRAND),
    "revlon": LexiconEntry("revlon", KIND_BRAND),
    "ریمل لندن": LexiconEntry("rimmel", KIND_BRAND),
    "rimmel": LexiconEntry("rimmel", KIND_BRAND),
    "مکس فاکتور": LexiconEntry("max_factor", KIND_BRAND),
    "max_factor": LexiconEntry("max_factor", KIND_BRAND),
    "کاورگرل": LexiconEntry("covergirl", KIND_BRAND),
    "covergirl": LexiconEntry("covergirl", KIND_BRAND),
    "وت اند وایلد": LexiconEntry("wet_n_wild", KIND_BRAND),
    "wet_n_wild": LexiconEntry("wet_n_wild", KIND_BRAND),
    "درماکول": LexiconEntry("dermacol", KIND_BRAND),
    "dermacol": LexiconEntry("dermacol", KIND_BRAND),
    "گلدن رز": LexiconEntry("golden_rose", KIND_BRAND),
    "golden_rose": LexiconEntry("golden_rose", KIND_BRAND),
    "نوت": LexiconEntry("note", KIND_BRAND),
    "note": LexiconEntry("note", KIND_BRAND),
    "نیوا": LexiconEntry("nivea", KIND_BRAND),
    "nivea": LexiconEntry("nivea", KIND_BRAND),
    "گارنیه": LexiconEntry("garnier", KIND_BRAND),
    "garnier": LexiconEntry("garnier", KIND_BRAND),
    "ویشی": LexiconEntry("vichy", KIND_BRAND),
    "vichy": LexiconEntry("vichy", KIND_BRAND),
    "لاروش": LexiconEntry("la_roche_posay", KIND_BRAND),
    "لا روش": LexiconEntry("la_roche_posay", KIND_BRAND),
    "la_roche_posay": LexiconEntry("la_roche_posay", KIND_BRAND),
    "سراوی": LexiconEntry("cerave", KIND_BRAND),
    "cerave": LexiconEntry("cerave", KIND_BRAND),
    "اوسرین": LexiconEntry("eucerin", KIND_BRAND),
    "eucerin": LexiconEntry("eucerin", KIND_BRAND),
    "بایودرما": LexiconEntry("bioderma", KIND_BRAND),
    "bioderma": LexiconEntry("bioderma", KIND_BRAND),
    "اوریاژ": LexiconEntry("uriage", KIND_BRAND),
    "uriage": LexiconEntry("uriage", KIND_BRAND),
    "آون": LexiconEntry("avon", KIND_BRAND),
    "avon": LexiconEntry("avon", KIND_BRAND),
    "اوریفلیم": LexiconEntry("oriflame", KIND_BRAND),
    "oriflame": LexiconEntry("oriflame", KIND_BRAND),
    "رکسونا": LexiconEntry("rexona", KIND_BRAND),
    "rexona": LexiconEntry("rexona", KIND_BRAND),
    # Fragrance and sports-fashion brands.
    "دیویدوف": LexiconEntry("davidoff", KIND_BRAND),
    "davidoff": LexiconEntry("davidoff", KIND_BRAND),
    "کالوین کلین": LexiconEntry("calvin_klein", KIND_BRAND),
    "calvin": LexiconEntry("calvin_klein", KIND_BRAND),
    "klein": LexiconEntry("calvin_klein", KIND_BRAND),
    "آدیداس": LexiconEntry("adidas", KIND_BRAND),
    "adidas": LexiconEntry("adidas", KIND_BRAND),
    "نایک": LexiconEntry("nike", KIND_BRAND),
    "nike": LexiconEntry("nike", KIND_BRAND),
    "هوگو باس": LexiconEntry("hugo_boss", KIND_BRAND),
    "hugo": LexiconEntry("hugo_boss", KIND_BRAND),
    "boss": LexiconEntry("hugo_boss", KIND_BRAND),
    "ورساچه": LexiconEntry("versace", KIND_BRAND),
    "versace": LexiconEntry("versace", KIND_BRAND),
    "آرمانی": LexiconEntry("armani", KIND_BRAND),
    "armani": LexiconEntry("armani", KIND_BRAND),
    "گوچی": LexiconEntry("gucci", KIND_BRAND),
    "gucci": LexiconEntry("gucci", KIND_BRAND),
    "پرادا": LexiconEntry("prada", KIND_BRAND),
    "prada": LexiconEntry("prada", KIND_BRAND),
    "بربری": LexiconEntry("burberry", KIND_BRAND),
    "burberry": LexiconEntry("burberry", KIND_BRAND),
    "هرمس": LexiconEntry("hermes", KIND_BRAND),
    "hermes": LexiconEntry("hermes", KIND_BRAND),
    "پاکو رابان": LexiconEntry("paco_rabanne", KIND_BRAND),
    "paco": LexiconEntry("paco_rabanne", KIND_BRAND),
    "rabanne": LexiconEntry("paco_rabanne", KIND_BRAND),
    "مونت بلانک": LexiconEntry("montblanc", KIND_BRAND),
    "montblanc": LexiconEntry("montblanc", KIND_BRAND),
    "لالیک": LexiconEntry("lalique", KIND_BRAND),
    "lalique": LexiconEntry("lalique", KIND_BRAND),
    "تام فورد": LexiconEntry("tom_ford", KIND_BRAND),
    "tom": LexiconEntry("tom_ford", KIND_BRAND),
    "ford": LexiconEntry("tom_ford", KIND_BRAND),
    "ژیوانشی": LexiconEntry("givenchy", KIND_BRAND),
    "givenchy": LexiconEntry("givenchy", KIND_BRAND),
    "بولگاری": LexiconEntry("bvlgari", KIND_BRAND),
    "bvlgari": LexiconEntry("bvlgari", KIND_BRAND),
    "بولگاری": LexiconEntry("bvlgari", KIND_BRAND),
    "لاکست": LexiconEntry("lacoste", KIND_BRAND),
    "lacoste": LexiconEntry("lacoste", KIND_BRAND),
    # Model and line names.
    "گلکسی": LexiconEntry("galaxy", KIND_MODEL),
    "galaxy": LexiconEntry("galaxy", KIND_MODEL),
    "اولترا": LexiconEntry("ultra", KIND_MODEL),
    "ultra": LexiconEntry("ultra", KIND_MODEL),
    "پرو": LexiconEntry("pro", KIND_MODEL),
    "pro": LexiconEntry("pro", KIND_MODEL),
    "پرومکس": LexiconEntry("pro_max", KIND_MODEL),
    "pro_max": LexiconEntry("pro_max", KIND_MODEL),
    "ایر": LexiconEntry("air", KIND_MODEL),
    "air": LexiconEntry("air", KIND_MODEL),
    "آیفون": LexiconEntry("iphone", KIND_MODEL),
    "iphone": LexiconEntry("iphone", KIND_MODEL),
    "آیپد": LexiconEntry("ipad", KIND_MODEL),
    "ipad": LexiconEntry("ipad", KIND_MODEL),
    "مکبوک": LexiconEntry("macbook", KIND_MODEL),
    "macbook": LexiconEntry("macbook", KIND_MODEL),
    "واچ": LexiconEntry("watch", KIND_MODEL),
    "watch": LexiconEntry("watch", KIND_MODEL),
    "استودیو": LexiconEntry("studio", KIND_MODEL),
    "studio": LexiconEntry("studio", KIND_MODEL),
    "فیکس": LexiconEntry("fix", KIND_MODEL),
    "fix": LexiconEntry("fix", KIND_MODEL),
    "studio_fix": LexiconEntry("studio_fix", KIND_MODEL),
    "کول": LexiconEntry("cool", KIND_MODEL),
    "cool": LexiconEntry("cool", KIND_MODEL),
    "واتر": LexiconEntry("water", KIND_MODEL),
    "water": LexiconEntry("water", KIND_MODEL),
    "cool_water": LexiconEntry("cool_water", KIND_MODEL),
    "اکوا": LexiconEntry("aqua", KIND_MODEL),
    "aqua": LexiconEntry("aqua", KIND_MODEL),
    "بلو": LexiconEntry("blue", KIND_MODEL),
    "blue": LexiconEntry("blue", KIND_MODEL),
    "بلک": LexiconEntry("black", KIND_MODEL),
    "black": LexiconEntry("black", KIND_MODEL),
    "وایت": LexiconEntry("white", KIND_MODEL),
    "white": LexiconEntry("white", KIND_MODEL),
    "مت": LexiconEntry("matte", KIND_MODEL),
    "matte": LexiconEntry("matte", KIND_MODEL),
    "مات": LexiconEntry("matte", KIND_MODEL),
    # Product categories and descriptors.
    "کرم": LexiconEntry("cream", KIND_CATEGORY),
    "cream": LexiconEntry("cream", KIND_CATEGORY),
    "پودر": LexiconEntry("powder", KIND_CATEGORY),
    "powder": LexiconEntry("powder", KIND_CATEGORY),
    "foundation": LexiconEntry("foundation", KIND_CATEGORY),
    "فاندیشن": LexiconEntry("foundation", KIND_CATEGORY),
    "ریمل": LexiconEntry("mascara", KIND_CATEGORY),
    "mascara": LexiconEntry("mascara", KIND_CATEGORY),
    "دئودورانت": LexiconEntry("deodorant", KIND_CATEGORY),
    "دودورانت": LexiconEntry("deodorant", KIND_CATEGORY),
    "deodorant": LexiconEntry("deodorant", KIND_CATEGORY),
    "عطر": LexiconEntry("perfume", KIND_CATEGORY),
    "ادکلن": LexiconEntry("cologne", KIND_CATEGORY),
    "perfume": LexiconEntry("perfume", KIND_CATEGORY),
    "fragrance": LexiconEntry("perfume", KIND_CATEGORY),
    "ادو": LexiconEntry("edp", KIND_CATEGORY),
    "پرفیوم": LexiconEntry("perfume", KIND_CATEGORY),
    "edp": LexiconEntry("edp", KIND_CATEGORY),
    "تویلت": LexiconEntry("edt", KIND_CATEGORY),
    "edt": LexiconEntry("edt", KIND_CATEGORY),
    "شامپو": LexiconEntry("shampoo", KIND_CATEGORY),
    "shampoo": LexiconEntry("shampoo", KIND_CATEGORY),
    "ضدآفتاب": LexiconEntry("sunscreen", KIND_CATEGORY),
    "آفتاب": LexiconEntry("sunscreen", KIND_CATEGORY),
    "sunscreen": LexiconEntry("sunscreen", KIND_CATEGORY),
    "سرم": LexiconEntry("serum", KIND_CATEGORY),
    "serum": LexiconEntry("serum", KIND_CATEGORY),
    "تونر": LexiconEntry("toner", KIND_CATEGORY),
    "toner": LexiconEntry("toner", KIND_CATEGORY),
    "رژ": LexiconEntry("lipstick", KIND_CATEGORY),
    "lipstick": LexiconEntry("lipstick", KIND_CATEGORY),
    "لب": LexiconEntry("lip", KIND_CATEGORY),
    "lip": LexiconEntry("lip", KIND_CATEGORY),
    "مداد": LexiconEntry("pencil", KIND_CATEGORY),
    "pencil": LexiconEntry("pencil", KIND_CATEGORY),
    "چشم": LexiconEntry("eye", KIND_CATEGORY),
    "eye": LexiconEntry("eye", KIND_CATEGORY),
    "سایه": LexiconEntry("eyeshadow", KIND_CATEGORY),
    "eyeshadow": LexiconEntry("eyeshadow", KIND_CATEGORY),
    "کانسیلر": LexiconEntry("concealer", KIND_CATEGORY),
    "concealer": LexiconEntry("concealer", KIND_CATEGORY),
    "پنکک": LexiconEntry("compact_powder", KIND_CATEGORY),
    "لوسیون": LexiconEntry("lotion", KIND_CATEGORY),
    "lotion": LexiconEntry("lotion", KIND_CATEGORY),
    "اسپری": LexiconEntry("spray", KIND_CATEGORY),
    "spray": LexiconEntry("spray", KIND_CATEGORY),
    "بادی": LexiconEntry("body", KIND_CATEGORY),
    "body": LexiconEntry("body", KIND_CATEGORY),
    "اسپلش": LexiconEntry("splash", KIND_CATEGORY),
    "splash": LexiconEntry("splash", KIND_CATEGORY),
    "موبایل": LexiconEntry("mobile", KIND_CATEGORY),
    "گوشی": LexiconEntry("mobile", KIND_CATEGORY),
    "mobile": LexiconEntry("mobile", KIND_CATEGORY),
    "phone": LexiconEntry("mobile", KIND_CATEGORY),
    "تلفن": LexiconEntry("mobile", KIND_CATEGORY),
    "تبلت": LexiconEntry("tablet", KIND_CATEGORY),
    "tablet": LexiconEntry("tablet", KIND_CATEGORY),
    "لپتاپ": LexiconEntry("laptop", KIND_CATEGORY),
    "لپ": LexiconEntry("laptop", KIND_CATEGORY),
    "تاپ": LexiconEntry("laptop", KIND_CATEGORY),
    "laptop": LexiconEntry("laptop", KIND_CATEGORY),
    "هدفون": LexiconEntry("headphone", KIND_CATEGORY),
    "headphone": LexiconEntry("headphone", KIND_CATEGORY),
    "هندزفری": LexiconEntry("earphone", KIND_CATEGORY),
    "earphone": LexiconEntry("earphone", KIND_CATEGORY),
    "ساعت": LexiconEntry("watch_category", KIND_CATEGORY),
    "تلویزیون": LexiconEntry("tv", KIND_CATEGORY),
    "tv": LexiconEntry("tv", KIND_CATEGORY),
    "دوربین": LexiconEntry("camera", KIND_CATEGORY),
    "camera": LexiconEntry("camera", KIND_CATEGORY),
    # Qualifiers.
    "مردانه": LexiconEntry("men", KIND_QUALIFIER),
    "مرد": LexiconEntry("men", KIND_QUALIFIER),
    "آقایان": LexiconEntry("men", KIND_QUALIFIER),
    "اقایان": LexiconEntry("men", KIND_QUALIFIER),
    "men": LexiconEntry("men", KIND_QUALIFIER),
    "mens": LexiconEntry("men", KIND_QUALIFIER),
    "male": LexiconEntry("men", KIND_QUALIFIER),
    "زنانه": LexiconEntry("women", KIND_QUALIFIER),
    "زن": LexiconEntry("women", KIND_QUALIFIER),
    "بانوان": LexiconEntry("women", KIND_QUALIFIER),
    "خانم": LexiconEntry("women", KIND_QUALIFIER),
    "women": LexiconEntry("women", KIND_QUALIFIER),
    "womens": LexiconEntry("women", KIND_QUALIFIER),
    "female": LexiconEntry("women", KIND_QUALIFIER),
    "یونیسکس": LexiconEntry("unisex", KIND_QUALIFIER),
    "unisex": LexiconEntry("unisex", KIND_QUALIFIER),
    "کودک": LexiconEntry("kids", KIND_QUALIFIER),
    "کودکان": LexiconEntry("kids", KIND_QUALIFIER),
    "اطفال": LexiconEntry("kids", KIND_QUALIFIER),
    "kids": LexiconEntry("kids", KIND_QUALIFIER),
    "چرب": LexiconEntry("oily", KIND_QUALIFIER),
    "oily": LexiconEntry("oily", KIND_QUALIFIER),
    "خشک": LexiconEntry("dry", KIND_QUALIFIER),
    "dry": LexiconEntry("dry", KIND_QUALIFIER),
    "حساس": LexiconEntry("sensitive", KIND_QUALIFIER),
    "sensitive": LexiconEntry("sensitive", KIND_QUALIFIER),
    "مختلط": LexiconEntry("combination", KIND_QUALIFIER),
    "combination": LexiconEntry("combination", KIND_QUALIFIER),
    "معمولی": LexiconEntry("normal", KIND_QUALIFIER),
    "normal": LexiconEntry("normal", KIND_QUALIFIER),
}

COMPOUND_DICTIONARY: dict[tuple[str, ...], CompoundEntry] = {
    ("استودیو", "فیکس"): CompoundEntry("studio_fix", KIND_MODEL),
    ("studio", "fix"): CompoundEntry("studio_fix", KIND_MODEL),
    ("کول", "واتر"): CompoundEntry("cool_water", KIND_MODEL),
    ("cool", "water"): CompoundEntry("cool_water", KIND_MODEL),
    ("پرو", "مکس"): CompoundEntry("pro_max", KIND_MODEL),
    ("pro", "max"): CompoundEntry("pro_max", KIND_MODEL),
    ("ایو", "سن", "لوران"): CompoundEntry("ysl", KIND_BRAND),
    ("استی", "لادر"): CompoundEntry("estee_lauder", KIND_BRAND),
    ("بابی", "براون"): CompoundEntry("bobbi_brown", KIND_BRAND),
    ("مکس", "فاکتور"): CompoundEntry("max_factor", KIND_BRAND),
    ("گلدن", "رز"): CompoundEntry("golden_rose", KIND_BRAND),
    ("کالوین", "کلین"): CompoundEntry("calvin_klein", KIND_BRAND),
    ("هوگو", "باس"): CompoundEntry("hugo_boss", KIND_BRAND),
    ("پاکو", "رابان"): CompoundEntry("paco_rabanne", KIND_BRAND),
    ("تام", "فورد"): CompoundEntry("tom_ford", KIND_BRAND),
    ("مونت", "بلانک"): CompoundEntry("montblanc", KIND_BRAND),
    ("لا", "روش"): CompoundEntry("la_roche_posay", KIND_BRAND),
    ("ضد", "آفتاب"): CompoundEntry("sunscreen", KIND_CATEGORY),
    ("body", "splash"): CompoundEntry("body_splash", KIND_CATEGORY),
    ("بادی", "اسپلش"): CompoundEntry("body_splash", KIND_CATEGORY),
}

# Explicit category descriptor map retained as a public static resource.
PRODUCT_CATEGORY_DESCRIPTOR_MAP: dict[str, str] = {
    "cream": "cream",
    "powder": "powder",
    "foundation": "foundation",
    "mascara": "mascara",
    "deodorant": "deodorant",
    "perfume": "perfume",
    "cologne": "cologne",
    "edp": "edp",
    "edt": "edt",
    "shampoo": "shampoo",
    "sunscreen": "sunscreen",
    "serum": "serum",
    "toner": "toner",
    "lipstick": "lipstick",
    "lip": "lip",
    "pencil": "pencil",
    "eye": "eye",
    "eyeshadow": "eyeshadow",
    "concealer": "concealer",
    "compact_powder": "compact_powder",
    "lotion": "lotion",
    "spray": "spray",
    "body": "body",
    "splash": "splash",
    "body_splash": "body_splash",
    "mobile": "mobile",
    "tablet": "tablet",
    "laptop": "laptop",
    "headphone": "headphone",
    "earphone": "earphone",
    "watch_category": "watch_category",
    "tv": "tv",
    "camera": "camera",
}

QUALIFIER_CANONICALS: frozenset[str] = frozenset(
    entry.canonical
    for entry in TRANSLITERATION_DICTIONARY.values()
    if entry.kind == KIND_QUALIFIER
)
BRAND_CANONICALS: frozenset[str] = frozenset(
    entry.canonical
    for entry in TRANSLITERATION_DICTIONARY.values()
    if entry.kind == KIND_BRAND
)
MODEL_CANONICALS: frozenset[str] = frozenset(
    entry.canonical
    for entry in TRANSLITERATION_DICTIONARY.values()
    if entry.kind == KIND_MODEL
)
CATEGORY_CANONICALS: frozenset[str] = frozenset(PRODUCT_CATEGORY_DESCRIPTOR_MAP.values())

# Compiled regexes. They are deliberately anchored when used as token classifiers.
PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
HARAKAT_RE = re.compile(r"[\u064B-\u0652\u0670]")
FORMAT_CONTROL_RE = re.compile(r"[\u200B\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")
SEPARATOR_RE = re.compile(r"[()\[\]{}،,;:؛/\\|\"“”'`~!؟?]+")
HYPHEN_SEPARATOR_RE = re.compile(r"[-‐-‒–—―]+")
MULTISPACE_RE = re.compile(r"\s+")
ATOMIC_CODE_RE = re.compile(r"^[A-Za-z]{1,4}\d{1,4}[A-Za-z0-9]{0,4}$")
DIGIT_LETTER_CODE_RE = re.compile(r"^\d{1,3}[A-Za-z]{1,4}\d{0,3}$")
SPF_RE = re.compile(r"^spf\d{1,3}\+?$", re.IGNORECASE)
VERSION_MODEL_RE = re.compile(r"^[A-Za-z]\d{1,3}$")
UNIT_BOUND_QUANTITY_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)?)(?P<unit>ml|m\s*l|l|lt|g|gr|kg|mg|oz|cc|iu|pcs|pc)$",
    re.IGNORECASE,
)
CANONICAL_QUANTITY_RE = re.compile(r"^\d+(?:\.\d+)?_(?:ml|l|g|kg|mg|oz|iu|count)$")
PURE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
PURE_INTEGER_2_TO_4_RE = re.compile(r"^\d{1,4}$")
ASCII_ALNUM_RE = re.compile(r"^[a-z0-9_]+$")
ASCII_WORD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SPF_CANONICAL_RE = re.compile(r"^spf\d{1,3}$")
SKU_LIKE_RE = re.compile(
    r"^(?:[a-z]{1,4}\d{1,4}[a-z0-9]{0,4}|\d{1,3}[a-z]{1,4}\d{0,3})$"
)


def _ascii_fold(value: str) -> str:
    """Return lowercase ASCII text by removing diacritics and non-ASCII marks."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_text.encode("ascii", "ignore").decode("ascii").lower()


def _normalize_unicode(raw: str) -> str:
    """Apply Stage A Unicode, Persian character, digit, and separator normalization."""
    value = unicodedata.normalize("NFKC", raw)
    value = value.replace("ي", "ی")
    value = value.replace("ك", "ک")
    value = HARAKAT_RE.sub("", value)
    value = value.replace("\u0640", "")
    value = value.replace("\u200c", " ")
    value = value.replace("\u200d", "")
    value = value.translate(_DIGIT_TRANSLATION)
    value = FORMAT_CONTROL_RE.sub("", value)
    value = HYPHEN_SEPARATOR_RE.sub(" ", value)
    value = SEPARATOR_RE.sub(" ", value)
    value = MULTISPACE_RE.sub(" ", value).strip()
    return value


def _contains_persian(value: str) -> bool:
    """Return True when the value contains Arabic-script letters."""
    return bool(PERSIAN_CHAR_RE.search(value))


def _contains_latin(value: str) -> bool:
    """Return True when the value contains Latin letters."""
    return bool(LATIN_CHAR_RE.search(value))


def _is_atomic_token(token: str) -> bool:
    """Return True for structural tokens that must not be script-split."""
    token_l = token.lower()
    if UNIT_BOUND_QUANTITY_RE.fullmatch(token_l):
        return True
    if SPF_RE.fullmatch(token_l):
        return True
    if ATOMIC_CODE_RE.fullmatch(token):
        return True
    if DIGIT_LETTER_CODE_RE.fullmatch(token):
        return True
    if VERSION_MODEL_RE.fullmatch(token):
        return True
    return False


def _split_on_persian_boundaries(token: str) -> list[str]:
    """Split a raw token where Persian script abuts non-Persian script."""
    if not token:
        return []
    if _is_atomic_token(token):
        return [token]
    pieces: list[str] = []
    current_chars: list[str] = []
    current_is_persian: bool | None = None
    for ch in token:
        ch_is_persian = _contains_persian(ch)
        if current_is_persian is None:
            current_chars.append(ch)
            current_is_persian = ch_is_persian
        elif ch_is_persian == current_is_persian:
            current_chars.append(ch)
        else:
            piece = "".join(current_chars).strip()
            if piece:
                pieces.append(piece)
            current_chars = [ch]
            current_is_persian = ch_is_persian
    final_piece = "".join(current_chars).strip()
    if final_piece:
        pieces.append(final_piece)
    return pieces


def _whitespace_and_script_tokenize(value: str) -> list[str]:
    """Tokenize by whitespace and then split mixed Persian/non-Persian tokens."""
    raw_tokens = value.split()
    tokens: list[str] = []
    for raw in raw_tokens:
        for piece in _split_on_persian_boundaries(raw):
            clean = piece.strip("._+")
            if clean:
                tokens.append(clean)
    return tokens


def _normalize_number(number_text: str) -> str:
    """Normalize a numeric string for canonical quantity and SKU comparison."""
    if "." not in number_text:
        stripped = number_text.lstrip("0")
        return stripped if stripped else "0"
    integer_part, fractional_part = number_text.split(".", 1)
    integer_part = integer_part.lstrip("0") or "0"
    fractional_part = fractional_part.rstrip("0")
    if not fractional_part:
        return integer_part
    return f"{integer_part}.{fractional_part}"


def _normalize_unit(unit: str) -> str | None:
    """Return the canonical unit for a unit surface form, or None when unknown."""
    key = _ascii_fold(unit.replace(".", "").strip()) if _contains_latin(unit) else unit.strip()
    key = MULTISPACE_RE.sub(" ", key.lower())
    return UNIT_NORMALIZATION.get(key)


def _quantity_from_unit_bound(token: str) -> str | None:
    """Return a canonical quantity token when a token combines number and unit."""
    match = UNIT_BOUND_QUANTITY_RE.fullmatch(token.lower())
    if not match:
        return None
    unit = _normalize_unit(match.group("unit"))
    if unit is None:
        return None
    number = _normalize_number(match.group("num"))
    return f"{number}_{unit}"


def _consolidate_quantities(tokens: list[str]) -> list[str]:
    """Merge adjacent number and unit tokens into canonical quantity tokens."""
    consolidated: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        bound_quantity = _quantity_from_unit_bound(token)
        if bound_quantity is not None:
            consolidated.append(bound_quantity)
            i += 1
            continue

        if PURE_NUMBER_RE.fullmatch(token):
            number = _normalize_number(token)
            if i + 2 < len(tokens):
                two_token_unit = f"{tokens[i + 1]} {tokens[i + 2]}"
                unit = _normalize_unit(two_token_unit)
                if unit is not None:
                    consolidated.append(f"{number}_{unit}")
                    i += 3
                    continue
            if i + 1 < len(tokens):
                unit = _normalize_unit(tokens[i + 1])
                if unit is not None:
                    consolidated.append(f"{number}_{unit}")
                    i += 2
                    continue

        consolidated.append(token)
        i += 1
    return consolidated


def _remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remove Persian and English stopwords after tokenization."""
    filtered: list[str] = []
    for token in tokens:
        key = token.lower() if _contains_latin(token) else token
        if key not in PERSIAN_STOPWORDS:
            filtered.append(token)
    return filtered


def _normalize_surface_for_compound(token: str) -> str:
    """Normalize a token surface for compound dictionary lookup."""
    if CANONICAL_QUANTITY_RE.fullmatch(token):
        return token
    return _ascii_fold(token) if _contains_latin(token) else token


def _apply_compound_dictionary(tokens: list[str]) -> list[str]:
    """Collapse static multi-token lexical entries into one canonical surface."""
    result: list[str] = []
    max_len = max((len(key) for key in COMPOUND_DICTIONARY), default=1)
    i = 0
    normalized = [_normalize_surface_for_compound(token) for token in tokens]
    while i < len(tokens):
        matched = False
        upper = min(max_len, len(tokens) - i)
        for size in range(upper, 1, -1):
            key = tuple(normalized[i : i + size])
            entry = COMPOUND_DICTIONARY.get(key)
            if entry is not None:
                result.append(entry.canonical)
                i += size
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1
    return result


def _romanize_persian(value: str) -> str:
    """Romanize Persian script to a lowercase ASCII approximation."""
    pieces: list[str] = []
    for ch in value:
        if ch in ROMANIZATION_MAP:
            pieces.append(ROMANIZATION_MAP[ch])
        elif ch.isdigit():
            pieces.append(ch)
        elif _contains_latin(ch):
            pieces.append(_ascii_fold(ch))
    romanized = "".join(pieces)
    romanized = re.sub(r"[^a-z0-9]+", "", romanized.lower())
    return romanized


def _fallback_phonetic_key(value: str) -> str:
    """Return a conservative deterministic phonetic skeleton when the optional package is absent."""
    folded = _ascii_fold(value)
    folded = re.sub(r"[^a-z0-9]+", "", folded)
    if not folded:
        return ""
    replacements = (
        ("ph", "f"),
        ("gh", "g"),
        ("kh", "k"),
        ("sh", "x"),
        ("ch", "c"),
        ("zh", "j"),
        ("ck", "k"),
        ("q", "k"),
        ("c", "k"),
        ("w", "v"),
        ("y", "i"),
    )
    key = folded
    for old, new in replacements:
        key = key.replace(old, new)
    first = key[0]
    tail = re.sub(r"[aeiou]+", "", key[1:])
    collapsed: list[str] = [first]
    for ch in tail:
        if ch != collapsed[-1]:
            collapsed.append(ch)
    return "".join(collapsed)


def _phonetic_codes(value: str) -> frozenset[str]:
    """Return Double Metaphone codes, using a deterministic fallback when unavailable."""
    folded = _ascii_fold(value)
    if not folded:
        return frozenset()
    if _doublemetaphone_impl is not None:
        codes = _doublemetaphone_impl(folded)
        return frozenset(code.lower() for code in codes if code)
    fallback = _fallback_phonetic_key(folded)
    return frozenset({fallback}) if fallback else frozenset()


def _build_phonetic_index(kind: str) -> dict[str, LexiconEntry]:
    """Build a phonetic lookup from canonical entries of one kind."""
    index: dict[str, LexiconEntry] = {}
    canonical_values = sorted(
        {entry.canonical for entry in TRANSLITERATION_DICTIONARY.values() if entry.kind == kind}
    )
    for canonical in canonical_values:
        if len(canonical.replace("_", "")) < 4:
            continue
        for code in _phonetic_codes(canonical):
            index.setdefault(code, LexiconEntry(canonical, kind))
    return index


BRAND_PHONETIC_INDEX: dict[str, LexiconEntry] = _build_phonetic_index(KIND_BRAND)
MODEL_PHONETIC_INDEX: dict[str, LexiconEntry] = _build_phonetic_index(KIND_MODEL)


def _lookup_phonetic_entry(romanized: str) -> LexiconEntry | None:
    """Resolve a romanized unknown token to a known brand or model by phonetic code."""
    if len(romanized) < 4:
        return None
    for code in _phonetic_codes(romanized):
        brand = BRAND_PHONETIC_INDEX.get(code)
        if brand is not None:
            return brand
        model = MODEL_PHONETIC_INDEX.get(code)
        if model is not None:
            return model
    return None


def _canonicalize_structural_token(token: str) -> tuple[str, str, str] | None:
    """Canonicalize SKU, SPF, and quantity tokens before lexical lookup."""
    quantity = _quantity_from_unit_bound(token)
    if quantity is not None:
        return quantity, KIND_QUANTITY, SOURCE_STRUCTURAL

    lower = _ascii_fold(token)
    if CANONICAL_QUANTITY_RE.fullmatch(lower):
        return lower, KIND_QUANTITY, SOURCE_STRUCTURAL

    if SPF_RE.fullmatch(lower):
        canonical = lower.rstrip("+")
        return canonical, KIND_QUALIFIER, SOURCE_STRUCTURAL

    if SKU_LIKE_RE.fullmatch(lower) or (PURE_INTEGER_2_TO_4_RE.fullmatch(lower) and len(lower) >= 2):
        return lower, KIND_SKU, SOURCE_STRUCTURAL

    return None


def _resolve_token(token: str) -> tuple[str, str, str]:
    """Resolve a surface token to canonical form, kind, and resolution source."""
    structural = _canonicalize_structural_token(token)
    if structural is not None:
        return structural

    dictionary_key = _normalize_surface_for_compound(token)
    entry = TRANSLITERATION_DICTIONARY.get(dictionary_key)
    if entry is not None:
        return entry.canonical, entry.kind, SOURCE_DICT

    if _contains_persian(token):
        romanized = _romanize_persian(token)
        phonetic_entry = _lookup_phonetic_entry(romanized)
        if phonetic_entry is not None:
            return phonetic_entry.canonical, phonetic_entry.kind, SOURCE_PHONETIC
        if romanized:
            return romanized, KIND_RESIDUAL, SOURCE_PHONETIC
        return token, KIND_RESIDUAL, SOURCE_RESIDUAL

    folded = _ascii_fold(token)
    if not folded:
        return token.lower(), KIND_RESIDUAL, SOURCE_RESIDUAL

    entry = TRANSLITERATION_DICTIONARY.get(folded)
    if entry is not None:
        return entry.canonical, entry.kind, SOURCE_DICT

    return folded, KIND_RESIDUAL, SOURCE_LATIN


def _classify_token(canonical: str, kind: str) -> int:
    """Assign the specification's discriminativity class to a canonical token."""
    if CANONICAL_QUANTITY_RE.fullmatch(canonical) or kind == KIND_QUANTITY:
        return CLASS_QUANTITY
    if SPF_CANONICAL_RE.fullmatch(canonical) or canonical in QUALIFIER_CANONICALS or kind == KIND_QUALIFIER:
        return CLASS_QUALIFIER
    if kind == KIND_SKU or SKU_LIKE_RE.fullmatch(canonical) or (
        PURE_INTEGER_2_TO_4_RE.fullmatch(canonical) and len(canonical) >= 2
    ):
        return CLASS_SKU
    if kind == KIND_BRAND or canonical in BRAND_CANONICALS:
        return CLASS_BRAND
    if kind == KIND_MODEL or canonical in MODEL_CANONICALS:
        return CLASS_MODEL
    if kind == KIND_CATEGORY or canonical in CATEGORY_CANONICALS:
        return CLASS_CATEGORY
    if ASCII_WORD_RE.fullmatch(canonical):
        return CLASS_RESIDUAL
    return CLASS_RESIDUAL


def preprocess(raw: str) -> tuple[Token, ...]:
    """Run complete Stage A preprocessing and return classified canonical tokens."""
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    normalized = _normalize_unicode(raw)
    if not normalized:
        return tuple()

    tokens = _whitespace_and_script_tokenize(normalized)
    tokens = _remove_stopwords(tokens)
    tokens = _consolidate_quantities(tokens)
    tokens = _remove_stopwords(tokens)
    tokens = _apply_compound_dictionary(tokens)

    output: list[Token] = []
    for surface in tokens:
        canonical, kind, source = _resolve_token(surface)
        if not canonical:
            continue
        canonical = canonical.lower()
        token_class = _classify_token(canonical, kind)
        weight = WEIGHTS[token_class]
        output.append(Token(surface, canonical, token_class, weight, source, kind, surface))
    return tuple(output)


def _canonical_set(tokens: Iterable[Token]) -> frozenset[str]:
    """Return the canonical token set for a token sequence."""
    return frozenset(token.canonical for token in tokens)


def _best_token_by_canonical(tokens: Iterable[Token]) -> dict[str, Token]:
    """Return the strongest token for each canonical form."""
    best: dict[str, Token] = {}
    for token in tokens:
        current = best.get(token.canonical)
        if current is None or token.weight > current.weight:
            best[token.canonical] = token
    return best


def _tokens_by_class(tokens: Iterable[Token], token_class: int) -> frozenset[str]:
    """Return canonical tokens of a particular discriminativity class."""
    return frozenset(token.canonical for token in tokens if token.token_class == token_class)


def _overlap_coefficient(query_tokens: tuple[Token, ...], product_tokens: tuple[Token, ...]) -> float:
    """Compute the Stage B canonical token-set overlap coefficient."""
    q_set = _canonical_set(query_tokens)
    p_set = _canonical_set(product_tokens)
    if not q_set or not p_set:
        return 0.0
    intersection_size = len(q_set & p_set)
    denominator = min(len(q_set), len(p_set))
    return intersection_size / denominator if denominator else 0.0


def _hard_gate(query_tokens: tuple[Token, ...], product_tokens: tuple[Token, ...]) -> bool:
    """Apply Stage C hard gates for SKU, quantity, and qualifier classes."""
    for token_class in (CLASS_SKU, CLASS_QUANTITY, CLASS_QUALIFIER):
        query_critical = _tokens_by_class(query_tokens, token_class)
        if not query_critical:
            continue
        product_critical = _tokens_by_class(product_tokens, token_class)
        if not query_critical.issubset(product_critical):
            return False
    return True


def _query_weight_sum(query_best: dict[str, Token]) -> float:
    """Return total unique canonical query weight."""
    return sum(token.weight for token in query_best.values())


def _has_phonetic_source(canonical: str, query_token: Token, product_best: dict[str, Token]) -> bool:
    """Return True when an exact match used a PHONETIC-resolved token."""
    product_token = product_best.get(canonical)
    if product_token is None:
        return query_token.source == SOURCE_PHONETIC
    return query_token.source == SOURCE_PHONETIC or product_token.source == SOURCE_PHONETIC


def _asymmetric_query_coverage(query_tokens: tuple[Token, ...], product_tokens: tuple[Token, ...]) -> float:
    """Compute S1: asymmetric query coverage with the PHONETIC confidence discount."""
    query_best = _best_token_by_canonical(query_tokens)
    product_best = _best_token_by_canonical(product_tokens)
    denominator = _query_weight_sum(query_best)
    if denominator <= 0.0:
        return 0.0

    numerator = 0.0
    for canonical, query_token in query_best.items():
        if canonical in product_best:
            multiplier = PHONETIC_CONFIDENCE_DISCOUNT if _has_phonetic_source(canonical, query_token, product_best) else 1.0
            numerator += query_token.weight * multiplier
    return min(1.0, numerator / denominator)


def _weighted_token_jaccard(query_tokens: tuple[Token, ...], product_tokens: tuple[Token, ...]) -> float:
    """Compute S2: weighted token Jaccard over canonical token sets.

    Query-side critical tokens are included. Product-only critical tokens are not
    penalized in the denominator because the hard gate defines them as extra
    product specificity rather than contradiction when absent from the query.
    """
    query_best = _best_token_by_canonical(query_tokens)
    product_best = _best_token_by_canonical(product_tokens)
    if not query_best or not product_best:
        return 0.0

    query_keys = set(query_best)
    product_keys = set(product_best)
    union_keys = set(query_keys)
    for key in product_keys:
        product_token = product_best[key]
        if key not in query_best and product_token.token_class in {CLASS_SKU, CLASS_QUANTITY, CLASS_QUALIFIER}:
            continue
        union_keys.add(key)

    denominator = 0.0
    for key in union_keys:
        q_token = query_best.get(key)
        p_token = product_best.get(key)
        if q_token is not None and p_token is not None:
            denominator += max(q_token.weight, p_token.weight)
        elif q_token is not None:
            denominator += q_token.weight
        elif p_token is not None:
            denominator += p_token.weight

    if denominator <= 0.0:
        return 0.0

    numerator = 0.0
    for key in query_keys & product_keys:
        numerator += min(query_best[key].weight, product_best[key].weight)
    return min(1.0, numerator / denominator)


def _trigram_counter(value: str) -> Counter[str]:
    """Return a boundary-padded character trigram multiset."""
    padded = f"  {value} "
    if len(padded) < 3:
        return Counter()
    return Counter(padded[i : i + 3] for i in range(len(padded) - 2))


def _trigram_dice(left: str, right: str) -> float:
    """Compute multiset Dice similarity over boundary-padded character trigrams."""
    if left == right:
        return 1.0
    if len(left) < 4 or len(right) < 4:
        return 0.0
    left_counter = _trigram_counter(left)
    right_counter = _trigram_counter(right)
    if not left_counter or not right_counter:
        return 0.0
    intersection = sum((left_counter & right_counter).values())
    denominator = sum(left_counter.values()) + sum(right_counter.values())
    return (2.0 * intersection / denominator) if denominator else 0.0


def _soft_trigram_coverage(query_tokens: tuple[Token, ...], product_tokens: tuple[Token, ...]) -> float:
    """Compute S3: exact query coverage plus tiered trigram soft matches."""
    query_best = _best_token_by_canonical(query_tokens)
    product_best = _best_token_by_canonical(product_tokens)
    denominator = _query_weight_sum(query_best)
    if denominator <= 0.0:
        return 0.0

    exact_weight = sum(token.weight for key, token in query_best.items() if key in product_best)
    product_by_class: dict[int, list[Token]] = {}
    for token in product_best.values():
        product_by_class.setdefault(token.token_class, []).append(token)

    soft_weight = 0.0
    for canonical, query_token in query_best.items():
        if canonical in product_best:
            continue
        if query_token.token_class not in {CLASS_BRAND, CLASS_MODEL, CLASS_CATEGORY, CLASS_RESIDUAL}:
            continue
        candidates = product_by_class.get(query_token.token_class, [])
        if not candidates:
            continue
        best_dice = 0.0
        for product_token in candidates:
            dice = _trigram_dice(query_token.canonical, product_token.canonical)
            if dice > best_dice:
                best_dice = dice
        if best_dice >= 0.80:
            soft_weight += 0.85 * query_token.weight
        elif best_dice >= 0.50:
            soft_weight += 0.50 * query_token.weight

    return min(1.0, (exact_weight + soft_weight) / denominator)


def _stage_c_score(query_tokens: tuple[Token, ...], product_tokens: tuple[Token, ...]) -> float:
    """Run Stage C hard gates and soft scoring, returning a score in [0, 1]."""
    if not _hard_gate(query_tokens, product_tokens):
        return 0.0

    s1 = _asymmetric_query_coverage(query_tokens, product_tokens)
    s2 = _weighted_token_jaccard(query_tokens, product_tokens)
    s3 = _soft_trigram_coverage(query_tokens, product_tokens)
    score = 0.50 * s1 + 0.25 * s2 + 0.25 * s3

    query_weight = _query_weight_sum(_best_token_by_canonical(query_tokens))
    if query_weight < WEAK_QUERY_WEIGHT_THRESHOLD:
        score = min(score, WEAK_QUERY_SCORE_CAP)

    if len(_canonical_set(product_tokens)) == 1:
        score = min(score, SINGLE_TOKEN_PRODUCT_SCORE_CAP)

    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


@lru_cache(maxsize=4096)
def _preprocess_query_cached(search_query: str) -> tuple[Token, ...]:
    """Memoize query preprocessing across repeated product comparisons.

    functools.lru_cache is sufficient here because preprocessing is pure,
    idempotent, keyed only by the raw query string, bounded by maxsize, and
    internally synchronized for concurrent access in normal CPython use.
    """
    return preprocess(search_query)


# ─────────────────────────────────────────────────────────────────────────────
# Brand-lexicon hot-loading
#
# brands.json schema
# ──────────────────
# {
#   "trans_dict":   { "<surface>": ["<canonical>", "<kind>"], … },
#   "compound_dict": { "<tok1>,<tok2>[,…]": ["<canonical>", "<kind>"], … }
# }
#
# Loading strategy
# ────────────────
# • Eager: attempted automatically at import time from the sibling
#   brands.json (same directory as this module).
# • Explicit: callers may call initialize_brands(path) before the first
#   similarity_score() call to override the default path.
# • Singleton: whichever path is tried first wins; subsequent calls are
#   no-ops (fast path without acquiring the lock).
# • Thread-safe: a threading.Lock + double-checked sentinel guards the
#   critical section so concurrent first-callers converge without races.
# • Cache coherence: the LRU query cache is cleared after a successful
#   merge so stale preprocessed tokens are never reused.
# ─────────────────────────────────────────────────────────────────────────────

_brands_lock: threading.Lock = threading.Lock()
_brands_loaded: bool = False          # written only inside _brands_lock
_brands_load_error: str | None = None # informational; set on failure


def _merge_brands_data(data: dict) -> None:  # noqa: C901
    """Merge a parsed brands.json payload into the live module dictionaries.

    Existing entries are never overwritten so the hand-curated in-module
    lexicon always takes precedence over the JSON data.  Must be called
    while the caller holds *_brands_lock* or during single-threaded import.
    """
    global BRAND_CANONICALS, MODEL_CANONICALS, QUALIFIER_CANONICALS, CATEGORY_CANONICALS

    changed = False

    # ── trans_dict → TRANSLITERATION_DICTIONARY ───────────────────────────
    #
    # Each entry is:  "<surface>": ["<canonical>", "<kind>"]
    #
    # Surface keys may be Persian or ASCII-folded Latin; the lookup path in
    # _resolve_token normalises via _normalize_surface_for_compound before
    # hitting the dictionary, so keys must already be in their normalised
    # form (which brands.json is expected to provide).
    for surface, payload in data.get("trans_dict", {}).items():
        if not (isinstance(payload, (list, tuple)) and len(payload) >= 2):
            continue
        canonical, kind = str(payload[0]).lower(), str(payload[1]).lower()
        if not canonical or not kind:
            continue
        if surface not in TRANSLITERATION_DICTIONARY:
            TRANSLITERATION_DICTIONARY[surface] = LexiconEntry(canonical, kind)
            changed = True
        # Register the canonical string itself so that when a compound entry
        # collapses multiple tokens down to this canonical, the subsequent
        # _resolve_token() call (which looks up the collapsed string) returns
        # the correct kind rather than falling through to KIND_RESIDUAL.
        # This mirrors the pattern used by the hard-coded entries, e.g.
        # "studio_fix": LexiconEntry("studio_fix", KIND_MODEL).
        if canonical not in TRANSLITERATION_DICTIONARY:
            TRANSLITERATION_DICTIONARY[canonical] = LexiconEntry(canonical, kind)
            changed = True

    # ── compound_dict → COMPOUND_DICTIONARY ──────────────────────────────
    #
    # Each entry is:  "<tok1>,<tok2>[,<tok3>]": ["<canonical>", "<kind>"]
    #
    # Comma-separated tokens map directly to the tuple keys used by
    # _apply_compound_dictionary.  Tokens are stripped of surrounding
    # whitespace; empty parts are skipped.
    for key_str, payload in data.get("compound_dict", {}).items():
        if not (isinstance(payload, (list, tuple)) and len(payload) >= 2):
            continue
        canonical, kind = str(payload[0]).lower(), str(payload[1]).lower()
        if not canonical or not kind:
            continue
        key_tuple = tuple(part.strip() for part in key_str.split(",") if part.strip())
        if len(key_tuple) < 2:
            # A single-token "compound" belongs in trans_dict, not here.
            continue
        if key_tuple not in COMPOUND_DICTIONARY:
            COMPOUND_DICTIONARY[key_tuple] = CompoundEntry(canonical, kind)
            changed = True
        # Same self-registration as trans_dict: compound resolution leaves
        # a bare canonical string that _resolve_token must recognise.
        if canonical not in TRANSLITERATION_DICTIONARY:
            TRANSLITERATION_DICTIONARY[canonical] = LexiconEntry(canonical, kind)
            changed = True

    if not changed:
        return

    # ── rebuild canonical fast-lookup frozensets ──────────────────────────
    #
    # _classify_token references these module-level names at call time, so
    # reassigning them (via global) is picked up immediately by all callers.
    BRAND_CANONICALS = frozenset(
        e.canonical
        for e in TRANSLITERATION_DICTIONARY.values()
        if e.kind == KIND_BRAND
    )
    MODEL_CANONICALS = frozenset(
        e.canonical
        for e in TRANSLITERATION_DICTIONARY.values()
        if e.kind == KIND_MODEL
    )
    QUALIFIER_CANONICALS = frozenset(
        e.canonical
        for e in TRANSLITERATION_DICTIONARY.values()
        if e.kind == KIND_QUALIFIER
    )
    # CATEGORY_CANONICALS merges both the static descriptor map and any new
    # category entries that brands.json may contribute.
    CATEGORY_CANONICALS = frozenset(PRODUCT_CATEGORY_DESCRIPTOR_MAP.values()) | frozenset(
        e.canonical
        for e in TRANSLITERATION_DICTIONARY.values()
        if e.kind == KIND_CATEGORY
    )

    # ── rebuild phonetic indexes in-place ────────────────────────────────
    #
    # _lookup_phonetic_entry holds direct references to these dicts, so
    # mutating them in-place (rather than rebinding the names) is the
    # correct update strategy — no global declaration needed.
    BRAND_PHONETIC_INDEX.clear()
    BRAND_PHONETIC_INDEX.update(_build_phonetic_index(KIND_BRAND))
    MODEL_PHONETIC_INDEX.clear()
    MODEL_PHONETIC_INDEX.update(_build_phonetic_index(KIND_MODEL))

    # ── clear the query preprocessing LRU cache ───────────────────────────
    #
    # Cached tokens may reference the pre-load vocabulary.  Clearing is
    # cheap relative to how rarely a load occurs vs. how often the cache
    # is queried.
    _preprocess_query_cached.cache_clear()


def initialize_brands(json_path: str) -> bool:
    """Load the brand lexicon from *json_path* exactly once (thread-safe).

    This function is idempotent: once the lexicon has been loaded (or a
    load attempt has failed) every subsequent call returns ``False``
    immediately without touching the lock.

    Parameters
    ----------
    json_path:
        Absolute or relative path to ``brands.json``.

    Returns
    -------
    bool
        ``True`` when the file was parsed and merged successfully.
        ``False`` when loading was skipped (already done) or failed
        (warning printed to *stderr*).
    """
    global _brands_loaded, _brands_load_error

    # Fast path — check without the lock first (common case after first call).
    if _brands_loaded:
        return False

    with _brands_lock:
        # Re-check inside the lock — another thread may have loaded between
        # the fast-path check and acquiring the lock.
        if _brands_loaded:
            return False

        try:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            _merge_brands_data(data)
            _brands_loaded = True
            return True

        except FileNotFoundError:
            _brands_load_error = f"brands file not found: {json_path}"
        except json.JSONDecodeError as exc:
            _brands_load_error = f"brands JSON parse error in '{json_path}': {exc}"
        except Exception as exc:  # noqa: BLE001
            _brands_load_error = f"brands load failed for '{json_path}': {exc}"

        # Mark as attempted so we don't retry on every similarity_score() call.
        _brands_loaded = True
        print(
            f"[similarity_score] WARNING: {_brands_load_error}",
            file=sys.stderr,
        )
        return False


def brands_load_status() -> dict[str, object]:
    """Return a snapshot of the brand-loader state for diagnostics.

    Example::

        >>> from similarity_score import brands_load_status
        >>> brands_load_status()
        {'loaded': True, 'error': None,
         'trans_dict_size': 312, 'compound_dict_size': 28}
    """
    return {
        "loaded": _brands_loaded,
        "error": _brands_load_error,
        "trans_dict_size": len(TRANSLITERATION_DICTIONARY),
        "compound_dict_size": len(COMPOUND_DICTIONARY),
    }


# ── Auto-load: look for brands.json next to this source file ─────────────
#
# ``__file__`` may be a .pyc path in some deployment layouts; resolve to the
# real directory so the sibling lookup is always relative to the source tree.
_DEFAULT_BRANDS_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "brands.json"
)
if os.path.isfile(_DEFAULT_BRANDS_PATH):
    initialize_brands(_DEFAULT_BRANDS_PATH)


def similarity_score(search_query: str, product_name: str) -> float:
    """Return the Persian-English product-name similarity score in [0, 1]."""
    query_tokens = _preprocess_query_cached(search_query)
    product_tokens = preprocess(product_name)
    if not query_tokens or not product_tokens:
        return 0.0

    if _overlap_coefficient(query_tokens, product_tokens) < PREFILTER_THRESHOLD:
        return 0.0

    return _stage_c_score(query_tokens, product_tokens)


def run_verification() -> None:
    """Run specification-driven assertions for cross-script, near-miss, and edge cases."""
    score = similarity_score("samsung galaxy s25 ultra", "سامسونگ گلکسی s25 اولترا")
    assert score >= 0.95, "Stage A resolves Persian tokens to samsung/galaxy/ultra and the s25 SKU gate succeeds."

    score = similarity_score("کرم پودر مک NC41", "کرم پودر مک NC40")
    assert score == 0.0, "Stage C Class 1 SKU gate rejects nc41 because product Class 1 set contains nc40."

    score = similarity_score("ادو پرفیوم دیویدوف 40 میلی لیتر", "ادو پرفیوم دیویدوف 50 میلی لیتر")
    assert score == 0.0, "Stage C Class 2 quantity gate rejects 40_ml because product quantity is 50_ml."

    score = similarity_score("دئودورانت رکسونا مردانه", "دئودورانت رکسونا زنانه")
    assert score == 0.0, "Stage C Class 3 qualifier gate rejects men because product qualifier is women."

    score = similarity_score("ریمل پیپا مدل 503", "ریمل پیپا مدل 507")
    assert score == 0.0, "Stage C Class 1 numeric model-code gate rejects 503 because product has 507."

    score = similarity_score("کرم پودر مک NC41", "کرم پودر مک استودیو فیکس NC41 SPF15 30ml")
    assert score >= 0.95, "Stage C accepts the query SKU and treats product-only SPF and quantity as extra specificity."

    score = similarity_score("کرم پودر مک NC41", "موبایل سامسونگ گلکسی s24")
    assert score == 0.0, "Stage B rejects the pair because canonical token overlap is zero."

    score = similarity_score("mac studio fix nc41", "mac studio fix nc41 foundation")
    assert score >= 0.95, "Latin-only equivalent reaches Stage C with all query tokens covered and one low-weight extra category token."

    score = similarity_score("", "سامسونگ گلکسی s25")
    assert score == 0.0, "Stage D returns zero for an empty query after Stage A emits no tokens."

    score = similarity_score("سامسونگ گلکسی s25", "")
    assert score == 0.0, "Stage D returns zero for an empty product after Stage A emits no tokens."

    score = similarity_score("کرم", "کرم پودر مک")
    assert score <= 0.85, "Stage C weak-query cap limits underspecified category-only queries below strong-match confidence."

    score = similarity_score("سامسونگ", "سامسونگ")
    assert score <= 0.70, "Stage C single-token product-name cap prevents a one-token product from scoring as a full semantic match."

    score = similarity_score("ادو پرفیوم دیویدوف کول واتر", "ادو پرفیوم دیویدوف کول واتر 125ml")
    assert score >= 0.95, "Stage C quantity gate is asymmetric, so product-only 125_ml does not reject an unspecified-size query."

    score = similarity_score("MAC NC41", "مک nc41")
    assert score >= 0.95, "Stage A lowercases Latin SKU text and resolves Persian مک to the same canonical brand token."


if __name__ == "__main__":
    run_verification()