#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_brands_json.py
────────────────────
Parses brands.txt (one brand per line, format: "Persian display (English canonical)")
and produces brands.json consumed by similarity_score.py at import time.

Output schema
─────────────
{
  "trans_dict": {
    "<normalised_token>": ["<canonical>", "brand"],
    ...
  },
  "compound_dict": {
    "<tok1>,<tok2>": ["<canonical_compound>", "brand"],
    ...
  }
}

Keys in trans_dict are normalised surface forms (both Persian and Latin).
Keys in compound_dict are comma-joined normalised token tuples.
All values are ["<lowercase_ascii_canonical>", "brand"] pairs.

Usage
─────
    python build_brands_json.py brands.txt brands.json
    python build_brands_json.py              # defaults: brands.txt → brands.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# §1  Persian normalisation (mirrors Stage-A of similarity_score.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

_RE_STRIP_CHARS = re.compile(r"[\u200B\uFEFF\u200D\u200E\u200F\u202A-\u202E]")
_RE_PERSIAN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_RE_WHITESPACE = re.compile(r"\s+")


def _unicode_normalize(text: str) -> str:
    """
    Apply the same seven-step Stage-A normalisation used by similarity_score.py.
    Ensures tokens produced here match tokens produced at runtime.
    """
    # Step 1: NFKC — resolves Arabic Presentation Forms
    text = unicodedata.normalize("NFKC", text)
    # Step 2: Arabic Ye (U+064A) → Persian Ye (U+06CC)
    text = text.replace("\u064a", "\u06cc")
    # Step 3: Arabic Kaf (U+0643) → Persian Kaf (U+06A9)
    text = text.replace("\u0643", "\u06a9")
    # Step 4: Strip harakat / tashkil (U+064B–U+0652) and shadda
    for cp in range(0x064B, 0x0653):
        text = text.replace(chr(cp), "")
    # Step 5: Strip tatweel / kashida (U+0640)
    text = text.replace("\u0640", "")
    # Step 6: ZWNJ (U+200C) → space
    text = text.replace("\u200c", " ")
    # Step 7: Persian digits (U+06F0–U+06F9) → ASCII digits
    for i in range(10):
        text = text.replace(chr(0x06F0 + i), str(i))
    # Step 7b: Arabic-Indic digits (U+0660–U+0669) → ASCII digits
    for i in range(10):
        text = text.replace(chr(0x0660 + i), str(i))
    # Step 8: Strip remaining control / formatting chars
    text = _RE_STRIP_CHARS.sub("", text)
    return text


def _is_latin_only(token: str) -> bool:
    """True if *token* contains no Persian/Arabic script codepoints."""
    return not bool(_RE_PERSIAN.search(token))


def _safe_canonical(raw: str) -> str:
    """
    Produce a lowercase ASCII canonical from a raw English brand name.
    Strips non-alphanumeric characters other than spaces, then converts
    spaces to underscores.

    Examples:
        "Atelier Cologne"  → "atelier_cologne"
        "3Q Beauty"        → "3q_beauty"
        "L'Oréal"         → "loreal"
        "8*4"             → "8_4"
    """
    # Lowercase first
    name = raw.lower().strip()
    # NFD + strip combining marks (diacritics)
    name = unicodedata.normalize("NFD", name)
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Mn")
    # Replace non-alphanumeric (except space) with underscore
    name = re.sub(r"[^a-z0-9 ]", "_", name)
    # Collapse multiple spaces / underscores
    name = re.sub(r"[ _]+", "_", name).strip("_")
    return name


# ══════════════════════════════════════════════════════════════════════════════
# §2  Line parser
# ══════════════════════════════════════════════════════════════════════════════

# Matches: <anything> ( <anything> )   with optional trailing whitespace
_LINE_RE = re.compile(r"^(.+?)\s*\((.+?)\)\s*$")


def _parse_line(line: str) -> tuple[str, str] | None:
    """
    Parse one line from brands.txt.

    Returns (persian_display, english_canonical_raw) or None for blank/malformed lines.

    brands.txt format (examples):
        آبان (aban)
        آتلیه کلون (Atelier Cologne)
        3q بیوتی (3Q Beauty)
        121 (121)
        8*4 (8*4)
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = _LINE_RE.match(line)
    if not m:
        return None
    persian_display = m.group(1).strip()
    english_raw = m.group(2).strip()
    return persian_display, english_raw


# ══════════════════════════════════════════════════════════════════════════════
# §3  Token extraction
# ══════════════════════════════════════════════════════════════════════════════

def _tokenise_persian_display(display: str) -> list[str]:
    """
    Normalise and whitespace-split the Persian display name.
    Returns a list of normalised surface-form tokens (may include Latin sub-tokens
    if the brand name is a mix like '3q بیوتی').
    """
    normed = _unicode_normalize(display)
    tokens = [t for t in _RE_WHITESPACE.split(normed.strip()) if t]
    return tokens


def _tokenise_english(english_raw: str) -> list[str]:
    """
    Lowercase and whitespace-split the English canonical name.
    Returns a list of individual word tokens (not yet joined with underscores).
    """
    return [t for t in _RE_WHITESPACE.split(english_raw.strip().lower()) if t]


# ══════════════════════════════════════════════════════════════════════════════
# §4  JSON entry builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_entries(
    persian_tokens: list[str],
    english_tokens: list[str],
    full_canonical: str,
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Derive trans_dict and compound_dict entries for one brand.

    Rules
    ─────
    Single-token brand (both sides have exactly 1 token):
        • Add persian_token  → [canonical, "brand"]   to trans_dict
        • Add english_token  → [canonical, "brand"]   to trans_dict

    Multi-token brand (≥2 tokens on either side):
        • If Persian count == English count:
              add aligned 1:1 individual token pairs to trans_dict
        • Always add the full compound key (comma-joined Persian tokens)
              → [full_canonical, "brand"]             to compound_dict
        • Add Latin compound key (comma-joined English tokens)
              → [full_canonical, "brand"]             to compound_dict

    Latin-only display names (no Persian script at all):
        • Treat as Latin synonym; add to trans_dict with english canonical.

    Returns (new_trans_entries, new_compound_entries).
    """
    trans: dict[str, list] = {}
    compound: dict[str, list] = {}

    n_persian = len(persian_tokens)
    n_english = len(english_tokens)

    if n_persian == 1 and n_english == 1:
        # Simple single-token case
        p_tok = persian_tokens[0]
        e_tok = english_tokens[0]
        trans[p_tok] = [full_canonical, "brand"]
        if _is_latin_only(p_tok):
            # Persian display is already Latin (e.g. "121")
            trans[p_tok.lower()] = [full_canonical, "brand"]
        else:
            trans[e_tok] = [full_canonical, "brand"]

    elif n_persian >= 2 or n_english >= 2:
        # Multi-token compound brand
        if n_persian == n_english:
            # Aligned 1:1 — add individual token entries
            for p_tok, e_tok in zip(persian_tokens, english_tokens):
                word_canonical = _safe_canonical(e_tok)
                trans[p_tok] = [word_canonical, "brand"]
                if _is_latin_only(p_tok):
                    trans[p_tok.lower()] = [word_canonical, "brand"]
                else:
                    trans[e_tok.lower()] = [word_canonical, "brand"]
        else:
            # Counts don't align — still add each Persian token mapped to full canonical
            for p_tok in persian_tokens:
                trans[p_tok] = [full_canonical, "brand"]
            for e_tok in english_tokens:
                trans[e_tok.lower()] = [full_canonical, "brand"]

        # Always add compound entries for multi-word brands
        persian_key = ",".join(persian_tokens)
        compound[persian_key] = [full_canonical, "brand"]

        latin_key = ",".join(english_tokens)
        if latin_key != persian_key:  # skip if identical (Latin-only display names)
            compound[latin_key] = [full_canonical, "brand"]

    return trans, compound


# ══════════════════════════════════════════════════════════════════════════════
# §5  Main build routine
# ══════════════════════════════════════════════════════════════════════════════

def build(input_path: Path, output_path: Path) -> None:
    all_trans: dict[str, list] = {}
    all_compound: dict[str, list] = {}

    skipped = 0
    processed = 0

    with input_path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            parsed = _parse_line(raw_line)
            if parsed is None:
                skipped += 1
                continue

            persian_display, english_raw = parsed
            persian_tokens = _tokenise_persian_display(persian_display)
            english_tokens = _tokenise_english(english_raw)
            full_canonical = _safe_canonical(english_raw)

            if not persian_tokens or not english_tokens or not full_canonical:
                print(f"  [WARN] line {lineno}: skipping — empty tokens: {raw_line!r}",
                      file=sys.stderr)
                skipped += 1
                continue

            new_trans, new_compound = _build_entries(
                persian_tokens, english_tokens, full_canonical
            )

            # Merge without overwriting; first occurrence wins (preserves ordering)
            for k, v in new_trans.items():
                all_trans.setdefault(k, v)
            for k, v in new_compound.items():
                all_compound.setdefault(k, v)

            processed += 1

    payload = {
        "trans_dict": all_trans,
        "compound_dict": all_compound,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(
        f"Done: {processed} brands processed, {skipped} lines skipped.\n"
        f"  trans_dict entries  : {len(all_trans)}\n"
        f"  compound_dict entries: {len(all_compound)}\n"
        f"  Output              : {output_path}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# §6  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build brands.json from brands.txt for use in similarity_score.py"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="brands.txt",
        help="Path to brands.txt  (default: brands.txt)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="brands.json",
        help="Path to output brands.json  (default: brands.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    build(input_path, output_path)


if __name__ == "__main__":
    _cli()