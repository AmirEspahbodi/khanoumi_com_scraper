from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse

from PIL import Image
from playwright.async_api import Page

from src.config import CFG
from src.data import ProductData

logger = logging.getLogger("result_manager")

# Characters forbidden in Windows path components, plus ASCII control chars
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_HYPHEN = re.compile(r"-{2,}")

# PNG compression level: 0 = no compression (largest file, original quality preserved).
# PNG is always lossless; level only affects file size vs. encode speed.
_PNG_COMPRESS_LEVEL = 0


def _sanitize_folder_name(name: str) -> str:
    """
    Produce a Windows-safe directory name component from *name*.

    - Replaces forbidden characters (< > : " / \\ | ? *) and ASCII control
      characters (0x00–0x1F) with a hyphen.
    - Collapses runs of hyphens into a single hyphen.
    - Strips leading/trailing whitespace and dots.
    - Truncates to 120 characters.
    """
    result = _FORBIDDEN.sub("-", name)
    result = _MULTI_HYPHEN.sub("-", result)
    result = result.strip(" .")
    return result[:120]


def _extract_product_slug(url: str) -> str:
    """
    Return the last non-empty path segment of *url*, URL-decoded.
    Falls back to 'unknown-product' if no segment can be found.
    """
    try:
        parsed = urlparse(url)
        segments = [s for s in parsed.path.split("/") if s]
        if not segments:
            return "unknown-product"
        return unquote(segments[-1])
    except Exception:
        return "unknown-product"


def _new_uuid() -> str:
    """Return a new random UUID4 string (e.g. '3d6f4e2a-…')."""
    return str(uuid.uuid4())


def _to_png_bytes(raw: bytes) -> bytes:
    """
    Convert *raw* image bytes (any Pillow-supported format) to PNG bytes at
    maximum quality (compress_level=0, lossless).

    Colour-mode handling:
      • Palette (P) / Palette+Transparency (PA) → RGBA  (preserves transparency)
      • All other modes with alpha (e.g. LA, RGBA)  → RGBA
      • Everything else                              → RGB

    Returns the original *raw* bytes unchanged if Pillow cannot open them,
    so the caller always gets *something* to write to disk.
    """
    try:
        img = Image.open(io.BytesIO(raw))

        # Normalise palette images first so conversions below are clean.
        if img.mode in ("P", "PA"):
            img = img.convert("RGBA")

        # Keep alpha channel when present; otherwise use plain RGB.
        if img.mode in ("RGBA", "LA"):
            target_mode = "RGBA"
        else:
            target_mode = "RGB"

        if img.mode != target_mode:
            img = img.convert(target_mode)

        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
        return buf.getvalue()
    except Exception as exc:
        logger.warning(
            "_to_png_bytes: Pillow conversion failed (%s) — saving raw bytes.", exc
        )
        return raw


class ResultManager:
    """
    Owns all result I/O: image downloads, product JSON persistence, and
    cross-run URL deduplication. Nothing outside this class performs raw
    file I/O for results.
    """

    def __init__(self) -> None:
        # Create the top-level result directory once at construction time.
        Path("result").mkdir(parents=True, exist_ok=True)

        self._scraped_urls: set[str] = set()
        self._url_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # URL deduplication
    # ------------------------------------------------------------------

    async def load_scraped_urls(self) -> None:
        """
        Populate self._scraped_urls from all .json files already on disk.
        Corrupt / incomplete files are skipped with a WARNING.
        """
        urls: set[str] = set()

        # Check all .json files to ensure we deduplicate failed-product files too
        for metadata_path in Path("result").rglob("*.json"):
            try:
                text = await asyncio.to_thread(
                    metadata_path.read_text, encoding="utf-8"
                )
                data = json.loads(text)
                url = data.get("url")
                if url:
                    urls.add(url)
            except json.JSONDecodeError:
                logger.warning(
                    "load_scraped_urls: malformed JSON in %s — skipping.", metadata_path
                )
            except Exception as exc:
                logger.warning(
                    "load_scraped_urls: could not read %s: %s — skipping.",
                    metadata_path,
                    exc,
                )

        self._scraped_urls = urls
        logger.info("load_scraped_urls: found %d previously scraped URLs.", len(urls))

    async def is_scraped(self, url: str) -> bool:
        """Return True if *url* has already been scraped (and saved to disk)."""
        async with self._url_lock:
            return url in self._scraped_urls

    async def mark_scraped(self, url: str) -> None:
        """Mark *url* as scraped so subsequent checks skip it."""
        async with self._url_lock:
            self._scraped_urls.add(url)

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------

    async def download_images(self, product: ProductData, page: Page) -> None:
        """
        Download all images referenced by *product* using the page's
        authenticated request context.

        Each image is:
          • Fetched from the network.
          • Converted to PNG at maximum (lossless) quality via Pillow.
          • Saved as ``<uuid4>.png`` inside the product directory.

        The UUID (without extension) is appended to ``product.image_ids``.

        Images are saved to: result/<product_uuid>/
        """
        # --- DO NOT download images for failed products ---
        if product.is_failed:
            logger.debug(
                "Skipping image download for failed product: %s",
                product.url,
            )
            return

        # Page-closed guard
        if page.is_closed():
            logger.warning(
                "Page closed before image download for %s — skipping images.",
                product.url,
            )
            return

        # Assign product_id here so the directory is known before saving images.
        if not product.product_id:
            product.product_id = _new_uuid()

        # Flat layout: result/<product_uuid>/
        product_dir = Path("result") / product.product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        # Build deduplicated, ordered list of image URLs (main first, then thumbnails)
        urls: list[str] = []
        seen: set[str] = set()
        candidates = [item for _, value in product.image_urls.items() for item in value]

        for u in candidates:
            if u and u not in seen:
                urls.append(u)
                seen.add(u)

        # Nothing to download
        if not urls:
            return

        for image_url in urls:
            # Relative URL guard
            if not image_url.startswith("http"):
                image_url = urljoin(CFG.base_url.rstrip("/"), image_url)

            max_retries = 3
            success = False
            last_exc: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    response = await page.request.get(image_url, timeout=30000)

                    if not response.ok:
                        last_exc = RuntimeError(f"HTTP {response.status}")
                        await asyncio.sleep(2.0)
                        continue

                    body = await response.body()

                    if len(body) == 0:
                        last_exc = RuntimeError("Empty body returned")
                        await asyncio.sleep(2.0)
                        continue

                    # ── Convert to PNG (lossless, max quality) ───────────
                    png_bytes = await asyncio.to_thread(_to_png_bytes, body)

                    # ── Assign a UUID filename ────────────────────────────
                    image_id = _new_uuid()
                    file_path = product_dir / f"{image_id}.png"

                    await asyncio.to_thread(file_path.write_bytes, png_bytes)

                    product.image_ids.append(image_id)

                    success = True
                    break

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0)

            if not success:
                logger.warning(
                    "Image download error for %s after %d attempts: %s — skipping.",
                    image_url,
                    max_retries,
                    last_exc,
                )

    # ------------------------------------------------------------------
    # Product JSON storage
    # ------------------------------------------------------------------

    async def save_product(self, product: ProductData) -> None:
        """
        Assign a UUID product_id (if not already set), serialise *product* to
        JSON, and write it atomically to disk.

        Failed products  → ``result/failed-product/<uuid4>.json``
        Success products → ``result/<product_uuid>/metadata.json``

        Also marks the product's URL as scraped.
        """
        if product.is_failed:
            # ── Failed Product Routing ───────────────────────────────────
            failed_dir = Path("result") / "failed-product"
            failed_dir.mkdir(parents=True, exist_ok=True)

            fail_id = _new_uuid()
            target_path = failed_dir / f"{fail_id}.json"
        else:
            # ── Success Product Routing ──────────────────────────────────
            # Flat layout: result/<product_uuid>/metadata.json
            # Only assign product_id if download_images hasn't done so already.
            if not product.product_id:
                product.product_id = _new_uuid()

            product_dir = Path("result") / product.product_id
            product_dir.mkdir(parents=True, exist_ok=True)

            target_path = product_dir / "metadata.json"

        json_str = json.dumps(dataclasses.asdict(product), ensure_ascii=False, indent=2)

        # Atomic write: write to .tmp then rename
        tmp_path = target_path.with_suffix(".tmp")
        await asyncio.to_thread(tmp_path.write_text, json_str, encoding="utf-8")
        await asyncio.to_thread(tmp_path.rename, target_path)

        status_flag = "failed" if product.is_failed else "successfully"
        logger.info(
            "Saved %s product: %s → %s",
            status_flag,
            product.product_id or target_path.stem,
            target_path,
        )

        # Mark URL as scraped (belt-and-suspenders with producer-side check)
        await self.mark_scraped(product.url)
