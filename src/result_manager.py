from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from playwright.async_api import Page

from src.config import CFG
from src.data import ProductData

logger = logging.getLogger("result_manager")

# Content-Type → file extension mapping
_CONTENT_TYPE_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

# Characters forbidden in Windows path components, plus ASCII control chars
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_HYPHEN = re.compile(r"-{2,}")


def _sanitize_folder_name(name: str, max_len: int = 50) -> str:
    """
    Produce a Windows-safe directory name component from *name*.

    - Replaces forbidden characters (< > : " / \\ | ? *) and ASCII control
      characters (0x00–0x1F) with a hyphen.
    - Collapses runs of hyphens into a single hyphen.
    - Strips leading/trailing whitespace and dots.
    - Truncates to max_len characters to avoid MAX_PATH issues.
    """
    result = _FORBIDDEN.sub("-", name)
    result = _MULTI_HYPHEN.sub("-", result)
    result = result.strip(" .")
    return result[:max_len]


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


def _make_uid10() -> str:
    """
    Return exactly 10 alphanumeric characters ([A-Za-z0-9]) drawn from
    secrets.token_urlsafe, retrying until enough characters are available.
    """
    chars: list[str] = []
    while len(chars) < 10:
        token = secrets.token_urlsafe(16)
        chars.extend(c for c in token if c.isalnum())
    return "".join(chars[:10])


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

    def __extract_image_filename(self, url: str) -> str:
        """
        Extracts the base filename from either a Next.js encoded URL or a direct image URL.
        """
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)

            if "url" in query_params:
                target_url = query_params["url"][0]
                target_path = urlparse(target_url).path
            else:
                target_path = parsed_url.path

            target_path = unquote(target_path)

            # Use split to safely grab the base name regardless of OS
            base_name = target_path.split("/")[-1]

            # Remove extension to avoid mid-string dots
            base_name_no_ext, _ = os.path.splitext(base_name)

            # Sanitize to prevent illegal chars and restrict length
            safe_base = _sanitize_folder_name(base_name_no_ext, max_len=40)

            # Fallback if empty
            if not safe_base:
                safe_base = "img"

            return f"{safe_base}__"

        except Exception as e:
            logger.warning("Error parsing URL %s: %s", url, e)
            return "img__"

    async def download_images(self, product: ProductData, page: Page) -> None:
        """
        Download all images referenced by *product* using the page's
        authenticated request context. Appends bare image IDs (no extension)
        to product.image_ids on success; logs warnings and continues on failure.
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
            slug = _sanitize_folder_name(_extract_product_slug(product.url))
            uid = _make_uid10()
            product.product_id = f"{slug}__{uid}"

        query_dir = _sanitize_folder_name(product.query_name) or "unknown-query"
        product_dir = Path("result") / query_dir / product.product_id
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

        url_to_id: dict[str, str] = {}

        for image_url in urls:
            # Relative URL guard
            if not image_url.startswith("http"):
                image_url = urljoin(CFG.base_url.rstrip("/"), image_url)

            max_retries = 3
            success = False
            last_exc = None

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

                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";")[0]
                        .strip()
                        .lower()
                    )
                    ext = _CONTENT_TYPE_EXT.get(content_type, "")

                    if not ext:
                        url_path = urlparse(image_url).path
                        suffix = Path(url_path).suffix.lower()
                        if suffix and len(suffix) <= 5:
                            ext = suffix

                    if not ext:
                        ext = ".jpg"

                    # Replaced secrets.token_hex(25) with 8 to prevent MAX_PATH overload
                    image_id = f"{self.__extract_image_filename(image_url)}{secrets.token_hex(8)}"
                    file_path = product_dir / f"{image_id}{ext}"

                    await asyncio.to_thread(file_path.write_bytes, body)

                    url_to_id[image_url] = image_id
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
        Assign a product_id (if not already set), serialise *product* to JSON,
        and write it atomically to disk.
        Failed products are routed to 'result/failed-product/{slug}_{uid10}.json'.
        Success products route to 'result/{query_name}/{product_id}/metadata.json'.
        Also marks the product's URL as scraped.
        """
        slug = _sanitize_folder_name(_extract_product_slug(product.url))
        uid = _make_uid10()

        if product.is_failed:
            # ── Failed Product Routing ───────────────────────────────────
            failed_dir = Path("result") / "failed-product"
            failed_dir.mkdir(parents=True, exist_ok=True)

            target_path = failed_dir / f"{slug}_{uid}.json"
        else:
            # ── Success Product Routing ──────────────────────────────────
            # Only assign product_id if download_images hasn't done so already.
            if not product.product_id:
                product.product_id = f"{slug}__{uid}"

            query_dir = _sanitize_folder_name(product.query_name) or "unknown-query"
            product_dir = Path("result") / query_dir / product.product_id
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
            product.product_id or f"{slug}_{uid}",
            target_path,
        )

        # Mark URL as scraped (belt-and-suspenders with producer-side check)
        await self.mark_scraped(product.url)
