from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Optional

from playwright.async_api import Error as PWError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PWTimeoutError

from src.browser_manager import BrowserManager
from src.config import CFG
from src.data import ProductData
from src.errors import BotChallengeDetected
from src.helpers import (
    detect_bot_challenge,
    human_delay,
    mouse_jitter,
)
from src.result_manager import ResultManager

logger = logging.getLogger("scrape_product")

_INTRO_BTN_XPATH = "//html/body/section/main/div/div[1]/div[4]/div[1]/button"
_INTRO_PANEL_XPATH = "//html/body/section/main/div/div[1]/div[4]/div[1]/div/div"

# HeadlessUI generates :rN: IDs deterministically per render but they can
# shift if the component tree changes, so we keep a text-based fallback.
_USAGE_BTN_ID = "headlessui-disclosure-button-:r6:"
_USAGE_PANEL_ID = "headlessui-disclosure-panel-:r7:"


def _parse_price(raw: str) -> Optional[int]:
    """Strip Persian/Arabic numerals, commas, spaces → int.  Returns None if empty."""
    # Normalise Eastern-Arabic digits to ASCII
    normalized = raw.translate(
        str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    )
    digits = re.sub(r"[^\d]", "", normalized)
    return int(digits) if digits else None


def _upscale_img_url(src: str, width: int = 800) -> str:
    """Replace the ?w=NNN thumbnail param with a higher-resolution value."""
    return re.sub(r"(\?w=)\d+", rf"\g<1>{width}", src)


def _clean_block(raw: str) -> str:
    """
    Normalise a single h2/h3/p text block:
      • \xa0 (non-breaking space) → regular space
      • bullet chars (·  •  ·) stripped
      • internal \n / \r / \t collapsed to single space
      • runs of spaces collapsed
      • leading/trailing whitespace stripped
    """
    text = raw.replace("\xa0", " ")
    text = re.sub(r"[·•·]\s*", "", text)  # remove bullet chars
    text = re.sub(r"[\n\r\t]+", " ", text)  # collapse line breaks
    text = re.sub(r" {2,}", " ", text)  # collapse runs of spaces
    return text.strip()


async def _extract_text_blocks(container) -> list[str]:
    """
    Walk every <h2>, <h3>, <p> inside *container* in DOM order,
    clean each one, and return non-empty results as a list[str].
    """
    blocks: list[str] = []
    for el in await container.locator("h2, h3, p").all():
        cleaned = _clean_block(await el.inner_text())
        if cleaned:  # drop blank / whitespace-only tags
            blocks.append(cleaned)
    return blocks


async def _safe_click(loc, timeout: int = 4_000) -> bool:
    """Click *loc* if it exists; return True on success, False otherwise."""
    try:
        await loc.wait_for(state="visible", timeout=timeout)
        await loc.click(timeout=timeout)
        return True
    except (PWTimeoutError, PWError):
        return False


async def scrape_product(page: Page, url: str) -> ProductData:
    """
    Navigate to a single product page and extract all structured data.
    All logic is wrapped in a single try-except block to gracefully handle
    infrastructure crashes, timeouts, and parsing anomalies.
    """
    product = ProductData(url=url)

    try:
        await page.goto(url, wait_until="domcontentloaded")

        # ── Pre-existing anti-bot routines ────────────────────────────────────
        await detect_bot_challenge(page)
        await human_delay(0.5, 1.2)
        await mouse_jitter(page, steps=4)

        # ── Anchor on the PDP container ───────────────────────────────────────
        root = page.locator("//html/body/section/main/div/div[1]/div[2]")
        await root.wait_for(state="visible", timeout=15_000)

        # ── 1. Names ──────────────────────────────────────────────────────────
        h1 = root.locator("h1").first
        if await h1.count():
            product.name_fa = (await h1.inner_text()).strip()

        # The English name is in a <span> that is hidden on mobile / shown on lg.
        # It sits directly below h1 inside the same mb-3 flex column.
        # Selector targets the specific Tailwind combo used in the source.
        en_span = root.locator("div.mb-3 span.text-text-mediumGray.font-medium").first
        if await en_span.count():
            product.name_en = (await en_span.inner_text()).strip()

        # ── 2. Social-proof bar ───────────────────────────────────────────────
        # Rating  →  "4.6 امتیاز"
        star_span = root.locator('[data-sentry-component="StarRate"] span').first
        if await star_span.count():
            m = re.search(r"[\d.]+", await star_span.inner_text())
            if m:
                product.rate = float(m.group())

        # Review count  →  "70 دیدگاه"
        comment_span = root.locator(
            '[data-sentry-component="CommentsCounter"] span'
        ).first
        if await comment_span.count():
            m = re.search(r"\d+", await comment_span.inner_text())
            if m:
                product.review_count = int(m.group())

        # Wishlist / favourites  →  "334 علاقه‌مندی"
        wishlist_span = root.locator(
            '[data-sentry-component="WishlistCount"] span'
        ).first
        if await wishlist_span.count():
            m = re.search(r"\d+", await wishlist_span.inner_text())
            if m:
                product.wishlist_count = int(m.group())

        # ── 3. Availability ───────────────────────────────────────────────────
        buy_box = root.locator('[data-test="pdp-buy-box"]').first
        if await buy_box.count():
            buy_box_text = await buy_box.inner_text()
            product.is_out_of_stuck = "این محصول ناموجود است" not in buy_box_text

        # ── 4. Pricing ────────────────────────────────────────────────────────
        # Struck-through original price (only present when discounted)
        original_el = root.locator("span.line-through").first
        if await original_el.count():
            product.real_price = _parse_price(await original_el.inner_text())
            product.has_discount = True

        # Discount badge  →  "50٪"  (bg-alert-error-dark pill)
        discount_el = root.locator("span.bg-alert-error-dark").first
        if await discount_el.count():
            m = re.search(r"\d+", await discount_el.inner_text())
            if m:
                product.discount_percentage = int(m.group())

        # Final / actual price  — the <span data-test="pdp-price"> inside
        # the pricing column (not the wrapper <div> that shares the attr).
        final_el = root.locator("span[data-test='pdp-price']").first
        if await final_el.count():
            product.discounted_price = _parse_price(await final_el.inner_text())
            # No discount → original == final
            if not product.has_discount:
                product.real_price = product.discounted_price

        # ── 5. Images (with per-colour variant support) ───────────────────────

        async def _collect_gallery() -> list[str]:
            """Snapshot all <figure> image srcs currently rendered in the slider."""
            imgs = page.locator('figure[aria-label="open-gallery-modal"] img')
            urls: list[str] = []
            for img in await imgs.all():
                src = await img.get_attribute("src")
                if src and src not in urls:
                    urls.append(_upscale_img_url(src))
            return urls

        color_buttons = root.locator('[data-sentry-component="ColorVariant"]')
        color_count = await color_buttons.count()

        if color_count == 0:
            # Single-colour or no variant product — just grab whatever is rendered.
            product.image_urls["default"] = await _collect_gallery()

        else:
            for i in range(color_count):
                btn = color_buttons.nth(i)

                # The colour label lives in aria-label  →  e.g. "T06"
                colour_label = (
                    await btn.get_attribute("aria-label") or f"color_{i}"
                ).strip()

                # Record the first image src *before* clicking so we can detect
                # when the gallery has actually refreshed.
                first_img = page.locator(
                    'figure[aria-label="open-gallery-modal"] img'
                ).first
                src_before = (
                    await first_img.get_attribute("src")
                    if await first_img.count()
                    else ""
                )

                await btn.click()

                # Wait until the gallery image swaps out (proves a re-render happened).
                try:
                    await page.wait_for_function(
                        """(before) => {
                            const img = document.querySelector(
                                'figure[aria-label="open-gallery-modal"] img'
                            );
                            return img && img.src !== before;
                        }""",
                        arg=src_before,
                        timeout=4_000,
                    )
                except PWTimeoutError:
                    # Gallery didn't change (maybe this colour is already selected
                    # or images are the same). Still harvest what's there.
                    pass

                await human_delay(0.2, 0.5)  # brief settle for lazy-loaded siblings
                product.image_urls[colour_label] = await _collect_gallery()

        # ── 6.  Expand  "معرفی محصول"  ──────────────────────────────────────
        intro_btn = page.locator(f"xpath={_INTRO_BTN_XPATH}")
        clicked_intro = await _safe_click(intro_btn)
        if clicked_intro:
            await human_delay(0.3, 0.6)

        # ── 7.  Expand  "نحوه مصرف"  ────────────────────────────────────────
        # Primary: stable HeadlessUI ID supplied by the site.
        usage_btn = page.locator(f'xpath=//*[@id="{_USAGE_BTN_ID}"]')
        if not await usage_btn.count():
            # Fallback: find any <button> whose visible text contains the label.
            # This survives ID shifts when component order changes.
            usage_btn = page.get_by_role("button", name=re.compile("نحوه مصرف"))

        clicked_usage = await _safe_click(usage_btn)
        if clicked_usage:
            await human_delay(0.3, 0.6)

        # ── 8.  Zoom to 25 % — forces lazy content into viewport ─────────────
        await page.evaluate("document.body.style.zoom = '0.25'")
        await page.wait_for_timeout(random.randint(2_000, 3_000))

        # ── 9.  Extract  "معرفی محصول"  ──────────────────────────────────────
        intro_panel = page.locator(f"xpath={_INTRO_PANEL_XPATH}")
        if await intro_panel.count():
            product.product_intro = await _extract_text_blocks(intro_panel)

        # ── 10. Extract  "نحوه مصرف"  ─────────────────────────────────────────
        # Primary: stable panel ID.
        usage_panel = page.locator(f'xpath=//*[@id="{_USAGE_PANEL_ID}"]')
        if not await usage_panel.count():
            # Fallback: grab the last disclosure panel currently marked as open.
            usage_panel = page.locator('[data-headlessui-state~="open"]').last

        if await usage_panel.count():
            product.usage_instructions = await _extract_text_blocks(usage_panel)

        # ── 11. Restore zoom so the page is normal for any further actions ────
        await page.evaluate("document.body.style.zoom = '1'")

    except BotChallengeDetected:
        # 1. Custom Exception: Propagate up so the worker applies backoff.
        raise
    except PWTimeoutError:
        # 2. Playwright Timeouts: Navigation took too long, or critical element absent.
        raise
    except PWError:
        # 3. Core Playwright Errors: Catches `TargetClosedError` and DOM detachments.
        raise
    except Exception as e:
        # 4. Catch-all for Python logic (IndexError, TypeError) or unexpected DOM shifts.
        logger.error(f"Unexpected parsing logic error on {url}", exc_info=True)
        product.error = f"{type(e).__name__}: {str(e)}"

    return product


async def worker(
    worker_id: int,
    queue: asyncio.Queue[Optional[tuple[str, str]]],
    results: list[ProductData],
    manager: BrowserManager,
    result_manager: ResultManager,
) -> None:
    """
    Consumer coroutine.
    Pulls URLs from the queue, scrapes each with retries + exponential back-off,
    and appends results to the shared list.
    """
    log = logging.getLogger(f"worker-{worker_id:02d}")
    log.info("Started.")

    while True:
        item = await queue.get()

        # Sentinel → this worker is done
        if item is None:
            log.info("Received shutdown sentinel. Exiting.")
            queue.task_done()
            break

        url, query_name = item

        product: Optional[ProductData] = None
        last_exc: Optional[Exception] = None

        for attempt in range(1, CFG.max_retries + 1):
            try:
                log.debug("Attempt %d/%d — %s", attempt, CFG.max_retries, url)

                async with manager.new_page() as page:
                    page.set_default_timeout(CFG.navigation_timeout)
                    product = await scrape_product(page, url)
                    product.query_name = query_name

                    # Download images while the page (and its request context) is still open
                    await result_manager.download_images(product, page)

                # Tab is auto-closed by the context manager; save product after close
                await result_manager.save_product(product)

                log.info(
                    "OK  %-60s | %-40s | %s",
                    url[-60:],
                    product.name_en[:40] if product.name_en else "N/A",
                    product.real_price if product.real_price else "N/A",
                )
                break  # success — exit retry loop

            except BotChallengeDetected as exc:
                last_exc = exc
                log.warning(
                    "Bot challenge on %s — attempt %d/%d", url, attempt, CFG.max_retries
                )
                await asyncio.sleep(CFG.backoff_base**attempt + random.uniform(0, 1))

            except PWTimeoutError as exc:
                last_exc = exc
                log.warning(
                    "Timeout on %s — attempt %d/%d", url, attempt, CFG.max_retries
                )
                await asyncio.sleep(CFG.backoff_base**attempt)

            except PWError as exc:
                last_exc = exc
                log.warning(
                    "Playwright rendering/connection error on %s — attempt %d/%d",
                    url,
                    attempt,
                    CFG.max_retries,
                )
                await asyncio.sleep(CFG.backoff_base**attempt)

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Unexpected error on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    CFG.max_retries,
                    exc,
                )
                await asyncio.sleep(CFG.backoff_base**attempt)

        if product is None:
            log.error(
                "FAILED after %d attempts: %s", CFG.max_retries, url, exc_info=last_exc
            )
            product = ProductData(url=url, query_name=query_name, error=str(last_exc))
            await result_manager.save_product(product)

        results.append(product)
        queue.task_done()

    log.info("Worker exiting.")
