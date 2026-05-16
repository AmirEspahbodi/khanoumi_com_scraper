from __future__ import annotations

import logging
import random
from typing import Any

from playwright.async_api import Locator

from src.browser_manager import BrowserManager
from src.similarity_score_gpt5 import similarity_score as similarity_score_gpt5
from src.similarity_score import similarity_score as similarity_score_claude
from src.config import CFG, get_search_queries
from src.helpers import (
    detect_bot_challenge,
    human_delay,
    human_write,
    mouse_jitter,
    scroll_to_element,
)
from src.producer_store import append_entry, load_producer_json

logger = logging.getLogger("producer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def get_product_name(card_locator: Locator) -> str:
    """
    Extract the product name from a product card locator.

    The name is the text content of the first <h3> element inside the card.
    Returns an empty string on any failure.
    """
    try:
        h3 = card_locator.locator("h3").first
        text = await h3.inner_text()
        return text.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


async def producer(manager: BrowserManager) -> int:
    """
    Navigate search-results pages for every query loaded from Book1.xlsx,
    score each product card, and persist new entries to producer.json.

    Resume support: entries already present in producer.json are skipped.

    Returns the total number of NEW entries written this session.
    """
    website_url: str = CFG.base_url

    # ── Load queries ─────────────────────────────────────────────────────
    search_queries = get_search_queries()
    if not search_queries:
        logger.error("Producer: no search queries loaded — aborting.")
        return 0

    logger.info(
        "Producer starting — base URL: %s | %d queries loaded.",
        website_url,
        len(search_queries),
    )

    # ── Load existing producer.json for deduplication ────────────────────
    existing_entries: list[dict[str, Any]] = await load_producer_json()
    existing_urls: set[str] = {
        entry["product_url"] for entry in existing_entries if "product_url" in entry
    }
    logger.info(
        "Producer: loaded %d existing entries from producer.json.",
        len(existing_urls),
    )

    # In-session seen set (across all queries)
    seen_urls: set[str] = set(existing_urls)
    new_entries_written: int = 0

    for query in search_queries:
        logger.info("Starting search for query: '%s'", query)
        page_num = 1
        is_product_found = False

        async with manager.new_page() as page:
            page.set_default_timeout(CFG.navigation_timeout)

            # ── Navigate to site and submit search ───────────────────────
            await page.goto(website_url, wait_until="domcontentloaded")
            await detect_bot_challenge(page)
            await human_delay(1.0, 2.0)

            await human_write(page, CFG.search_input_xpath, query)
            search_locator = page.locator(CFG.search_input_xpath).first
            await search_locator.press("Enter")
            await human_delay(1.0, 2.0)

            # Click "Page 1" button if present (some result pages show it)
            page1 = page.locator("//a[@aria-label='Page 1']")
            if await page1.count() != 0:
                await page1.click()
                await human_delay(2.0, 3.0)

            await page.evaluate("document.body.style.zoom = '25%'")

            # ── Pagination loop ──────────────────────────────────────────
            while True:
                await human_delay(4.0, 5.0)
                logger.info(
                    "Producer scraping page %d for query '%s' …", page_num, query
                )

                # Empty-results guard
                check_empty = page.locator(CFG.check_no_product_container_xpath)
                if await check_empty.count() > 0:
                    logger.info(
                        "Producer: no products found in container for query '%s'.",
                        query,
                    )
                    break

                # Collect all product card locators on this page
                products = await page.locator(CFG.each_product_xpath).all()
                logger.info(
                    "Page %d — found %d product cards.", page_num, len(products)
                )

                if not products:
                    logger.info(
                        "Producer: no product cards on page %d for query '%s'.",
                        page_num,
                        query,
                    )
                    break

                page_new = 0
                for card in products:
                    # Extract href from the <a> tag inside the card
                    link_locator = card.locator("a")
                    href = await link_locator.get_attribute("href")
                    if not href:
                        continue

                    # Normalise to absolute URL
                    if href.startswith("http"):
                        product_url = href
                    else:
                        product_url = f"{CFG.base_url}{href.lstrip('/')}"

                    # Dedup against all seen URLs (existing + this session)
                    if product_url in seen_urls:
                        continue
                    seen_urls.add(product_url)

                    # Extract product name and compute similarity score
                    name = await get_product_name(card)
                    score1 = similarity_score_claude(query, name)
                    score2 = similarity_score_gpt5(query, name)

                    # Only persist entries with a positive score
                    if score1 < 0.80 and score2 < 0.80:
                        logger.info(
                            "Skipping low-score product (%.4f), (%.4f): query = %s _ name = %s", score1, score2, query, name
                        )
                        continue

                    entry: dict[str, Any] = {
                        "search_query": query,
                        "product_url": product_url,
                        "similarity_score1": score1,
                        "similarity_score2": score2,
                        "is_scraped": False,
                        "scrap_directory": "",
                    }

                    # Real-time persistence: write after every single product
                    await append_entry(entry)
                    is_product_found=True
                    new_entries_written += 1
                    page_new += 1

                    logger.debug(
                        "Appended entry #%d — score=%.4f,%.4f url=%s",
                        new_entries_written,
                        score1,
                        score2,
                        product_url,
                    )
                    if is_product_found:
                        break
                if is_product_found:
                    break

                logger.info(
                    "  Page %d → %d new entries written (running total: %d).",
                    page_num,
                    page_new,
                    new_entries_written,
                )

                # ── Next page ────────────────────────────────────────────
                next_btn = page.locator(CFG.sel_next_page)
                if await next_btn.count() == 0:
                    logger.info("Producer: no more pages for query '%s'.", query)
                    break

                is_disabled = await next_btn.get_attribute("aria-disabled")
                if is_disabled == "true":
                    logger.info(
                        "Producer: next-page button disabled for query '%s'.", query
                    )
                    break

                await scroll_to_element(page, CFG.sel_next_page)
                await mouse_jitter(page)
                await human_delay()
                await next_btn.click()
                await page.wait_for_load_state("domcontentloaded")
                await detect_bot_challenge(page)
                page_num += 1
    if is_product_found:
        pass
    else:
        pass

    logger.info(
        "Producer finished — %d new entries written to producer.json.",
        new_entries_written,
    )
    return new_entries_written
