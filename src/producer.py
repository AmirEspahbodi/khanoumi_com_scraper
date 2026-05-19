from __future__ import annotations

import logging
import random
from cmath import log
from pathlib import Path  # Added for state tracking
from typing import Any

from playwright.async_api import Locator

from src.browser_manager import BrowserManager
from src.config import CFG, get_search_data
from src.helpers import (
    detect_bot_challenge,
    human_delay,
    human_write,
    mouse_jitter,
    scroll_to_element,
)
from src.producer_store import append_entry, load_producer_json
from src.similarity_score_claude import similarity_score as similarity_score_claude
from src.similarity_score_gemini import similarity_score as similarity_score_gemini
from src.similarity_score_gemini2 import similarity_score as similarity_score_gemini2

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
    Row resume support: successfully completed rows are tracked in producer_state.txt.

    Returns the total number of NEW entries written this session.
    """
    website_url: str = CFG.base_url

    # ── Load queries ─────────────────────────────────────────────────────
    search_queries: tuple[tuple[str, ...]] = get_search_data()
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

    # ── Load State Tracker for Excel Rows ────────────────────────────────
    state_file = Path("producer_state.txt")
    processed_rows = set()
    if state_file.exists():
        processed_rows = set(state_file.read_text(encoding="utf-8").splitlines())
        logger.info(
            "Producer: loaded %d completed rows from state file. Resuming where left off...",
            len(processed_rows),
        )

    # In-session seen set (across all queries)
    seen_urls: set[str] = set(existing_urls)
    new_entries_written: int = 0

    for data in search_queries:
        original_product_names = data[0]
        query = data[1]
        normalized_product_names = data[2:]

        # Create a unique identifier for this row to track completion
        row_id = f"{original_product_names}::{query}".replace("\n", " ").strip()

        # Skip if we already finished this row in a previous run
        if row_id in processed_rows:
            logger.info("Skipping already processed row: '%s'", original_product_names)
            continue

        print("--" * 50)
        logger.info(f"original_product_names = {original_product_names}")
        logger.info(f"query = {query}")
        logger.info(f"normalized_product_names = {normalized_product_names}")
        logger.info("Starting search for query: '%s'", query)

        page_num = 1
        candidate_product_entities: list[dict] = []

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
                await human_delay(9.0, 10.0)
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

                page_new_entity = 0

                for card in products:
                    # Extract href from the <a> tag inside the card
                    link_locator = card.locator("a")
                    href = await link_locator.get_attribute("href")
                    if not href:
                        logger.info("url of product didn't found!")
                        logger.debug(
                            f"{'---' * 50}\nhtml content of this card is: \n{await card.inner_html()}\n {'---' * 50}\n"
                        )
                        continue

                    # Normalise to absolute URL
                    if href.startswith("http"):
                        product_url = href
                    else:
                        product_url = f"{CFG.base_url}{href.lstrip('/')}"

                    name = await get_product_name(card)
                    # Extract product name and compute similarity score
                    temp_avg_score = []
                    avg_score = 0
                    max_score = 0
                    for temp_name in [
                        original_product_names,
                        *normalized_product_names,
                    ]:
                        score_gemini_2 = similarity_score_gemini2(name, temp_name)
                        score_gemini_1 = similarity_score_gemini(name, temp_name)
                        score_claude_1 = similarity_score_claude(name, temp_name)

                        if (
                            score_gemini_1 > 0.60
                            or score_gemini_2 > 0.60
                            or score_claude_1 > 0.60
                        ):
                            temp_avg_score.append(
                                (score_gemini_2 + score_gemini_1 + score_claude_1) / 3
                            )
                            max_score = max(
                                max_score,
                                score_gemini_2,
                                score_gemini_1,
                                score_claude_1,
                            )
                    if temp_avg_score:
                        avg_score = max(temp_avg_score)
                    # Only persist entries with a positive score
                    if avg_score > 0.70:
                        candidate_product_entities.append(
                            {
                                "excel_product_name": original_product_names,
                                "search_query": query,
                                "website_product_name": name,
                                "product_url": product_url,
                                "avg_similarity_score": avg_score,
                                "max_similarity_score": max_score,
                                "is_scraped": False,
                                "scrap_directory": "",
                            }
                        )
                    elif max_score > 0.70:
                        candidate_product_entities.append(
                            {
                                "excel_product_name": original_product_names,
                                "search_query": query,
                                "website_product_name": name,
                                "product_url": product_url,
                                "avg_similarity_score": avg_score,
                                "max_similarity_score": max_score,
                                "is_scraped": False,
                                "scrap_directory": "",
                            }
                        )
                    else:
                        logger.info(
                            "Skipping low-score product, avg-score = (%.4f), max-score = (%.4f): query = %s _ name = %s",
                            avg_score,
                            max_score,
                            original_product_names,
                            name,
                        )
                        continue
                    page_new_entity += 1

                    logger.info(
                        "Appended entry — avg score=%.4f, max score = %.4f, url=%s",
                        avg_score,
                        max_score,
                        product_url,
                    )

                logger.info(
                    "  Page %d → %d new entries appended to list.",
                    page_num,
                    page_new_entity,
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

                if page_num > 30:
                    break

            best_avg_entity = max(
                candidate_product_entities,
                key=lambda entity: entity.get("avg_similarity_score", float("-inf")),
                default=None,
            )
            best_max_entity = max(
                candidate_product_entities,
                key=lambda entity: entity.get("max_similarity_score", float("-inf")),
                default=None,
            )

            if best_avg_entity and best_avg_entity["avg_similarity_score"] >= 0.70:
                await append_entry(best_avg_entity)

                logger.info(
                    "an entries written for product -- %s -- by avg score.",
                    original_product_names,
                )
                logger.info(
                    f"Found best match: {best_avg_entity['product_url']} with avg score {best_avg_entity['avg_similarity_score']}"
                )
                new_entries_written += 1

            if best_max_entity and best_max_entity["max_similarity_score"] >= 0.7:
                if not (
                    best_avg_entity
                    and best_avg_entity["product_url"] == best_max_entity["product_url"]
                ):
                    await append_entry(best_max_entity)

                    logger.info(
                        "an entries written for product -- %s -- by max score.",
                        original_product_names,
                    )
                    logger.info(
                        f"Found best match: {best_max_entity['product_url']} withmax  score {best_max_entity['avg_similarity_score']}"
                    )
                    new_entries_written += 1

            if not best_max_entity and not best_avg_entity:
                logger.info("No candidate entities found.")

        # ── MARK ROW AS DONE ──────────────────────────────────────────────
        # We append to state tracking immediately after closing the page for this query.
        # This prevents duplicate work if the script shuts down at any point.
        with state_file.open("a", encoding="utf-8") as sf:
            sf.write(row_id + "\n")
        processed_rows.add(row_id)
        logger.info("Row marked as fully completed: '%s'", original_product_names)

    logger.info(
        "Producer finished — %d new entries written to producer.json.",
        new_entries_written,
    )
    return new_entries_written
