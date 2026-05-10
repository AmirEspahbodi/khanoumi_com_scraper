from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from src.browser_manager import BrowserManager
from src.config import CFG
from src.helpers import (
    detect_bot_challenge,
    human_delay,
    human_write,
    mouse_jitter,
    scroll_to_element,
)
from src.result_manager import ResultManager

logger = logging.getLogger("producer")


async def producer(
    queue: asyncio.Queue[Optional[tuple[str, str]]],
    manager: BrowserManager,
    result_manager: ResultManager,
) -> int:
    """
    Navigate the search results pages for multiple queries sequentially.
    Collects all product URLs, deduplicates them (both in-session and against
    previously scraped runs), and pushes them into the queue.

    Returns the total number of unique URLs queued this session.
    """
    website_url = CFG.base_url
    seen_urls: set[str] = set()

    logger.info("Producer starting — base search URL: %s", website_url)

    for query in CFG.search_queries:
        logger.info("Starting search for query: '%s'", query)
        page_num = 1

        # Open a fresh context (and rotate User-Agent) for *each* query
        async with manager.new_page() as page:
            page.set_default_timeout(CFG.navigation_timeout)

            await page.goto(
                website_url,
                wait_until="domcontentloaded",
            )
            await detect_bot_challenge(page)
            await human_delay(1.0, 2.0)

            search_xpath = CFG.search_input_xpath
            await human_write(page, search_xpath, query)
            search_button_locator = page.locator(CFG.search_input_xpath).first
            await search_button_locator.press("Enter")
            await human_delay(1.0, 2.0)
            page1 = page.locator("//a[@aria-label='Page 1']")
            await page1.click()
            await human_delay(2.0, 3.0)
            await page.evaluate("document.body.style.zoom = '25%'")

            while True:
                await human_delay(4.0, 5.0)
                logger.info(
                    "Producer scraping page %d for query '%s' …", page_num, query
                )

                # await page.evaluate("document.body.style.zoom = '25%'")
                # await human_delay(4.0, 5.0)

                await page.wait_for_selector(
                    CFG.one_product_xpath,
                    timeout=CFG.element_timeout,
                )

                product_links = []

                # check if there is a product in container or not
                check_product_in_container = page.locator(
                    CFG.check_no_product_container_xpath
                )
                if await check_product_in_container.count() > 0:
                    logger.info("no product in container")
                    break

                products = await page.locator(CFG.each_product_xpath).all()
                logger.info("product number of this page: %d", len(products))

                for product in products:
                    link_locator = product.locator("a")
                    href = await link_locator.get_attribute("href")
                    if href:
                        product_links.append(f"{CFG.base_url}{href[1:]}")
                if not product_links:
                    break

                # await page.evaluate("document.body.style.zoom = '100%'")
                # await human_delay(1.0, 2.0)

                # Deduplicate against all previously seen URLs across all queries (in-session)
                new_urls = [u for u in product_links if u not in seen_urls]
                seen_urls.update(new_urls)

                # Filter out URLs already scraped in a previous run
                urls_to_queue: list[str] = []
                for url in new_urls:
                    if await result_manager.is_scraped(url):
                        logger.info("Skipping already-scraped URL: %s", url)
                    else:
                        urls_to_queue.append(url)

                logger.info(
                    "  Page %d → %d new URLs (%d skipped as already scraped, total unique seen: %d)",
                    page_num,
                    len(urls_to_queue),
                    len(new_urls) - len(urls_to_queue),
                    len(seen_urls),
                )

                for url in urls_to_queue:
                    await queue.put((url, query))

                # ── Pagination / infinite-scroll detection ─────────────────
                next_btn = page.locator(CFG.sel_next_page)
                if await next_btn.count() == 0:
                    logger.info("Producer: no more pages found for query '%s'.", query)
                    break

                is_disabled = await next_btn.get_attribute("aria-disabled")
                if is_disabled == "true":
                    logger.info(
                        "Producer: next-page button is disabled for query '%s'.", query
                    )
                    break

                await scroll_to_element(page, CFG.sel_next_page)
                await mouse_jitter(page)
                await human_delay()
                await next_btn.click()
                await page.wait_for_load_state("domcontentloaded")
                await detect_bot_challenge(page)
                page_num += 1

    logger.info("Producer finished — %d unique URLs seen in total.", len(seen_urls))
    return len(seen_urls)
