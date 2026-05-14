from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

from src.browser_manager import BrowserManager
from src.config import CFG
from src.data import ProductData
from src.producer_store import load_producer_json
from src.result_manager import ResultManager
from src.worker import worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("consumer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_consumer")


async def main() -> None:
    """
    Entry point for the consumer phase.

    1. Boot BrowserManager.
    2. Initialise ResultManager and restore cross-run dedup state.
    3. Read producer.json; collect entries where is_scraped == false.
    4. Push (product_url, search_query) tuples onto an asyncio.Queue,
       followed by CFG.num_workers sentinel Nones.
    5. Spawn CFG.num_workers worker tasks and await completion.
    6. Shut down BrowserManager in a finally block.
    7. Log final counts (attempted, succeeded, failed).
    """
    manager = await BrowserManager.get_instance()

    results: list[ProductData] = []

    try:
        # ── Initialise ResultManager ─────────────────────────────────────
        result_manager = ResultManager()
        await result_manager.load_scraped_urls()

        # ── Load unscraped entries from producer.json ────────────────────
        all_entries = await load_producer_json()
        pending_entries = [e for e in all_entries if not e.get("is_scraped", False)]

        logger.info(
            "Consumer: %d total entries in producer.json, %d unscraped.",
            len(all_entries),
            len(pending_entries),
        )

        if not pending_entries:
            logger.info("Consumer: nothing to process — all entries already scraped.")
            return

        # ── Build the work queue ─────────────────────────────────────────
        queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue(
            maxsize=CFG.queue_maxsize
        )

        for entry in pending_entries:
            url: str = entry.get("product_url", "")
            query: str = entry.get("search_query", "")
            if url:
                await queue.put((url, query))

        # Sentinels — one per worker to trigger graceful shutdown
        for _ in range(CFG.num_workers):
            await queue.put(None)

        logger.info(
            "Consumer: enqueued %d URLs for %d workers.",
            len(pending_entries),
            CFG.num_workers,
        )

        # ── Spawn workers ────────────────────────────────────────────────
        worker_tasks = [
            asyncio.create_task(
                worker(i, queue, results, manager, result_manager),
                name=f"worker-{i:02d}",
            )
            for i in range(CFG.num_workers)
        ]

        # ── Await queue drain then worker completion ──────────────────────
        await queue.join()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    except Exception as exc:
        logger.critical("Consumer crashed: %s", exc, exc_info=True)
    finally:
        instance = BrowserManager._instance
        if instance:
            await instance.shutdown()

    # ── Final summary ────────────────────────────────────────────────────
    attempted = len(results)
    succeeded = sum(1 for p in results if not p.is_failed)
    failed = attempted - succeeded

    logger.info(
        "Consumer run complete — attempted: %d | succeeded: %d | failed: %d.",
        attempted,
        succeeded,
        failed,
    )


if __name__ == "__main__":
    asyncio.run(main())
