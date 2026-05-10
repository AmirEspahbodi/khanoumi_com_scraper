from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

from src.browser_manager import BrowserManager
from src.config import CFG
from src.data import ProductData
from src.producer import producer
from src.result_manager import ResultManager
from src.worker import worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scraper")


async def run_scraper() -> list[ProductData]:
    """
    Main orchestration coroutine:
      1. Boot the BrowserManager singleton.
      2. Initialise ResultManager and load previously scraped URLs.
      3. Start the producer to populate the queue.
      4. Spin up N consumer workers in parallel.
      5. Wait for the queue to drain and collect results.
    """
    manager = await BrowserManager.get_instance()
    queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue(
        maxsize=CFG.queue_maxsize
    )
    results: list[ProductData] = []

    # ── Initialise ResultManager and restore cross-run dedup state ────
    result_manager = ResultManager()
    await result_manager.load_scraped_urls()

    # ── Start workers (they block on the queue immediately) ──────────
    worker_tasks = [
        asyncio.create_task(
            worker(i, queue, results, manager, result_manager),
            name=f"worker-{i:02d}",
        )
        for i in range(CFG.num_workers)
    ]
    logger.info("Spawned %d workers.", CFG.num_workers)

    # ── Run producer ─────────────────────────────────────────────────
    try:
        total_urls = await producer(queue, manager, result_manager)
    except Exception as exc:
        logger.critical("Producer crashed: %s", exc, exc_info=True)
        total_urls = 0
    # ── Send one sentinel per worker to trigger graceful shutdown ─────
    for _ in range(CFG.num_workers):
        await queue.put(None)

    # ── Wait for all workers to finish ────────────────────────────────
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    logger.info(
        "Scrape complete — %d URLs discovered, %d results collected.",
        total_urls,
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Graceful Shutdown
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()


def _register_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT / SIGTERM to trigger a clean shutdown."""

    def _handle(sig_name: str) -> None:
        logger.warning("Signal %s received — initiating graceful shutdown …", sig_name)
        _shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _handle(s))
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals
            signal.signal(sig, lambda *_: _handle(sig.name))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


async def main() -> None:
    loop = asyncio.get_running_loop()
    _register_signal_handlers(loop)

    scraper_task = asyncio.create_task(run_scraper())

    # Race between the scraper finishing and a shutdown signal
    done, pending = await asyncio.wait(
        [scraper_task, asyncio.create_task(_shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel anything still running
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Retrieve results if the scraper finished normally
    results: list[ProductData] = []
    if scraper_task in done and not scraper_task.cancelled():
        try:
            results = scraper_task.result()
        except Exception as exc:
            logger.error("Scraper task raised: %s", exc, exc_info=True)

    # Always tear down the browser
    instance = BrowserManager._instance
    if instance:
        await instance.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
