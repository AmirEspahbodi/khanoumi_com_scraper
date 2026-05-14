from __future__ import annotations

import asyncio
import logging
import sys

from src.browser_manager import BrowserManager
from src.producer import producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("producer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_producer")


async def main() -> None:
    """
    Entry point for the producer phase.

    1. Boot BrowserManager.
    2. Run the producer — discovers all product URLs across all search queries
       and writes them to producer.json with ``is_scraped: false``.
    3. Shut down BrowserManager regardless of outcome.
    4. Log the total number of entries written.
    """
    manager = await BrowserManager.get_instance()
    total_written: int = 0

    try:
        total_written = await producer(manager)
    except Exception as exc:
        logger.critical("Producer crashed: %s", exc, exc_info=True)
    finally:
        instance = BrowserManager._instance
        if instance:
            await instance.shutdown()

    logger.info(
        "Producer run complete — %d new entries written to producer.json.",
        total_written,
    )


if __name__ == "__main__":
    asyncio.run(main())
