from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("producer_store")

PRODUCER_JSON_PATH = Path("producer.json")

# Module-level lock shared by both producer and consumer.
_producer_json_lock: asyncio.Lock = asyncio.Lock()


async def load_producer_json() -> list[dict[str, Any]]:
    """
    Load and return the list of entry dicts from producer.json.
    Returns an empty list when the file does not yet exist.
    Acquires _producer_json_lock for the duration of the read.
    """
    async with _producer_json_lock:
        if not PRODUCER_JSON_PATH.exists():
            return []
        try:
            text = await asyncio.to_thread(
                PRODUCER_JSON_PATH.read_text, encoding="utf-8"
            )
            data = json.loads(text)
            if not isinstance(data, list):
                logger.warning(
                    "load_producer_json: expected a JSON array, got %s — resetting.",
                    type(data).__name__,
                )
                return []
            return data
        except json.JSONDecodeError as exc:
            logger.error(
                "load_producer_json: malformed JSON in %s (%s) — returning empty list.",
                PRODUCER_JSON_PATH,
                exc,
            )
            return []
        except Exception as exc:
            logger.error(
                "load_producer_json: unexpected error reading %s: %s",
                PRODUCER_JSON_PATH,
                exc,
            )
            return []


async def append_entry(entry: dict[str, Any]) -> None:
    """
    Append *entry* to the in-file list and atomically rewrite producer.json.
    Acquires _producer_json_lock for the duration of the read-modify-write.
    """
    async with _producer_json_lock:
        # Read current contents (without re-acquiring the lock — we already hold it).
        if PRODUCER_JSON_PATH.exists():
            try:
                text = await asyncio.to_thread(
                    PRODUCER_JSON_PATH.read_text, encoding="utf-8"
                )
                entries: list[dict[str, Any]] = json.loads(text)
                if not isinstance(entries, list):
                    entries = []
            except Exception as exc:
                logger.warning(
                    "append_entry: could not read existing file (%s) — starting fresh.",
                    exc,
                )
                entries = []
        else:
            entries = []

        entries.append(entry)
        await _atomic_write(entries)
        logger.debug(
            "append_entry: wrote %d entries (added %s).",
            len(entries),
            entry.get("product_url", "<unknown>"),
        )


async def update_entry_scraped(url: str, scrap_dir: str) -> None:
    """
    Find the entry whose ``product_url`` matches *url*, set its
    ``is_scraped`` flag to ``True`` and ``scrap_directory`` to *scrap_dir*,
    then atomically rewrite producer.json.

    Acquires _producer_json_lock for the duration of the read-modify-write.
    No-op (with a warning) when *url* is not found.
    """
    async with _producer_json_lock:
        if not PRODUCER_JSON_PATH.exists():
            logger.warning(
                "update_entry_scraped: producer.json not found — cannot update %s.", url
            )
            return

        try:
            text = await asyncio.to_thread(
                PRODUCER_JSON_PATH.read_text, encoding="utf-8"
            )
            entries: list[dict[str, Any]] = json.loads(text)
        except Exception as exc:
            logger.error("update_entry_scraped: could not read producer.json: %s", exc)
            return

        found = False
        for entry in entries:
            if entry.get("product_url") == url:
                entry["is_scraped"] = True
                entry["scrap_directory"] = scrap_dir
                found = True
                break

        if not found:
            logger.warning(
                "update_entry_scraped: URL not found in producer.json — %s", url
            )
            return

        await _atomic_write(entries)
        logger.debug("update_entry_scraped: marked %s as scraped → %s", url, scrap_dir)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


async def _atomic_write(entries: list[dict[str, Any]]) -> None:
    """Serialise *entries* to JSON and atomically replace producer.json."""
    json_str = json.dumps(entries, ensure_ascii=False, indent=2)
    tmp_path = PRODUCER_JSON_PATH.with_suffix(".tmp")
    await asyncio.to_thread(tmp_path.write_text, json_str, encoding="utf-8")
    await asyncio.to_thread(tmp_path.replace, PRODUCER_JSON_PATH)
