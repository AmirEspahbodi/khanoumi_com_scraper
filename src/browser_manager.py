from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright_stealth import Stealth

from src.config import CFG

logger = logging.getLogger("scraper")


import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    async_playwright,
)
from playwright_stealth import Stealth

from src.config import CFG

logger = logging.getLogger("scraper")


class BrowserManager:
    """
    Thread-safe Singleton that owns the Playwright instance and browser.
    Creates an isolated context per page to prevent session bleeding.
    """

    _instance: Optional["BrowserManager"] = None
    _lock: Optional[asyncio.Lock] = None

    # Internal state
    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _initialized: bool = False

    @classmethod
    async def get_instance(cls) -> "BrowserManager":
        # Lazily instantiate the lock inside the running event loop
        if cls._lock is None:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance._init()
            return cls._instance

    async def _init(self) -> None:
        if self._initialized:
            return

        logger.info("Initializing Playwright + Chrome …")
        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            channel="chrome",  # real Chrome binary
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--window-size=1920,1080",
            ],
        )

        self._initialized = True
        logger.info("BrowserManager ready.")

    async def shutdown(self) -> None:
        logger.info("BrowserManager shutting down …")
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        BrowserManager._instance = None
        self._initialized = False
        logger.info("BrowserManager stopped.")

    @asynccontextmanager
    async def new_page(self) -> AsyncIterator[Page]:
        """
        Yield a new page from an ISOLATED context.
        This prevents cookie/session bleeding and allows safe User-Agent rotation.
        """
        if not self._initialized or not self._browser:
            raise RuntimeError("BrowserManager.get_instance() must be called first.")

        ua = random.choice(CFG.user_agents)

        # Create a fresh, isolated context for this specific task
        context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=ua,
            locale="en-US",
            timezone_id="America/New_York",
            java_script_enabled=True,
            accept_downloads=False,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        )

        page = await context.new_page()

        # Rely solely on playwright-stealth to handle evasions cleanly
        await Stealth().apply_stealth_async(page)

        try:
            yield page
        finally:
            if not page.is_closed():
                await page.close()
            # Clean up the context so we don't leak memory
            await context.close()
