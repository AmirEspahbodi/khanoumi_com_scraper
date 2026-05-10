from __future__ import annotations

import asyncio
import logging
import random

from playwright.async_api import Page

from src.config import CFG
from src.errors import BotChallengeDetected

logger = logging.getLogger("scraper")


async def human_delay(
    min_s: float = CFG.min_delay, max_s: float = CFG.max_delay
) -> None:
    """Async sleep for a random human-like duration."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def mouse_jitter(page: Page, steps: int = 5) -> None:
    """Move the mouse in small random increments to emulate human motion."""
    x = random.randint(200, 800)
    y = random.randint(200, 600)
    for _ in range(steps):
        x += random.randint(-30, 30)
        y += random.randint(-20, 20)
        await page.mouse.move(
            max(0, x),
            max(0, y),
            steps=random.randint(3, 8),
        )
        await asyncio.sleep(random.uniform(0.03, 0.12))


async def scroll_to_element(page: Page, selector: str) -> None:
    """Scroll the element smoothly into the viewport."""
    try:
        await page.locator(selector).first.scroll_into_view_if_needed(
            timeout=CFG.element_timeout
        )
        await asyncio.sleep(random.uniform(0.2, 0.5))
    except Exception:
        pass  # Non-fatal — best effort


async def detect_bot_challenge(page: Page) -> None:
    """
    Inspect the page title and body for known bot-challenge fingerprints.
    Raises BotChallengeDetected if any are found.
    """
    content = (await page.title() + await page.content()).lower()
    for keyword in CFG.bot_challenge_keywords:
        if keyword in content:
            raise BotChallengeDetected(
                f"Bot challenge detected: '{keyword}' found on {page.url}"
            )


async def human_write(
    page: Page,
    xpath: str,
    search_text: str,
    *,
    click_before_type: bool = True,
    clear_before_type: bool = False,
    min_char_delay: float = 0.07,
    max_char_delay: float = 0.22,
    typo_probability: float = 0.06,
    pre_type_delay: tuple[float, float] = (0.4, 0.9),
    post_type_delay: tuple[float, float] = (0.3, 0.7),
) -> None:
    """
    Type *search_text* into an input element (located by XPath) with
    realistic human-like behaviour:

      • Moves the mouse to the element and clicks it naturally.
      • Optional triple-click to select & clear any pre-filled value.
      • Types each character one-by-one with randomised inter-key delays.
      • Randomly inserts a typo then immediately corrects it with Backspace.
      • Adds brief random pauses mid-word to simulate thinking.

    Parameters
    ----------
    page              : Playwright ``Page`` object.
    xpath             : XPath expression that uniquely identifies the input.
    search_text       : The final text that should appear in the input.
    click_before_type : Move mouse to element + click before typing.
    clear_before_type : Triple-click to select all, then Delete existing value.
    min_char_delay    : Minimum seconds between keystrokes.
    max_char_delay    : Maximum seconds between keystrokes (before jitter).
    typo_probability  : 0–1 chance of making (then fixing) a typo per character.
    pre_type_delay    : (min, max) pause after the click, before first keystroke.
    post_type_delay   : (min, max) pause after the last keystroke.
    """
    locator = page.locator(f"xpath={xpath}").first

    # ── 1. Scroll element into view ──────────────────────────────────────
    await locator.scroll_into_view_if_needed(timeout=CFG.element_timeout)
    await asyncio.sleep(random.uniform(0.15, 0.35))

    # ── 2. Human mouse-move + click ──────────────────────────────────────
    if click_before_type:
        bounding_box = await locator.bounding_box()
        if bounding_box:
            # Aim at a slightly random spot inside the element, not dead-centre
            target_x = bounding_box["x"] + bounding_box["width"] * random.uniform(
                0.3, 0.7
            )
            target_y = bounding_box["y"] + bounding_box["height"] * random.uniform(
                0.3, 0.7
            )
            await page.mouse.move(target_x, target_y, steps=random.randint(8, 18))
            await asyncio.sleep(random.uniform(0.05, 0.18))
            await page.mouse.click(target_x, target_y)
        else:
            await locator.click()

    # ── 3. Clear existing content ────────────────────────────────────────
    if clear_before_type:
        await locator.triple_click()  # select all pre-filled text
        await asyncio.sleep(random.uniform(0.1, 0.25))
        await page.keyboard.press("Delete")  # wipe selection
        await asyncio.sleep(random.uniform(0.1, 0.2))

    # ── 4. Pre-typing pause (human "thinking" before starting) ───────────
    await asyncio.sleep(random.uniform(*pre_type_delay))

    # ── 5. Type character-by-character with typos ────────────────────────
    # Build a pool of "nearby" keys on a QWERTY layout for realistic typos
    _QWERTY_NEIGHBOURS: dict[str, str] = {
        "a": "sqwz",
        "b": "vghn",
        "c": "xdfv",
        "d": "srfce",
        "e": "wrds",
        "f": "drtgc",
        "g": "ftyhb",
        "h": "gyujn",
        "i": "uojk",
        "j": "huikm",
        "k": "jilom",
        "l": "kop",
        "m": "njk",
        "n": "bhjm",
        "o": "iklp",
        "p": "ol",
        "q": "wa",
        "r": "etdf",
        "s": "aqwdez",
        "t": "ryfg",
        "u": "yhij",
        "v": "cfgb",
        "w": "qase",
        "x": "zsdc",
        "y": "tugh",
        "z": "asx",
    }

    for char in search_text:
        # Occasionally make a typo on letter characters
        if (
            typo_probability > 0
            and char.isalpha()
            and random.random() < typo_probability
        ):
            neighbours = _QWERTY_NEIGHBOURS.get(char.lower(), "")
            if neighbours:
                typo_char = random.choice(neighbours)
                await page.keyboard.type(typo_char)
                await asyncio.sleep(random.uniform(0.08, 0.18))  # brief "oh wait"
                await page.keyboard.press("Backspace")
                await asyncio.sleep(random.uniform(0.06, 0.14))  # correction pause

        # Type the correct character
        await page.keyboard.type(char)

        # Randomised inter-key delay
        delay = random.uniform(min_char_delay, max_char_delay)

        # Inject an occasional longer mid-word pause (simulates hesitation)
        if random.random() < 0.08:
            delay += random.uniform(0.25, 0.55)

        await asyncio.sleep(delay)

    # ── 6. Post-typing pause (human glances at what they typed) ─────────
    await asyncio.sleep(random.uniform(*post_type_delay))
    logger.debug("human_write: typed %r into xpath=%r", search_text, xpath)
