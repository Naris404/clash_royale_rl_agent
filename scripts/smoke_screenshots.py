"""Screenshots of the web app (visual verification) — requires a running server.

  python scripts/smoke_screenshots.py --url http://localhost:8000 --out screenshots
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="screenshots")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        page.goto(args.url)
        page.wait_for_timeout(800)
        page.screenshot(path=out / "1_home.png")

        # start coach training
        page.click("text=Training with coach")
        page.wait_for_timeout(2500)
        page.screenshot(path=out / "2_game_start.png")

        # play the first card in hand: click the card, then the middle of own half
        page.click(".card:first-child")
        page.wait_for_timeout(300)
        box = page.locator(".arena-canvas canvas").bounding_box()
        if box:
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.72)
        page.locator(".toast").first.wait_for(state="visible", timeout=5000)
        assert page.locator(".card-selected").count() == 0, "Card remained selected after being played"
        page.wait_for_timeout(2500)
        page.screenshot(path=out / "3_after_play.png")

        # pauza
        page.click("text=Pauza")
        page.wait_for_timeout(400)
        page.screenshot(path=out / "4_paused.png")

        browser.close()

    print(f"Saved screenshots to {out}/")


if __name__ == "__main__":
    main()
