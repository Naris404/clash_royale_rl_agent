"""Fast-forwarded match — screenshots of the endgame and the summary modal.

  python scripts/fast_forward.py --url http://localhost:8000 --out screenshots
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="screenshots")
    parser.add_argument("--timeout-s", type=int, default=240)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.url)
        page.wait_for_timeout(600)
        page.click("text=Training with coach")
        page.wait_for_timeout(2000)

        # 2× tempo
        page.click(".speed-group >> text=2×")
        page.wait_for_timeout(300)

        # a few plays during the match
        box = page.locator(".arena-canvas canvas").bounding_box()
        plays = 0
        deadline = time_left = args.timeout_s
        import time

        start = time.time()
        modal_shot = False
        while time.time() - start < deadline:
            if page.locator(".modal").count() > 0:
                page.screenshot(path=out / "5_summary_modal.png")
                modal_shot = True
                break
            # play a random available card every ~2.5 s
            cards = page.locator(".card:not(.card-disabled)")
            if box and cards.count() > 0 and plays < 40:
                try:
                    cards.first.click(timeout=500)
                    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.75)
                    plays += 1
                except Exception:
                    pass
            page.wait_for_timeout(2500)

        page.screenshot(path=out / "6_end_state.png")
        print(f"modal: {modal_shot}, cards played: {plays}")
        browser.close()


if __name__ == "__main__":
    main()
