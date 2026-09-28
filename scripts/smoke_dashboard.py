"""Browser smoke test for dashboard and route navigation."""

from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(f"{args.url}/dashboard")
        page.wait_for_selector(".dashboard-page")
        page.wait_for_selector(".dashboard-summary")

        page.get_by_role("button", name="Menu").click()
        page.wait_for_url(f"{args.url}/")
        page.locator(".mode-card").first.click()
        page.wait_for_url(f"{args.url}/game")
        assert page.locator(".game-screen").count() == 1
        browser.close()


if __name__ == "__main__":
    main()
