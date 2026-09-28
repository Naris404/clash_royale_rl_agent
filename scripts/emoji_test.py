"""Test renderowania emoji w headless Chromium."""

from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page()
    page.set_content("<div style='font-size:30px'>🧱 🗿 ⚔️ 🎯 🔥 👑 🏰 🐗 🔫 💥 🪖</div>")
    page.wait_for_timeout(500)
    page.screenshot(path="screenshots/emoji_test.png")
    browser.close()
print("ok")
