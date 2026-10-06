"""One-time fetch of official card art into the web assets.

Reads the Clash Royale API key from `.env`, downloads the icon of every
playable card and writes a manifest next to them. The output is committed,
so the app and the Docker image never need the key at runtime.

Usage: python scripts/fetch_card_art.py
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cr_rl.game.cards import PLAYABLE_CARDS  # noqa: E402

logger = logging.getLogger(__name__)

API_URL = "https://api.clashroyale.com/v1/cards"
ENV_FILE = ROOT / ".env"
ENV_KEYS = ("CLASH_ROYALE_API_KEY", "CLASH_ROYALE_API_KEY_MCP")
OUT_DIR = ROOT / "web" / "src" / "assets" / "cards"


def read_api_key(env_file: Path = ENV_FILE) -> str:
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            values[key.strip()] = value.strip().strip("\"'")
    for key in ENV_KEYS:
        if values.get(key):
            return values[key]
    raise SystemExit(f"No API key in {env_file} (expected one of: {', '.join(ENV_KEYS)})")


def fetch_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=20) as response:
        path.write_bytes(response.read())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    items = fetch_json(API_URL, read_api_key())["items"]
    by_name = {item["name"]: item for item in items}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cards: dict[str, dict] = {}
    for card in PLAYABLE_CARDS:
        item = by_name.get(card.replace("_", " "))
        if item is None:
            logger.warning("%s: not found in the API response", card)
            continue
        file_name = f"{card}.png"
        download(item["iconUrls"]["medium"], OUT_DIR / file_name)
        cards[card] = {"id": item["id"], "file": file_name}
        logger.info("%s -> %s", card, file_name)

    manifest = {"source": API_URL, "cards": cards}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d cards to %s", len(cards), OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
