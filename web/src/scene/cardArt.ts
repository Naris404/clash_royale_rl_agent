import manifest from "../assets/cards/manifest.json";

// Official card art cached by scripts/fetch_card_art.py; missing entries fall back to emoji.
interface ArtManifest {
  cards: Record<string, { file: string }>;
}

const files = import.meta.glob<string>("../assets/cards/*.png", {
  eager: true,
  query: "?url",
  import: "default",
});

export function cardArtUrl(card: string): string | null {
  const entry = (manifest as ArtManifest).cards[card];
  return entry ? files[`../assets/cards/${entry.file}`] ?? null : null;
}
