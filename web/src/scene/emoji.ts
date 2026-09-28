export const CARD_EMOJI: Record<string, string> = {
  Knight: "⚔️",
  Giant: "🧱", // 🗿 nie renderuje się w środowiskach bez rozszerzonych fontów emoji
  Cannon: "🎯",
  Musketeer: "🔫",
  Hog_Rider: "🐗",
  Fireball: "🔥",
};

export const TOWER_EMOJI: Record<string, string> = {
  King_Tower: "👑",
  Tower: "🏰",
};

export function cardEmoji(card: string): string {
  return CARD_EMOJI[card] ?? "❔";
}

export const OWNER_COLORS = [0x3b82f6, 0xef4444] as const;
