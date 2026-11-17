import { formatTime, gradeLabel, zoneLabel } from "../i18n";
import type { MoveLogEntry } from "../net/types";
import { cardEmoji } from "../scene/emoji";

/** One move as "time · emoji · card → zone · grade tag". */
export function MoveLabel({ move, lang }: { move: MoveLogEntry; lang: "pl" | "en" }) {
  return (
    <>
      <span className="move-time">{formatTime(move.time)}</span> {cardEmoji(move.card)} {move.card} →{" "}
      {zoneLabel(move.zone, lang)}{" "}
      <span className={`grade-tag grade-${move.grade}`}>{gradeLabel(move.grade, lang)}</span>
    </>
  );
}
