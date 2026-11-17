import { formatTime, gradeLabel, Strings, zoneLabel } from "../i18n";
import type { MoveLogEntry } from "../net/types";
import { cardEmoji } from "../scene/emoji";

export function signed(value: number): string {
  return (value >= 0 ? "+" : "") + value.toFixed(2);
}

interface Props {
  moves: MoveLogEntry[];
  strongest: number[]; // move_log indices
  weakest: number[];
  selected: number | null;
  onSelect: (index: number) => void;
  strings: Strings;
  lang: "pl" | "en";
}

export function MoveHighlights({ moves, strongest, weakest, selected, onSelect, strings, lang }: Props) {
  const sections = [
    { title: strings.strongest_moves, indices: strongest },
    { title: strings.weakest_moves, indices: weakest },
  ];
  return (
    <>
      {sections
        .filter(({ indices }) => indices.length > 0)
        .map(({ title, indices }) => (
          <div key={title} className="summary-section">
            <h3>{title}</h3>
            <ul>
              {indices.map((i) => {
                const move = moves[i];
                return (
                  <li key={i}>
                    <button
                      className={`highlight-move${selected === i ? " move-selected" : ""}`}
                      onClick={() => onSelect(i)}
                    >
                      <span className="move-time">{formatTime(move.time)}</span> {cardEmoji(move.card)}{" "}
                      {move.card} → {zoneLabel(move.zone, lang)}{" "}
                      <span className={`grade-tag grade-${move.grade}`}>{gradeLabel(move.grade, lang)}</span>
                      {move.delta_value != null && ` (${signed(move.delta_value)})`}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
    </>
  );
}
