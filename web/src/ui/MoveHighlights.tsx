import { signed, Strings } from "../i18n";
import type { MoveLogEntry } from "../net/types";
import { MoveLabel } from "./MoveLabel";

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
                      className={`highlight-move${selected === i ? " selected" : ""}`}
                      onClick={() => onSelect(i)}
                    >
                      <MoveLabel move={move} lang={lang} />
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
