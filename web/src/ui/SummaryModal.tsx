import { useEffect, useRef, useState } from "react";
import { formatTime, gradeLabel, signed, Strings, zoneLabel } from "../i18n";
import type { MatchEndDetail, MoveLogEntry } from "../net/types";
import { cardEmoji } from "../scene/emoji";
import { EvaluationGraph } from "./EvaluationGraph";
import { MoveHighlights } from "./MoveHighlights";
import { MoveLabel } from "./MoveLabel";

interface Props {
  detail: MatchEndDetail & { winner?: number | null };
  strings: Strings;
  lang: "pl" | "en";
  onRematch: () => void;
  onMenu: () => void;
}

interface MoveRowProps {
  move: MoveLogEntry;
  strings: Strings;
  lang: "pl" | "en";
  selected: boolean;
  onSelect: () => void;
}

function MoveRow({ move, strings, lang, selected, onSelect }: MoveRowProps) {
  const { value_before: before, value_after: after, delta_value: delta } = move;
  const row = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (selected) row.current?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  return (
    <li
      ref={row}
      className={selected ? "selected" : undefined}
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => e.key === "Enter" && onSelect()}
    >
      <div>
        <MoveLabel move={move} lang={lang} />
      </div>
      {before != null && after != null && delta != null && (
        <div className="move-detail">
          {strings.evaluation}: {signed(before)} → {signed(after)} ({signed(delta)})
        </div>
      )}
      <div className="move-detail">
        {move.grade === "best" ? (
          strings.same_as_yours
        ) : (
          <>
            {move.best_card
              ? `${strings.agent_move}: ${cardEmoji(move.best_card)} ${move.best_card} → ${zoneLabel(move.best_zone, lang)}`
              : strings.agent_waits}
            {move.reason && <div>{move.reason}</div>}
          </>
        )}
      </div>
    </li>
  );
}

export function SummaryModal({ detail, strings, lang, onRematch, onMenu }: Props) {
  const title =
    detail.winner === 0 ? strings.you_won : detail.winner === 1 ? strings.you_lost : strings.draw;
  const grades = detail.grade_counts ?? {};
  const moves = detail.move_log ?? [];
  const [selected, setSelected] = useState<number | null>(null); // move_log index

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="review-label">{strings.match_review}</div>
        <h2>{title}</h2>
        <div className="summary-stats">
          <div>
            {strings.crowns}: <strong>{detail.crowns[0]} – {detail.crowns[1]}</strong>
          </div>
          <div>
            {strings.moves}: <strong>{detail.moves}</strong>
          </div>
          {detail.mean_score != null && (
            <div>
              {strings.mean_quality}: <strong>{(detail.mean_score * 100).toFixed(0)}%</strong>
            </div>
          )}
        </div>

        {Object.keys(grades).length > 0 && (
          <div className="grade-chips">
            {(["best", "good", "inaccuracy", "blunder"] as const)
              .filter((g) => grades[g])
              .map((g) => (
                <span key={g} className={`grade-chip grade-${g}`}>
                  {gradeLabel(g, lang)} × {grades[g]}
                </span>
              ))}
          </div>
        )}

        <EvaluationGraph detail={detail} strings={strings} lang={lang} selected={selected} onSelect={setSelected} />

        {detail.key_moments && detail.key_moments.length > 0 && (
          <div className="summary-section">
            <h3>{strings.key_moments}</h3>
            <ul>
              {detail.key_moments.map((km, i) => (
                <li key={i} className={km.delta_value >= 0 ? "km-good" : "km-bad"}>
                  {formatTime(km.time)} — {km.note}
                </li>
              ))}
            </ul>
          </div>
        )}

        <MoveHighlights
          moves={moves}
          strongest={detail.strongest ?? []}
          weakest={detail.weakest ?? []}
          selected={selected}
          onSelect={setSelected}
          strings={strings}
          lang={lang}
        />

        <div className="summary-section move-history">
          <h3>{strings.move_history}</h3>
          {moves.length === 0 ? (
            <p>{strings.no_moves}</p>
          ) : (
            <ul>
              {moves.map((move, i) => (
                <MoveRow
                  key={i}
                  move={move}
                  strings={strings}
                  lang={lang}
                  selected={i === selected}
                  onSelect={() => setSelected(i)}
                />
              ))}
            </ul>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-primary" onClick={onRematch}>
            {strings.rematch}
          </button>
          <button className="btn" onClick={onMenu}>
            {strings.back_to_menu}
          </button>
        </div>
      </div>
    </div>
  );
}
