import { formatTime, gradeLabel, Strings, zoneLabel } from "../i18n";
import type { MatchEndDetail } from "../net/types";
import { cardEmoji } from "../scene/emoji";

interface Props {
  detail: MatchEndDetail & { winner?: number | null };
  strings: Strings;
  lang: "pl" | "en";
  onRematch: () => void;
  onMenu: () => void;
}

export function SummaryModal({ detail, strings, lang, onRematch, onMenu }: Props) {
  const title =
    detail.winner === 0 ? strings.you_won : detail.winner === 1 ? strings.you_lost : strings.draw;
  const grades = detail.grade_counts ?? {};
  const moves = detail.move_log ?? [];

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>{title}</h2>
        <div className="summary-stats">
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

        <div className="summary-section move-history">
          <h3>{strings.move_history}</h3>
          {moves.length === 0 ? (
            <p>{strings.no_moves}</p>
          ) : (
            <ul>
              {moves.map((move, i) => (
                <li key={i}>
                  <span className="move-time">{formatTime(move.time)}</span>{" "}
                  {cardEmoji(move.card)} {move.card} → {zoneLabel(move.zone, lang)}{" "}
                  <span className={`grade-tag grade-${move.grade}`}>{gradeLabel(move.grade, lang)}</span>
                </li>
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
