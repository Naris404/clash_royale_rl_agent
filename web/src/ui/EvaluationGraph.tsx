import { gradeLabel, Strings } from "../i18n";
import type { MatchEndDetail } from "../net/types";

interface Props {
  detail: MatchEndDetail;
  strings: Strings;
  lang: "pl" | "en";
  selected: number | null;
  onSelect: (index: number) => void;
}

const WIDTH = 320;
const HEIGHT = 120;
const PAD = 8;
const VALUE_RANGE = 1; // the evaluation lives roughly in [-1, 1]

/** Evaluation from the player's side over the match; the player's moves are clickable dots. */
export function EvaluationGraph({ detail, strings, lang, selected, onSelect }: Props) {
  const samples = detail.evaluation ?? [];
  if (samples.length === 0) {
    return <p className="graph-note">{strings.evaluation_needs_model}</p>;
  }

  const duration = Math.max(detail.time, samples[samples.length - 1].time, 1);
  const x = (time: number) => PAD + (time / duration) * (WIDTH - 2 * PAD);
  const y = (value: number) => {
    const clamped = Math.max(-VALUE_RANGE, Math.min(VALUE_RANGE, value));
    return HEIGHT / 2 - (clamped / VALUE_RANGE) * (HEIGHT / 2 - PAD);
  };
  const line = samples.map((s) => `${x(s.time).toFixed(1)},${y(s.value).toFixed(1)}`).join(" ");

  return (
    <div className="summary-section">
      <h3>{strings.evaluation_graph}</h3>
      <svg className="eval-graph" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={strings.evaluation_graph}>
        <line className="eval-zero" x1={PAD} x2={WIDTH - PAD} y1={y(0)} y2={y(0)} />
        <polyline className="eval-line" points={line} />
        {detail.key_moments.map((km, i) => (
          <polygon
            key={`km-${i}`}
            className={km.delta_value >= 0 ? "km-good" : "km-bad"}
            points={`${x(km.time)},${y(km.value) - 5} ${x(km.time) + 4},${y(km.value)} ${x(km.time)},${y(km.value) + 5} ${x(km.time) - 4},${y(km.value)}`}
          >
            <title>{km.note}</title>
          </polygon>
        ))}
        {detail.move_log.map((move, i) =>
          move.value_before == null ? null : (
            <g
              key={`move-${i}`}
              className={`eval-dot grade-${move.grade}${i === selected ? " selected" : ""}`}
              role="button"
              tabIndex={0}
              aria-label={`${move.card}, ${gradeLabel(move.grade, lang)}`}
              onClick={() => onSelect(i)}
              onKeyDown={(e) => e.key === "Enter" && onSelect(i)}
            >
              <circle className="eval-hit" cx={x(move.time)} cy={y(move.value_before)} r={10} />
              <circle className="eval-mark" cx={x(move.time)} cy={y(move.value_before)} r={i === selected ? 6 : 4} />
            </g>
          ),
        )}
      </svg>
      <div className="move-detail">{strings.evaluation_graph_hint}</div>
    </div>
  );
}
