import type { GameStore } from "../game/store";
import { formatTime, Strings } from "../i18n";

interface Props {
  store: GameStore;
  strings: Strings;
}

/** Panel trenera: podpowiedź, pasek oceny pozycji, kluczowe momenty. */
export function CoachPanel({ store, strings }: Props) {
  const hint = store.hint;
  const evalValue = store.normalizedValue();

  return (
    <aside className="coach-panel">
      <h2 className="coach-title">🧠 {strings.coach}</h2>

      <div className="coach-hint">
        {hint ? (
          <>
            <div className="coach-hint-main">{hint.hint}</div>
            <div className="coach-hint-reason">{hint.reason}</div>
            {hint.top && hint.top.length > 1 && (
              <div className="coach-alts">
                {hint.top.slice(1, 3).map((alt, i) => (
                  <span key={i} className="coach-alt">
                    {alt.card ?? "·"} {(alt.prob * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="coach-hint-reason">{strings.coach_waiting}</div>
        )}
      </div>

      {evalValue != null && (
        <div className="eval-section">
          <div className="eval-label">{strings.eval_bar}</div>
          <div className="eval-bar" title={hint?.value != null ? hint.value.toFixed(4) : undefined}>
            <div className="eval-mid" />
            <div
              className={`eval-fill ${evalValue >= 0.5 ? "eval-good" : "eval-bad"}`}
              style={{
                height: `${Math.abs(evalValue - 0.5) * 200}%`,
                [evalValue >= 0.5 ? "bottom" : "top"]: "50%",
              } as React.CSSProperties}
            />
          </div>
        </div>
      )}

      <div className="key-moments">
        <div className="eval-label">{strings.key_moments}</div>
        {store.keyMoments.length === 0 ? (
          <div className="coach-hint-reason">{strings.no_key_moments}</div>
        ) : (
          <ul>
            {[...store.keyMoments].reverse().map((moment, i) => (
              <li key={i} className={moment.delta_value >= 0 ? "km-good" : "km-bad"}>
                {formatTime(moment.time)} — {moment.note} ({moment.delta_value >= 0 ? "+" : ""}
                {moment.delta_value.toFixed(3)})
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
