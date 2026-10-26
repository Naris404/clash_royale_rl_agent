import { formatTime, Strings } from "../i18n";
import type { Snapshot } from "../net/types";

interface Props {
  snap: Snapshot | null;
  paused: boolean;
  speed: number;
  opponent: string | null; // e.g. "vs RL · Shipped model", only in a match vs RL
  onPause: (paused: boolean) => void;
  onSpeed: (speed: number) => void;
  onMenu: () => void;
  strings: Strings;
}

const SPEEDS = [0.5, 1, 2];

export function Hud({ snap, paused, speed, opponent, onPause, onSpeed, onMenu, strings }: Props) {
  const clock = snap?.clock ?? 0;
  const phase = snap?.phase ?? "regulation";
  const crowns = snap?.crowns ?? [0, 0];
  return (
    <div className="hud">
      <div className={`hud-timer hud-phase-${phase}`} title={strings[`phase_${phase}`]}>
        ⏱ {formatTime(Math.ceil(clock))} · {strings[`phase_${phase}`]}
      </div>
      {opponent && <div className="hud-opponent">{opponent}</div>}
      <div className="hud-crowns" title={strings.crowns}>
        👑 {crowns[0]} – {crowns[1]}
      </div>
      <div className="hud-controls">
        <button className="btn" onClick={onMenu}>⌂ {strings.menu}</button>
        <button className="btn" onClick={() => onPause(!paused)}>
          {paused ? `▶ ${strings.resume}` : `⏸ ${strings.pause}`}
        </button>
        <div className="speed-group" title={strings.speed}>
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={`btn btn-small ${speed === s ? "btn-active" : ""}`}
              onClick={() => onSpeed(s)}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
