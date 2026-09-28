import { formatTime, Strings } from "../i18n";
import type { Snapshot } from "../net/types";

interface Props {
  snap: Snapshot | null;
  paused: boolean;
  speed: number;
  onPause: (paused: boolean) => void;
  onSpeed: (speed: number) => void;
  onMenu: () => void;
  strings: Strings;
}

const SPEEDS = [0.5, 1, 2];

export function Hud({ snap, paused, speed, onPause, onSpeed, onMenu, strings }: Props) {
  const time = snap?.time ?? 0;
  const limit = snap?.time_limit ?? 180;
  return (
    <div className="hud">
      <div className="hud-timer" title="Timer">
        ⏱ {formatTime(Math.min(time, limit))} / {formatTime(limit)}
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
