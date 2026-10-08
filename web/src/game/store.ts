import type {
  ArenaConfig,
  EventMsg,
  Feedback,
  Hint,
  KeyMoment,
  MatchEndDetail,
  ServerMessage,
  Snapshot,
} from "../net/types";

export interface Toast {
  id: number;
  feedback: Feedback;
  expiresAt: number;
}

/**
 * Mutable store — Pixi reads it every frame without React; React subscribes
 * through `subscribe`/`getVersion` (useSyncExternalStore) for the HUD only.
 */
export class GameStore {
  config: ArenaConfig | null = null;
  coachEnabled = false;
  sessionId: string | null = null;
  mode: string | null = null;

  prevSnap: Snapshot | null = null;
  currSnap: Snapshot | null = null;
  currAt = 0;

  hint: Hint | null = null;
  toasts: Toast[] = [];
  keyMoments: KeyMoment[] = [];
  matchEnd: (MatchEndDetail & { winner?: number | null }) | null = null;
  lastError: string | null = null;
  connected = false;
  paused = false;
  speed = 1;

  /** card selected from hand (targeting mode) */
  selectedCard: string | null = null;

  private toastSeq = 1;
  private valueMin = Infinity;
  private valueMax = -Infinity;
  private listeners = new Set<() => void>();
  private version = 0;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getVersion = (): number => this.version;

  private emit(): void {
    this.version++;
    this.listeners.forEach((listener) => listener());
  }

  apply(msg: ServerMessage): void {
    switch (msg.type) {
      case "hello":
        this.config = msg.config;
        this.coachEnabled = msg.coach_enabled;
        this.sessionId = msg.session_id;
        this.mode = msg.mode;
        this.connected = true;
        this.matchEnd = null;
        this.keyMoments = [];
        this.toasts = [];
        this.valueMin = Infinity;
        this.valueMax = -Infinity;
        break;
      case "snapshot":
        this.prevSnap = this.currSnap;
        this.currSnap = msg;
        this.currAt = performance.now();
        if (msg.done && this.matchEnd == null) {
          // match_end event usually arrives in the same tick; the snapshot alone
          // also ends the game (safety net)
          this.selectedCard = null;
        }
        break;
      case "hint":
        this.hint = msg;
        if (msg.value != null && Number.isFinite(msg.value)) {
          this.valueMin = Math.min(this.valueMin, msg.value);
          this.valueMax = Math.max(this.valueMax, msg.value);
        }
        break;
      case "feedback":
        this.toasts = [
          ...this.toasts.slice(-3),
          { id: this.toastSeq++, feedback: msg, expiresAt: performance.now() + 4500 },
        ];
        this.selectedCard = null;
        break;
      case "key_moment":
        this.keyMoments = [...this.keyMoments.slice(-19), msg];
        break;
      case "event":
        this.applyEvent(msg);
        break;
      case "error":
        this.lastError = msg.message;
        this.toasts = [
          ...this.toasts.slice(-3),
          {
            id: this.toastSeq++,
            feedback: { type: "feedback", grade: "unknown", score: 0, card: "", zone: 0, reason: msg.message },
            expiresAt: performance.now() + 3500,
          },
        ];
        break;
      case "pong":
        return; // no emit
    }
    this.emit();
  }

  private applyEvent(msg: EventMsg): void {
    if (msg.kind === "match_end") {
      this.matchEnd = msg.detail as MatchEndDetail;
      this.selectedCard = null;
    }
  }

  /** Positions for rendering: prev→curr interpolation (render ~1 snapshot behind). */
  interpolationAlpha(now: number): number {
    if (!this.prevSnap || !this.currSnap || !this.config) return 1;
    const intervalMs = (this.config.tick_dt * 1000) / this.speed;
    return Math.min(1, Math.max(0, (now - this.currAt) / intervalMs));
  }

  /** Normalized position eval 0..1 (0.5 = even) for the eval bar. */
  normalizedValue(): number | null {
    const value = this.hint?.value;
    if (value == null || !Number.isFinite(value)) return null;
    const span = this.valueMax - this.valueMin;
    if (span < 1e-6) return 0.5;
    return Math.min(1, Math.max(0, (value - this.valueMin) / span));
  }

  selectCard(card: string | null): void {
    this.selectedCard = card;
    this.emit();
  }

  /** Local reset after rematch — the server does not resend hello. */
  resetMatchLocal(): void {
    this.matchEnd = null;
    this.keyMoments = [];
    this.toasts = [];
    this.hint = null;
    this.prevSnap = null;
    this.currSnap = null;
    this.valueMin = Infinity;
    this.valueMax = -Infinity;
    this.selectedCard = null;
    this.emit();
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    this.emit();
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    this.emit();
  }

  pruneToasts(now: number): void {
    const alive = this.toasts.filter((t) => t.expiresAt > now);
    if (alive.length !== this.toasts.length) {
      this.toasts = alive;
      this.emit();
    }
  }
}
