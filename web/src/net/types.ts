// Typy wiadomości WebSocket — lustro server/protocol.py

export interface ArenaConfig {
  arena_width: number;
  arena_length: number;
  river_y: number;
  river_half_width: number;
  bridge_lane_x: number[];
  deploy_zones: Record<string, number[][]>;
  tower_layout: Record<string, Record<string, number[]>>;
  cards: Record<string, CardStats>;
  playable_cards: string[];
  tick_dt: number;
  time_limit: number;
}

export interface CardStats {
  damage?: number;
  hp?: number;
  elisir?: number;
  type?: string;
  radius?: number;
  [key: string]: unknown;
}

export interface TowerState {
  id: string;
  name: string;
  owner: number;
  x: number;
  y: number;
  hp: number;
  max_hp: number;
  alive: boolean;
}

export interface TroopState {
  id: number;
  card: string;
  owner: number;
  x: number;
  y: number;
  hp: number;
  max_hp: number;
}

export interface SpellState {
  id: number;
  card: string;
  owner: number;
  x: number;
  y: number;
  start_x: number;
  start_y: number;
  radius: number;
  age: number;
  travel_time: number;
}

export interface Snapshot {
  type: "snapshot";
  tick: number;
  time: number;
  time_limit: number;
  elixir: number[];
  towers: TowerState[];
  troops: TroopState[];
  spells: SpellState[];
  hand: string[];
  next_card?: string | null;
  done: boolean;
  winner?: number | null;
}

export interface HintTop {
  card?: string | null;
  slot?: number | null;
  zone?: number | null;
  prob: number;
}

export interface Hint {
  type: "hint";
  card?: string | null;
  slot?: number | null;
  zone?: number | null;
  prob?: number | null;
  value?: number | null;
  reason: string;
  hint: string;
  top: HintTop[];
  source: string;
}

export type Grade = "best" | "good" | "inaccuracy" | "blunder" | "unknown";

export interface Feedback {
  type: "feedback";
  grade: Grade;
  score: number;
  card: string;
  zone: number;
  human_prob?: number | null;
  delta_value?: number | null;
  best_card?: string | null;
  best_zone?: number | null;
  reason: string;
}

export interface KeyMoment {
  type: "key_moment";
  time: number;
  delta_value: number;
  note: string;
}

export interface MoveLogEntry {
  time: number;
  card: string;
  zone: number;
  grade: Grade;
  score: number;
  delta_value?: number | null;
}

export interface MatchEndDetail {
  winner?: number | null;
  time: number;
  moves: number;
  mean_score?: number | null;
  grade_counts: Record<string, number>;
  key_moments: Array<{ time: number; delta_value: number; note: string }>;
  move_log: MoveLogEntry[];
}

export interface EventMsg {
  type: "event";
  kind: "tower_down" | "match_end" | string;
  detail: Record<string, unknown> & Partial<MatchEndDetail>;
}

export interface Hello {
  type: "hello";
  session_id: string;
  mode: string;
  coach_enabled: boolean;
  lang: string;
  config: ArenaConfig;
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export type ServerMessage =
  | Snapshot
  | Hint
  | Feedback
  | KeyMoment
  | EventMsg
  | Hello
  | { type: "pong" }
  | ErrorMsg;

export type ClientMessage =
  | { type: "play"; card: string; x: number; y: number }
  | { type: "pause"; paused: boolean }
  | { type: "speed"; value: number }
  | { type: "rematch" }
  | { type: "ping" };
