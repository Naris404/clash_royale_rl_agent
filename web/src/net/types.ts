// WebSocket message types — mirror of server/protocol.py

export interface ArenaConfig {
  arena_width: number;
  arena_length: number;
  river_y: number;
  river_half_width: number;
  bridge_lane_x: number[];
  bridge_half_width: number;
  back_edge_margin: number;
  deploy_zones: Record<string, DeployZone[]>;
  tower_layout: Record<string, Record<string, number[]>>;
  cards: Record<string, CardStats>;
  deck: string[];
  tick_dt: number;
}

export interface DeployZone {
  name: string;
  x: number;
  y: number;
  card_types: string[];
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
  deploy_remaining: number;
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

export interface HitState {
  owner: number;
  from_x: number;
  from_y: number;
  x: number;
  y: number;
  damage: number;
  ranged: boolean;
}

export interface PocketState {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}

export type MatchPhase = "regulation" | "double_elixir" | "overtime";

export interface Snapshot {
  type: "snapshot";
  tick: number;
  time: number;
  phase: MatchPhase;
  clock: number; // seconds left in the current period
  elixir: number[];
  king_active: boolean[];
  towers: TowerState[];
  troops: TroopState[];
  spells: SpellState[];
  hits: HitState[];
  pockets: PocketState[]; // extra legal deploy areas for the human (a fallen enemy princess tower)
  hand: string[];
  next_card: string;
  crowns: number[];
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
  aim?: [number, number] | null; // where the move lands now (a push zone's target moves)
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
  crowns: number[];
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

// REST — mirror of the Model catalogue in server/model_catalogue.py and server/app.py
export interface ModelInfo {
  id: string;
  label: Record<"pl" | "en", string>;
  description: Record<"pl" | "en", string>;
  available: boolean;
}

export interface ModelCatalogue {
  default: string;
  models: ModelInfo[];
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
