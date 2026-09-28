# RL v2 architecture

Status: proposed (2026-09-28) — awaiting plan approval; no training runs before then.

The v1 agent (MaskablePPO, flat observation, 13 actions, tower-HP-only reward,
trained only against `LogicAgent`) overfits its opponent and suffers from an
extremely sparse reward. The v2 rewrite keeps MaskablePPO — action masks are
essential and the algorithm is not the bottleneck — and changes everything around
it. The v1 checkpoint stays frozen as the thesis baseline; v2 ships in the game
only if it beats v1 under the eval protocol.

## Observation

Flat vector + MLP retained (small board, CPU-friendly). Enriched feature groups:
elixir advantage, per-lane threat mass (HP-weighted), distance-to-bridge features,
card-cycle state. Enemy elixir stays visible, enemy hand stays hidden (as v1).

## Actions — 45 = noop + 4 hand slots × 11 zones

Troop/building zones (8), per side, P0 coordinates (P1 mirrors `y → 32 − y`):

| Zone | Coord | Purpose |
|------|-------|---------|
| back-L / back-R | (3.0, 4.5) / (14.0, 4.5) | slow buildup behind towers (Giant) |
| mid-L / mid-R | (3.0, 9.0) / (14.0, 9.0) | standard lane play |
| bridge-L / bridge-R | (3.0, 14.0) / (14.0, 14.0) | aggressive bridge pressure |
| pull-L / pull-R | (5.5, 9.5) / (12.5, 9.5) | center plant pulling building-targeters |

Spell-only zones (3), absolute enemy-side coordinates:

| Zone | Coord (vs P1) |
|------|---------------|
| spell-tower-L / spell-tower-R | (3.5, 24.0) / (14.5, 24.0) |
| spell-king | (9.0, 30.0) |

Pull-zone derivation (left lane, P0): enemy Hog enters at bridge exit
E = (3.0, 14.5); princess tower T = (3.5, 8.0); d(E,T) ≈ 6.52. Pull requires
d(E,C) < d(E,T); user rule: maximize d(B,C) with d(B,C) < d(B,T) ≈ 8.02, shifted
toward center. C = (5.5, 9.5) gives d(E,C) ≈ 5.59 (margin ≈ 0.93) and
d(B,C) ≈ 6.96. Right lane margin is larger (≈ 1.30).

Masking keeps effective choices small (≤8 troops, 3 spells), so the larger nominal
space costs little. The old `play_card` hacks (building `y = RIVER_Y − 5`, spell
y-mirroring) are replaced by explicit absolute zone coordinates with per-zone
allowed card types.

## Reward

Tower-HP delta stays dominant. Shaped auxiliaries, each toggleable for ablations:
troop trade value (damage dealt/taken by units), elixir-leak penalty, kill/loss
delta. Ablation matrix (full vs no-shaping vs no-curriculum) is part of the plan
and runs only after plan approval.

## Training

Curriculum: RandomAgent → LogicAgent. Self-play hook kept in code, deferred.
SubprocVecEnv with ~12 envs on the 14-core CPU; 1–3h budget per run.

## Eval protocol

Fixed seeds; 200 episodes vs LogicAgent + 200 vs RandomAgent; report win rate,
mean reward, match length. v2 replaces v1 in the game only if it beats v1 vs
LogicAgent under this protocol.

## Consequences

- Observation/action contract changes: v1 checkpoints are incompatible (accepted;
  v1 is frozen baseline).
- Coach zone mapping and hint/feedback messages update to the 11-zone layout.
- `NUM_ACTIONS`/`OBS_DIM` become versioned constants; tests pin the new layout.
