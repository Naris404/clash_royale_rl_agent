# RL v2 architecture

Status: proposed (2026-09-28, revised 2026-10-21 after the research review in
[`docs/research/rl-architecture-review.md`](../research/rl-architecture-review.md));
no training runs before plan approval.

The v1 agent (MaskablePPO, flat observation, 13 actions, tower-HP-only reward,
trained only against `LogicAgent`) overfits its opponent and suffers from an
extremely sparse reward. The v2 rewrite keeps MaskablePPO with full action masks
and a flat MLP policy (separate pi/vf networks, Tanh), which is what the closest
published card-placement agents used at a similar budget, and changes everything
around it. v2 trains and is evaluated on env-v1.1 (ADR-0005).

## Observation

Flat vector + MLP (small board, CPU-friendly). Card identity is one-hot everywhere.

- **Unit grid** replaces the distance-sorted unit slots: 2 lanes × 8 depth bins
  (4 rows each, the river on a bin edge) × 6 unit cards × 2 owners = 192 floats. A cell
  is the summed HP of those units divided by the card's max HP, capped at 1. Lane is
  decided by x against the board's center line. Spells are not units and have no cells.
- **Own Cycle**: the 4 hand cards and the next card, each one-hot over the 8 Deck cards.
- **Opponent Cycle**: the opponent's last 4 plays, oldest first, each one-hot over the
  8 Deck cards plus "none". The Cycle is FIFO, so after 4 plays this gives the opponent's
  hand and next card exactly, the same information a human gets by watching.
- Enemy elixir stays visible. Elixir advantage, tower and time features stay.

Player 1 sees a y-only mirror (`y' = L − y`, x unchanged), the same mirror its deploy
zones and tower features use. This fixes the P1 lane mismatch, which blocked the web RL
opponent and any play against frozen policies.

## Actions: 53 = noop + 4 hand slots × 13 zones

Troop/building zones (8), P0 coordinates (P1 mirrors `y → 32 − y`):

| # | Zone | Coord | Purpose |
|---|------|-------|---------|
| 0 / 1 | back-L / back-R | (7.5, 0.5) / (10.5, 0.5) | slow buildup behind the king tower (Giant); troops only, no buildings |
| 2 / 3 | mid-L / mid-R | (3.5, 10.5) / (14.5, 10.5) | standard lane play, just in front of the princess tower |
| 4 / 5 | bridge-L / bridge-R | (3.5, 14.0) / (14.5, 14.0) | aggressive bridge pressure |
| 6 / 7 | pull-L / pull-R | (8.0, 11.0) / (10.0, 11.0) | center plant pulling building-targeters |

Spell zones (5). Zones 8–10 are absolute enemy-side coordinates (shown vs P1):

| # | Zone | Coord / aim |
|---|------|-------------|
| 8 / 9 | spell-tower-L / spell-tower-R | (3.5, 24.0) / (14.5, 24.0) |
| 10 | spell-king | (9.0, 29.0) |
| 11 / 12 | spell-push-L / spell-push-R | best enemy cluster in that lane, anywhere on the board |

Push-zone aim, computed on the tick of the play: candidate centres are the positions of
the enemy units and buildings in that lane. Each is scored by Σ min(spell damage, current
HP) over enemies inside that spell's radius. An enemy princess tower inside the radius
adds min(crown tower damage, tower HP). Ties go to the unit placed earliest. A push zone
is legal only while its lane holds at least one enemy unit or building. Towers never
make it legal; finishing a tower is zones 8 and 9.

Pull-zone derivation (left lane, P0): enemy Hog Rider leaves the bridge at
E = (3.5, 14.5); princess tower T = (3.5, 8.0); d(E,T) = 6.50. C = (8.0, 11.0) gives
d(E,C) ≈ 5.70, a margin of ≈ 0.80. The layout is mirror-symmetric about x = 9, so the
right lane has the same margin.

Masking keeps effective choices small, so the larger nominal space costs little.
Zones carry explicit absolute coordinates with per-zone allowed card types.

## Decision points

The agent is asked for an action only at a Decision point: a card play is legal and no
noop hold is running. Ticks where only noop is legal advance without asking, and their
rewards are summed into the step. A noop starts a hold of k = 3 ticks (0.3 s). The
schedule is one shared component used by training and by the RL opponent in a Session,
so the live policy plays at the timing it learned. There is no switch to turn it off;
k = 1 gives near per-tick play.

## Reward

Outcome-driven, with potential-based shaping (Ng et al. 1999) so shaping cannot change
the optimal policy:

```
r_t = λ_win   · outcome                              (last step only: win +1, loss −1, draw 0)
    + λ_crown · (γ·Φ_crown(s′) − Φ_crown(s))         Φ_crown = my crowns − enemy crowns
    + λ_hp    · (γ·Φ_hp(s′)    − Φ_hp(s))            Φ_hp = (my tower HP − enemy tower HP) / MAX_TOWER_HP_PER_PLAYER
```

γ is the PPO γ, applied per decision step. Φ(terminal) = 0. Defaults: λ_win = 1.0,
λ_crown = 0.2, λ_hp = 0.5, all configurable. Troop trade value, kill/loss delta and the
elixir-leak penalty are dropped: the potentials already give a dense signal, and the
leak penalty is not potential-based. Ablations are judged by win rate, not shaped return.

## Training

- Phase 1: all envs against RandomAgent (short). Phase 2: Training opponents assigned
  per SubprocVecEnv index, 8 LogicAgent, 2 ε-LogicAgent, 2 RandomAgent. ε-LogicAgent
  plays a random legal action (noop included) with probability ε = 0.2 at each tick it
  would act, drawn from the env's seeded RNG.
- The env accepts a frozen policy as a Training opponent (hook only; no snapshot pool).
- Every training episode gets its own derived seed.
- SubprocVecEnv with ~12 envs on the 14-core CPU; masks come from the env's
  `action_masks()`, not the `ActionMasker` wrapper. 1–3 h budget per run.

## Eval protocol

- Fixed evaluation seeds, disjoint from training seeds; 200 episodes vs LogicAgent and
  200 vs RandomAgent. Report win rate with a 95% Wilson CI, mean return and match length.
- The final configuration is trained with 3 seeds. The ablation arms (full, no-shaping
  = λ_crown = λ_hp = 0, no-curriculum = phase 2's mix from step 0) use 1 seed each on an
  equal, shorter budget and are reported as indicative, which the thesis states under
  limitations.
- **Ship gate:** v2 replaces the rule-based bot as the web RL opponent when the lower
  bound of its 95% Wilson CI vs LogicAgent is above 50%.

## Consequences

- Observation/action contract changes (`OBS_DIM`, `NUM_ACTIONS = 53`); tests pin the
  new layout.
- Coach zone mapping and hint/feedback copy (PL/EN) cover 13 zones, including the
  state-dependent push zones.
- V(s) moves to the outcome scale, so the coach's grading thresholds are retuned once a
  trained v2 model exists.

## Future work

Kept out of v2 on purpose: snapshot pool / self-play league, privileged critic, CNN
encoder, LSTM or frame stacking, autoregressive heads, behaviour-cloning warm start.
See the research review for the evidence and costs.
