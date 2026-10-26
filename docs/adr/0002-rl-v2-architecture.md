# RL v2 architecture

Status: proposed (2026-09-28, revised 2026-10-21 after the research review in
[`docs/research/rl-architecture-review.md`](../research/rl-architecture-review.md),
revised again 2026-10-23 after the training review in
[`docs/research/maskable-ppo-training-review.md`](../research/maskable-ppo-training-review.md));
no training runs before plan approval. The Training-opponent mix and checkpoint selection
for v3 are amended by [ADR-0007](0007-v3-self-play-fine-tuning.md).

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
- Enemy elixir stays visible. Elixir advantage, tower and time features stay. The match-time
  feature keeps its meaning, `min(1, t / 180)`, so it saturates in Overtime.
- **Overtime progress** is one new last feature,
  `overtime_progress = clip((t − 180) / 120, 0, 1)`, so the agent can tell how much of
  Overtime is left. `OBS_DIM` grows by one and the layout test pins it.

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

## Match end and termination

Every Match end is a termination: a king tower falls, Regulation ends with unequal
Crowns, the first Crown is scored in Overtime, or Overtime runs out (Tiebreak or draw).
The time limit is part of the task, so the env returns `terminated=True` for all of them
and nothing is ever truncated. Stable-Baselines3 then never bootstraps from the state
after a Match end, which keeps the value targets and the potential-based shaping
consistent (Pardo et al. 2018).

## Training

Training runs in three Training phases, split by the share of the step budget. The 12
`SubprocVecEnv` envs get their Training opponent by env index, and the mix is fixed
within a phase.

| Training phase | Budget share | Training opponents (12 envs) |
|---|---|---|
| Warm-up | 0–20% | 12 RandomAgent |
| Scripted mix | 20–60% | 8 LogicAgent, 2 ε-LogicAgent, 2 RandomAgent |
| Self-play mix | 60–100% | 4 Past self, 6 LogicAgent, 1 ε-LogicAgent, 1 RandomAgent |

- ε-LogicAgent plays a random legal action (noop included) with probability ε = 0.2 at
  each tick it would act, drawn from the env's seeded RNG.
- **Past-self pool.** 3 Past selves, seeded from the last 3 selection checkpoints. Every
  100k steps the current agent plays the pool in an admission check: 50 matches, and if it
  wins at least 45% it joins the pool and replaces the oldest Past self. Past selves play
  deterministically and are loaded through the env's frozen-policy opponent hook. A
  pool of older Past selves is more stable than playing only the latest self, and the
  scripted bots in the mix keep the agent from forgetting the rule-based baseline
  (Bansal et al. 2018; Straka & Schmid 2025).
- Every training episode gets its own derived seed.
- SubprocVecEnv with ~12 envs on the 14-core CPU; masks come from the env's
  `action_masks()`, not the `ActionMasker` wrapper.

## Eval protocol

Two fixed seed sets, disjoint from each other and from the training seeds:

- **Selection seeds** (100) choose the checkpoint of a run. The agent plays them
  against LogicAgent every 50k steps and once for the final model. The highest win rate
  wins; a tie goes to the later checkpoint. Selection never uses shaped reward.
- **Report seeds** (200) produce every reported number. The same 200 seeds are used for
  the LogicAgent and the RandomAgent runs. Each win rate comes with a 95% Wilson
  confidence interval, next to mean return and match length.
- Among the 3 seeds of the final configuration, the shipped run is the one with the best
  selection win rate.
- A head-to-head between the full and no-self-play Ablation arms is played on the Report
  seeds, each seed from both sides, to show what the Past-self pool adds beyond the
  scripted opponents.
- **Ship gate:** v2 replaces the rule-based bot as the web RL opponent when, on the Report
  seeds only, the lower bound of its 95% Wilson CI vs LogicAgent is above 50%.

## Runs

- **Final configuration:** 3 seeds × 1M steps.
- **Ablation arms:** full, no-shaping, no-curriculum and no-self-play, each × 3 seeds ×
  500k steps. Every arm differs from full in exactly one thing:
  - no-shaping: λ_crown = λ_hp = 0;
  - no-curriculum: drops the warm-up and runs the scripted mix from 0 to 60%;
  - no-self-play: runs the scripted mix from 20% to 100%.

  Ablations are judged by win rate, not shaped return.
- **Budget rule:** if the selection win rate of the first final-configuration run rises
  by more than 10 percentage points over the last 20% of its steps, it is still learning
  and all final runs move to 3M steps.
- **Curriculum metrics**, compared between the full and no-curriculum arms:
  - jumpstart: selection win rate at the first round after the warm-up, and at the same
    step for no-curriculum;
  - time to a 50% selection win rate, warm-up steps counted;
  - final win rate on the Report seeds.

## Consequences

- Observation/action contract changes (`OBS_DIM` including `overtime_progress`,
  `NUM_ACTIONS = 53`); tests pin the new layout.
- The env stops returning `truncated`; `info["winner"]` is unchanged.
- Training and evaluation gain the Past-self pool, the Selection and Report seed sets,
  Wilson intervals and the Ablation arms above.
- Coach zone mapping and hint/feedback copy (PL/EN) cover 13 zones, including the
  state-dependent push zones.
- V(s) moves to the outcome scale, so the coach's grading thresholds are retuned once a
  trained v2 model exists.

## Future work

Kept out of v2 on purpose: `target_kl`, learning-rate decay, multi-length noop, a
privileged critic, a behaviour-cloning warm start, and CNN, LSTM or autoregressive
heads. See the research reviews for the evidence and costs.
