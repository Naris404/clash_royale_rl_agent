# RL agent

Source document for the thesis method chapter. It describes the RL agent exactly as
the code implements it today, and it is a living document: every change to the
observation, actions, reward, network, hyperparameters or training setup updates
this file in the same PR (see [Change log](#change-log)).

Vocabulary follows [`CONTEXT.md`](../CONTEXT.md). Decisions: ADR-0002 (RL v2
architecture), ADR-0003 (environment first, frozen at **env-v1**), ADR-0004 (fixed
mirror deck), ADR-0005 (arena geometry reopened as **env-v1.1**).

## Status

| | |
|---|---|
| Documented layout | 279-dim observation (version 2), 53 actions (13 deploy zones) |
| Simulation | env-v1.1 (env-v1 plus the arena geometry of ADR-0005, below): 8-card Deck, air units and splash; crowns decide the match; double elixir from 2:00, overtime and a weakest-tower tiebreak. Playtested and signed off ([checklist](playtest-env-v1.md#env-v11-sign-off)); frozen, tagged `env-v1.1` |
| Trained model for this layout | none yet; the **v2 model** is trained only on env-v1.1, after its sign-off (ADR-0005) |
| **v1 model** | historical result only (13 actions); cannot run in env-v1, see [Results](#results) |

The observation and action layout above is the contract: the observation is pinned by
`tests/test_observation.py` and the actions by `tests/test_env_layout.py`, and any change
to either needs a new environment version.

## Agent at a glance

- **Algorithm:** MaskablePPO (`sb3-contrib`), PPO with invalid-action masking.
- **Learner:** player 0 (bottom half of the arena). The opponent plays player 1.
- **Training opponents:** a mix of the rule-based bot (`LogicAgent`), a noisy variant
  (ε-`LogicAgent`) and the random bot (`RandomAgent`), assigned per parallel env (see
  [Training setup](#training-setup)); a frozen policy can also be plugged in.
- **Decision rate:** the agent is asked only at **Decision points** (see
  [Decision points](#decision-points)); the simulation runs at 10 Hz (`TICK_DT = 0.1` s).
- **Episode:** one match, at most 3001 simulation ticks (180 s regulation + 120 s overtime),
  fewer agent steps.

Code: `src/cr_rl/game/board.py` (observation, actions, mask), `src/cr_rl/game/rewards.py` (reward),
`src/cr_rl/env/gym_env.py` (Gymnasium wrapper), `src/cr_rl/game/decisions.py`
(decision schedule), `src/cr_rl/training/train.py`
(training and evaluation), `src/cr_rl/agents/rl.py` (inference).

## Environment interface

`ClashRoyaleEnv` wraps `Board` as a Gymnasium environment with
`observation_space = Box(-1, 1, (279,), float32)` and `action_space = Discrete(53)`,
and exposes `action_masks()` for MaskablePPO.

One `step(action)` registers the action with the decision schedule and then simulates
ticks until the next Decision point or the end of the match. Every tick:

1. The opponent picks its action for player 1 (the rule-based bot submits its play
   through `set_pending_play` and returns noop). The opponent acts on every tick.
2. Player 0's action is applied (noop on the skipped ticks), then player 1's.
3. Elixir regenerates for both players (start 5, max 10, +1 per 2.8 s; twice as fast from 2:00 through overtime).
4. Combat runs for one tick.
5. Dead units are removed and the clock advances by 0.1 s.
6. End check: **terminated** when a king tower is destroyed (3 crowns for its
   destroyer). Otherwise nothing happens before 180 s; then also **terminated** as soon as the
   crowns differ (end of regulation, or the first crown of the up-to-120 s overtime), so the
   player with more crowns wins (a princess tower is 1 crown). Tied crowns at 180 s start
   overtime; if it ends without a crown, the player whose weakest standing tower has less
   HP loses and equal weakest towers are a draw. 1800 ticks of 0.1 s accumulate to just
   under 180.0 in floating point, so a regulation match lasts 1801 ticks. Every Match end is a
   termination, so `truncated` is always `False` and the value of a final state is 0 for PPO.

The returned reward is computed once per step, from the position at the start and at the
end of the step (see [Reward](#reward)). `reset()` also advances to the first Decision point.

### Decision points

The agent is asked for an action only at a **Decision point**: a tick where at least one
card play is legal and no noop hold is running. Ticks where only noop is legal advance
without asking. Choosing noop starts a hold of `k = 3` ticks (0.3 s, `NOOP_HOLD_TICKS`)
before the agent is asked again; `ClashRoyaleEnv(hold_ticks=...)` configures `k`, and
`k = 1` asks on every tick where a play is legal. Choosing a card play starts no hold.
The env therefore never returns a state where only noop is legal, except the terminal one.

`DecisionSchedule` (`src/cr_rl/game/decisions.py`) is the single component behind this
schedule. The env uses it for player 0 and the Session uses it for the RL opponent
(player 1), so the live policy plays at the timing it was trained with; the rule-based
bot is not gated. There is no switch to turn the schedule off. Coach Hints and move grades
are computed from the Board at any tick and are unaffected.

Because one step spans several ticks, `gamma` and `n_steps` below count agent steps,
not ticks.

When the match ends, `info["winner"]` holds the winner (0 or 1) or `None` for a draw;
it is `None` on every earlier step.

## Observation

A flat `float32` vector of `OBS_DIM = 279` features (observation version 2), built by
`Board.get_observation(player)`. Segment offsets are named constants in `board.py`
(`OBS_GLOBAL_FEATURES`, `OBS_TOWER_BASE`, `OBS_HAND_BASE`, `OBS_NEXT_BASE`, `OBS_PLAYS_BASE`,
`OBS_GRID_BASE`, `OBS_OVERTIME_INDEX`), pinned by `tests/test_observation.py`. Card identity is one-hot everywhere.
No `VecNormalize` is applied; every feature is already scaled to [-1, 1].

| Indices | Count | Feature | Encoding |
|---|---|---|---|
| 0 | 1 | Own elixir | `elixir / 10` |
| 1 | 1 | Enemy elixir | `elixir / 10` (visible, unlike the real game) |
| 2 | 1 | Elixir advantage | `(own − enemy elixir) / 10` |
| 3 | 1 | Match time | `min(1, time / 180)` (stays 1 during overtime; see overtime progress) |
| 4–6 | 3 | Own towers: left princess, right princess, king | `hp / max_hp`; 0 once destroyed |
| 7–9 | 3 | Enemy towers: left princess, right princess, king | `hp / max_hp`; 0 once destroyed |
| 10–41 | 32 | Hand: 4 slots × 8 cards | one-hot per slot |
| 42–49 | 8 | Next card | one-hot over the 8 Deck cards |
| 50–85 | 36 | Opponent's last 4 plays: 4 × (8 cards + none) | one-hot per play, oldest first |
| 86–277 | 192 | Unit grid: 2 owners × 2 lanes × 8 depth bins × 6 unit cards | see below |
| 278 | 1 | Overtime progress | `clip((time − 180) / 120, 0, 1)`: 0 throughout Regulation, 1 at the end of Overtime |

**Hand.** Slot `s` occupies indices `10 + 8·s … 17 + 8·s`; the hot index is the
card ID in Deck order: Knight 0, Giant 1, Hog Rider 2, Musketeer 3, Cannon 4,
Fireball 5, Arrows 6, Baby Dragon 7. The hand is always full (4 cards). Slots are
positional: playing the card in slot `s` shifts slots `s+1…3` down by one and the
drawn card lands in slot 3.

**Next card.** Indices `42 … 49` hold the one-hot of the visible next card, the card
drawn after the next play (the Cycle's queue head).

**Opponent's last plays.** The Board keeps each player's play history (`Board.play_history`,
every card played, spells included). The observation holds the opponent's last 4 plays as
four blocks of 9 (8 Deck cards plus "none"), oldest first; the latest play is always the last
block, and plays that have not happened yet read "none" in the leading blocks. The Cycle is
FIFO and the queue holds exactly 4 cards, so after the opponent's 4th play these four cards
are their queue: the other 4 Deck cards are their hand, and the oldest of the four is their
next card. The agent's own plays are not part of its observation.

**Unit grid.** Every living troop and building (towers excluded) adds to one cell, indexed
`[owner][lane][depth][unit card]` in row-major order, with `owner` 0 = own and 1 = enemy:

- *Lane:* 0 (left) when `x < 9`, 1 (right) otherwise, against the board's centre line.
- *Depth:* `min(7, floor(y / 4))` in the observer's view, so 8 bins of 4 rows each; the
  river (`y = 16`) lies on the edge between bins 3 and 4.
- *Unit card:* the 6 non-spell Deck cards in Deck order (Knight, Giant, Hog Rider,
  Musketeer, Cannon, Baby Dragon). Spells are not units and have no cells.
- *Value:* the summed `hp / max_hp` of the units in the cell, capped at 1, so one full-HP
  unit reads 1.0, a wounded one less, and two Knights in one cell read 1.0.

The grid does not reshuffle when units spawn or die: a unit always occupies the same
cell. Unit attack cooldown, deploy state, remaining Cannon lifetime and spell effects are
not observed.

**Player 1 view.** For `player = 1` the vector is built from player 1's perspective,
so a policy trained as player 0 can drive player 1: own elixir first, own towers
first, own hand, the opponent's plays, and depth measured in y mirrored only
(`y' = 32 − y`, x unchanged). This matches player 1's deploy zones (also mirrored in y
only) and its tower order (left princess first), so left stays left for both players and a
unit deployed at a zone appears in the same lane and depth of its owner's view
(`tests/test_board.py`, `TestMirroredObservation`).

## Actions

`NUM_ACTIONS = 53 = 1 + 4 hand slots × 13 deploy zones`.

| Action | Meaning |
|---|---|
| 0 | noop (always legal) |
| 1–52 | play the card in hand slot `s` at zone `z`, where `s, z = divmod(action − 1, 13)` |

So actions 1–13 are slot 0 at zones 0–12, actions 14–26 are slot 1, and so on. For
the agent each **deploy zone** is a placement point (ADR-0002), fixed except for the two
push zones; humans in the web client place at continuous coordinates. Player 0 coordinates; for player 1
the y coordinate is mirrored (`y → 32 − y`):

| Zone | Name | (x, y) | Accepts | Purpose |
|---|---|---|---|---|
| 0 | back-L | (7.5, 0.5) | troops | behind the king tower: slow buildup (a Giant push); walks to the left Bridge lane |
| 1 | back-R | (10.5, 0.5) | troops | walks to the right Bridge lane |
| 2 | mid-L | (3.5, 10.5) | troops, buildings | standard lane play, just in front of the princess tower |
| 3 | mid-R | (14.5, 10.5) | troops, buildings | |
| 4 | bridge-L | (3.5, 14.0) | troops, buildings | bridge pressure |
| 5 | bridge-R | (14.5, 14.0) | troops, buildings | |
| 6 | pull-L | (8.0, 11.0) | troops, buildings | **pull zone** for building-targeters |
| 7 | pull-R | (10.0, 11.0) | troops, buildings | |
| 8 | spell-tower-L | (3.5, 24.0) | spells | **spell zone** on the left enemy princess tower |
| 9 | spell-tower-R | (14.5, 24.0) | spells | right enemy princess tower |
| 10 | spell-king | (9.0, 29.0) | spells | enemy king tower |
| 11 | spell-push-L | state-dependent | spells | **push zone**: the best enemy cluster in the left lane, wherever it stands |
| 12 | spell-push-R | state-dependent | spells | the same in the right lane |

The push zones are the only zones whose target moves and whose legality depends on the
board (`tests/test_push_zones.py`). On the tick of the play a push zone aims at the
position of an enemy unit or building in its lane (lane by x against the centre line
x = 9, the same lane for both players) that scores highest. A candidate's score is
Σ min(spell damage, current HP) over every enemy unit and building inside the spell's
radius, plus min(Crown tower damage, HP) for each enemy princess tower inside it. Ties
go to the unit placed earliest, and the aim is clamped into the spell area. A push zone
is legal only while its lane holds at least one enemy unit or building: towers alone do
not unmask it, so finishing a tower stays on zones 8–10. Push zones accept spells only.

### Arena geometry (env-v1.1)

Pinned by `tests/test_arena_geometry.py` (ADR-0005):

- King towers are 4×4 (collision radius 2.0) at y = 3.0 (player 0) and y = 29.0
  (player 1), leaving one free tile behind them; their activation and targeting are
  unchanged.
- Troops and buildings may be deployed from `BACK_EDGE_MARGIN` = 0.5 tiles off the back
  edge to the river bank (player 0: y ≥ 0.5, player 1: y ≤ 31.5), so behind the king;
  spells reach the same 0.5 … 31.5 span. Humans and agents share these bounds, except for
  the pocket below.
- Pocket (ADR-0006, `tests/test_pocket_zone.py`): once an enemy princess tower is down, its
  destroyer may also deploy troops and buildings on the enemy side of that lane, 4 tiles
  deep from the bank and 9 wide. The server sends the rectangles in each snapshot
  (`pockets`). **Humans only**: the action mask and the 13 zones are unchanged, so agents
  never play there.
- Bridge lanes are at x = 3.5 and 14.5, in line with the princess towers, so the arena is
  mirror-symmetric about x = 9. A troop behind its king walks to the lane on its side of
  the centre line.

The zones are the ADR-0002 layout on these geometry constants: indices 0–12 are pinned
by `tests/test_board.py` (`TestDeployZones`). "Troops" are ground and air units; the back
zones refuse buildings because a Cannon behind the king would be wasted. No zone starts
a unit inside a tower, a Giant in a back zone included
(`tests/test_arena_geometry.py`). Spell-king follows the enemy king tower.

The pull-zone placement is derived in ADR-0002: a Cannon at pull-L is closer to an
enemy Hog Rider leaving the left bridge than the princess tower is, and the pull zones
sit symmetrically about x = 9, so both lanes have the same margin
(`tests/test_scenarios.py`, both lanes).

### Targeting

Pinned by `tests/test_target_lock.py`. A troop or tower that is attacking a target keeps it
until the target dies or leaves its range; a closer enemy appearing does not switch its
aim. A unit with no target, or one that just lost its target, picks the closest valid
enemy within sight range (towers: within range), ties going to the earlier unit on the
board. A unit still walking toward its target re-picks the closest each tick, so the lock
only forms once it is in range. The King tower activation rules are unchanged.

### Action masking

`Board.valid_action_mask(player)` returns a 53-element boolean mask. Noop is always
legal; play `(s, z)` is legal when the player can afford the card in slot `s` and
zone `z` accepts that card type (troops: zones 0–7, buildings: zones 2–7, spells: zones
8–12) and, for a push zone, its lane holds an enemy unit or building. With the Deck a
troop (Baby Dragon included) has 8 legal zones, the Cannon 6 and each of Fireball and
Arrows 3, or 5 with enemies in both lanes, so at most 33 of the 53 actions are legal at once.

The mask is applied during training (each env serves its own `action_masks()`), during
evaluation and in-game inference (`model.predict(..., action_masks=mask)`).
MaskablePPO sets the logits of masked actions to a large negative value before the
softmax, so they get zero probability, contribute nothing to the policy gradient, and
are excluded from the entropy. The agent therefore never spends samples on illegal
plays.

## Reward

Outcome-driven, with potential-based shaping (Ng et al. 1999). It is computed per
decision step by `cr_rl.game.rewards.step_reward`, from player 0's perspective:

```
r = λ_win   · outcome                              (last step only: win +1, loss −1, draw 0)
  + λ_crown · (γ·Φ_crown(s′) − Φ_crown(s))         Φ_crown = my Crowns − enemy Crowns
  + λ_hp    · (γ·Φ_hp(s′)    − Φ_hp(s))            Φ_hp = (my tower HP − enemy tower HP) / MAX_TOWER_HP_PER_PLAYER
```

with `MAX_TOWER_HP_PER_PLAYER = 2 × 3052 + 4824 = 10928` (one player's total tower HP).
`s` is the state at the Decision point where the step started and `s′` the state where
it ended, so a step spanning several ticks pays the net change.

- **Outcome.** Follows the match rules: more Crowns at the end of Regulation, the first
  Crown in Overtime, a king tower kill, then the Tiebreak (weakest standing tower); a
  draw is 0. The outcome is paid on the step that ends the match.
- **γ.** PPO's γ, applied per decision step. `ClashRoyaleEnv(gamma=...)` takes it;
  `train()` passes its own `gamma` on, default `DEFAULT_GAMMA = 0.995`.
- **Φ(terminal) = 0.** The last step therefore pays `−Φ(s)` of shaping on top of the
  outcome. With γ = 1 the shaping terms telescope to zero over a match (Φ at the start is
  0, as both players have full towers), so the undiscounted return equals the outcome and
  shaping cannot change the optimal policy.
- **Weights.** `RewardWeights(win=1.0, crown=0.2, hp=0.5)`, passed as
  `ClashRoyaleEnv(reward_weights=...)`. `crown = hp = 0` is the outcome-only reward of
  the no-shaping ablation; no code changes are needed.
- **Symmetry.** Every term is antisymmetric between the players, so player 0 and
  player 1 get negated rewards for the same transition.
- **Removed.** Troop trade value, the kill/loss delta and the elixir-leak penalty: the
  potentials already give a dense signal and those terms are not potential-based.
  Ablations are judged by win rate, not shaped return.

The Coach's blunder and Key moment thresholds were tuned to the old tower-HP reward
scale and are retuned once a v2 model exists.

## Network architecture

`MlpPolicy` of MaskablePPO with `net_arch = dict(pi=[256, 256], vf=[256, 256])`:
separate policy and value networks with no shared layers, Tanh activations, and
orthogonal initialization (SB3 defaults).

```mermaid
flowchart LR
    obs["Observation<br/>279"] --> p1["Linear 279→256<br/>Tanh"]
    obs --> v1["Linear 279→256<br/>Tanh"]
    subgraph policy["Policy network (actor)"]
        p1 --> p2["Linear 256→256<br/>Tanh"] --> logits["Linear 256→53<br/>action logits"]
    end
    subgraph value["Value network (critic)"]
        v1 --> v2["Linear 256→256<br/>Tanh"] --> val["Linear 256→1<br/>V(s)"]
    end
    mask["Action mask<br/>53"] --> dist
    logits --> dist["Masked categorical<br/>π(a | s)"]
```

| Part | Parameters |
|---|---|
| Policy MLP (279→256→256) | 137,472 |
| Value MLP (279→256→256) | 137,472 |
| Action head (256→53) | 13,621 |
| Value head (256→1) | 257 |
| **Total** | **288,822** |

A flat MLP fits the problem size: the observation is small and fixed-length, and the
network is small enough to train on a laptop CPU (ADR-0002 keeps it for v2).
Separate networks keep the value loss from distorting the policy features, which
matters because the value target, the net tower damage, is noisy and mostly zero.

## PPO hyperparameters

Values are the defaults of `train()` in `src/cr_rl/training/train.py`; the v1 model
was trained with the same values.

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 3e-4, constant | Standard Adam step size for PPO (SB3 default); the sweep tests 1e-4 and 1e-3. |
| `n_envs` | 12 | Twelve environments, one process each (`SubprocVecEnv`), use the 14-core CPU and decorrelate the samples in a rollout; the 8 / 2 / 2 opponent mix of phase 2 needs 12. |
| `n_steps` (per env) | `max(2048 // n_envs, 256)` = 256 | The floor of 256 steps per environment keeps each segment longer than the discount horizon (3072 transitions per rollout with 12 envs). |
| `batch_size` | 256 | 12 minibatches per epoch over the 3072-transition rollout (256 steps × 12 envs): enough gradient steps per rollout while each minibatch still spans several environments and game phases. |
| `n_epochs` | 10 | SB3 default. Simulation steps are the expensive part, so each rollout is reused for 10 passes; the clipped objective keeps that reuse from moving the policy too far. |
| `gamma` | 0.995 | Applied per decision step. A play pays off only when units reach a tower: a Giant from back-L walks about 15 s (150 ticks, roughly 50 steps while the agent holds noop for 3 ticks) to the enemy princess tower. γ = 0.995 gives an effective horizon of 1/(1−γ) = 200 steps and still weights that payoff by 0.995^50 ≈ 0.78; the 0.99 default would weight it by 0.99^50 ≈ 0.61, and the match outcome even less. |
| `gae_lambda` | 0.95 | SB3 default. Advantages still credit rewards some tens of decision steps after the action (λγ = 0.945 per step) while damping the variance of the return, which is dominated by the sparse match outcome. |
| `clip_range` | 0.2 | SB3 default. Limits each update to a ±20% change in action probabilities, which keeps the rare card-play actions from being suppressed by one bad rollout. |
| `ent_coef` | 0.02 | SB3 uses 0. Noop is legal on every tick and right most of the time, so without an entropy bonus the policy tends to collapse to waiting early; 0.02 keeps card plays explored. The sweep tests 0.005, 0.01 and 0.05. |
| `vf_coef` | 0.5 | SB3 default. With separate networks it only scales the value loss against the policy loss in the shared Adam step. |
| `max_grad_norm` | 0.5 | SB3 default. Clips the rare large gradients from steps where a tower falls and the reward spikes. |
| `net_arch` | pi [256, 256], vf [256, 256] | See [Network architecture](#network-architecture); the sweep tests 128 and 512 units. |
| `total_timesteps` | 1,000,000 | Decision steps, not ticks: about 5,000 to 38,000 matches, depending on how often the policy waits (26 steps per match for random plays, 210 when it waits 90% of the time); small enough that several variants fit a laptop budget. |
| `seed` | 42 | The run seed; every episode's seed is derived from it (see [Training setup](#training-setup)). |

## Training setup

`cr-rl-train` (`train()`):

- **Environments:** 12 `ClashRoyaleEnv`s in a `SubprocVecEnv`, one process each. Action
  masks come from the env's own `action_masks()`; there is no `ActionMasker` wrapper.
- **Training phases** (`cr_rl/training/phases.py`): the step budget is split into phases,
  each with its own Training opponents, assigned by env index and fixed within the phase.

  | Training phase | Steps | Training opponents (12 envs) |
  |---|---|---|
  | Warm-up | 0–20% | 12 `RandomAgent` |
  | Scripted mix | 20–60% | 8 `LogicAgent`, 2 ε-`LogicAgent`, 2 `RandomAgent` |
  | Self-play mix | 60–100% | 4 Past self, 6 `LogicAgent`, 1 ε-`LogicAgent`, 1 `RandomAgent` |

  `--layout` (`train(layout=...)`, one setting) selects the arrangement:
  `full` (the table), `no-curriculum` (no warm-up: scripted mix 0–60%, self-play mix
  60–100%) or `no-self-play` (warm-up 0–20%, scripted mix 20–100%); `PHASE_LAYOUTS` holds
  them. The two non-default layouts are the curriculum and self-play Ablation arms.
- **Past-self pool** (`cr_rl/training/self_play.py`): holds 3 Past selves, checkpoint files
  seeded at the start of the self-play phase from the last 3 selection checkpoints (the
  model at that point is evaluated first, so it is among them). Every 100,000 steps
  (`POOL_UPDATE_INTERVAL`, `pool_interval`) the current model is a candidate: it saves
  itself to `models/pool/`, plays 50 matches (`ADMISSION_MATCHES`) against the pool, spread
  over the members in rotation, and replaces the oldest Past self only if it wins at least
  45% (`ADMISSION_WIN_RATE`; draws do not count). Past selves play deterministically as
  player 1 at Decision points. The 4 self-play envs take pool members in rotation and the
  envs are rebuilt after every check, each Past self loaded from its checkpoint file in the
  env's own process. The admission seeds derive from the run seed and the step
  (`admission_seeds()`), so the decisions follow from the run seed and the checkpoint files;
  they are logged to `models/pool.json`.
- **Checkpoint selection:** `SelectionCallback` (`cr_rl/training/selection.py`) plays the
  current policy deterministically vs `LogicAgent` on the 100 Selection seeds
  (`SELECTION_SEEDS`, `1_000_000 + i`, the same every round) every 50,000 steps
  (`SELECTION_INTERVAL`) and once on the final model. Each evaluated checkpoint is kept in
  `models/checkpoints/ppo_cr_<step>_steps.zip`; its win rate is logged with its 95% Wilson
  interval (`selection/win_rate` in TensorBoard) and the list of `CheckpointResult(step,
  wins, episodes, path)` is written to `models/selection.json` (`load_selection()`). The
  highest win rate becomes the run's model, a tie going to the later checkpoint
  (`select_checkpoint()`): it is copied to `models/best_model.zip` and, for the main run, to
  `models/ppo_cr_best.zip`, the path the game loads by default. Shaped reward plays no part.
  Selection seeds are never used for reported numbers.
- **Final model:** saved to `models/ppo_cr_final.zip`.
- **Logging:** TensorBoard in `models/tb/`.
  - `rollout/win_rate_vs_<opponent>`: player 0's win rate over the last 200 decisive
    matches (draws skipped) against each Training opponent type separately: `random`,
    `logic`, `logic-eps` and, once self-play exists, `past-self`. Each env reports its
    opponent in `info["opponent"]`. Logged once at least 5,000 env steps have passed since
    the previous log, so the cadence does not depend on `--n-envs`. A pooled rate would
    mix opponents of very different strength.
  - Update-stability checks: watch SB3's `train/approx_kl` and `train/clip_fraction`.
    Approximate KL that regularly exceeds about 0.02, or a clip fraction that stays
    well above 0.2, means the 10 epochs per rollout move the policy too far. No
    `target_kl` is set; the two metrics are only monitored, and `target_kl` or a lower
    learning rate is the remedy if they drift.
- **CLI:** `--timesteps`, `--n-envs`, `--seed`, `--lr`, `--layout`,
  `--resume`, and
  `--eval-only MODEL.zip --eval-episodes N`.

**Training opponents.** `ClashRoyaleEnv(opponent=...)` takes `"logic"`, `"logic-eps"`,
`"random"`, a frozen `MaskablePPO` policy, or any object with `choose_action(board,
player)` and `reset()`.

- *ε-`LogicAgent`* (`EpsilonLogicAgent`): at each tick it would act, with probability
  ε = 0.2 it plays a random legal action (noop included) instead of the rule-based
  choice. It draws from the Board's seeded RNG, so it is reproducible under a fixed
  seed but differs from `LogicAgent` and across seeds.
- *Frozen policy* (a Past self, see the Past-self pool above): a loaded `MaskablePPO` plays as
  player 1 through the fixed P1 view (observation mirrored in y only) and its action
  mask, and, like the Session RL opponent, is asked only at Decision points
  (`ScheduledAgent`).

**Seeding and determinism.** The simulation is deterministic given the board seed, which
shuffles both players' starting card order and seeds the Board RNG (ε-`LogicAgent`,
`RandomAgent`). An env built with a run seed gives every episode its own seed,
`derive_episode_seed(run_seed, env_index, episode)` (a `numpy.random.SeedSequence` of
the three), so consecutive episodes see different openings and a whole run is
reproducible from its run seed. `reset(seed=...)` overrides it, which evaluation uses for
fixed seeds; `reset()` returns the episode's seed as `info["seed"]`. Envs built
later in a run continue the env indices of the earlier ones, so no seed is replayed.

**Experiments** (`src/cr_rl/experiments/`, results under `experiment_runs/`):

- `cr-rl-sweep`: one training run per configuration (`baseline`, `low_lr`,
  `high_lr`, `high_entropy`, `mid_entropy` (`ent_coef` 0.01), `low_entropy`, `gamma_99`, `big_net`, `small_net`), each
  evaluated vs the rule-based bot.

No sweep results are committed yet.

### Evaluation

Every evaluation plays its matches through `play_matches()` (`cr_rl/training/matches.py`): a policy as player 0
against any Training opponent on an explicit seed list, returning each match's seed, winner, return and
length in decision steps. `wilson_interval()` gives the 95% Wilson interval of a win count.

`evaluate_model()` (`cr-rl-evaluate`, `cr-rl-train --eval-only`; the reported match length is
in agent steps) plays N matches as
player 0 vs the rule-based bot with the deterministic policy and seeds
`seed, seed + 1, …, seed + N − 1`, and reports wins, losses, draws, win rate, mean
return and mean match length.

Checkpoint selection during training uses `play_matches()` on the Selection seeds (see
[Training setup](#training-setup)).

The protocol for the v2 model (ADR-0002, amended by ADR-0003): fixed seeds,
200 matches vs the rule-based bot and 200 vs the random bot, reporting win rate, mean
return and match length. The v1 model is not part of that comparison.

## Results

### v1 model (historical)

The **v1 model** was trained on the pre-env-v1 game (commit `de8d9f4`, June 2026)
with the hyperparameters above for about 1,000,000 steps (SB3 finishes the last
2048-step rollout, so roughly 1,001,472). It cannot run in env-v1 or in the
current 45-action layout (ADR-0003). It is kept as a historical result, not as a
comparison point for v2.

Differences from the current game and layout:

- 3 deploy zones per player, (3, 6), (9, 6) and (14, 6) for player 0, giving 13
  actions. There were no dedicated building or spell zones: `play_card` moved a
  building to y = 11 (5 tiles from the river) and cast a spell at the zone mirrored
  onto the enemy side, e.g. (3, 26).
- The mask filtered by elixir only: every zone was legal for any affordable card.
- Units walked L-shaped paths to the bridges; diagonal bridge-bank pathing came later
  (ADR-0001), which changed the dynamics the v1 model was trained on.
- The player 0 observation (153 features: the 6-card layout without a next card) and
  the reward were the same as then; the player 1 view was not mirrored yet.

Evaluation recorded in [`docs/thesis/ocena.ipynb`](thesis/ocena.ipynb): 30 matches as
player 0 vs the rule-based bot, seeds 0–29, deterministic policy.

| Agent | Wins | Losses | Draws | Win rate | Mean return | Mean length |
|---|---|---|---|---|---|---|
| v1 model, final weights (1M steps) | 21 | 9 | 0 | **70.0%** | 0.065 | 1801 |
| v1 model, eval-callback best (650k steps) | 2 | 28 | 0 | 6.7% | −0.201 | 1794 |
| `RLAgent` without weights (random legal action on 50% of ticks; not the random bot) | 14 | 16 | 0 | 46.7% | −0.056 | 1801 |
| Rule-based bot vs random bot (50 matches) | 50 | 0 | 0 | 100.0% | — | 1051 |

How to read the result:

- The final weights' mean length of 1801 is the maximum, so every one of its 30
  matches ran to the time limit: all 21 wins are total-tower-HP wins, none a
  king-tower win.
- `RLAgent` without weights, playing random legal actions, already wins 46.7% against
  the rule-based bot, so 70% is a moderate improvement over chance-level play, from a
  small sample (30 matches).
- The training-time win rate (`rollout/win_rate_vs_logic`) ended at 25.5% (peak
  28%). It measures the stochastic policy on the 8 fixed training seeds, while the 70%
  is the deterministic policy on 30 other seeds.
- The eval-callback "best" model performed far worse than the final weights because
  every one of its 50 evaluation episodes was the same match (see
  [Known limitations](#known-limitations)), so "best" was picked on a single match.

The model files currently in `models/` come from a later run on the same v1 layout
(2,000,896 steps, August 2026) that has no recorded evaluation. They are not the
weights behind the 70% result.

### v2 model

Not trained yet. It is trained on env-v1 and evaluated against the rule-based and
random bots under the protocol above.

## Known limitations

Facts about the current code that the v2 work should address:

- **No pocket for agents.** After an enemy princess tower falls, humans can deploy in a
  4 × 9 pocket on that side (ADR-0006); agents and the rule-based bot cannot, because the
  action layout has no zone there.
- **Partial observation.** No unit cooldown or deploy state, and opponent elixir is
  visible. The opponent's hand is unknown until its 4th play (then it follows from the
  last 4 plays).

## Change log

| Date | Change |
|---|---|
| 2026-10-17 | First version: documents the 45-action / 11-zone layout and records the v1 model as a historical result. |
| 2026-10-17 | Card stats moved to `card_stats.json` at tournament standard (level 11); tower HP, and so `MAX_TOWER_HP_PER_PLAYER`, changed from 15880 to 10928. |
| 2026-10-17 | Deck of 8 (Knight, Giant, Hog Rider, Musketeer, Cannon, Fireball, Arrows, Baby Dragon) with air units and splash (#19): `OBS_DIM` 153 → 169 (hand one-hot 4×8, new next-card one-hot, card IDs in Deck order), action space unchanged at 45. v1 and 153-dim checkpoints are ignored at load. |
| 2026-10-17 | The win rule counts crowns instead of comparing total tower HP at the time limit; the reward is unchanged. |
| 2026-10-17 | env-v1 playtest ([checklist](playtest-env-v1.md)): troops and buildings can no longer be placed in the river water (the deploy region ends at the bank, `RIVER_Y ∓ 1.2`); the 11 deploy zones, the 45 actions and the 169-dim observation are unchanged and pinned by `tests/test_env_layout.py`. |
| 2026-10-21 | env-v1.1 arena geometry (ADR-0005, #35): king towers move to y = 3.0 / 29.0, deploy and spell areas reach y = 0.5 / 31.5, bridge lanes move to x = 3.5 / 14.5; the spell-king zone follows the king to (9.0, 29.0). Zone count, actions and observation are unchanged. |
| 2026-10-21 | Decision points (#36, ADR-0002): the agent is asked only when a card play is legal and no noop hold runs; noop holds 3 ticks (configurable); skipped-tick rewards are summed into the step; the Session RL opponent follows the same schedule. Observation and action layout unchanged. |
| 2026-10-21 | Target locking (#48): troops and towers keep their target until it dies or leaves range instead of re-picking the closest every tick. Observation and action layout unchanged. |
| 2026-10-21 | Pocket deploy zone (#50, ADR-0006): a fallen enemy princess tower opens a 4 × 9 pocket for its destroyer, humans only; the snapshot carries `pockets`. Action mask, zones and observation are unchanged. |
| 2026-10-23 | Zone layout v2 and the P1 lane fix (#38, ADR-0002): zones 0–10 keep their indices on the new coordinates (back zones behind the king, troops only; mid at the princess towers; centred pull zones). The player 1 observation mirrors only y (x unchanged), fixing the lane mismatch. 45 actions and 169 dims unchanged, but a model trained on the old zones or the 180° view is not valid. |
| 2026-10-23 | Outcome reward with potential shaping (#41, ADR-0002): win/loss/draw plus Crown and tower-HP shaping per decision step, all weights configurable; the tower-HP-only per-tick reward is removed and `StepResult` no longer carries a reward. Observation and action layout unchanged. |
| 2026-10-23 | Push Spell zones (#39, ADR-0002): zones 11 and 12 (spell-push-L / -R) aim a spell at the best enemy cluster in a lane and are legal only while that lane holds an enemy unit or building. `NUM_ZONES` 11 → 13, `NUM_ACTIONS` 45 → 53, `divmod(action − 1, 13)`; the observation is unchanged. A model trained on the 45-action layout is not valid. |
| 2026-10-23 | Observation v2 (#40, ADR-0002): `OBS_DIM` 169 → 278. The 24 distance-sorted unit slots are replaced by a permutation-invariant grid (2 owners × 2 lanes × 8 depth bins × 6 unit cards, summed HP / max HP capped at 1); an elixir-advantage feature and the opponent's last 4 plays (from the Board's per-player play history) are added. Actions are unchanged. v1 and 169-dim checkpoints are ignored at load. |
| 2026-10-23 | Training opponents, curriculum and seeds (#42, ADR-0002): `"logic-eps"` (ε-`LogicAgent`, ε = 0.2) and frozen-policy opponents; `cr-rl-train` runs 12 `SubprocVecEnv` envs with env-side masks, phase 1 vs `RandomAgent` then phase 2 with 8 / 2 / 2 `LogicAgent` / ε-`LogicAgent` / `RandomAgent` (`--no-curriculum` starts in phase 2); every episode gets a seed derived from the run seed, env index and episode counter, which removes the fixed-seed limitation. The separate `cr-rl-curriculum` experiment is removed. Observation and action layout unchanged. |
| 2026-10-23 | Overtime progress (#59, ADR-0002): one feature appended as the last observation element, `clip((time − 180) / 120, 0, 1)`, so the agent can tell how much Overtime is left; `OBS_DIM` 278 → 279, match time keeps its meaning. 278-dim checkpoints are ignored at load. |
| 2026-10-23 | Checkpoint selection (#62, ADR-0002): a run picks its model by win rate vs `LogicAgent` on 100 fixed Selection seeds every 50k steps and on the final model, ties to the later checkpoint; every evaluated checkpoint and its win rate are kept (`models/checkpoints/`, `models/selection.json`). The shaped-reward `MaskableEvalCallback` and the 100k `CheckpointCallback` are removed. Observation and action layout unchanged. |
| 2026-10-23 | Self-play Training phase (#64, ADR-0002): a run has three Training phases (warm-up, scripted mix, self-play mix) selected by `--layout full / no-curriculum / no-self-play`, which replaces `--no-curriculum` and `--phase1-fraction`. The self-play mix trains against a pool of 3 Past selves seeded from the last 3 selection checkpoints; every 100k steps the current model joins by winning at least 45% of 50 matches against the pool. Decisions are logged to `models/pool.json`. Observation and action layout unchanged. |
| 2026-10-23 | Documentation fix (#57): minibatches per epoch 8 → 12 (3072-transition rollout) and 1M steps ≈ 5,000–38,000 matches instead of 555, both checked against the code. ADR-0002 records the training-review decisions; they are not implemented yet. |
