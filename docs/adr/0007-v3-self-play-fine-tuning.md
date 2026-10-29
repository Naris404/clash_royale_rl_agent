# v3: self-play fine-tuning with Exploiters, selected by a reference pool

Status: accepted (2026-10-26). Amends ADR-0002 (training opponents and checkpoint selection).

The v2 runs showed that self-play makes the agent stronger, but ADR-0002's selection protocol
throws that away. Selection scores checkpoints only against LogicAgent, and that score
falls during the self-play phase, so every final-configuration run selected a checkpoint
from before self-play. The Final models (after self-play) beat those Selected models in
75–84% of Report-seed matches. A full-arm model also beat its no-self-play counterpart
64–84% head-to-head, yet the full arm scored lower against LogicAgent.

LogicAgent also taught the Shipped model a habit that no opponent punished. In every one
of the 200 Report seeds, the agent sends Arrows at a tower as soon as it reaches 3 elixir,
about 4 times per match, while the enemy holds about 8 elixir. Blocking only that play lowers
its win rate against LogicAgent (93.5% → 87.5%) and raises it against the v2 Final model
(23.0% → 28.5%). Neither difference is significant on its own.

The **v3 model** is therefore a v2 Final model fine-tuned further, not a new run from
scratch:

- **Two generations per seed**, from each of the 3 v2 Final models. Each generation first
  trains an **Exploiter**: a copy of the current main agent trained only against that frozen
  agent, until it wins at least 70% of matches or reaches 300k steps (AlphaStar's main
  exploiters, Vinyals et al. 2019). The main agent then fine-tunes for 500k steps with the
  Exploiter in its pool.
- **Training opponents (12 envs):** 10 from the pool (Past selves plus Exploiters), sampled by
  PFSP with hardest-first weighting (1 − win rate)², the win rates taken from the pool
  admission check; 2 LogicAgent as the only scripted anchor.
- **Unchanged from v2:** reward weights and PPO hyperparameters, so the only difference
  from v2 is who the agent trains against.
- **Selection:** after training, the generation-2 checkpoints are scored every 100k steps by
  their average win rate against a reference pool: LogicAgent, the v2 Shipped model and the
  3 v2 Final models. Selection uses the Selection seeds; reported numbers use the Report
  seeds.
- **Target:** about 90% or more against LogicAgent, and a head-to-head win over the v2 Shipped
  model. If v3 meets it, v3 becomes the Shipped model and the v2 Shipped model stays in
  the Model catalogue. If not, v2 is re-selected from its saved checkpoints with the
  reference pool and v3 is reported as a negative result.

## Considered options

- **Train longer from scratch:** rejected. More steps of the same mix would not raise the
  LogicAgent score, which falls during self-play, and the deadline allows one night of compute.
- **A stronger rule-based bot:** rejected. It would teach a new set of habits to exploit; an
  Exploiter finds weaknesses by RL without hand-written rules.
- **Pure self-play with no LogicAgent:** rejected. AlphaStar and Generals.io anchor
  self-play with a model trained on human games, which we lack; 2 LogicAgent envs are our
  anchor.
- **GPU training:** rejected. A PPO update on the 256×256 MLP is about 7% of the wall-clock
  time; the CPU game simulation is the rest.

## Outcome (2026-10-29)

v3 missed the target, and the fallback rule was not applied as written.

- The v3 run with the best selection score (gen2-seed2) wins 73.5% [67.0, 79.1] against LogicAgent on the
  Report seeds and 78.0% [73.7, 81.8] head-to-head against the v2 Shipped model. It meets the
  head-to-head half of the target, not the LogicAgent half.
- The re-selected v2 fallback (#75) misses the target too: the reference pool picks a checkpoint after
  self-play in every v2 run (71.5–78.0% against LogicAgent). The fallback is weaker against LogicAgent than
  the v2 Shipped model (93.5%), and weaker against learned models than v3, so shipping it would make
  the default opponent worse on both measures.
- v3 is also a worse model for the Coach. Graded by v3, a LogicAgent proxy's moves no longer
  track the outcome (`cr-rl-coach-validate`, 50 matches: correlation −0.17 and win/loss pair accuracy 40%,
  against +0.28 and 66% with the v2 Shipped model). v3's value function rates most proxy moves as
  gains (median ΔV +0.13).

The **v2 Shipped model therefore stays the Shipped model**: the default RL Opponent and the Coach's
model. v3 is reported as a negative result against the target. Because it is the strongest model
against learned opponents, it joins the Model catalogue as its own entry (`v3`,
`models/ppo_cr_v3.zip`) so a player can choose it.
