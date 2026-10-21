# RL v2 architecture review against published game-RL research

Research note, 2026-10-18. It reviews the agent design planned in issue #6
("05: RL v2 implementation (no training)") and ADR-0002 against published work on
Clash Royale and similar games. It does not change any decision. Open questions are
listed under [Recommendation](#recommendation) for the ADR owner.

Vocabulary follows [`CONTEXT.md`](../../CONTEXT.md). Repo facts come from
[`docs/rl-agent.md`](../rl-agent.md), ADR-0002/0003/0004 and the code at commit `bd6fa04`.
Literature claims link to the paper or official docs they come from. Anything not
checked against a primary source is marked **(unverified)**.

## TL;DR verdict

- **#6 is a reasonable baseline, but not the best design available at this budget.**
  Its core is MaskablePPO with a flat `Discrete(45)` masked action and a small MLP. That
  matches what the closest published systems used, and they ran on similar or smaller
  budgets: the Heroic card-placement RTS agent (PPO, masked joint card→position head,
  small networks) [[Warchalski et al. 2020]](https://arxiv.org/abs/2002.06290), and
  Gym-μRTS (PPO + full invalid-action masking on one machine)
  [[Huang et al. 2021]](https://arxiv.org/abs/2105.13807). Nothing in the literature
  says a CNN, LSTM, transformer or autoregressive head is needed for a 45-action,
  single-deck problem.
- **#6 leaves out the three things the closest prior work found decisive:**
  1. **Decision frequency and no-op handling.** Heroic found that querying the policy
     every 0.2–1 s "can result in poor strategies", and fixed it by returning a state only
     when a non-trivial action is available. Our agent decides on every 0.1 s tick.
  2. **A reward aligned with winning.** The current reward is net tower HP with no
     win/crown term, while env-v1 decides matches by crowns. Every comparable system
     includes the match outcome (AlphaStar, OpenAI Five, Gym-μRTS, Generals.io, Heroic).
     Shaping should be potential-based so it cannot change which policy is optimal
     [[Ng et al. 1999]](https://mlanthology.org/icml/1999/ng1999icml-policy/).
  3. **Opponent diversity.** A Random → Logic sequence still ends on one deterministic
     opponent, which is how v1 overfit. Gym-μRTS gained from mixing several bots across
     its parallel envs, and Heroic and Generals.io gained from small self-play pools.
- **The observation has fixable encoding problems** that matter more than the choice of
  network: unit card is an ordinal scalar (`card_id / 7`), the unit list is a
  distance-sorted slot list rather than a permutation-invariant summary, and the
  opponent's hand is treated as hidden even though the Cycle makes it fully deducible
  from the opponent's last 4 plays.
- **What to do:** keep MaskablePPO, the flat 45-action head and the MLP. Add decision-point
  stepping, a terminal outcome reward with potential-based shaping, better observation
  encoding and a mixed opponent pool. Defer the CNN encoder, LSTM, privileged critic and
  autoregressive heads. They are defensible, but cost more than they are likely to return
  on a CPU-only thesis budget.

## Our requirements

These are taken from the repo, not assumed:

| Requirement | Source in repo |
|---|---|
| Deterministic 18×32 simulation, 10 Hz ticks (`TICK_DT = 0.1`), match ≤ 3001 steps | `src/cr_rl/game/board.py`, `docs/rl-agent.md` |
| Two-player zero-sum, mirror match, one fixed 8-card Deck | ADR-0004 |
| Partial information: opponent hand and Cycle hidden; opponent elixir visible | `docs/rl-agent.md` § Observation |
| Flat `Box(-1, 1, (169,))` observation; `Discrete(45)` = noop + 4 hand slots × 11 deploy zones; action masks | `board.py` (`OBS_DIM`, `NUM_ACTIONS`, `valid_action_mask`) |
| Spell zones only on enemy towers and king (zones 8–10), so no defensive spells | `docs/rl-agent.md` § Actions |
| MaskablePPO (sb3-contrib), separate pi/vf MLPs `[256, 256]`, Tanh | `src/cr_rl/training/train.py` |
| CPU-only, 14 cores, ~12 envs, 1–3 h per run | ADR-0002 |
| Frozen env-v1: mechanics fixed; obs/action layout versioned and pinned by tests | ADR-0003, `tests/test_env_layout.py` |
| The coach reads `π(a|s)` and `V(s)` from the policy (`PolicyInspector`), maps actions 1:1 to (card, zone), and grades with thresholds tuned to the reward scale | `src/cr_rl/coach/inference.py`, `evaluator.py` |
| The RL policy must also play as player 1 (web opponent); the P1 view is rotated 180° while P1 zones mirror only y (the known lane mismatch) | `docs/rl-agent.md` § Known limitations |

## Literature survey

### Clash Royale and card-placement real-time games

**SEAT: "Playing Card-Based RTS Games with Deep Reinforcement Learning"**, Liu, Zheng,
Li, Bian, Song, IJCAI 2019, [PDF](https://www.ijcai.org/proceedings/2019/0631.pdf).
This is the only peer-reviewed RL paper on a Clash Royale simulator that we found.

- Observation: a 3-layer "state image" on a **32×18 grid**, the same size as our Board
  (layers for health, class and faction), plus hand-card features.
- Model: two conv layers on the image and two FC layers on the card features. They are
  concatenated into **DQN Q-values for 5 actions** (4 hand cards + "none").
- Position: chosen by an **untrained "attention" heuristic**, the argmax cell of the last
  conv layer on the enemy-unit image. Troop placement is restricted to the own half, and
  some cards have hard-coded positions.
- Reward is shaped: +2 per enemy unit killed, −3 per card played, +20/−30 per
  enemy/own tower destroyed.
- The agent decides every 15 game rounds (0.3 s). An "abandoning rate" ζ forces a no-op
  with probability ζ, and ζ = 0.5 learned faster and more stably than ζ = 0.
- Results: 90% / 90% / 70% against aggressive rule-based, defensive rule-based and
  decision-tree agents, from only **10 battles per pairing**.
- Lessons for us: a small conv net on the 32×18 grid is enough, and limiting how often
  the agent acts helps. The evaluation is too small to be strong evidence.

**"Deep RL Agent for a Real-Time Action Strategy Game"** (Heroic: Magic Duel),
Warchalski, Radojevic, Milosevic (Nordeus), 2020,
[arXiv:2002.06290](https://arxiv.org/abs/2002.06290). This is the closest analogue to
our setup: 1v1, real-time, cards cast onto lanes, a mana resource and a 4-card hand.

- Observation: spatial tensor **D=10 bins × L=3 lanes × 2U channels**, where each cell
  holds the **sum of health of own/enemy units of each type** in that bin. A non-spatial
  vector holds action cooldowns, castle HP and time.
- Network: Atari-net style, with one conv layer (32 filters of size 1×3) and FC 32,
  concatenated, then a 512-unit head.
- Action head: the policy outputs a distribution over action types and, **for each
  action type, its own distribution over (bin, lane)**. Invalid actions are masked and
  renormalized. This conditioned joint head beat an Atari-net head with independent
  x and y by **about 10% win rate**.
- Algorithm: PPO (clip 0.2, lr 3e-4 for the policy and 1e-3 for the value, γ = 0.99,
  λ = 0.97, KL early stop at 0.01), 60,000 samples per iteration (~3,000 matches), on
  12 processes and 2 Titan V GPUs.
- **No-op handling:** "naively triggering the network too often (every 0.2, 0.5 or 1
  second), even when no move is available, can result in poor strategies." They return
  a state only when a non-trivial action is available, and a no-op lasts 4 s. Matches
  took ~14 decisions without spells and ~25 with spells.
- Reward: only ±1 for the outcome.
- Opponents: training against the existing AI is fast but "fitting its policy against a
  particular type of opponent". Self-play with an **ensemble of n = 3 policies** and
  opponent sampling from the last (δv, v] iterations, δ = 0.5
  [[Bansal et al. 2018]](https://arxiv.org/abs/1710.03748). The best self-play agent won
  64.5% against the rule-based AI, 37.3% against the tree-search AI, and 50% against a
  top-5% human over 20 matches.
- Spells made the game "considerably more complex". Against the tree-search AI the spell
  agent needed **pretraining against the rule-based AI**, "since otherwise the reward
  signal is not strong enough".
- Memory: they used an MLP policy. Stacking past observations gave "no immediate
  benefit", and LSTM was left as future work.

**"Playing Non-Embedded Card-Based Games with Reinforcement Learning"** (KataCR),
Wu, Wan, Wang, Wan, Lan, 2025, [arXiv:2504.04783](https://arxiv.org/abs/2504.04783).

- Setup: real Clash Royale on a phone, with perception by YOLO and OCR.
- State: an **18×32 grid with a 15-dim feature per cell**, plus hand, elixir and time.
- Action: card slot + grid position.
- Learning: offline, from 105 expert episodes (113,981 frames) with StARformer or
  Decision Transformer, which the authors call "more akin to imitation learning".
- Only **4.01% of frames are action frames**. They predict a delay until the next action
  and resample action frames.
- Reward: tower-HP delta + tower-destroyed ±1 (king ×3) + king-activation ±0.1 +
  **elixir-overflow penalty of 0.05 per second**. This is close to #6's terms.
- Results: win rates against the built-in AI were **0–10%**. The authors say the agents
  "still cannot consistently defeat the built-in AI" and point to online RL as future
  work.
- Lessons for us: confirms the extreme no-op sparsity and the reward vocabulary. It is a
  negative result for offline imitation only.

**"Learning to Play Imperfect-Information Games by Imitating an Oracle Planner"**,
Boney, Ilin, Kannala, Seppänen, 2020, [arXiv:2012.12186](https://arxiv.org/abs/2012.12186).
This is Aalto University with Supercell. A full-information planner (fixed-depth search
with decoupled Thompson sampling, because naive MCTS did poorly in the combinatorial
action space) is distilled into a partial-observation follower on Clash Royale and
Pommerman. The abstract gives no quantitative results. The approach needs a forward
model, and our deterministic Board is one, but it is outside #6's scope.

**Community repositories (unverified).**
[Jaso1024/Real-Time-Strategy-RL-Clash-Royale](https://github.com/Jaso1024/Real-Time-Strategy-RL-Clash-Royale)
describes PPO with a factored card agent and two tile agents over autoencoded screen
images. Its README reports no results. We found no peer-reviewed 2023–2026 paper on
online PPO or self-play for Clash Royale.

### Large-scale game agents

**AlphaStar: "Grandmaster level in StarCraft II using multi-agent reinforcement learning"**,
Vinyals et al., Nature 575, 2019, doi:[10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z)
([author PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)).

- Encoders: a scalar MLP, an **entity transformer** over units and a **spatial ResNet**,
  joined by "scatter connections", feeding a **deep LSTM** core.
- Action heads: **autoregressive** (action type → delay → queued → selected units by
  pointer net → target unit → target point by deconv ResNet). The agent "decide[s] ahead
  of time when to observe next", 370 ms on average.
- Training: supervised pretraining on human replays, then RL with V-trace + TD(λ) +
  UPGO. Reward is ±1 for the outcome, undiscounted, plus human-statistic pseudo-rewards
  and a KL penalty toward the supervised policy.
- Value: during training it uses information from both players' perspectives.
- League: main agents with prioritized fictitious self-play (PFSP), main exploiters and
  league exploiters.
- Compute: per agent, 16,000 concurrent matches, 16 TPU-v3 actors and a 128-core TPU
  learner, trained for 44 days.

**OpenAI Five: "Dota 2 with Large Scale Deep Reinforcement Learning"**, Berner et al.
(OpenAI), 2019, [arXiv:1912.06680](https://arxiv.org/abs/1912.06680).

- Observation: unit-centred arrays (~1,200 categorical and 14,534 continuous values),
  with categorical data embedded.
- Network: a single-layer **4096-unit LSTM**, 159M parameters.
- Action: one primary action plus factored parameter heads (delay, unit selection,
  offset). The agent acts every 4th frame.
- Reward: a hand-designed table (win = 5, kills, gold, HP, towers…) that is "modeled
  loosely after potential-based shaping … though the guarantees therein do not apply".
  Rewards are made **zero-sum by subtracting the enemies' average**.
- Self-play: 80% against the latest policy, 20% against older ones.
- Horizon: increasing the GAE horizon from 180 s (γ = 0.9993) improved play.
- Compute: thousands of GPUs over 10 months.

**Honor of Kings 1v1: "Mastering Complex Control in MOBA Games with Deep Reinforcement Learning"**,
Ye et al. (Tencent), AAAI 2020, [arXiv:1912.09729](https://arxiv.org/abs/1912.09729).

- Encoders: conv on image features, FC on unit vectors with **max-pooling per unit
  type** (permutation-invariant), FC on game state, then an LSTM (1024 units, 16 steps).
- Action heads: decoupled multi-label heads, **target attention** for unit targets, and
  an action mask.
- Algorithm: dual-clip PPO (ε = 0.2, c = 3), γ = 0.997, λ = 0.95, with a reward table
  modelled on OpenAI Five's.
- Ablation against the base model: +action mask gave 50.5% win rate but converged in
  65 h instead of 80 h; +target attention 75%; +LSTM 73%; full model 90%.
- Compute: 48 P40 GPUs and 18,000 CPU cores per hero.

**Honor of Kings full game: "Towards Playing Full MOBA Games with Deep Reinforcement Learning"**,
Ye et al., NeurIPS 2020, [arXiv:2011.12692](https://arxiv.org/abs/2011.12692).

- **Curriculum self-play learning:** small teacher models are trained on fixed lineups
  and distilled into one student, which then continues training.
- **Multi-head value:** five reward groups (farming, KDA, damage, pushing, win/lose),
  each with its own value head.
- The **value function sees hidden information during training only**.

**PySC2 baselines: "StarCraft II: A New Challenge for Reinforcement Learning"**,
Vinyals et al., 2017, [arXiv:1708.04782](https://arxiv.org/abs/1708.04782).

- Categorical feature layers are one-hot encoded ("equivalent to … one-hot encoding in
  the channel dimension followed by a 1×1 convolution"). Numerical features are
  log-scaled.
- Atari-net samples spatial x and y independently. FullyConv keeps spatial resolution.
- In supervised policy prediction, top-5 screen-argument accuracy was 2.9% for Atari-net,
  18.5% for FullyConv and 22.1% for the autoregressive arFullyConv. The authors explain
  arFullyConv's edge by its knowing "which action identifier the argument will be used
  for".

### Affordable grid RTS and board-like games

**"Gym-μRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep RL"**,
Huang, Ontañón, Bamford, Grela, IEEE CoG 2021, [arXiv:2105.13807](https://arxiv.org/abs/2105.13807).

- Observation: grid with **27 one-hot feature planes**.
- Two action designs: per-unit (UAS) and per-cell (Gridnet), with PPO and action
  composition.
- **Full invalid-action masking** was one of two augmentations "essential for having an
  agent that even starts learning". Masking only the action type (PySC2-style) "has
  little impact".
- **Diverse opponents:** across 24 envs, 18 played CoacAI and 2 each played
  RandomBiasedAI, WorkerRush and LightRush. This gave "a rather significant performance
  boost" for Gridnet and a milder one for UAS.
- Network: IMPALA-CNN (residual) helped UAS, and an encoder-decoder helped Gridnet.
  Model size mattered less than these techniques.
- Shaped reward: ±10 win/loss plus resource, production and attack terms. The agents
  "sometimes overfit to the shaped rewards instead of sparse rewards".
- Self-play (appendix): naive self-play against the latest self "do[es] not perform
  particularly well", which suggests league-style methods are needed.
- Compute: the best agent reached a 91% cumulative win rate against past competition
  bots in about 60 h on 1 GPU and 3 vCPUs. Ablations used 4 seeds each.

**"Comparing Observation and Action Representations for Deep RL in μRTS"**, Huang &
Ontañón, AIIDE 2019 workshop, [arXiv:1910.12134](https://arxiv.org/abs/1910.12134). On
harvesting tasks, local (per-unit) representations beat global ones, but "this advantage
… does not necessarily hold in larger maps". This has limited relevance to us, since
our player controls no units after deployment.

**"Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning"**,
Straka & Schmid, 2025, [arXiv:2507.06825](https://arxiv.org/abs/2507.06825).

- Observation: grid feature planes plus a **hand-crafted "memory stack"** (revealed
  positions, explored cells, the last 7 moves of each side) instead of recurrence.
- Network: U-Net torso.
- Training: behaviour cloning from 16,320 filtered replays, then PPO self-play against a
  **pool of N = 3 recent predecessors**. A candidate enters the pool after reaching a 45%
  win rate against it.
- Reward: ±1 plus **potential-based shaping** with Φ = 0.3·land + 0.3·army + 0.4·castle
  log-ratios.
- Ablations: self-play beat BC 68.8%; shaping beat naive self-play 71.9%; the population
  beat shaping alone 70.2%.
- Compute: 36 h on one H100. The environment ran at ~3,500 FPS on a 12-core CPU with
  12 envs.
- Results: top 0.003% of human 1v1 players.
- A follow-up, "Superhuman AI for Generals.io Using Self-Play RL" (Straka & Schmid,
  [ICML 2026](https://icml.cc/virtual/2026/78760)), reports reaching #1 on the
  leaderboard *without* behaviour cloning, potential-based shaping or population
  self-play. Its architecture and compute were not checked **(unverified beyond the
  abstract)**.

**"Google Research Football: A Novel RL Environment"**, Kurach et al., AAAI 2020,
[arXiv:1907.11180](https://arxiv.org/abs/1907.11180). This is the cleanest published
**flat-vector vs spatial-grid** comparison. With IMPALA and the checkpoint reward on the
easy 11v11 task, average goal advantage was:

| Representation | 100M steps | 500M steps |
|---|---|---|
| Floats (115-dim) | 2.42 | 5.73 |
| SMM (4 binary 72×96 planes) | 5.94 | 9.75 |
| SMM stacked | 7.62 | 12.89 |

The shaped "checkpoint" reward sped up policy-gradient methods. The comparison is
informative, but it was run at 100–500M steps, roughly two orders of magnitude beyond our
budget.

### Card games (turn-based)

**DouZero**, Zha et al., ICML 2021, [arXiv:2106.06135](https://arxiv.org/abs/2106.06135).
Self-play Deep Monte-Carlo with action encoding, trained on one 4-GPU server in days,
ranked first on Botzone among 344 agents (abstract only; the architecture was not
checked). For Hearthstone and other collectible card games we did not find a primary
PPO source. Search hits (Cardsformer, ECAI 2023, and Legends of Code and Magic papers)
were not read **(unverified)**. Lux AI evidence lives in Kaggle write-ups, not papers,
and was not checked **(unverified)**. Turn-based card-game agents mainly inform action
encoding for variable legal-action sets. A fixed 45-action space does not need that.

### Methodology

- **Invalid action masking:** "A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms", Huang & Ontañón, FLAIRS 2022,
  [arXiv:2006.14171](https://arxiv.org/abs/2006.14171).
  - The masked gradient is a valid policy gradient.
  - Masking scales as the invalid-action share grows. A negative reward for invalid
    actions "fails to scale", and on 10×10 maps those agents spent 3.43% of training just
    finding the first reward, versus ~0.06% with masking.
  - "Naive" masking (only at sampling time) made the KL divergence explode.
  - MaskablePPO cites this paper
    ([sb3-contrib docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)).
- **"What Matters in On-Policy RL? A Large-Scale Empirical Study"**, Andrychowicz et al.,
  ICLR 2021, [arXiv:2006.05990](https://arxiv.org/abs/2006.05990). Separate policy and
  value networks were better on 4 of 5 environments, and tanh worked best. Its
  recommendations: initialize the last policy layer with 100× smaller weights; "always
  use observation normalization"; GAE with λ = 0.9; tune γ, which is "one of the most
  important hyperparameters"; "try frame skip". **Caveat:** the study used continuous
  control (MuJoCo) only.
- **"The 37 Implementation Details of PPO"**, Huang, Dossa, Raffin, Kanervisto, Wang,
  ICLR 2022 Blog Track,
  [link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
  - Orthogonal init with policy-output gain 0.01. SB3 does the same: `action_net: 0.01`
    in [`policies.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py).
  - Per-minibatch advantage normalization.
  - MultiDiscrete heads are treated as independent factors.
  - LSTM handling details; separate networks do better in simple environments.
- **"Action Space Shaping in Deep RL"**, Kanervisto, Scheller, Hautamäki, IEEE CoG 2020,
  [arXiv:2004.00980](https://arxiv.org/abs/2004.00980). Removing actions and discretizing
  continuous ones "can be crucial". Discrete vs MultiDiscrete made no significant
  difference except that MultiDiscrete scales better to many actions.
- **Potential-based shaping:** "Policy Invariance Under Reward Transformations",
  Ng, Harada, Russell, ICML 1999, pp. 278–287,
  [mlanthology](https://mlanthology.org/icml/1999/ng1999icml-policy/).
  - Adding F = γΦ(s′) − Φ(s) preserves optimal policies.
  - The abstract argues this is essentially the only general form that does, and
    attributes known shaping "bugs" to non-potential rewards.
  - The full PDF did not text-extract, so the theorem wording was checked only via the
    abstract.
- **Self-play opponent sampling:** "Emergent Complexity via Multi-Agent Competition",
  Bansal et al., ICLR 2018, [arXiv:1710.03748](https://arxiv.org/abs/1710.03748).
  Sampling opponents from history beat always using the latest one. The dense
  "exploration" reward was annealed to 0, and keeping it on was worse.
- **Evaluation statistics:** "Deep RL at the Edge of the Statistical Precipice",
  Agarwal et al., NeurIPS 2021, [arXiv:2108.13264](https://arxiv.org/abs/2108.13264).
  With few runs, report interval estimates (and IQM), not point estimates.
- **sb3-contrib constraints**
  ([MaskablePPO docs source](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.md)):
  - Action spaces: Discrete, MultiDiscrete and MultiBinary only.
  - **Recurrent policies: ❌.**
  - "In order to use `SubprocVecEnv` with `MaskablePPO`, you must implement the
    `action_masks` inside the environment (`ActionMasker` cannot be used)."
  - RecurrentPPO exists separately, but its docs do not mention masking.
  - SB3's `NatureCNN` uses an 8×8 stride-4 first conv, and `CombinedExtractor` only
    routes `uint8` image spaces to it
    ([`torch_layers.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py)).
    An 18×32 float grid therefore needs a custom features extractor.

## Comparison table

| System | Obs encoding | Action head | Algorithm | Opponents / curriculum | Reward | Memory | Compute | Relevance to us |
|---|---|---|---|---|---|---|---|---|
| **#6 / ADR-0002 (planned)** | Flat 169 + new groups → MLP | Flat masked `Discrete(45)` (slot×zone joint) | MaskablePPO | Random → Logic; self-play hook deferred | Tower-HP delta + toggled trade, leak, kill terms; **no win term** | None | 14-core CPU, ~12 envs, 1–3 h | — |
| SEAT (CR, IJCAI'19) | 3×32×18 grid CNN + card FC | DQN over 5 (card/none); position by untrained heuristic | DQN | Fixed rule/DT bots | Kills, card cost, towers | None | n/a | High (same game and grid); weak eval (10 games) |
| Heroic (Nordeus '20) | Lane×bin×unit-type HP-sum grid (conv 1×3) + flat | Per-action conditioned spatial softmax, masked | PPO | Rule AI, then 3-policy self-play ensemble; pretrain vs rule AI for spells | ±1 only | None (stacking: no gain) | 2 GPUs, 12 procs | **Highest**: closest game; low budget |
| KataCR (CR, '25) | 18×32×15 grid tokens (ViT patches) | Card + position + delay prediction | Offline (StARformer/DT) | Built-in AI eval only | Tower Δ, destroy, activation, elixir overflow | Transformer over time | n/a | Medium: reward terms; no-op sparsity (4%) |
| AlphaStar ('19) | Entity transformer + spatial ResNet + scalar | Autoregressive incl. delay, pointer net | SL → V-trace/TD(λ)/UPGO | League (PFSP, exploiters) | ±1 + human pseudo-rewards | Deep LSTM | Thousands of TPU-days | Low (scale); ideas: delay, privileged value |
| OpenAI Five ('19) | Unit-centred arrays, embeddings | Primary action + factored params; frameskip 4 | PPO (huge batch) | 80% latest self / 20% past | Shaped table, zero-sum, win = 5 | 4096 LSTM | Thousands of GPUs, 10 months | Low (scale); ideas: zero-sum shaping, horizon |
| HoK 1v1 (AAAI'20) | Conv image + max-pooled unit vectors + state | Decoupled heads, target attention, mask | Dual-clip PPO | Self-play | OpenAI-Five-like | LSTM 1024 | 18k CPU cores / hero | Low-medium: permutation-invariant pooling, mask |
| Gym-μRTS (CoG'21) | 27 one-hot planes, IMPALA-CNN | MultiDiscrete composition (UAS/Gridnet), full mask | PPO | **Diverse bot mix across envs**; naive self-play weak | ±10 + shaped | None | 1 GPU + 3 vCPU, ~60 h | **High**: affordable PPO + masking + opponent mix |
| Generals.io ('25) | Grid planes + memory stack, U-Net | Per-cell masked heads | BC → PPO self-play | Pool of 3, 45% gate | ±1 + **potential-based** | Hand-crafted memory | 1 H100, 36 h | High: PBRS, small pool, memory features |
| GRF ('19) | Floats vs SMM grid vs pixels | Discrete 19 | IMPALA/PPO/DQN | Built-in AI levels | Scoring vs checkpoint | Stacking | 100–500M steps | Medium: grid > flat, but at large budgets |
| PySC2 ('17) | One-hot feature layers; FullyConv | Independent vs autoregressive args | A3C | Built-in AI | Game score | FullyConv-LSTM | Large | Medium: one-hot categorical; conditioning matters |

## Evaluation of #6, point by point

**Algorithm: MaskablePPO. Keep.** Every affordable comparable system uses PPO with full
masking (Heroic, Gym-μRTS, Generals.io, HoK's PPO variant). The masking paper shows full
masks matter, and MaskablePPO masks during both sampling and the gradient, which avoids
the "naive" failure mode. IMPALA/V-trace and dual-clip PPO address distributed off-policy
lag, which we do not have. One implementation detail: per the sb3-contrib docs, the
SubprocVecEnv path must drop the `ActionMasker` wrapper `train.py` uses today. That is
fine here, because `ClashRoyaleEnv` already implements `action_masks()`.

**Action head: flat masked Discrete(45). Keep.** A flat softmax over slot×zone pairs
already *is* the fully conditioned joint distribution that Heroic found ~10% better than
independent heads, and that PySC2 found better as arFullyConv. Factored or autoregressive
heads only pay off when the joint space is too large to enumerate. Kanervisto et al.
found no significant Discrete vs MultiDiscrete difference except at scale, and
MultiDiscrete masking in sb3-contrib cannot express "zone valid *given* card" anyway. The
flat head also suits the coach best, since each probability is one (card, zone)
suggestion.

The real weakness is **expressivity**. Spell zones exist only on enemy towers and the
king, so Arrows or Fireball can never be used defensively or on a push at the bridge.
SEAT aimed spells at enemy-unit concentrations, and Heroic reported spells as the hard,
precision-dependent part. This is a deliberate ADR-0002 design and a limit on what the
agent can learn, not a bug.

**Network: flat MLP. Keep as the primary model, with caveats.** Grid encoders beat flat
vectors in GRF (SMM vs floats) and are the norm in SEAT, Heroic and Gym-μRTS, but those
results come from 10–100× our sample budgets. Heroic's "conv" is a 1×3 filter over
lane-bin cells, close to a hand-built binned summary. An MLP over a **binned,
permutation-invariant** encoding captures most of that benefit at MLP cost. Separate
pi/vf networks and tanh match Andrychowicz et al. A CNN is a valid ablation but not
necessary (see [Local measurements](#local-measurements)).

**Observation: enriched flat features. Partly right; three encoding gaps.**

1. **Unit card is encoded as `card_id / 7`**, an ordinal scalar for a categorical
   variable. PySC2, Gym-μRTS and OpenAI Five all one-hot or embed categoricals. An MLP
   has to carve Giant (1/7) from Hog Rider (2/7) out of a continuum.
2. **The unit list is 24 slots sorted by distance**, so a unit's slot index shifts
   whenever another unit spawns or dies. HoK max-pools per unit type, AlphaStar uses
   attention, and Heroic sums HP per type per lane bin; all of these are
   permutation-invariant. #6's "per-lane HP-weighted threat mass" is a coarse version of
   Heroic's encoding. Extending it to **lane × depth bin × card × owner HP sums** is a
   cheap, flat, MLP-compatible step. The size is roughly 2 × 8 × 8 × 2 = 256 floats,
   with bin count a free choice.
3. **The opponent hand is not actually hidden information.** In `Board._cycle_hand_after_play`
   the queue is FIFO with 4 cards, so after a player's 4th play the queue holds exactly
   their last 4 plays, in order. Their hand is the other 4 Deck cards, and their next card
   is the oldest of those 4 plays. A "last 4 opponent plays" feature, which a human
   player also observes, gives the policy the enemy hand without an LSTM. This is the
   Generals.io "memory stack" idea applied to a case where memory makes the state
   exactly Markov after 4 plays. sb3-contrib's MaskablePPO cannot be recurrent anyway,
   and Heroic saw no gain from frame stacking. #6's "card-cycle state" should state
   whether it covers the opponent's cycle.

**Reward: tower-HP dominant with toggled shaping. Needs an outcome term and a
potential-based form.**

- **No term rewards winning.** env-v1 decides by crowns and tiebreak, but the return is
  net tower HP. That is why v1's wins were all at the time limit (`docs/rl-agent.md`
  § Results). AlphaStar (±1), OpenAI Five (win = 5), Gym-μRTS (±10), Heroic and
  Generals.io (±1) all include the outcome.
- **Tower-HP delta is already a potential difference.** With Φ = (own tower-HP fraction
  − enemy tower-HP fraction), the per-step reward is Φ(s′) − Φ(s). With γ close to 1 it
  is (almost) potential-based shaping, so adding a terminal outcome reward turns the
  current reward into "outcome + PBRS" with little conceptual change.
- **#6's new terms can mostly take the same form.**
  - Troop trade value and kill/loss delta can be written as Φ = elixir value of own
    living units − the enemy's, and stay policy-invariant (Ng et al. 1999).
  - The **elixir-leak penalty cannot**: it is a state penalty, a non-potential term that
    can change the optimal policy. KataCR used it, but as offline-IL reward shaping.
    Keep it toggled and small, or anneal it to 0 (Bansal et al.).
- Gym-μRTS warns that agents overfit shaped rewards while sparse win rate varies widely.
  The ablation should therefore report **win rate**, not shaped return.
- Making all terms own − enemy (zero-sum symmetrized, as in OpenAI Five) keeps P0 and P1
  training consistent.
- **Coach impact:** `BLUNDER_DELTA_V` and the other thresholds are tuned to the current
  reward scale and must be retuned when the reward changes.

**Curriculum: Random → Logic. Reasonable as phase 1; insufficient as the whole plan.**

- Pretraining against a weak scripted bot before a stronger one matches Heroic's
  experience with spells. Starting against Random is plausible but weakly motivated: v1
  data show the rule-based bot beats Random 50–0, while v1's random-legal-action
  `RLAgent` won 46.7% against Logic.
- The overfitting ADR-0002 diagnoses in v1 is exactly what Heroic ("fitting its policy
  against a particular type of opponent") and Gym-μRTS (losing to WorkerRush after
  training on CoacAI) describe. Their fixes were cheap: **mix opponents across the
  parallel envs**, then a **small frozen-snapshot pool**.
  - Heroic: n = 3, δ = 0.5.
  - Generals.io: pool of 3, 45% gate.
  - OpenAI Five: 80/20 latest/past.
  - Gym-μRTS found naive latest-self self-play weak.
- `ClashRoyaleEnv` already accepts an opponent object, so this is wiring, not
  architecture.
- **Self-play is blocked by the player 1 lane mismatch** (Known limitations): a policy
  playing P1 sees rotated lanes but deploys into unrotated zones. #6 does not list this
  fix.

**Decision timing: not addressed by #6, and probably the largest lever.**

- The agent is queried every 0.1 s for up to 3001 steps.
- Heroic: per-second querying "can result in poor strategies"; matches ran ~14–25
  decisions.
- KataCR: 4% of frames contain an action.
- SEAT: decides every 0.3 s and gains from forced no-ops.
- OpenAI Five: acts every 4th frame. AlphaStar: learns when to act next.
- Our `ent_coef = 0.02` rationale ("noop is right most of the time, so the policy
  collapses to waiting") is a symptom of this.
- In a local probe, 72% of ticks had **only noop legal** (see
  [Local measurements](#local-measurements)). Those transitions carry no policy-gradient
  signal but still fill rollouts and stretch credit assignment over γ.

**Vectorization: SubprocVecEnv with ~12 envs. Fine.** With the masking note above.
Env-side seeding should vary per episode; today each env replays one opening, which
`docs/rl-agent.md` lists as a known limitation.

**Eval protocol: 200 vs Logic + 200 vs Random with fixed seeds. Keep, and add
uncertainty and seeds.**

- Report Wilson or bootstrap CIs on win rate. Generals.io reports a 95% Wilson CI.
- Train **≥ 3 seeds** per configuration, following Agarwal et al. 2021 and Gym-μRTS's
  4 seeds. For the thesis, the variance across training seeds matters more than the
  episode count.
- The board seed only shuffles the opening hand, and Logic plus the deterministic policy
  are deterministic, so each eval seed is one fixed game. That is reproducible, but 200
  seeds sample 200 deals, not 200 independent behaviours.
- Consider also reporting a stochastic-policy evaluation.

## Recommendation

Ranked by benefit per cost. "Contract" marks changes to `OBS_DIM` or `NUM_ACTIONS`;
#6 already versions those, so observation changes fit its scope.

### Keep

- MaskablePPO with full masks, env-side `action_masks()` (drop `ActionMasker` for
  SubprocVecEnv).
- Flat `Discrete(45)` joint head. It already gives a conditioned card→zone distribution
  and maps 1:1 onto coach suggestions.
- Separate pi/vf MLPs with tanh, SB3's 0.01 policy-head init, and features pre-scaled
  to [-1, 1].
- Per-term reward toggles and the ablation matrix (full / no-shaping / no-curriculum).
- The fixed-seed eval script and its metadata.

### Change in #6 (cheap, high value)

1. **Decision-point stepping** (no contract change). Add an env wrapper that advances
   ticks without querying the policy while only noop is legal, summing rewards over the
   skipped ticks (Heroic; SEAT; OpenAI Five frameskip).
   - Optional: a noop holds for *k* ticks, a hyperparameter (Heroic used 4 s; try
     0.3–1 s, cf. SEAT's 0.3 s).
   - Re-tune γ per decision, because one step now spans variable time.
   - The coach still works: it queries any state, and at noop-only ticks the policy is
     trivially noop.
2. **Outcome-aligned reward:**
   - Add a terminal win/loss/draw term, optionally crown difference.
   - Express HP, trade and kill terms as potentials, Φ(s′) − Φ(s), computed own − enemy.
   - Keep the elixir-leak term small, toggled and preferably annealed.
   - Judge every ablation by win rate, not shaped return.
   - Retune the coach thresholds.
3. **Observation encoding (contract):**
   - One-hot the unit card.
   - Replace or augment the sorted unit slots with lane × depth-bin × card × owner HP
     sums, a generalization of "threat mass".
   - Add the opponent's last 4 plays, which makes the opponent hand known after 4 plays.
   - Keep unit-level features the coach explainer needs.
4. **Opponent mixture, then a snapshot pool:**
   - Assign opponents per SubprocVecEnv index, e.g. 8 Logic, 2 Random, 2 Logic-with-ε
     noise, instead of only a sequential switch.
   - After the Logic phase, add 2–3 frozen own checkpoints sampled from recent history
     (Bansal δ ≈ 0.5; Generals.io 45% gate).
   - This requires **fixing the P1 lane mismatch first**. That fix also matters for the
     web RL opponent.
5. **Evaluation statistics:** ≥ 3 training seeds per configuration, CIs on every win
   rate, and per-episode seed variation during training.

### Change if time allows (medium cost)

6. **Defensive or unit-targeted spell zones (contract):** for example own-side lane
   points, or "on the densest enemy cluster per lane" (SEAT's idea). This adds
   expressivity the current layout cannot learn. It needs a `NUM_ACTIONS` bump, coach
   zone mapping and explainer copy (PL/EN).
7. **Privileged critic:** the value net also sees the opponent hand and elixir during
   training (AlphaStar; HoK full game). It needs a custom SB3 policy with split feature
   extractors. The coach's `V(s)` still works because the server has the full Board.
   Item 3 already removes most hidden information, so the gain may be small.

### Defer (document as future work in the thesis)

- **Hybrid CNN encoder** over 18×32 occupancy/HP planes plus flat scalars (SEAT, GRF
  SMM, Gym-μRTS IMPALA-CNN). It needs a Dict observation and a custom extractor (NatureCNN
  does not fit 18×32), and costs roughly 5–30× more per update on CPU (below). Run it as
  one ablation arm only if the MLP plateaus.
- **LSTM / frame stacking:** sb3-contrib's MaskablePPO does not support recurrence.
  Heroic saw no gain from stacking, and memory features (item 3) cover the known
  partial observability.
- **Autoregressive or factored heads:** only needed if placement becomes a fine grid
  (e.g. 4 slots × 18×15 cells).
- **League training, exploiters, multi-head value, behaviour-cloning warm start from
  LogicAgent:** effective at scale (AlphaStar, HoK, Generals.io) but outside a 1–3 h CPU
  budget.

## Local measurements

These were measured on this machine (14 threads, CPU only) on 2026-10-18 with throwaway
scripts. They come from no paper and are indicative only.

- `ClashRoyaleEnv` against LogicAgent in one process: ~7,800 env steps/s with P0 playing
  a random legal card on 5% of ticks. Under that policy, **28% of ticks had any legal
  card play** and 72% were noop-only. The share depends on how the policy spends elixir.
- Forward + backward pass, batch 256, one network:

  | Network | Parameters | Time |
  |---|---|---|
  | MLP 169→256→256→45 | ~121k | ~1.8 ms |
  | 3-layer CNN (12×32×18 input) | ~361k | ~61 ms |
  | 2-layer strided CNN | ~346k | ~40 ms |

  At batch 12 (rollout inference) the forward pass took ~0.2 ms for the MLP and ~1.2 ms
  for the CNN.
- Implication: a 1M-step PPO run (10 epochs, batch 256, separate pi/vf) needs only
  minutes of gradient time with the MLP but on the order of an hour with a CNN. Within a
  1–3 h budget, that means either ~10–30M MLP steps or ~1–3M CNN steps. Decision-point
  stepping (recommendation 1) cuts steps per match and helps both.

## Unverified or limited items

- Ng et al. 1999: the PDF did not text-extract, so the theorem was checked only via the
  abstract. The "necessary and sufficient" wording comes from secondary sources.
- The Generals.io ICML 2026 follow-up was checked from the abstract page only.
- Community Clash Royale repos: README claims only, with no results.
- Hearthstone and other CCG RL papers, and Lux AI solutions: not read.
- AlphaStar figures come from DeepMind's unformatted author PDF, not the Nature typeset
  version, which was paywalled.
- Oh et al. 2019 (no-op skipping in a fighting game) is cited second-hand through the
  Heroic paper.
