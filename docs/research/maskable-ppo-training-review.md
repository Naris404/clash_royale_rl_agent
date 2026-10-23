# MaskablePPO training review against published game-RL research

Research note, 2026-10-23. It reviews the training design that landed in #42
("training opponents, curriculum and per-episode seeds", commit `30e7dbb`): the PPO
setup, the opponent schedule, termination handling, model selection and evaluation. It
then compares that design with published work on invalid-action-masked PPO and on
similar games, and proposes changes. It does not change any decision or any code.

This note follows [`rl-architecture-review.md`](rl-architecture-review.md) (2026-10-18),
which reviewed the observation, action layout and reward that ADR-0002 then adopted.
Where the two overlap, this note refers to the earlier one instead of repeating it.
Vocabulary follows [`CONTEXT.md`](../../CONTEXT.md).

Every literature claim cites the paper and the section, table or figure it comes from.
Every listed paper was opened and read, as full text unless noted otherwise. Library
documentation and source code are cited only to describe implementation behaviour, and
are labelled **[impl]**. Anything not checked against a primary source is marked
**(unverified)**.

## 1. Summary

**Verdict:** the design is close to what the literature recommends for an affordable
PPO agent. Two problems are worth fixing before the first real v2 training run. Neither
one touches the observation/action contract.

**Already matches best practice (keep):**

- Full invalid-action masking in both the sample and the gradient
  [Huang & Ontañón 2022][r1]; [Huang et al. 2021, §V-B][r2].
- A flat joint card×zone head. Heroic found a conditioned head about 10% better than
  independent heads [Warchalski et al. 2020, "Effect of Conditioning Spatial
  Actions"][r4].
- Decision-point stepping with a noop hold [Warchalski et al. 2020, "Experiments"][r4];
  [Liu et al. 2019, §4][r5].
- An outcome reward plus potential-based shaping that uses PPO's own γ and sets
  Φ(terminal) = 0 [Ng et al. 1999, Thm. 1][r26].
- Separate tanh policy and value MLPs with SB3's 0.01 policy-head init and no value
  clipping [Andrychowicz et al. 2020, §3.2, §3.4][r21]; [Huang et al. 2022][r22].
- Scripted opponents mixed across the parallel envs
  [Huang et al. 2021, §V-C][r2].

**Problems found:**

1. **Every time-limit match end is treated as truncation, so SB3 bootstraps from a
   terminal state.** `Board._check_end` returns `truncated=True` at the end of regulation
   and in overtime (`board.py:825`, `:828`), and the env passes that flag through
   (`gym_env.py:117`). sb3-contrib then adds `γ·V(terminal_obs)` to the last reward of
   every such match **[impl]**. Crowns, the Tiebreak and draws all decide the match by
   the rules, so these ends are true terminations. Bootstrapping there is wrong
   [Pardo et al. 2018, §2][r25]. In a local probe, 9 to 24 of every 30 matches ended
   this way (§8).
2. **Model selection is noisy and biased.** The saved "best" model is the maximum of
   about 40 evaluations of *shaped mean reward*. Each evaluation uses only 50 episodes,
   and every evaluation round sees different seeds. Taking the maximum of noisy
   estimates favours lucky rounds. The repo already saw this happen: the v1 "best"
   model won 6.7% while the final one won 70%. The literature asks for interval
   estimates and several training seeds [Agarwal et al. 2021, Table 1][r35];
   [Henderson et al. 2018, Fig. 5][r36].

**Top recommendations:**

1. **(High)** Return `terminated=True` whenever the Board ends a match. This is one line
   in `ClashRoyaleEnv.step`; the Board is untouched.
2. **(High)** Select and report by win rate:
   - fixed held-out evaluation seeds;
   - Wilson confidence intervals;
   - all three scripted opponents, not only `LogicAgent`;
   - at least 3 training seeds per arm.
3. **(Medium)** After the scripted phase, add a small pool of frozen snapshots. Play
   them deterministically, sample them δ-uniformly and admit each one through a
   win-rate gate [Bansal et al. 2018, §5.4][r27];
   [Straka & Schmid 2025, "Self-Play Fine-Tuning"][r14]. The frozen-policy hook already
   exists.
4. **(Medium)** Log PPO's approximate KL and clip fraction. Consider `target_kl` and
   fewer epochs, because the opponent mix makes the data non-stationary
   [Yu et al. 2022, §5][r23]. Fix the stale budget and minibatch numbers in
   `docs/rl-agent.md`.
5. **(Medium)** Switch curriculum phases on a win-rate threshold instead of a fixed 20%
   of steps. Report the curriculum ablation with time-to-threshold and asymptotic
   metrics [Narvekar et al. 2020, §3.3][r32].

## 2. Our current training design

These facts come from the code at `30e7dbb`. Line numbers refer to that commit.

### Algorithm and network

| Item | Value | Source |
|---|---|---|
| Library | `sb3-contrib` MaskablePPO; installed SB3 / sb3-contrib 2.8.0, gymnasium 1.2.3, torch 2.12 (CPU) | `pyproject.toml`; local `pip` |
| Policy | `"MlpPolicy"`, `net_arch = dict(pi=[256, 256], vf=[256, 256])`, separate networks, Tanh, orthogonal init (SB3 defaults) | `train.py:121`, `:152-166` |
| Parameters | 288,310 in total | `docs/rl-agent.md` § Network architecture |
| Observation | `Box(-1, 1, (278,))`, prescaled, no `VecNormalize` | `gym_env.py:63`; `board.py:131-142` |
| Action | `Discrete(53)` = noop + 4 hand slots × 13 zones | `gym_env.py:62`; `board.py:122` |
| Masking | Env-side `action_masks()` → `Board.valid_action_mask(0)`; no `ActionMasker` wrapper | `gym_env.py:70-71`; `train.py:54-68` |

The observation holds:

- elixir, the elixir advantage and time;
- 6 tower HP values;
- the own hand and next card, one-hot;
- the opponent's last 4 plays, one-hot;
- a 2 owner × 2 lane × 8 depth × 6 card HP-sum grid.

### PPO hyperparameters

| Hyperparameter | Value | Source |
|---|---|---|
| `learning_rate` | 3e-4, constant | `train.py:102` |
| `n_steps` per env | `max(2048 // 12, 256)` = 256, so 3,072 transitions per rollout | `train.py:156` |
| `batch_size` | 256, so **12** minibatches per epoch | `train.py:104`, `:157` |
| `n_epochs` | 10, so 120 gradient steps per rollout | `train.py:158` |
| `gamma` | 0.995 per decision step (env and PPO share it) | `rewards.py:17`; `train.py:159` |
| `gae_lambda` | 0.95 | `train.py:160` |
| `ent_coef` | 0.02 | `train.py:105` |
| `clip_range` | 0.2 | `train.py:162` |
| Other settings | `vf_coef` 0.5, `max_grad_norm` 0.5, `normalize_advantage=True`, `clip_range_vf=None`, `target_kl=None` (sb3-contrib defaults) | `sb3_contrib/ppo_mask/ppo_mask.py:87-94` **[impl]** |
| `total_timesteps` | 1,000,000 decision steps | `train.py:99` |
| `n_envs` | 12, one `SubprocVecEnv` process each | `train.py:36` |

### Decision schedule and episode

- The agent is asked only at a Decision point, where a card play is legal and no noop
  hold runs (`decisions.py:30-31`).
- Noop starts a 3-tick (0.3 s) hold (`decisions.py:15`, `:33-36`).
- The env simulates the skipped ticks inside `step()` (`gym_env.py:105-112`).
- The opponent acts on every tick (`gym_env.py:123`).
- A match lasts at most 3,001 ticks. In decision steps it ran 12–344 steps in the local
  probe (§8).

### Reward

- `r = 1.0·outcome + (γ·Φ(s′) − Φ(s))`, with
  `Φ = 0.2·(crown difference) + 0.5·(tower-HP difference)/10,928` (`rewards.py:21-56`).
- Φ = 0 once the Board is done (`rewards.py:36-37`).
- The reward is computed once per decision step (`gym_env.py:113-114`).

### Termination flags

- A king-tower kill returns `terminated=True` (`board.py:819`).
- The end of regulation with unequal crowns, the first crown in overtime, and the end of
  overtime (Tiebreak or draw) all return **`truncated=True`** (`board.py:825`, `:828`).
- The env passes both flags unchanged (`gym_env.py:117`).

### Opponents and curriculum

- **Phase 1:** the first 20% of steps, all 12 envs against `RandomAgent` (`train.py:37`,
  `:41-45`, `:118-119`). `RandomAgent` plays a random legal card on 45% of ticks
  (`rl.py:40`).
- **Phase 2:** opponents by env index: 8 `LogicAgent`, 2 ε-`LogicAgent` (ε = 0.2), 2
  `RandomAgent` (`train.py:38`).
- `--no-curriculum` starts in phase 2.
- A new `SubprocVecEnv` is built per phase, with env indices offset by phase
  (`train.py:144-148`).
- **Frozen-policy hook:** a loaded `MaskablePPO` opponent is wrapped in `ScheduledAgent`
  and plays deterministically by default (`gym_env.py:141-142`; `rl.py:70`). No snapshot
  pool exists yet (ADR-0002 § Future work).
- `LogicAgent` places cards at continuous positions on its own half, not at the 13 zones
  (`logic.py:95-103`).

### Seeding

- `set_random_seed(seed)` seeds the libraries (`train.py:116`).
- Each episode's Board seed is `SeedSequence([run_seed, env_index, episode])`
  (`gym_env.py:28`, `:80-86`).
- `reset(seed=...)` overrides it for evaluation.

### Training-time evaluation and model selection

- `MaskableEvalCallback` runs every ~25,000 steps: 50 deterministic episodes against
  `LogicAgent` on an env built with seed `run_seed + 10,000` (`train.py:120`, `:124-132`).
- SB3's `EvalCallback` keeps the checkpoint with the highest **mean episode reward**.
  The training loop copies it to `models/ppo_cr_best.zip`, the model the game loads
  (`train.py:180-186`).
- The eval env never gets fixed seeds, so its episode counter keeps advancing and each
  evaluation round plays **50 different seeds**. This follows from `gym_env.py:82-84`
  and from SB3's `DummyVecEnv.reset`, which passes `seed=None` after the first reset
  **[impl]**.
- `WinRateCallback` pools all opponents of the current phase. It logs only when
  `num_timesteps % 5000 == 0`, and `num_timesteps` grows by 12 per vec step, so with
  12 envs it logs every 15,000 steps (`train.py:91`).

### Final evaluation

- `evaluate_model()` plays N matches as P0 **against `LogicAgent` only**, with seeds
  `seed … seed+N−1` and the deterministic policy (`train.py:192-242`).
- It reports wins, losses, draws, win rate, mean return and length. It reports no
  confidence interval.
- ADR-0002 asks for 200 matches against `LogicAgent` *and* 200 against `RandomAgent`,
  with 95% Wilson CIs. That protocol is not implemented yet.

### Experiments

- `cr-rl-sweep` runs one seed per configuration: baseline, lr 1e-4 / 1e-3, entropy
  0.005 / 0.05, γ 0.99, nets 128 / 512 (`experiments/sweep.py:25-32`).
- ADR-0002 plans 3 seeds for the final configuration and 1 seed for each ablation arm.

### Documentation drift (`docs/rl-agent.md`)

- Line 325 says "8 minibatches per epoch over the 2048-transition rollout". The code
  gives 12 minibatches over 3,072 transitions.
- Line 334 says 1M steps is "about 555 full-length matches". That was the per-tick
  count. With decision points, 1M steps is about 5,000–38,000 matches (§8).

## 3. Literature review

### 3.1 Closest systems: Clash Royale and card-placement real-time games

[`rl-architecture-review.md`](rl-architecture-review.md#clash-royale-and-card-placement-real-time-games)
already summarizes SEAT, Heroic, KataCR and Boney et al. This section adds only the
training-related details that note left out.

**Heroic: Magic Duel** [Warchalski et al. 2020][r4] is the closest analogue: 1v1,
real-time, cards cast on lanes, mana, a 4-card hand, PPO.

- *Decision timing.* "Naively triggering the network too often (every 0.2, 0.5 or 1
  second), even when no move is available, can result in poor strategies." The fix is to
  return a state only when a non-trivial action exists, with a 4 s no-op ("Experiments",
  first paragraph).
- *No-op ablation.* In the no-spell game, agents with and without the card no-op
  "perform almost the same", and the agent "learns to avoid them over time". With
  spells, no-op use did *not* drop during training ("Effect of Card No-Op", "Effect of
  Enabling Spells").
- *PPO settings* ("Experiment Details"):
  - clip 0.2;
  - separate policy and value learning rates (3e-4 / 1e-3);
  - γ = 0.99, λ = 0.97;
  - 60,000 samples (~3,000 matches) per iteration, minibatches of 2,500;
  - early stopping when the approximate KL exceeds 0.01.

  The authors say "collecting more samples per iteration helped stabilize the training
  and improved exploration".
- *Opponents* ("Training Curriculum"):
  - Training against the existing AI gives "quicker progress" but risks "fitting its
    policy against a particular type of opponent".
  - Self-play uses an ensemble of n = 3 policies and samples opponent weights from
    (δv, v) with δ = 0.5, following Bansal et al.
  - Decks are sampled from a pool of 200 for both sides.
- *Curriculum for the hard case.* The spell agent was "pretrained against the rule-based
  AI … since otherwise the reward signal is not strong enough to initiate any learning"
  ("Effect of Enabling Spells").
- *Evaluation:* 64.5% against the rule-based AI and 37.3% against the tree-search AI; 50%
  against a top-5% human over 20 matches ("Effect of Self-Play").

**SEAT** [Liu et al. 2019][r5] is DQN on a Clash Royale simulator.

- It decides every 15 rounds (0.3 s, "the same as human's fastest reaction time") and
  uses γ = 1 (§4).
- With an "abandoning rate" of 0.5 (random forced no-ops), it learned faster than with 0
  (§4, Fig. 3).
- Each agent pairing was evaluated in only 10 battles (§4, Table 2).

**KataCR** [Wu et al. 2025][r6] is offline RL on real Clash Royale.

- Only ~4% of frames are action frames (§4.3).
- Its reward includes an elixir-overflow penalty (§4.1). This is the non-potential term
  ADR-0002 dropped.
- The agents "still cannot consistently defeat the built-in AI" (Conclusion).

**Boney et al.** [Boney et al. 2020][r7] (Aalto with Supercell) used a full-state
planner distilled into a partial-observation policy.

- They describe Clash Royale as having "cyclic strategies" (§I).
- They note that "waiting for a good deploy time is an important part of strategy"
  (§II).
- They contrast their self-play-trained agent with prior work that "achieved high
  win-rates against the rule-based opponent by directly training against it" (§VI).

No 2023–2026 peer-reviewed paper on online PPO or self-play for Clash Royale was found.
Searches turned up only community repositories, which are not used as evidence.

**Collectible card games.**

- The Legends of Code and Magic (LOCM) champion agent uses an action mask, a ±1 terminal
  reward, and a deep-RL best response inside an "optimistic smooth fictitious play" loop
  [Xi et al. 2023, §2.1, §2.4][r15].
- That agent is ByteRL, the state of the art in LOCM and Hearthstone. A best-response
  PPO agent fine-tuned against it beats it, which shows it is "highly exploitable". In
  the same study, behaviour-cloning pre-training reached a 75% win rate in less than
  half the time of random initialization [Haluška 2024, §5][r16].
- DouZero handles DouDizhu's huge, turn-varying set of legal moves with *action
  encoding* and Deep Monte-Carlo instead of PPO [Zha et al. 2021, abstract][r17]. A fixed
  53-action space does not need that.

### 3.2 Invalid action masking and action-space design

**Theory.**

- Masking replaces invalid logits with a large negative number before the softmax. It is
  "a state-dependent differentiable function", so the masked gradient "is the policy
  gradient" of the masked policy [Huang & Ontañón 2022, Prop. 1][r1].
- The canonical implementation also zeroes the gradient of the masked logits
  ([r1], "Invalid Action Masking").

**Scaling.**

- On μRTS harvesting, masking solved the task in ~12% of training on every map size
  (4×4 to 24×24), and found the first reward in 0.05–0.08% of steps.
- The invalid-action penalty "does not scale". At 10×10 with r = −0.01, it spent 3.43%
  of training just finding the first reward, and r = −1 "discourag[ed] exploration"
  ([r1], "Evaluation Results", Table 2).

**Naive masking** masks only at sampling time and computes the gradient on the unmasked
distribution. Its KL divergence between updates "explodes", and its time to solve was
volatile (49% of training at 24×24) ([r1], "Evaluation Results").

**Full versus partial masks.**

- In Gym-μRTS, masking action types only (PySC2-style) "has little impact", while the
  full mask "considerably improves performance".
- Action composition and full masking were "essential for having an agent that even
  starts learning" [Huang et al. 2021, §V-B, §V-C][r2].
- On SC2 minigames, masking was "crucial" on BuildMarines, helped on
  CollectMineralsAndGas, and did not help on two others
  [Kanervisto et al. 2020, §III, Fig. 6][r3].
- In Honor of Kings, the action mask reached the same strength as the base model (50.5%)
  but "can largely reduce the training time" [Ye et al. 2020a, Ablation Study, Table 5][r10].

**Joint versus factored heads.**

- Heroic's head conditions the spatial action on the chosen card and samples (x, y)
  jointly. It beat an Atari-net head with independent factors by "about 10%" win rate
  [Warchalski et al. 2020, "Effect of Conditioning Spatial Actions"][r4].
- PySC2's autoregressive arFullyConv beat FullyConv "presumably because it knows which
  action identifier the argument will be used for" [Vinyals et al. 2017, §5.2, Table 2][r12].
- Converting MultiDiscrete to Discrete "has no clear positive effect" and scales poorly
  only for large spaces [Kanervisto et al. 2020, §IV][r3].
- Removing actions "can lower the overall performance (Doom), but it can be an important
  step to make environments learnable" ([r3], §IV).

**Library behaviour [impl].**

- sb3-contrib's `MaskableCategorical` sets masked logits to −1e8. Its entropy sums only
  over unmasked actions ("masked actions possess no uncertainty")
  (`sb3_contrib/common/maskable/distributions.py:59-61`, `:72-82`).
- The docs list Discrete, MultiDiscrete and MultiBinary action spaces, no recurrent
  policies, and require `action_masks()` inside the env when using `SubprocVecEnv` [D1].
- One consequence follows (our reasoning): masked entropy is bounded by log(#legal
  actions), which here ranges from log 2 to log 33. A fixed `ent_coef` therefore pulls
  hardest toward uniform play in states with many legal plays.

### 3.3 PPO implementation details and hyperparameters

**Reference settings.**

| Source | Settings |
|---|---|
| PPO Atari [Schulman et al. 2017, Table 5][r18] | horizon 128, 8 actors, 3 epochs, minibatch 32×8, γ 0.99, λ 0.95, clip 0.1·α, lr 2.5e-4·α, both annealed linearly to 0, value coeff 1, entropy 0.01 |
| PPO MuJoCo ([r18], Table 3) | horizon 2048, 10 epochs, minibatch 64, lr 3e-4 |
| Gym-μRTS [Huang et al. 2021, App. A, Table III][r2] | 24 envs × 256 steps, 4 minibatches, 4 epochs, γ 0.99, λ 0.95, clip 0.1, lr 2.5e-4 annealed to 0, entropy 0.01, 300M steps |
| Bansal et al. [2018, §5][r27] | γ 0.995, λ 0.95, clip 0.2, no entropy bonus |
| Honor of Kings 1v1 [Ye et al. 2020a, "Experiments"][r10] | dual-clip PPO (ε 0.2, c 3), γ 0.997 ("half-life of about 46 seconds"), λ 0.95, lr 1e-4 |

**γ and λ.**

- γ "most importantly determines the scale of the value function" and introduces bias
  "regardless of the value function's accuracy". λ < 1 adds bias only when V is
  inaccurate.
- The best λ is "much lower than the best value of γ" [Schulman et al. 2016, §3][r19].
- Andrychowicz et al. call γ "one of the most important hyperparameters" and recommend
  λ = 0.9 [Andrychowicz et al. 2020, §3.4, §3.6][r21].

**Code-level choices.**

- *Code-level optimizations matter.* They "are responsible for most of PPO's gain in
  cumulative reward over TRPO". Reward normalization, Adam annealing and network
  initialization "each significantly impact" results [Engstrom et al. 2020, abstract,
  §3, Fig. 1][r20].
- *Large on-policy study* (250,000 agents; continuous control only)
  [Andrychowicz et al. 2020][r21]:
  - separate policy and value networks were better on 4 of 5 environments (§3.2);
  - a 100× smaller last policy layer helps (§3.2);
  - "always use observation normalization" (§3.3);
  - per-minibatch advantage normalization matters little (§3.3);
  - PPO-style value clipping "hurts the performance" (§3.4);
  - many parallel environments shorten experience chunks and can hurt sample complexity
    (§3.5);
  - linear lr decay gives small gains (§3.7);
  - regularizers including entropy showed no significant benefit (§3.8).
- *The 37 implementation details* [Huang et al. 2022][r22]:
  - policy output init scale 0.01;
  - lr annealing;
  - per-minibatch advantage normalization;
  - value clipping kept only for fidelity, though studies find it does not help;
  - "the separate-networks architecture clearly outperforms in simpler environments";
  - the authors "did not implement the correct handling for truncated environments".

**Non-stationary data.** MAPPO makes five suggestions, motivated by non-stationarity from
co-learning agents [Yu et al. 2022, §5][r23]:

1. value normalization;
2. global information in the value input;
3. "at most 10 training epochs on difficult environments";
4. clip ratio "under 0.2";
5. large batches.

**Entropy.** Entropy regularization can smooth the optimization landscape and allow
larger learning rates. The authors stress that its effect depends on the environment
[Ahmed et al. 2019, abstract, §4][r24].

### 3.4 Episode termination, time limits and discounting

**Time limits.** Pardo et al. separate two cases [Pardo et al. 2018, §1–3][r25]:

1. **Time limit that is part of the task.** "The terminations due to time limits are in
   fact part of the environment", and "a notion of the remaining time should be included
   as part of the agent's input". Otherwise states that differ only in remaining time
   alias, which leads "to suboptimal policies and instability" (§2).
2. **Time limit used only to cut training episodes.** Bootstrapping from the last state's
   value (partial-episode bootstrapping) is correct (§3).

**Implementation [impl].**

- Gymnasium: "when an episode ends due to termination we don't bootstrap, when it ends
  due to truncation, we bootstrap" [D5].
- SB3's `SubprocVecEnv` sets `info["TimeLimit.truncated"] = truncated and not terminated`
  (`stable_baselines3/common/vec_env/subproc_vec_env.py:38`).
- MaskablePPO's rollout then adds `self.gamma * terminal_value` to the reward of any such
  step (`sb3_contrib/ppo_mask/ppo_mask.py:248-260`).
- The 37-details authors note that the reference PPO did not handle truncation at all
  [r22].
- Andrychowicz et al. found that correct handling did not matter for MuJoCo's 1000-step
  limits [r21, §3.6]. That study concerns *artificial* limits, not task-defining ones.

**Horizon in strategy games.**

- AlphaStar uses the match outcome "without a discount to accurately reflect the true
  goal of winning games" [Vinyals et al. 2019, Methods, "Reinforcement learning"][r8].
- In OpenAI Five, resuming a skilled agent with a longer horizon than 180 s (γ = 0.9993)
  "increases win rate", with diminishing returns [Berner et al. 2019, §4.5, Fig. 6][r9].

### 3.5 Reward design

**Potential-based shaping.**

- Adding F = γΦ(s′) − Φ(s) leaves the optimal policy unchanged. Without further
  knowledge of the MDP, this form is also *necessary*. Known shaping "bugs" (circling a
  bicycle, vibrating next to a ball) come from non-potential rewards
  [Ng et al. 1999, §1, Thm. 1][r26].
- In the undiscounted case the theorem uses Φ(absorbing state) = 0 ([r26], §3). Our
  `potential()` implements exactly that.

**Shaping in practice.**

- **OpenAI Five** used a hand-designed reward "modeled loosely after potential-based
  shaping … though the guarantees therein do not apply". All rewards were made zero-sum
  "by subtracting from each hero's reward the average of the enemies' rewards"
  [Berner et al. 2019, App. G][r9].
- **Generals.io** used ±1 plus PBRS. Against naive self-play, PBRS lifted the win rate to
  71.9% [Straka & Schmid 2025, "Reward Shaping", Table 2][r14].
- **Gym-μRTS** used a non-potential shaped reward. Its agents "sometimes overfit to the
  shaped rewards instead of sparse rewards", with little shaped-return variance across
  seeds but large win/loss variance [Huang et al. 2021, §VI][r2].
- **Bansal et al.** annealed a dense "exploration reward" to zero over the first 10–15% of
  training. Agents that kept the dense reward lost to agents trained on the sparse one
  [Bansal et al. 2018, §4.1, §5.3, Fig. 3][r27].
- **Outcome-only rewards** suffice for Heroic (±1) [r4] and LOCM (±1) [r15].

### 3.6 Opponents, self-play and populations

**Scripted opponents overfit; diversity helps.**

- Training only against CoacAI left Gym-μRTS agents "being defeated by AIs as simple as
  WorkerRush". A mix across the 24 envs (18 CoacAI, 2 each of three other bots) gave "a
  rather significant performance boost" for Gridnet and a milder one for UAS
  [Huang et al. 2021, §V-C][r2].
- Heroic reports the same risk ("fitting its policy against a particular type of
  opponent") [r4].
- Independent learners "can overfit to the other agents' policies". Even in an almost
  fully observable map, a policy could expect to lose 34.2% of its reward against
  partners it was not trained with [Lanctot et al. 2017, §4.1][r28].

**Naive self-play is unstable or cyclic.**

- Training against the latest opponent "leads to imbalance". Sampling "random old
  versions" works much better. The best δ for Uniform(δv, v) depends on the task: 0.5
  for Humanoid, 0 for Ant [Bansal et al. 2018, §4.2, §5.4, Table 1][r27].
- Latest-self self-play in Gym-μRTS did "not perform particularly well"
  [Huang et al. 2021, App. C][r2].
- Self-play variants paired with PPO "exhibit cyclic policy evolutions"
  [Hernandez et al. 2020, abstract, §VIII][r30].
- Self-play "may chase cycles". Fictitious self-play avoids cycles by playing a mixture
  of all past policies, which converges to a Nash equilibrium in two-player zero-sum
  games [Vinyals et al. 2019, main text][r8]; [Heinrich & Silver 2016, abstract][r29].
- Real-world games have a large non-transitive dimension at mid-level strength. This
  "clarifies why populations of strategies are necessary"
  [Czarnecki et al. 2020, abstract][r31].

**Concrete opponent-sampling recipes.**

| System | Recipe |
|---|---|
| OpenAI Five [Berner et al. 2019, App. N][r9] | 80% latest, 20% past versions; past opponents sampled by a softmax over learned quality scores |
| AlphaStar [Vinyals et al. 2019, Methods, "League"][r8] | main agents use 35% self-play, 50% PFSP against the league, 15% PFSP against forgotten players; f_hard(x) = (1 − x)^p focuses on the hardest opponents and f_var(x) = x(1 − x) on opponents near the agent's level; PFSP beat FSP on every measure (Ext. Data Fig. 5) |
| Heroic [r4] | 3-policy ensemble, δ = 0.5 |
| Generals.io [Straka & Schmid 2025, "Self-Play Fine-Tuning"][r14] | pool of N = 3 predecessors; a candidate replaces the oldest after a 45% win rate against the pool; opponents act by argmax, because "training against stochastic opponents makes new agents focus on exploiting mistakes caused by stochasticity" |

**Automatic curricula from competition.**

- Self-play can be seen as automatic curriculum learning over opponents
  [Portelas et al. 2020, §4][r33].
- Competition generates "a naturally emergent curriculum" [Leibo et al. 2019, abstract][r34].

### 3.7 Curriculum learning

- **Evaluating a curriculum.** Curricula should be judged against learning from scratch
  on the target task. The metrics are *jumpstart*, *asymptotic performance*, *total
  reward* and *time to threshold*, and one must decide whether the curriculum's own steps
  count as cost [Narvekar et al. 2020, §2.3, §3.3][r32].
- **Automatic curricula** adapt the training distribution to the agent's current
  capabilities instead of using a fixed schedule [Portelas et al. 2020, §1–3][r33].
- **Phase gates in practice:**
  - HoK's curriculum self-play learning advances phases "based on the convergence of Elo
    scores" [Ye et al. 2020b, §3.3][r11];
  - Generals.io gates pool entry at 45% [r14];
  - AlphaStar adds exploiters at 70% win rates [r8].
- **Warm-starting against the scripted bot** before a harder setting was needed for
  Heroic's spell agent [r4].

### 3.8 Observation encoders and memory (brief)

[`rl-architecture-review.md`](rl-architecture-review.md#network-flat-mlp-keep-as-the-primary-model-with-caveats)
covers this topic. New points:

- **Heroic's grid matches ours.** Its spatial input is D = 10 bins × L = 3 lanes × 2U
  channels, each holding "the sum of health (in percent)" of own and enemy units of a
  type. It feeds one 1×3 conv layer [r4, "Reinforcement Learning Environment",
  "Experiment Details"]. Our 2 × 2 × 8 × 6 grid is the same encoding.
- **Model size mattered less than the techniques.** Gym-μRTS found no strong correlation
  between parameter count and performance. Masking and the architecture choice (IMPALA-CNN
  for UAS) mattered more [r2, §VI].
- **Memory.**
  - Heroic saw "no immediate benefit" from stacking observations [r4, "Future Work"].
  - Generals.io used a hand-crafted memory stack, including the last 7 moves of each side
    [r14].
  - Recurrent model-free RL is "a strong baseline for many POMDPs" when tuned carefully
    [Ni et al. 2022, abstract][r39].
  - sb3-contrib's MaskablePPO cannot be recurrent [D1].
- **Entity encoders.**
  - AlphaStar uses self-attention over units, scatter connections and an LSTM
    [r8, main text].
  - Relational (self-attention) RL improved sample efficiency on SC2 minigames
    [Zambaldi et al. 2018, abstract][r38].
  - HoK max-pools unit features per type [r10, "Algorithm Design"].
- **Privileged critic.**
  - AlphaStar uses the opponent's observations for a lower-variance baseline [r8,
    Conclusion].
  - HoK feeds "full information about the game state, including observations hidden from
    the policy" to the value function during training [r11, §3.2].
  - MAPPO Suggestion 2 adds global features to the value input [r23].

### 3.9 Evaluation methodology

- **Few runs need interval estimates.** With 3–10 runs, report stratified-bootstrap
  confidence intervals, performance profiles and the interquartile mean instead of point
  estimates [Agarwal et al. 2021, §4, Table 1][r35].
- **Seeds change conclusions.** Ten runs of one configuration split into two groups of
  five gave "statistically different distributions just from varying random seeds"
  [Henderson et al. 2018, "Random Seeds and Trials", Fig. 5][r36].
- **Elo can fail under cycles.** Elo's predicted win probabilities "can fail in simple
  cases" such as rock-paper-scissors. Multidimensional Elo and Nash averaging handle
  cyclic interactions [Balduzzi et al. 2018, §2–3][r37].
- **Ratings in practice.**
  - OpenAI Five rated agents with TrueSkill against a pool of fixed reference agents
    [Berner et al. 2019, App. J][r9].
  - Generals.io reported a 95% Wilson CI over 529 games and an Elo table among bots
    [r14, "Results"].
  - Gym-μRTS evaluated 100 games against each of 13 bots [r2, §V].
- **A strong fixed agent can still be exploitable.** A best response to the
  state-of-the-art LOCM agent beats it, so a high win rate against one fixed opponent
  does not show general strength [Haluška 2024, §5–6][r16].

## 4. Comparison table

| Aspect | Our design | Literature | Assessment |
|---|---|---|---|
| Algorithm | MaskablePPO, on-policy, 12 synchronous envs | PPO is the norm in affordable game RL [r2][r4][r14]; IMPALA, V-trace and dual-clip address off-policy lag [r8][r10] | **Matches.** No lag here. |
| Masking | Full mask (affordability, zone type, push-zone legality), applied in the sample and the gradient | Full masking is essential; naive masking makes the KL explode [r1][r2] | **Matches.** |
| Action head | Flat `Discrete(53)` joint card×zone | Conditioned or joint heads beat independent ones by ~10% [r4][r12] | **Matches.** |
| Decision frequency | Ask only when a play is legal; noop holds 0.3 s | Skip trivial states [r4]; 0.3 s [r5]; frameskip [r9]; learned delay [r8] | **Matches.** Noop duration is fixed, not learned (open question). |
| Observation | Flat 278: one-hot cards, last 4 enemy plays, lane × depth × card HP-sum grid | The Heroic grid is the same encoding [r4]; memory features [r14] | **Matches** for this budget. Overtime time is aliased (§5, L3). |
| Network | Separate tanh MLPs 256×256 | Separate nets better in simple envs [r21][r22]; size matters less than technique [r2] | **Matches.** |
| Reward | ±1 outcome + PBRS (crowns, tower HP), same γ, Φ(terminal) = 0, antisymmetric | PBRS preserves optimal policies [r26] and helped [r14]; zero-sum symmetrization [r9]; non-potential shaping overfits [r2] | **Matches.** Its guarantees are undone by the truncation bootstrap below. |
| γ / λ | 0.995 per decision step / 0.95 | γ near 1 or undiscounted for outcome rewards [r8][r9][r10][r27]; λ 0.9–0.97 [r4][r21] | **Reasonable.** γ^50 ≈ 0.78; episodes of 26–210 steps fall inside the horizon. |
| Terminal handling | Time-limit match ends are `truncated` → SB3 bootstraps γ·V(s_T) | Task-defining time limits are terminal; bootstrap only on artificial cut-offs [r25][D5] | **Gap (bug).** High priority. |
| Time in observation | `min(1, t/180)`: saturates during the ≤120 s overtime | Remaining time should be observable in time-limited tasks [r25, §2] | **Minor gap.** Contract change. |
| Rollout / epochs | 3,072 transitions, 12 minibatches × 10 epochs, no `target_kl` | Atari 3 epochs [r18]; μRTS 4 × 4 [r2]; MAPPO ≤ 10 epochs under non-stationarity [r23]; Heroic KL stop 0.01, large batches [r4] | **Acceptable but unmonitored.** Add KL diagnostics; consider `target_kl`. |
| Learning rate | 3e-4 constant | 3e-4 is a safe default; linear decay gives small gains [r21, §3.7][r18][r2] | **Matches** (optional decay). |
| Entropy | 0.02, chosen for per-tick noop collapse | 0.01 in Atari and μRTS [r18][r2]; 0 in [r27]; no general benefit [r21]; landscape smoothing [r24] | **Rationale stale** since decision points; sweep. |
| Value / advantage normalization | Advantages normalized; no value clipping; no value normalization | Value clipping hurts [r21][r22]; value normalization helps for large return scales [r23] | **Matches.** Returns are bounded by about ±2. |
| Opponents | Phase 2: 8 Logic, 2 ε-Logic, 2 Random by env index; frozen-policy hook unused | Mixed bots across envs [r2]; past-self pools [r4][r9][r14][r27]; PFSP league [r8]; fixed-opponent overfitting [r4][r28][r16] | **Partly matches.** No pool after the scripted phase. |
| Curriculum | 20% steps vs Random, then the phase-2 mix; ablation `--no-curriculum` | Pretrain vs scripted AI when the signal is too weak [r4]; gate phases on performance [r11][r14]; judge by jumpstart / time-to-threshold / asymptote [r32] | **Partly matches.** The fixed switch is not performance-gated. |
| Seeding | Per-episode `SeedSequence(run, env, episode)`; eval seeds disjoint | Fixed seed sets for reproducibility [r25]; vary seeds across runs [r36] | **Matches.** |
| Training seeds | 3 for the final config, 1 per ablation | 3–10 runs with interval estimates [r35]; 4 seeds [r2]; seed variance can flip conclusions [r36] | **Gap** for ablations. |
| Model selection | Max shaped mean reward over ~40 rounds × 50 episodes; seeds change per round | Interval estimates [r35]; judge by sparse win/loss, not shaped return [r2] | **Gap.** Biased upward and by the wrong metric. |
| Final evaluation | 100 matches vs Logic only, no CI (ADR: 200 vs Logic + 200 vs Random, Wilson CI) | Many opponents [r2]; Wilson CI [r14]; TrueSkill / Elo vs reference pools [r9][r14]; exploitability [r16] | **Gap.** ADR protocol not implemented; no held-out opponents. |
| Budget | 1M decision steps ≈ 5k–38k matches (§8) | μRTS 300M steps [r2]; Heroic 60k samples per iteration [r4]; Generals 36 h H100 [r14] | **Small but cheap.** Increase if learning curves have not flattened. |

## 5. Proposed changes

Each item lists what to change, why, the expected impact, the cost and risk, and the
effect on the observation/action contract and determinism (AGENTS.md).

### High priority

**H1. Treat every Board match end as a termination.**

- *What:* in `ClashRoyaleEnv.step` (`gym_env.py:117`), return
  `terminated = result.terminated or result.truncated` and `truncated = False`.
  `Board._check_end` keeps its own flags; `tests/test_board.py:882`, `:978` pin them.
  Add an env-level test that a regulation-time win returns `terminated=True`.
- *Why:* a match that ends by Crowns, Tiebreak or draw ends by the game's rules. Its
  value after the end is 0, Φ(terminal) is already 0, and the outcome is already paid.
  Pardo et al. say task-defining time limits must be treated as terminal
  [r25, §2]. Gymnasium says not to bootstrap on termination [D5].
  - Today sb3-contrib adds γ·V(s_T) on these steps **[impl]**. V(s_T) is the critic's
    extrapolation to a post-match state it never trains on. In a won position it is
    likely close to +1.
  - Effect 1: regulation-time wins and losses get value targets about twice as large as
    king-kill wins and losses.
  - Effect 2: the telescoping that makes PBRS policy-invariant [r26] is broken.
  - Effect 3: the critic, which the coach also reads as V(s), is distorted near the end
    of the match.
- *Impact:* removes a systematic critic bias on most matches (9–24 of 30 ended this way
  in §8). Expect better-calibrated V(s) and cleaner advantages for late-game plays.
- *Cost / risk:* one line plus a test.
  - Callers that treat "truncated" as "time-limit end" see a different flag. Only
    `done = terminated or truncated` patterns exist in `tests/` and `src/`.
  - The rule-based `rollout/win_rate_vs_logic` and `info["winner"]` are unaffected.
- *Contract / determinism:* **no** observation or action change. Fully deterministic.
  Not observable by the coach or the web client.

**H2. Select and report models by win rate, with uncertainty, against several opponents.**

- *What:*
  1. **Training-time evaluation.**
     - Replace the shaped-reward best-model selection with a callback that plays a fixed
       held-out seed set (e.g. 100 seeds, disjoint from training) against `LogicAgent`.
     - Log the win rate with a 95% Wilson interval.
     - Keep every evaluated checkpoint.
  2. **Final evaluation.**
     - Implement the ADR-0002 protocol in `evaluate_model`: 200 vs `LogicAgent` and 200
       vs `RandomAgent`, both with Wilson CIs.
     - Add ε-`LogicAgent` as a third, held-out-style opponent.
     - Evaluate the *selected* checkpoint on a second seed set that was not used for
       selection.
  3. **Training seeds.** Train every arm (full, no-shaping, no-curriculum) with ≥ 3 seeds,
     and report interval estimates across seeds. Agarwal et al. recommend stratified
     bootstrap CIs and the IQM.
- *Why:*
  - Point estimates from few runs mislead [r35, §4], and seeds alone can produce
    "statistically different distributions" [r36, Fig. 5].
  - Shaped return can diverge from win rate [r2, §VI]; ADR-0002 itself says ablations are
    judged by win rate.
  - Picking the maximum of ~40 noisy 50-episode estimates on shifting seeds biases the
    selected score upward. The re-evaluation on fresh seeds removes that bias (our
    reasoning).
  - A win rate against one fixed bot does not show general strength [r16].
- *Impact:* thesis-grade numbers and a defensible ship gate (ADR-0002: "lower bound of
  its 95% Wilson CI vs LogicAgent is above 50%").
- *Cost / risk:* low code cost; more CPU for evaluation.
  - With decision points a match costs about 0.4 s in one process (§8), so 400 matches
    take minutes.
  - Three seeds per arm triples training time.
- *Contract / determinism:* none. Fixed seed lists keep evaluation reproducible.

### Medium priority

**M1. Add a small frozen-snapshot pool after the scripted phase.**

- *What:*
  - Add a phase 3, or a share of phase-2 envs, whose opponents are frozen checkpoints of
    the learner, loaded through the existing hook (`gym_env.py:141-142`).
  - Start with a pool of 3, sampled δ-uniformly with δ ≈ 0.5 from recent snapshots.
  - Admit a snapshot after it reaches ≥ 45% against the pool.
  - Snapshots play deterministically, which is already the `RLAgent` default.
  - Keep some `LogicAgent` envs so the agent does not forget the scripted baseline.
  - Rebuilding `SubprocVecEnv` with new snapshot paths per phase, or per pool update,
    fits the current per-phase env construction (`train.py:144-148`).
- *Why:*
  - Scripted-only training overfits [r2][r4][r28].
  - Sampling past selves stabilizes self-play [r27, §4.2], and every comparable
    affordable system used a small pool: Heroic n = 3, δ = 0.5 [r4]; Generals.io N = 3
    with a 45% gate and argmax opponents [r14]; OpenAI Five 20% past versions [r9].
  - Naive latest-self self-play is weak [r2, App. C] or cyclic [r30], and Clash Royale
    is described as having cyclic strategies [r7, §I].
- *Impact:* robustness to opponents other than `LogicAgent`. H2's multi-opponent
  evaluation measures it.
- *Cost / risk:*
  - Moderate: snapshot bookkeeping and evaluation of the gate.
  - It adds MLP inference per tick of the opponent; this is cheap, but `ScheduledAgent`
    asks only at Decision points.
  - Risk of chasing cycles. PFSP-style weighting [r8] is the next step if it happens.
  - ADR-0002 lists the pool as future work, so this needs an ADR amendment.
- *Contract / determinism:* no observation or action change. Deterministic if snapshot
  selection uses a seeded RNG and fixed checkpoint paths.

**M2. Monitor and bound the PPO update.**

- *What:*
  - Track `train/approx_kl` and `train/clip_fraction`, which SB3 logs already.
  - If the KL regularly exceeds ~0.02, set `target_kl` to 0.01–0.02, and/or lower
    `n_epochs` from 10 to 4–5, and/or use fewer, larger minibatches.
  - Fix the stale numbers in `docs/rl-agent.md` lines 325 and 334.
- *Why:*
  - The phase-2 mix makes the data non-stationary across envs.
  - MAPPO found fewer epochs better under non-stationarity [r23, §5].
  - Heroic used a 0.01 KL early stop and large batches "to stabilize the training"
    [r4].
  - μRTS used 4 epochs × 4 minibatches [r2, Table III]; Atari PPO used 3 epochs [r18].
  - Our 120 gradient steps per 3,072-transition rollout is at the high end.
- *Impact:* fewer destructive updates when opponents switch, especially at the phase
  boundary.
- *Cost / risk:* trivial configuration. Too tight a `target_kl` slows learning.
- *Contract / determinism:* none.

**M3. Gate curriculum phases on performance, and report curriculum metrics.**

- *What:*
  - Switch from phase 1 to phase 2 when the rolling win rate against `RandomAgent`
    passes a threshold (e.g. 80% over 200 matches), with the current 20% as a cap.
  - In the `--no-curriculum` ablation, report jumpstart, time to a win-rate threshold
    against `LogicAgent`, and the asymptotic win rate. Count phase-1 steps as cost.
- *Why:*
  - Fixed schedules ignore the agent's progress, and automatic curricula adapt to it
    [r33].
  - HoK advances phases on Elo convergence [r11, §3.3]; Generals.io uses a win-rate gate
    [r14].
  - The curriculum literature defines exactly these metrics for comparing against
    no-curriculum training [r32, §3.3].
  - Heroic needed scripted pre-training only when the signal was too weak [r4]. Our
    `LogicAgent` beat `RandomAgent` 50–0 (v1 data in `docs/rl-agent.md`), so phase 1 may
    finish early or barely matter.
- *Impact:* less wasted budget, and a curriculum ablation the thesis can interpret.
- *Cost / risk:* small. The phase loop already rebuilds envs. The threshold is a new
  hyperparameter.
- *Contract / determinism:* none. The switch is deterministic given seeds, because
  rollouts are synchronous.

**M4. Re-derive the training budget in matches and check for plateaus.**

- *What:*
  - Recompute matches per run from decision steps (§8).
  - Plot the held-out win rate (H2) against steps for the baseline.
  - Raise `total_timesteps` (e.g. 3–5M) if the curve has not flattened.
- *Why:*
  - 1M decision steps is small next to every comparable system [r2][r4][r14].
  - Collecting more samples helped Heroic [r4].
  - The MLP is cheap (`rl-architecture-review.md` § Local measurements), so environment
    simulation dominates the cost.
- *Impact:* avoids reporting an undertrained agent. *Cost:* CPU time.
- *Contract / determinism:* none.

### Low priority

**L1. Re-justify or retune `ent_coef`.**

- The 0.02 rationale in `docs/rl-agent.md:330` addresses noop collapse under per-tick
  stepping. Decision points now remove noop-only states.
- The masked entropy scale varies between log 2 and log 33 per state (§3.2).
- Comparable systems use 0.01 [r18][r2] or 0 [r27], and on-policy studies find no
  general benefit [r21, §3.8].
- The sweep already includes 0.005 and 0.05. Add 0.01, run multiple seeds (H2), and
  consider linear annealing.
- No contract impact.

**L2. Linear learning-rate decay.** Small, consistent gains in large studies
[r21, §3.7][r20, Fig. 1]. PPO Atari and μRTS used it [r18][r2]. SB3 accepts a schedule
callable. Mind that `learn()` is called once per phase. No contract impact.

**L3. Make overtime time observable.**

- `obs[3] = min(1, t/180)` stays at 1 for the up-to-120 s of overtime (`board.py:855`),
  so remaining overtime is aliased [r25, §2].
- Options: a separate `overtime remaining / 120` feature, or rescaling the time feature
  over 300 s.
- **Breaks the observation contract.** It changes `OBS_DIM` or the meaning of index 3,
  so it needs a new observation version, updated `tests/test_observation.py`, and
  invalidates checkpoints.
- Bundle it with any other observation change; it is not worth a version bump alone.

**L4. Learned noop duration.**

- Replace the single noop with 2–3 noop actions of different holds (e.g. 0.3 s / 1 s /
  3 s).
- AlphaStar predicts when to act next [r8]. Heroic named learning the no-op duration as
  future work, because spell agents kept using no-ops [r4]. Boney et al. stress waiting
  as strategy [r7].
- **Breaks the action contract** (`NUM_ACTIONS`, coach mapping, PL/EN copy) and changes
  `DecisionSchedule`. Defer unless agents visibly mistime plays.

**L5. Privileged critic.**

- The value network gets the full Board state: opponent hand and Cycle before 4 plays,
  plus cooldowns [r8][r11][r23].
- The policy observation is unchanged, but this needs a custom policy class with split
  extractors. The coach's `V(s)` would then need full-state input, which the server has.
- The expected gain is small, because the last-4-plays features already reveal most
  hidden state. Defer.

**L6. Behaviour-cloning warm start from `LogicAgent`.**

- BC pre-training more than halved the time to a 75% win rate in LOCM [r16, §5]. AlphaStar
  and Generals.io start from imitation [r8][r14].
- `LogicAgent` places at continuous coordinates (`logic.py:95-103`), so its plays must
  be mapped onto the 13 zones first.
- Imitating the main training opponent also risks a mirror policy.
- Defer.

**L7. Logging hygiene.**

- Log the win rate per opponent type instead of the pooled `rollout/win_rate_vs_logic`.
- Change the cadence check in `train.py:91` to something that fires every ~5,000 steps
  for any `n_envs`.
- No contract impact.

### Explicitly not proposed

These already match best practice, so the literature gives no reason to change them:

- MaskablePPO with full env-side masks;
- the flat joint head;
- separate tanh MLPs with SB3 init;
- no value clipping;
- prescaled observations without `VecNormalize`;
- PBRS with the PPO γ;
- antisymmetric rewards;
- per-episode seeds;
- the scripted-opponent mix by env index.

CNN, LSTM, attention and autoregressive heads stay deferred for the reasons in
`rl-architecture-review.md`.

## 6. Open questions

- **Is γ = 0.995 per decision step right when steps vary in duration?** A noop step
  lasts 0.3 s, while a card play can be followed by seconds of noop-only ticks, so the
  per-second discount varies with the policy. With γ this close to 1 and episodes of 26–210
  steps the effect is probably small. No paper found here settles semi-Markov
  discounting for PPO in games.
- **How deterministic should training opponents be?** Generals.io found stochastic
  opponents teach agents to exploit noise [r14]. Gym-μRTS and Heroic gained from opponent
  diversity [r2][r4]. Our mix has 4 of 12 stochastic envs (ε-Logic and Random). The
  literature does not give a ratio.
- **Do the 13 zones cost strength against a continuous-placement opponent?** Removing
  actions can help learnability or lower performance depending on the game [r3, §IV].
  `LogicAgent` and humans place freely (and humans also in the pocket, ADR-0006), but the
  agent cannot. Only an ablation would show the cost.
- **How to detect exploitation of `LogicAgent`?** The game is deterministic given a seed,
  and `LogicAgent` is deterministic. A high win rate could reflect a narrow exploit
  rather than general play. Cross-play against held-out opponents (H2, M1) and human
  playtests are the available checks [r16][r37]. A formal exploitability measure is out
  of reach for this game.
- **Shaping weights.** PBRS weights (crown 0.2, HP 0.5) do not change the optimal policy
  [r26], but they do change learning speed. No paper gives weights for a Clash-Royale-like
  game. The planned no-shaping ablation is the right test, given the seeds in H2.
- **Value-function scale for the coach.** After H1, V(s) should lie in [−1, 1] plus the
  shaping potential. The coach's thresholds need retuning anyway (ADR-0002). No
  literature addresses grading thresholds for move feedback.

## 7. References

Papers (all opened; full text read unless noted):

1. <a id="ref-1"></a>S. Huang, S. Ontañón. *A Closer Look at Invalid Action Masking in
   Policy Gradient Algorithms.* FLAIRS-35, 2022. [arXiv:2006.14171](https://arxiv.org/abs/2006.14171)
2. <a id="ref-2"></a>S. Huang, S. Ontañón, C. Bamford, L. Grela. *Gym-μRTS: Toward
   Affordable Full Game Real-time Strategy Games Research with Deep Reinforcement
   Learning.* IEEE CoG 2021. [arXiv:2105.13807](https://arxiv.org/abs/2105.13807)
3. <a id="ref-3"></a>A. Kanervisto, C. Scheller, V. Hautamäki. *Action Space Shaping in
   Deep Reinforcement Learning.* IEEE CoG 2020. [arXiv:2004.00980](https://arxiv.org/abs/2004.00980)
4. <a id="ref-4"></a>M. Warchalski, D. Radojevic, M. Milosevic. *Deep RL Agent for a
   Real-Time Action Strategy Game.* 2020. [arXiv:2002.06290](https://arxiv.org/abs/2002.06290)
5. <a id="ref-5"></a>T. Liu, Z. Zheng, H. Li, K. Bian, L. Song. *Playing Card-Based RTS
   Games with Deep Reinforcement Learning.* IJCAI 2019.
   [PDF](https://www.ijcai.org/proceedings/2019/0631.pdf)
6. <a id="ref-6"></a>T. Wu, L. Wan, Y. Wang, Q. Wan, X. Lan. *Playing Non-Embedded
   Card-Based Games with Reinforcement Learning.* 2025.
   [arXiv:2504.04783](https://arxiv.org/abs/2504.04783) (venue ICIRA 2024 per search
   results, **unverified**)
7. <a id="ref-7"></a>R. Boney, A. Ilin, J. Kannala, J. Seppänen. *Learning to Play
   Imperfect-Information Games by Imitating an Oracle Planner.* 2020.
   [arXiv:2012.12186](https://arxiv.org/abs/2012.12186)
8. <a id="ref-8"></a>O. Vinyals et al. *Grandmaster level in StarCraft II using
   multi-agent reinforcement learning.* Nature 575:350–354, 2019.
   [doi:10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z); read via the
   [author PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
9. <a id="ref-9"></a>C. Berner et al. (OpenAI). *Dota 2 with Large Scale Deep
   Reinforcement Learning.* 2019. [arXiv:1912.06680](https://arxiv.org/abs/1912.06680)
10. <a id="ref-10"></a>D. Ye et al. *Mastering Complex Control in MOBA Games with Deep
    Reinforcement Learning.* AAAI 2020. [arXiv:1912.09729](https://arxiv.org/abs/1912.09729)
11. <a id="ref-11"></a>D. Ye et al. *Towards Playing Full MOBA Games with Deep
    Reinforcement Learning.* NeurIPS 2020. [arXiv:2011.12692](https://arxiv.org/abs/2011.12692)
12. <a id="ref-12"></a>O. Vinyals et al. *StarCraft II: A New Challenge for Reinforcement
    Learning.* 2017. [arXiv:1708.04782](https://arxiv.org/abs/1708.04782)
13. <a id="ref-13"></a>K. Kurach et al. *Google Research Football: A Novel Reinforcement
    Learning Environment.* AAAI 2020. [arXiv:1907.11180](https://arxiv.org/abs/1907.11180)
    (cited via `rl-architecture-review.md`; representation table re-checked)
14. <a id="ref-14"></a>M. Straka, M. Schmid. *Artificial Generals Intelligence: Mastering
    Generals.io with Reinforcement Learning.* 2025. [arXiv:2507.06825](https://arxiv.org/abs/2507.06825)
15. <a id="ref-15"></a>W. Xi, Y. Zhang, S. Deng, H. Liang, C. Xiao, J. Chen, X. Huang et
    al. (ByteDance). *Mastering Strategy Card Game (Legends of Code and Magic) via
    End-to-End Policy and Optimistic Smooth Fictitious Play.* 2023.
    [arXiv:2303.04096](https://arxiv.org/abs/2303.04096)
16. <a id="ref-16"></a>R. Haluška. *Learning to Beat ByteRL: Exploitability of Collectible
    Card Game Agents.* 2024. [arXiv:2404.16689](https://arxiv.org/abs/2404.16689)
17. <a id="ref-17"></a>D. Zha et al. *DouZero: Mastering DouDizhu with Self-Play Deep
    Reinforcement Learning.* ICML 2021. [arXiv:2106.06135](https://arxiv.org/abs/2106.06135)
    (abstract only)
18. <a id="ref-18"></a>J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov.
    *Proximal Policy Optimization Algorithms.* 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
19. <a id="ref-19"></a>J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel.
    *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR
    2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
20. <a id="ref-20"></a>L. Engstrom, A. Ilyas, S. Santurkar, D. Tsipras, F. Janoos,
    L. Rudolph, A. Madry. *Implementation Matters in Deep Policy Gradients: A Case Study
    on PPO and TRPO.* ICLR 2020. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
21. <a id="ref-21"></a>M. Andrychowicz et al. *What Matters In On-Policy Reinforcement
    Learning? A Large-Scale Empirical Study.* 2020.
    [arXiv:2006.05990](https://arxiv.org/abs/2006.05990) (v1 read; the ICLR 2021 venue
    mentioned in the earlier note is **unverified** here)
22. <a id="ref-22"></a>S. Huang, R. F. J. Dossa, A. Raffin, A. Kanervisto, W. Wang.
    *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track 2022.
    [link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
23. <a id="ref-23"></a>C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, Y. Wu.
    *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.* NeurIPS 2022
    Datasets and Benchmarks. [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
24. <a id="ref-24"></a>Z. Ahmed, N. Le Roux, M. Norouzi, D. Schuurmans. *Understanding
    the Impact of Entropy on Policy Optimization.* ICML 2019.
    [arXiv:1811.11214](https://arxiv.org/abs/1811.11214) (abstract and introduction)
25. <a id="ref-25"></a>F. Pardo, A. Tavakoli, V. Levdik, P. Kormushev. *Time Limits in
    Reinforcement Learning.* ICML 2018. [arXiv:1712.00378](https://arxiv.org/abs/1712.00378)
26. <a id="ref-26"></a>A. Y. Ng, D. Harada, S. Russell. *Policy Invariance Under Reward
    Transformations: Theory and Application to Reward Shaping.* ICML 1999, pp. 278–287.
    [mlanthology](https://mlanthology.org/icml/1999/ng1999icml-policy/); full text read
    via the [author PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
27. <a id="ref-27"></a>T. Bansal, J. Pachocki, S. Sidor, I. Sutskever, I. Mordatch.
    *Emergent Complexity via Multi-Agent Competition.* ICLR 2018.
    [arXiv:1710.03748](https://arxiv.org/abs/1710.03748)
28. <a id="ref-28"></a>M. Lanctot, V. Zambaldi, A. Gruslys, A. Lazaridou, K. Tuyls,
    J. Pérolat, D. Silver, T. Graepel. *A Unified Game-Theoretic Approach to Multiagent
    Reinforcement Learning.* NIPS 2017. [arXiv:1711.00832](https://arxiv.org/abs/1711.00832)
29. <a id="ref-29"></a>J. Heinrich, D. Silver. *Deep Reinforcement Learning from
    Self-Play in Imperfect-Information Games.* 2016.
    [arXiv:1603.01121](https://arxiv.org/abs/1603.01121) (abstract and §4)
30. <a id="ref-30"></a>D. Hernandez, K. Denamganaï, S. Devlin, S. Samothrakis,
    J. A. Walker. *A Comparison of Self-Play Algorithms Under a Generalized Framework.*
    2020. [arXiv:2006.04471](https://arxiv.org/abs/2006.04471) (extends "A Generalized
    Framework for Self-Play Training", IEEE CoG 2019)
31. <a id="ref-31"></a>W. M. Czarnecki, G. Gidel, B. Tracey, K. Tuyls, S. Omidshafiei,
    D. Balduzzi, M. Jaderberg. *Real World Games Look Like Spinning Tops.* 2020.
    [arXiv:2004.09468](https://arxiv.org/abs/2004.09468) (abstract and introduction)
32. <a id="ref-32"></a>S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor,
    P. Stone. *Curriculum Learning for Reinforcement Learning Domains: A Framework and
    Survey.* JMLR 21, 2020. [arXiv:2003.04960](https://arxiv.org/abs/2003.04960)
33. <a id="ref-33"></a>R. Portelas, C. Colas, L. Weng, K. Hofmann, P.-Y. Oudeyer.
    *Automatic Curriculum Learning For Deep RL: A Short Survey.* 2020.
    [arXiv:2003.04664](https://arxiv.org/abs/2003.04664)
34. <a id="ref-34"></a>J. Z. Leibo, E. Hughes, M. Lanctot, T. Graepel. *Autocurricula
    and the Emergence of Innovation from Social Interaction.* 2019.
    [arXiv:1903.00742](https://arxiv.org/abs/1903.00742) (abstract)
35. <a id="ref-35"></a>R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville,
    M. G. Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical
    Precipice.* NeurIPS 2021. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
36. <a id="ref-36"></a>P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup,
    D. Meger. *Deep Reinforcement Learning that Matters.* AAAI 2018.
    [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
37. <a id="ref-37"></a>D. Balduzzi, K. Tuyls, J. Pérolat, T. Graepel. *Re-evaluating
    Evaluation.* NeurIPS 2018. [arXiv:1806.02643](https://arxiv.org/abs/1806.02643)
38. <a id="ref-38"></a>V. Zambaldi et al. *Relational Deep Reinforcement Learning.* 2018.
    [arXiv:1806.01830](https://arxiv.org/abs/1806.01830) (abstract)
39. <a id="ref-39"></a>T. Ni, B. Eysenbach, R. Salakhutdinov. *Recurrent Model-Free RL Can
    Be a Strong Baseline for Many POMDPs.* ICML 2022.
    [arXiv:2110.05038](https://arxiv.org/abs/2110.05038) (abstract)

Implementation documentation and source **[impl]** (describes library behaviour only):

- <a id="doc-1"></a>D1. sb3-contrib MaskablePPO docs,
  [`docs/modules/ppo_mask.md`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.md).
- D2. sb3-contrib 2.8.0 source, read from the installed package:
  `sb3_contrib/ppo_mask/ppo_mask.py:248-260` (timeout bootstrap) and
  `sb3_contrib/common/maskable/distributions.py:59-82` (masking, entropy).
  Repository: [Stable-Baselines-Team/stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib).
- D3. stable-baselines3 2.8.0 source, read from the installed package:
  `common/vec_env/subproc_vec_env.py:38`, `common/vec_env/dummy_vec_env.py:66-81`,
  `common/vec_env/base_vec_env.py:96-100`.
- <a id="doc-5"></a>D5. Gymnasium, *Handling Time Limits*,
  [tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## 8. Local measurements

These were measured with a throwaway script on this machine on 2026-10-23 (CPU, one
process). They come from no paper and are indicative only.

`ClashRoyaleEnv(seed=1, opponent="logic")` ran 30 matches per scripted P0 policy:

| P0 policy | Decision steps per match (mean / min / max) | Legal actions at a Decision point (mean / max) | Ended by king kill (`terminated`) | Ended by time / crowns (`truncated`) |
|---|---|---|---|---|
| Always a random legal play | 26 / 12 / 49 | 13.5 / 33 | 6 | **24** |
| Noop 50%, else a random play | 41 / 17 / 62 | 13.5 / 33 | 16 | **14** |
| Noop 90%, else a random play | 210 / 119 / 344 | 19.8 / 33 | 21 | **9** |

- 30 matches took about 11–13 s, about 0.4 s per match including `LogicAgent`.
- 1M decision steps is therefore roughly 5,000–38,000 matches, depending on how often the
  policy waits.
- At least 30% of matches, and up to 80% under these policies, end through the
  `truncated` path that H1 fixes.

## Unverified or limited items

- **Venues not confirmed from the PDFs.** KataCR's venue (ICIRA 2024) comes from search
  results. Andrychowicz et al.'s ICLR 2021 venue was not confirmed from the arXiv v1
  read here.
- **Read as abstracts or in part only.** DouZero, Zambaldi et al., Ni et al. and Leibo et
  al. were read as abstracts. Ahmed et al., Czarnecki et al. and Heinrich & Silver were
  read only in part (abstract, introduction and the cited sections). They support
  general statements only.
- **Garbled text.** The Honor of Kings 1v1 ablation table (Table 5) did not extract
  cleanly. Only the 50.5% action-mask result and the "largely reduce the training time"
  statement are cited.
- **Our reasoning, not the papers'.** The claim that the selected "best" checkpoint's
  score is biased upward is our statistical reasoning, supported by the repo's own v1
  result. No cited paper studies checkpoint selection under shifting evaluation seeds.
  The same holds for the claim that the masked-entropy scale varies with the number of
  legal actions.
- **No Clash Royale self-play paper found.** No peer-reviewed Clash Royale paper with
  online PPO or self-play was found. Community repositories were not used as evidence.

[r1]: #ref-1
[r2]: #ref-2
[r3]: #ref-3
[r4]: #ref-4
[r5]: #ref-5
[r6]: #ref-6
[r7]: #ref-7
[r8]: #ref-8
[r9]: #ref-9
[r10]: #ref-10
[r11]: #ref-11
[r12]: #ref-12
[r13]: #ref-13
[r14]: #ref-14
[r15]: #ref-15
[r16]: #ref-16
[r17]: #ref-17
[r18]: #ref-18
[r19]: #ref-19
[r20]: #ref-20
[r21]: #ref-21
[r22]: #ref-22
[r23]: #ref-23
[r24]: #ref-24
[r25]: #ref-25
[r26]: #ref-26
[r27]: #ref-27
[r28]: #ref-28
[r29]: #ref-29
[r30]: #ref-30
[r31]: #ref-31
[r32]: #ref-32
[r33]: #ref-33
[r34]: #ref-34
[r35]: #ref-35
[r36]: #ref-36
[r37]: #ref-37
[r38]: #ref-38
[r39]: #ref-39
[D1]: #doc-1
[D5]: #doc-5
