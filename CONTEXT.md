# Clash Royale RL Coach

A thesis project: an RL agent trained in a deterministic Clash Royale environment
built for it. The live browser game and the coach exist to show the agent in action.
This file is the shared vocabulary. The repo is English; web UI copy is bilingual
(EN default, PL optional).

## Game

**Board**:
The deterministic 18×32 arena simulation; the single source of truth for a match.
_Avoid_: arena (that is the rendered view), game state

**Deploy zone**:
A named region where a card may be played. Humans place at continuous coordinates;
agents and the coach reason in discrete zones. After an enemy princess tower falls, the
destroyer (humans only, ADR-0006) can also deploy in a 4×9 **pocket** on that side.
_Avoid_: drop zone, placement area, slot

**Deck**:
The fixed 8 cards both players use in every match (a mirror match): Knight, Giant,
Hog Rider, Musketeer, Cannon, Fireball, Arrows, Baby Dragon. Never chosen by a player.
_Avoid_: card pool, loadout

**Cycle**:
The deck rotating through a 4-card hand, one visible next card, and a queue;
playing a card sends it to the back of the queue and draws the next card.
_Avoid_: rotation

**Bridge lane**:
One of the two river crossings that ground units must path through (Hog Rider jumps).
_Avoid_: crossing, path

**Pull zone**:
A deploy zone positioned so an enemy building-targeter is drawn to a building
planted there instead of a tower (one per lane, near the center).
_Avoid_: bait zone

**Spell zone**:
A deploy zone that only spells may target: fixed on an enemy tower, or a push zone
that aims at the best enemy cluster in one lane wherever it stands. Masked off for
troops and buildings.
_Avoid_: target zone

**Crown tower damage**:
The reduced damage a spell deals to princess and king towers (`tower_damage` in the
card data file); units and buildings like the Cannon take the full spell damage.
_Avoid_: tower damage

**Deploy time**:
The delay after a card is placed before its unit can move or attack; the unit can
already be targeted and damaged. Spells have none.
_Avoid_: spawn delay, cast time

**Collision**:
Units and buildings occupy space: overlapping units push each other apart, the
heavier (by card mass) giving way less, while buildings and towers never move.
Pushing never carries a ground unit into the river off a bridge lane.
_Avoid_: overlap handling, separation

**King tower activation**:
The king tower stays idle until it takes any damage (including splash or spells) or
one of its princess towers is destroyed; once active it stays active.
_Avoid_: king wake-up, king trigger

## Match

**Crown**:
The score unit of a match: 1 per princess tower destroyed; destroying the king tower
gives 3 crowns and ends the match at once.
_Avoid_: point, star

**Regulation**:
The first 3:00 of a match; the player with more crowns at its end wins.
_Avoid_: normal time, main time

**Double elixir**:
Elixir regenerating at twice the normal rate, during the last 1:00 of regulation and
all of overtime.
_Avoid_: elixir boost, 2x

**Overtime**:
Up to 2:00 played when crowns are tied after regulation; the first crown wins
(sudden death).
_Avoid_: extra time, sudden death

**Tiebreak**:
The rule when overtime ends without a crown: the player whose weakest tower has less
HP loses; equal weakest towers give a draw.
_Avoid_: HP comparison, decision

**Match end**:
A match ends only by the rules: a king tower falls, regulation ends with unequal
crowns, the first crown is scored in overtime, or overtime runs out (Tiebreak or
draw). A match is never cut short.
_Avoid_: timeout, truncation

## Platform

**Session**:
A live browser match driven authoritatively by the server over WebSocket at 10 Hz.
_Avoid_: room, lobby, game instance

**Snapshot**:
The authoritative per-tick state sent to the client, which only renders it.
_Avoid_: frame, state update

**Coach**:
Move suggestions and post-hoc move grades surfaced during a session; heuristic
(rule-based) or neural (policy-based).
_Avoid_: assistant, advisor

**Hint**:
A proactive coach suggestion for the next move.
_Avoid_: tip, recommendation

**Key moment**:
A large evaluation swing surfaced to the player.
_Avoid_: highlight, turning point

**Opponent**:
Who the human plays against: the rule-based bot, the RL policy (a model from the
Model catalogue), or the bot with the coach enabled.
_Avoid_: enemy AI

## Model lineage

**v1 model**:
The original MaskablePPO agent trained on the pre-env-v1 game (70% win rate vs the
rule-based bot). A historical result only; it cannot run in env-v1.
_Avoid_: old model, legacy model, baseline

**v2 model**:
The rewritten RL agent, trained only on env-v1.1 and evaluated against the rule-based
and random bots.
_Avoid_: new model

**v3 model**:
A v2 Final model fine-tuned further by self-play against Past selves and Exploiters.
It is v2 plus a fine-tuning stage, not a new architecture.
_Avoid_: v2.1, retrained model

**Shipped model**:
The v2 checkpoint chosen by the selection protocol (ADR-0002) that passed the Ship
gate. It is the default RL Opponent and the only model the Coach uses.
_Avoid_: best model, default model

**Model catalogue**:
The fixed, curated list of models a player may choose as the RL Opponent before a
Session; the Shipped model is its default entry. A player picks an entry, never a file.
_Avoid_: model list, model registry

## Training

**Training opponent**:
A bot or frozen policy the agent plays against during training or evaluation.
_Avoid_: opponent (that is who the human plays in a Session), sparring partner

**Decision point**:
A tick at which the agent is asked for an action: at least one card play is legal
and no noop hold is running. Choosing noop starts a short hold (0.3 s by default)
before the agent is asked again. Training and the live RL opponent use the same
schedule.
_Avoid_: step, frame, agent tick

**Selection seeds**:
The fixed set of matches used only to choose which checkpoint of a training run
becomes its model. Never used for reported results.
_Avoid_: validation set, eval seeds

**Selected model**:
The checkpoint of a training run that the selection seeds choose as that run's model.
It may come from any Training phase, including one before self-play.
_Avoid_: best model, self-play model

**Reference pool**:
The opponents that score checkpoints during selection: LogicAgent plus any frozen
models. A checkpoint's selection score is its average win rate over the pool, and ties
go to the later checkpoint (ADR-0007).
_Avoid_: league, evaluation set

**Final model**:
The last checkpoint of a training run, whatever its selection score.
_Avoid_: latest model, self-play model

**Report seeds**:
The fixed set of matches, disjoint from the selection seeds, on which every reported
result and the ship gate are measured.
_Avoid_: test set, eval seeds

**Ship gate**:
The condition for the v2 model to replace the rule-based bot as the RL opponent in a
Session: on the report seeds, the lower bound of the 95% Wilson interval of its win
rate against the rule-based bot is above 50%.
_Avoid_: release criterion, acceptance

**Past self**:
A frozen earlier version of the agent used as a Training opponent.
_Avoid_: snapshot (that is the per-tick client state), old model

**Past-self pool**:
The small set of past selves the agent plays during the self-play phase; a new past
self joins only by beating the pool.
_Avoid_: snapshot pool, league

**Exploiter**:
An RL agent trained only to beat one frozen version of the main agent, then used as a
Training opponent to expose that agent's weaknesses.
_Avoid_: counter-bot, adversary

**Training phase**:
One stage of a training run with its own set of Training opponents: warm-up, scripted
mix, then self-play mix.
_Avoid_: stage, curriculum step

**Ablation arm**:
A training configuration that differs from the full configuration in exactly one
thing, trained on the shorter ablation budget.
_Avoid_: variant, experiment

## Environment lifecycle

**env-v1**:
The environment version frozen after the user's playtest sign-off; mechanics may
change before it and are frozen after it (see docs/adr/0003).
_Avoid_: final game, mechanics freeze

**env-v1.1**:
env-v1 reopened once, before any v2 training, for arena geometry fixes, then frozen
again after a new playtest sign-off (see docs/adr/0005).
_Avoid_: env-v2, patched env

**Playtest sign-off**:
The user's explicit statement, after playing in the web client, that a mechanic or
the whole environment works as intended. Passing tests alone is not a sign-off.
_Avoid_: QA, acceptance
