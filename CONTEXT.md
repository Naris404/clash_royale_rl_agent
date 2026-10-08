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
agents and the coach reason in discrete zones.
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
A deploy-only zone on the enemy side that spells may target; masked off for
troops and buildings.
_Avoid_: target zone

**Deploy time**:
The delay after a card is placed before its unit can move or attack; the unit can
already be targeted and damaged. Spells have none.
_Avoid_: spawn delay, cast time

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
Who the human plays against: the rule-based bot, the RL policy, or the bot with the
coach enabled.
_Avoid_: enemy AI

## Model lineage

**v1 model**:
The original MaskablePPO agent trained on the pre-env-v1 game (70% win rate vs the
rule-based bot). A historical result only; it cannot run in env-v1.
_Avoid_: old model, legacy model, baseline

**v2 model**:
The rewritten RL agent, trained only on env-v1 and evaluated against the rule-based
and random bots.
_Avoid_: new model

## Environment lifecycle

**env-v1**:
The environment version frozen after the user's playtest sign-off; mechanics may
change before it and are frozen after it (see docs/adr/0003).
_Avoid_: final game, mechanics freeze

**Playtest sign-off**:
The user's explicit statement, after playing in the web client, that a mechanic or
the whole environment works as intended. Passing tests alone is not a sign-off.
_Avoid_: QA, acceptance
