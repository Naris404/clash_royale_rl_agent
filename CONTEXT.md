# Clash Royale RL Coach

Deterministic Clash Royale simulation with RL training, a live browser game, and a
move coach. This file is the shared vocabulary. Code identifiers and comments are
English; web UI copy is bilingual (PL/EN).

## Game

**Board**:
The deterministic 18×32 arena simulation; the single source of truth for a match.
_Avoid_: arena (that is the rendered view), game state

**Deploy zone**:
A named region where a card may be played. Humans place at continuous coordinates;
agents and the coach reason in discrete zones.
_Avoid_: drop zone, placement area, slot

**Cycle**:
The fixed 6-card deck rotating through a 4-card hand; playing a card draws the next
one in queue. The deck is fixed, not user-chosen.
_Avoid_: deck, rotation

**Bridge lane**:
One of the two river crossings that ground units must path through (Hog Rider jumps).
_Avoid_: crossing, path

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
The original MaskablePPO checkpoint (`models/ppo_cr_best.zip`), frozen as the thesis
baseline after the refactor.
_Avoid_: old model, legacy model

**v2 model**:
The rewritten RL agent (new observation/action/reward/curriculum); it ships in the
game only once it beats the v1 model in evaluation.
_Avoid_: new model

**Mechanics freeze**:
The rule that simulation semantics stay identical during the refactor; the only
exception is the diagonal-movement fix (see docs/adr/0001).
