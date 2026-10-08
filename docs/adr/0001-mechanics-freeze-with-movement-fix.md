# Mechanics freeze with a diagonal-movement exception

Status: superseded by ADR-0003

During the `/game` refactor and the RL rewrite, simulation mechanics (board, cards,
elixir, combat, win conditions) stay semantically frozen so the v1 checkpoint and
coach grading remain interpretable. The single exception is the bridge waypoint
logic, which produced L-shaped movement ("first sideways, then straight"): units now
walk diagonally to the bridge bank, cross, then continue diagonally to the target.

## Consequences

- The movement fix changes dynamics, so the v1 checkpoint is stale: it stays frozen
  as the thesis baseline, and the RL opponent improves only once the v2 model is
  trained.
- Observation/action layout changes belong to the RL v2 work and land together with
  retraining, never inside the game refactor.
