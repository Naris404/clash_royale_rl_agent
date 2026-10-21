# Reopen env-v1 as env-v1.1 before any v2 training

Status: accepted (2026-10-21). Amends ADR-0003.

ADR-0003 froze the mechanics at env-v1. The v2 zone layout (ADR-0002) exposed three
geometry problems that only a mechanics change can fix, so the environment is reopened
once, before any v2 training run, and frozen again as **env-v1.1** after the user's
playtest sign-off. Reopening now invalidates no results, because no v2 model has been
trained; the risk ADR-0003 guarded against only starts with the first training run.

- **King tower 4×4 with one free tile behind it.** King towers move from y = 2.0 / 30.0
  to y = 3.0 / 29.0 (collision radius stays 2.0). This makes room for the back zones
  behind the king, and the king's range reaches one tile further into its own half.
- **Deploying behind the king.** The deploy area runs to the back edge: y ≥ 0.5 for
  player 0 and y ≤ 31.5 for player 1, for humans and agents alike. The spell area
  extends the same way.
- **Bridges at x = 3.5 / 14.5.** They were at 3.0 / 14.0 while the princess and king
  towers sit at 3.5 / 14.5 / 9.0, so the arena was not mirror-symmetric and identical
  zones pulled with different margins per lane. The bridges now line up with the towers.

## Consequences

- env-v1.1 is tagged only after a playtest sign-off; all v2 training and evaluation run
  on env-v1.1.
- Layout tests, the web arena (bridges come from the server config) and LogicAgent's
  lane logic follow the new constants.
