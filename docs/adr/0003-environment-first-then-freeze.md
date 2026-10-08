# Environment first, frozen at env-v1, then RL

Status: accepted (2026-10-08). Supersedes ADR-0001.

The thesis is the RL agent, and the environment it trains in is part of the thesis
work, so the environment is finished before any RL v2 training. Mechanics may change
until the user's playtest sign-off, at which point the environment is tagged
**env-v1** and frozen; all v2 training and evaluation run on env-v1. The alternative,
keeping the minimal 6-card game frozen (ADR-0001) and training now, was rejected: any
later mechanic change would invalidate the trained agent and its results.

## Consequences

- env-v1 adds unit collisions, crowns and the real win rule, king tower activation,
  double elixir and overtime, deploy time for every card, air units and splash.
- The v1 model cannot run in env-v1. It is kept as a historical result only, and the
  "v2 ships only if it beats v1" gate in ADR-0002 is replaced by evaluation against
  the rule-based and random bots.
- RL v2 work (observation, reward, training) is blocked until env-v1 is tagged.
