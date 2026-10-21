# Pocket deploy zone is humans-only

Status: accepted (2026-10-21). Amends ADR-0005.

In Clash Royale, destroying an enemy princess tower lets the destroyer deploy troops and
buildings in a pocket on the enemy side of that tower's lane. The Board implements it
(`Board.pocket_zones`): a rectangle 4 tiles deep from the enemy river bank and 9 tiles
wide (one lane half, x 0–9 or 9–18), measured in the real game. The usual 1-tile side
margin still applies, so the playable pocket is 8 wide (x 1–9 or 9–17).
The server validates placements against it and sends the rectangles in every snapshot, so
the web arena highlights and accepts them without a rule copy of its own.

**Exception to ADR-0005's "humans and agents share the legal area".** Agents keep the
53-action layout of #38: the pocket adds no zone, so `valid_action_mask` is unchanged and
agents and the rule-based bot never play in it. The pocket is a rare late-game state, and
a layout change would delay v2 training.

## Consequences

- A human who has taken a princess tower can deploy where an agent cannot, so agents are
  slightly under-powered in that state. This is a recorded limitation, not a bug.
- Lifting the exception later means adding pocket zones to the layout, which changes the
  action space and so invalidates trained checkpoints.
