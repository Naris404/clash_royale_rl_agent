# 02: 11-zone action space

**What to build:** The agent and coach reason over 11 meaningful deploy zones per
side instead of 3 coarse ones: 8 troop/building zones (back / mid / bridge per
lane, plus two mathematically derived pull zones that draw building-targeters like
Hog Rider away from towers) and 3 spell-only zones on the enemy side (both
princess towers + king). The nominal action space becomes 45 (noop + 4 hand slots
× 11 zones), but card-type and elixir masking keep effective choices small (≤8 for
troops, 3 for spells). Zone coordinates are absolute and carry allowed card types,
replacing the old building/spell placement hacks. Coach hints and grades map onto
the new zones.

**Blocked by:** 01 (same simulation module; sequential edits).

**Status:** ready-for-agent

- [ ] Zone layout pinned by tests: 11 zones per side with coordinates and allowed card types (pull zones at (5.5, 9.5) / (12.5, 9.5) for P0, mirrored for P1)
- [ ] `NUM_ACTIONS` = 45; action masks pin spell-only vs troop zones and elixir affordability
- [ ] Building/spell placement hacks removed; zones carry absolute coordinates and card-type rules
- [ ] A Cannon planted in a pull zone draws an enemy Hog Rider off the tower lane (behavior test)
- [ ] Coach hints and grades reference the new zones; coach tests updated
- [ ] `python -m pytest` green
