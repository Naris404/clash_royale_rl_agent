# env-v1 playtest checklist

The playtest that gates the **env-v1** freeze (ADR-0003, issue #20). Claude played in
the web client through Playwright; items it cannot judge are listed under
[Needs the user](#needs-the-user). A **Playtest sign-off** (see `CONTEXT.md`) is still the
user's explicit statement, so the freeze waits for it.

## How it was run

```
python -m uvicorn cr_rl.server.app:app --port 8123      # serves web/dist
PYTHONPATH=src python scripts/playtest_env_v1.py --url http://localhost:8123 --matches 4
```

`scripts/playtest_env_v1.py` opens real matches against the rule-based bot in headless
Chromium, plays random legal moves through the UI (select a card, click the arena), and
checks every snapshot the server streams. Re-run it after any mechanics change.
Targeted situations that a random match rarely produces are covered by the scenario
tests in `tests/`. Screenshots go to `--out`.

## Results

Status: ✅ passed, 🔧 found a bug and fixed it, 👤 needs the user.

| # | Mechanic | How it was checked | Status |
|---|---|---|---|
| 1 | English UI by default, PL toggle | Browser: home and HUD copy in both languages | ✅ |
| 2 | Deploy time: 1.0 s for every card, no walking while deploying, targetable meanwhile | Playtest script (every deployed unit, 4 matches); `TestDeployTime` | ✅ |
| 3 | Spells act on the tick they are cast | `TestDeployTime` | ✅ |
| 4 | Collisions: no stacking (the script allows 0.35 tile; the worst observed was 0.05), heavier unit gives way less, buildings never move | Playtest script; `tests/test_collisions.py` | ✅ |
| 5 | Ground units never stand in the river off a bridge | Playtest script | 🔧 see below |
| 6 | King tower asleep until damaged or a princess falls, then awake for good; a sleeping king never shoots | Playtest script (`king_active` vs tower HP in every snapshot); `TestKingTower`, including the Musketeer that finishes a princess tower | ✅ |
| 7 | Tower reach: no shot longer than range plus the largest unit radius (0.75); longest observed princess shot 8.25 (a Giant at the edge of 7.5), king 7.7 | Playtest script; `test_tower_shoots_only_in_range`. It cannot tell whether a shot is "too early" | ✅ range / 👤 feel |
| 8 | Crowns: 1 per princess, 3 for the king and the match ends at once | Playtest script (crowns recomputed from tower state every tick); `TestCrowns` | ✅ |
| 9 | Regulation winner by crowns, not tower HP | Playtest script (winner vs crowns in every match); `TestCrowns` | ✅ |
| 10 | Double elixir from 2:00 (regen 0.714/s vs 0.357/s), also in overtime, no triple | Playtest script (measured regen per phase); `TestDoubleElixir` | ✅ |
| 11 | Overtime: starts on tied crowns, first crown wins at once | Browser reached overtime and ended on a first crown; `TestOvertime` | ✅ |
| 12 | Overtime tiebreak by weakest tower, equal is a draw | `TestOvertime` only (a random bot match does not reach it) | ✅ unit tests / 👤 optional |
| 13 | HUD: timer, phase label (Regulation, Double elixir, Overtime), crowns, sleeping-king icon, end modal with Rematch | Browser screenshots | ✅ |
| 14 | Cycle: 4-card hand, next card shown, never repeats inside the hand | Playtest script (every snapshot) | ✅ |
| 15 | Arrows and Fireball: reduced damage on towers (93 / 159), full damage on units and the Cannon | Playtest script (damage of every spell hit by target kind); `TestSpells` | ✅ (Fireball value checked against the game by the owner) |
| 16 | Baby Dragon: flies over the river, splash, hit by Musketeer and towers only | `tests/test_air_and_splash.py` | ✅ |
| 17 | Ground-targeting troops attack enemy buildings; the Giant walks past troops | `TestTargetingAndCombat` | ✅ |
| 18 | Simultaneous attack resolution (a tick's hits land together) | `TestTargetingAndCombat` | ✅ |
| 19 | Unit just past the far bank keeps walking | `TestRiverAndBridges` | ✅ |
| 20 | Placement in the UI: water, enemy half and out-of-bounds clicks are ignored, bank accepted | Browser clicks | 🔧 see below |
| 21 | Coach mode starts and sends heuristic hints | Browser | ✅ |
| 22 | Match vs AI starts without a model | Browser (no `ppo_cr_best.zip` loaded; needs the v2 model, #6) | 👤 |

### Bug found and fixed: deploying in the river

The deploy zone reached `RIVER_Y ∓ 0.5`, which is inside the river water (it spans
`RIVER_Y ± 1.2`). A troop or building placed there, by a player or by the rule-based bot's
counter-play, stood in the water until it started moving and then **teleported to the
nearest bridge bank**. The zone now ends at the water's edge (`RIVER_Y ∓ RIVER_HALF_WIDTH`)
for troops and buildings; spells still reach everywhere. This shrinks the legal region near the
river by 0.7 tile, so the action mask differs slightly from before; no model is trained on env-v1 yet. The web client uses the same
boundary, and the bot's clamps follow it (its Hog Rider bridge play now deploys at the
bank). The 11 discrete deploy zones, so the action layout, are unchanged.

## Needs the user

Things Claude cannot judge from the simulation or the browser:

- **Tower reach and bridge feel** (issue #20 comment). Measured: a princess tower at y = 8
  hits a unit whose edge is within 7.5 tiles, so it reaches the middle of the bridge
  (y ≈ 16) and never fires beyond its range. Whether that matches how the game feels at the
  bridge, especially against a Musketeer, is a feel call.
- **How the arena looks and moves** (sprites, projectile arcs, spell animations,
  readability at 1×) was not evaluated.
- **Match vs AI** needs a model trained on env-v1 (#6, #7); this is expected, not a bug.
- The Musketeer that kills a princess tower and walks up to the king is covered by
  `TestKingTower`, but a random bot match did not produce it live. Play it once if you
  want to see it.
