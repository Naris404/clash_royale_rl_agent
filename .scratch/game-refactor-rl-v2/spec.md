# /game refactor & RL v2 — spec

Status: ready-for-agent

Vocabulary per `CONTEXT.md`. Decisions per `docs/adr/0001` (mechanics freeze) and
`docs/adr/0002` (RL v2 architecture). Phase detail in `docs/refactor-plan.md`.

## Problem Statement

The project grew organically and the playable web game (`/game`) shows it: units
walk L-shaped paths ("first sideways, then straight") instead of diagonally;
refreshing the page kills the match with no way back; placement rules are
duplicated between client and server and can drift; `vs_rl` mode silently degrades
to the rule-based bot when the model file is missing; the UI is ad-hoc emoji with
no opponent picker and no post-game analysis. Underneath, the v1 RL agent was
trained against a single opponent with a tower-HP-only reward and a coarse 3-zone
action space — it overfits `LogicAgent`, cannot express real Clash Royale plays
(Giant behind the king, center Cannon pulling a Hog Rider, Fireball on a tower),
and there is no systematic way to measure whether a new agent is actually better.

## Solution

A documented, gated four-phase refactor: (1) natural diagonal movement via bridge
banks; (2) backend protocol v2 with session resume, server-authoritative
placement, and honest opponent signaling; (3) a redesigned bilingual (PL/EN) UI
with official card art, an opponent picker, combat feedback (damage numbers,
hit flashes, death poofs, projectile arcs), and a post-game stats screen;
(4) an RL v2 rewrite — 11-zone masked action space with mathematically derived
pull zones, enriched observations, toggleable shaped rewards, curriculum training
— that replaces the frozen v1 baseline in the game only if it beats it under a
fixed-seed evaluation protocol.

## User Stories

1. As a player, I want units to walk diagonally toward their target, so that movement looks natural like in real Clash Royale.
2. As a player, I want units to cross the river only at bridge lanes (Hog Rider jumps), so that arena geometry matters.
3. As a player, I want to pick my opponent (rule-based bot / RL policy / bot with coach) before the match, so that I control the experience.
4. As a player, I want an honest indication when the RL model is unavailable and I'm facing the bot, so that I know what I'm playing against.
5. As a player, I want to refresh the page and resume my match, so that I don't lose progress.
6. As a player, I want a direct link to my match (`/game/{id}`), so that I can rejoin or share it.
7. As a player, I want valid placement areas rendered from server-provided geometry, so that I can't attempt illegal plays.
8. As a player, I want a rejected play to come back with a reason, so that I learn the rules.
9. As a player, I want official Clash Royale card art in my hand and on arena units, so that the game looks authentic.
10. As a player, I want emoji fallback when art isn't cached, so that the game always works offline.
11. As a player, I want damage numbers and hit flashes, so that combat feels responsive.
12. As a player, I want death poofs and projectile arcs, so that I can follow the battle.
13. As a player, I want tower HP bars at the arena edges, so that I can assess the match at a glance.
14. As a player, I want the elixir bar and next card next to my hand, so that I can plan plays.
15. As a player, I want pause and one consistent speed range everywhere, so that I can analyze or fast-forward.
16. As a player, I want a post-game stats screen (result, damage dealt/taken, elixir efficiency, grades, key moments), so that I can review my performance.
17. As a player, I want rematch to fully reset client state via a fresh `hello`, so that a new match starts cleanly.
18. As a player, I want the UI in Polish or English, so that I can play in my language.
19. As a player, I want a collapsible coach panel, so that it doesn't crowd the arena when I don't need it.
20. As a coached player, I want hints mapped to meaningful zones (including pull zones), so that suggestions are actionable.
21. As a coached player, I want the reason shown for each grade and suggestion, so that I learn why a move is good.
22. As a coached player, I want key moments logged during the match, so that I can review turning points afterwards.
23. As the thesis author, I want the v1 model frozen as a baseline, so that I can compare v1 vs v2 in the thesis.
24. As the thesis author, I want an ablation matrix (full / no-shaping / no-curriculum), so that each component's contribution is measured.
25. As the RL developer, I want an 11-zone action space (8 troop incl. 2 pull + 3 spell-only), so that the agent can express real CR strategies.
26. As the RL developer, I want action masking to keep effective choices small, so that training stays tractable despite 45 nominal actions.
27. As the RL developer, I want enriched observations (elixir advantage, per-lane threat mass, cycle state, bridge distances), so that the agent can make informed decisions.
28. As the RL developer, I want shaped rewards with toggles, so that credit assignment improves while ablations stay possible.
29. As the RL developer, I want curriculum training (Random → Logic) with a self-play hook kept for later, so that the agent doesn't overfit one opponent.
30. As the RL developer, I want a fixed-seed 200-episode evaluation protocol against multiple opponents, so that results are reproducible.
31. As the RL developer, I want v2 to ship in the game only if it beats v1, so that the game never regresses.
32. As a maintainer, I want one canonical way to run everything (console scripts, Docker script), so that documentation stays truthful.
33. As a maintainer, I want the server to be the only placement validator, so that rules can't drift between client and server.
34. As a maintainer, I want a protocol version field, so that future breaking changes are detectable.

## Implementation Decisions

- **Simulation module**: diagonal waypoint pathing — units walk diagonally to the
  bridge bank, cross, then diagonally to the target. River enforcement stays as a
  safety net but should no longer trigger. Hog Rider jump unchanged. Determinism
  preserved (same seed → same match).
- **Zone layout** (per ADR-0002): 11 zones per side — 8 troop/building
  (back/mid/bridge per lane + 2 pull) and 3 spell-only (enemy towers). Pull-zone
  coordinates derived from the pull condition `d(E,C) < d(E,T)` with a ~1-tile
  safety margin: (5.5, 9.5) and (12.5, 9.5) for P0, mirrored for P1. Zones carry
  allowed card types and absolute coordinates, replacing the building/spell
  placement hacks.
- **Action space**: 45 = noop + 4 hand slots × 11 zones; masking by card type and
  elixir keeps effective choices ≤ 8.
- **Observation**: flat vector + MLP retained; enriched with elixir advantage,
  per-lane HP-weighted threat mass, card-cycle state, bridge-distance features;
  enemy elixir visible, enemy hand hidden; `OBS_DIM`/`NUM_ACTIONS` versioned.
- **Reward**: tower-HP delta dominant; toggleable shaped terms (troop trade value,
  elixir-leak penalty, kill/loss delta) for the ablation matrix.
- **Training**: MaskablePPO retained; SubprocVecEnv ~12 envs on the 14-core CPU;
  curriculum RandomAgent → LogicAgent; self-play hook kept, deferred; 1–3h per run.
- **Evaluation**: fixed seeds; 200 episodes vs LogicAgent + 200 vs RandomAgent;
  win rate, mean reward, match length; v2 ships in the game only if it beats v1.
- **Server**: protocol v2 — `version` field in `hello`; `hello` resent on rematch;
  one speed clamp (0.25–4) shared by protocol, session, and UI; `error` messages
  carry reason codes; `hello` reports the effective opponent (`bot` | `rl`), never
  a silent fallback. Session resume by id: the URL carries `/game/{id}`, a WS
  reconnect rejoins the live session, expired sessions return a clean error.
  `hello.config` carries deploy/spell zone geometry; the client renders valid
  areas from it and the server remains the only validator.
- **Coach**: hint/feedback zone mapping updated to the 11-zone layout; grade and
  suggestion reasons surfaced in the UI.
- **Web client**: approved layout (top bar; arena with tower HP bars; collapsible
  coach panel; bottom hand bar with elixir and next card). Pre-match opponent
  picker; post-game stats screen; deep-linkable match route. Art pipeline: a
  one-time fetch script reads the official API key from `.env` and caches card
  PNGs + a manifest into the web assets (committed; runtime needs no key); emoji
  fallback stays. Arena units are icon-in-ring tokens. Damage numbers, hit
  flashes, death poofs, projectile arcs; snapshot interpolation retained.
- **Language rule**: code, comments, and scripts in English; web UI copy bilingual
  (PL/EN) via the strings module.
- **Runnability**: console scripts for Python entry points; Docker helper scripts
  build and run the whole app as one container.

## Testing Decisions

A good test exercises external behavior at the highest existing seam with
deterministic seeds — never implementation details.

- **Board seam** (prior art: `test_board.py`, `test_cards.py`) — diagonal movement
  regression (diagonal path shorter than the old L-path; river never entered
  off-bridge except Hog Rider); zone geometry pinned (11 zones, coordinates,
  allowed card types); action mask layout pinned (45 actions, spells masked to
  spell zones, troops masked off them); placement validation rejects illegal
  coordinates.
- **Gym env seam** — observation/action contract pinned (`OBS_DIM`, `NUM_ACTIONS`,
  mask shape and dtype); reward terms toggle cleanly; opponent interface
  (logic/random) unchanged.
- **Server seam** (prior art: `test_server.py`, TestClient + WebSocket) —
  protocol v2 handshake includes version and zone geometry; rematch resends
  `hello`; resume by id restores a live match and expired ids fail cleanly;
  rejected plays carry reason codes; `vs_rl` fallback is explicit in `hello`.
- **Coach seam** (prior art: `test_coach.py`) — zone mapping covers all 11 zones;
  hint/feedback payloads include reasons.
- **Experiments seam** (prior art: `test_m7_experiments.py`) — eval protocol
  writes reproducible metadata (seeds, episode counts, opponent types).
- **End-to-end (manual)**: Playwright smoke scripts against the redesigned
  `/game`; `npm run build`; Docker build + run. No new frontend unit-test
  framework is introduced.

## Out of Scope

- Deck building — the 6-card deck is fixed, not user-chosen.
- New cards, new mechanics, card stat changes (mechanics freeze, ADR-0001).
- Self-play training runs (the hook is kept in code, deferred).
- Fly.io/Railway platform configs (Docker-only, platform-agnostic deployment doc).
- Sound, accounts, persistence, replays, spectating.
- A frontend unit-test framework.
- Any training or evaluation runs before this spec is approved.

## Further Notes

- The v1 checkpoint stays frozen as the thesis baseline; `models/` is gitignored.
- The official API key lives in `.env` (gitignored) and is used only by the
  one-time art fetch; the cached PNGs are committed so the app and the Docker
  image need no key.
- Cleanup (Phase 0) is already committed: root shims removed, console-script entry
  points, thesis artifacts under `docs/thesis/`, Docker-only deploy.
