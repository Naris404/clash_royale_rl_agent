# /game refactor & RL v2 — implementation plan

Status: **awaiting approval**. No implementation and no training runs before
approval. Vocabulary per `CONTEXT.md`; RL decisions per `docs/adr/0002`.

## Phase 0 — cleanup (done)

Root shims removed; entry points are `cr-rl-*` console scripts; thesis artifacts in
`docs/thesis/`; Docker-only deploy (`scripts/docker.ps1` / `docker.sh`);
`CONTEXT.md` + ADR-0001/0002. Baseline commit `8ee37ea`.

## Phase 1 — movement fix

Units currently walk L-shaped when crossing the river (`_bridge_waypoint` returns
`(bx, troop.y)` — horizontal first, then straight). Fix: waypoint path with
diagonal segments — walk diagonally to the bridge bank point `(bx, bank_y)`, cross
to the far bank, then diagonally to the target. `_enforce_no_river_cut` stays as a
safety net but should no longer trigger.

- Files: `src/cr_rl/game/board.py`
- Tests: diagonal path is shorter than the L-path; river never entered off-bridge
  (except Hog Rider); Hog jump unchanged; determinism preserved (same seed → same
  result).
- Consequence per ADR-0001: dynamics change → v1 checkpoint stale → retrain (Phase 4).

## Phase 2 — backend refactor

- **Protocol v2**: `version` field in `hello`; `hello` resent on rematch; one speed
  clamp (0.25–4) shared by protocol/session/UI; `error` messages carry a reason
  code; `vs_rl` fallback to bot is signaled explicitly in `hello` (`opponent:
  "bot" | "rl"`, never silent).
- **Session resume**: session id in the URL (`/game/{id}`); on WS connect the
  server sends the current snapshot and resumes; expired sessions get a clean
  `error` + redirect to home.
- **Server-authoritative placement**: `hello.config` carries the deploy-zone /
  spell-zone geometry; the client renders valid areas from it and stops
  duplicating placement rules; the server remains the only validator.
- **Coach**: zone mapping updated to the 11-zone layout (ADR-0002); toasts surface
  `feedback.reason`; dead `GRADE_SCORE` import removed.
- Files: `src/cr_rl/server/{app,protocol,sessions}.py`, `src/cr_rl/coach/*`,
  `tests/test_server.py`.

## Phase 3 — frontend redesign

- **Layout** (approved mock): top bar (menu, timer, pause, speed, settings);
  arena center with tower HP bars; collapsible coach panel on the right
  (suggestion + reason, eval bar, key-moments log); hand bar at the bottom
  (elixir, 4 cards, next card).
- **Flow**: home → opponent picker (Bot / RL / Coach toggle, honest RL fallback)
  → match → post-game stats screen (result, damage dealt/taken, elixir
  efficiency, grades, key moments) → rematch / menu. Deep-linkable
  `/game/{id}` with resume.
- **Art pipeline**: `scripts/fetch_card_art.py` reads the API key from `.env`,
  fetches official card icons once, caches PNGs + a manifest into
  `web/public/assets/cards/` (committed; runtime needs no key). Emoji remains the
  offline fallback. Arena units are icon-in-ring tokens (no official walking-unit
  sprites exist).
- **Feel**: damage numbers, hit flashes, death poofs, projectile arcs; snapshot
  interpolation stays.
- **i18n**: all new copy in `STRINGS.pl` / `STRINGS.en`; code and comments in
  English.
- Files: `web/src/**`, `scripts/fetch_card_art.py`.

## Phase 4 — RL v2 (after approval, per ADR-0002)

1. Implement obs/action/reward v2 behind versioned constants; update coach zone
   mapping; tests pin the new layout.
2. Curriculum training runs (Random → Logic), SubprocVecEnv ×12, 1–3h budget.
3. Eval protocol (200 eps vs Logic + 200 vs Random, fixed seeds).
4. Ablation matrix (full / no-shaping / no-curriculum) if time allows.
5. v1-vs-v2 report for the thesis; swap the in-game opponent only if v2 wins.

## Verification gates

- `python -m pytest` green after every phase.
- `npm run build --prefix web` after Phase 3.
- Playwright smoke (`scripts/smoke_*.py`) on the redesigned `/game`.
- `docker build` + `scripts/docker.ps1` run at the end.
