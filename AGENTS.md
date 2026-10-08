## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `Naris404/clash_royale_rl_agent` (via `gh` CLI or the GitHub MCP). See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Project overview

Clash Royale RL Coach is an educational, thesis-oriented platform built around a deterministic 18×32 game simulation. It supports MaskablePPO training and evaluation, rule-based and RL opponents, live move suggestions and grading, a browser-playable match, experiment reporting, and Docker deployment.

The Python backend owns simulation state and is the single source of truth. The browser renders snapshots and sends player input; it must not duplicate game rules. Preserve deterministic seeding and observation/action compatibility when changing game mechanics.

## Technical stack

- Python 3.10+, NumPy, Gymnasium, PyTorch.
- FastAPI, Pydantic, Uvicorn.
- React 18, strict TypeScript, Vite, and PixiJS for the web client.
- pytest for Python tests; Docker uses a Node build stage and a non-root Python runtime.

## Repo structure

- `src/cr_rl/game/`: simulation, cards, observations, actions, and rewards.
- `src/cr_rl/agents/`, `env/`, `training/`: agents, Gymnasium integration, training, and evaluation.
- `src/cr_rl/coach/`: policy inspection, suggestions, explanations, and move grading.
- `src/cr_rl/server/`: FastAPI routes, WebSocket protocol, and game-session lifecycle.
- `src/cr_rl/experiments/`, `stats/`, `reports/`: reproducible thesis experiments and outputs.
- `web/src/`: React UI, PixiJS arena, networking, state store, and PL/EN copy.
- `tests/`: deterministic unit and integration tests; `scripts/`: development utilities.
- All Python source lives under `src/cr_rl/`; entry points are the `cr-rl-*` console scripts defined in `pyproject.toml`. New Python code and imports belong under `cr_rl`.

## Coding style

- Python: 4-space indentation, `from __future__ import annotations`, type hints on public APIs, dataclasses or Pydantic models for structured data, and module loggers instead of `print` in library code.
- Keep simulation logic deterministic and side effects explicit. Use named constants for observation/action layouts and update tests whenever those contracts change.
- TypeScript: strict types, functional React components, double quotes, semicolons, and shared protocol types matching the Pydantic WebSocket models.
- Keep rendering separate from game rules: React handles interface state, PixiJS renders the arena, and FastAPI validates all authoritative actions.
- Add focused pytest coverage for Python behavior and run `python -m pytest`; run `npm run build --prefix web` after frontend changes.
- Follow existing PL/EN behavior: user-facing web copy must remain translatable; code identifiers stay in English.
- KEEP CLEAN AND SIMPLE SOURCE CODE.

## IMPORTANT INFORMATION
- Card identity, elixir cost and icons come from the official Clash Royale API (key in `.env`). The API does not provide combat stats, so those come from a committed data file.
- Everything in the repo (code, docs, comments, commit messages) is English unless the user says otherwise. The UI stays bilingual (PL/EN) with English as the default.
