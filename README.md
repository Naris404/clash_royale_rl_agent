# Clash Royale RL Coach

A simplified [Clash Royale](https://supercell.com/en/games/clashroyale/) simulation for **reinforcement learning** training. The arena, elixir, card queue and combat are implemented in Python; the RL agent learns to play against a rule-based bot.

## Features

- **18×32 tile board** (matching the official CR arena)
- **Two bridges** — units cannot cross the river except at a bridge (Hog jumps)
- **6 cards in the deck**, **4 in hand** — a queue without duplicates (as in the game)
- **Towers** with ~7.5 tile range; they shoot enemies as soon as they enter range
- **`RLAgent`** — PPO (MaskablePPO) trained against `LogicAgent`
- **`LogicAgent`** — rule-based bot (defense, Hog, Cannon, etc.)

## Requirements

- Python 3.10+
- Node.js 20+ (web interface only)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Quick start

```bash
# PPO training vs LogicAgent (~1M steps, a few to a dozen minutes on CPU)
cr-rl-train

# Evaluate a trained model
cr-rl-evaluate --model models/ppo_cr_best.zip --episodes 100

# Statistics report (TB logs, win rate, plots)
cr-rl-stats
cr-rl-stats --plot --episodes 100
```

## Project structure

```
clash_royale_rl_agent/
├── src/cr_rl/
│   ├── game/          # board and cards
│   ├── agents/        # rule-based, random and PPO agents
│   ├── env/           # Gymnasium environment
│   ├── training/      # training and evaluation
│   ├── coach/         # move suggestions and grading
│   ├── server/        # FastAPI and WebSocket sessions
│   ├── experiments/   # sweep, curriculum
│   ├── stats/         # reports and plots
├── web/               # React + PixiJS
├── tests/             # pytest tests
├── scripts/           # development utilities
├── docs/              # project docs and thesis material (docs/thesis)
├── models/            # models and results (gitignored)
└── pyproject.toml      # package and cr-rl-* commands
```

All source code lives in `src/cr_rl/`; entry points are the `cr-rl-*` commands
defined in `pyproject.toml` (available after `pip install -e .`).

## Environment API (`Board`)

```python
from cr_rl.game.board import Board, NUM_ACTIONS, OBS_DIM

env = Board(seed=42)
obs = env.reset(seed=42)

# Action 0 = noop; 1–44 = hand slot (0–3) × deploy zone (0–10)
result = env.step(action_p0, action_p1=None)

obs = result.observation   # float32 vector, OBS_DIM
reward = result.reward
done = result.terminated or result.truncated
```

### Deck cards (default)

Knight, Giant, Cannon, Musketeer, Hog_Rider, Fireball

### RL agent (PPO)

```python
from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.agents.rl import RLAgent

# Training
# cr-rl-train --timesteps 1000000 --n-envs 8

# Playing with a model
agent = RLAgent(model_path="models/ppo_cr_best.zip")
env = ClashRoyaleEnv()
obs, _ = env.reset()
action = agent.choose_action(env.board, player=0)
obs, reward, done, trunc, info = env.step(action)
```

**Observation** (`OBS_DIM`): own and enemy elixir, time, HP of the 6 towers, one-hot hand (4×6), up to 24 units (sorted).

**Reward**: only the change in total tower HP `(enemy loss − own loss) / max_tower_HP`.

`LogicAgent` uses `board.set_pending_play()` — `gym_env` wires up both modes correctly.

## Configuration

| File | What to change |
|------|----------------|
| `src/cr_rl/game/board.py` → `TOWER_LAYOUT`, `BRIDGE_LANE_X` | Tower / bridge positions |
| `src/cr_rl/game/cards.py` → `cards_dic` | Card stats |
| `cr-rl-train --timesteps --n-envs` | Training length and parallelism |

## Web platform

The FastAPI backend is the source of truth for the simulation, and the React/PixiJS
interface renders the arena and talks to it over WebSocket. The landing page is a
bilingual (PL/EN, English by default) project portfolio with a thesis summary.

### Running without Docker

After the one-time installation from the "Requirements" section:

```powershell
cd web
npm ci
cd ..
.\scripts\dev.ps1
```

Open **http://localhost:5173**. Vite proxies `/api` and `/ws` to the FastAPI
backend on port **8000**.

You can also start the processes manually in two terminals:

```powershell
# terminal 1 — API
.\.venv\Scripts\Activate.ps1
uvicorn cr_rl.server.app:app --reload --port 8000

# terminal 2 — frontend
cd web
npm run dev
```

After `npm run build`, the production frontend is served by the API from `web/dist`
at http://localhost:8000.

## Thesis experiments

Each experiment saves its metadata and results as JSON under
`experiment_runs/`, so runs are reproducible and can be
compared with `cr-rl-stats`.

```bash
# sweep: learning rate, entropy, gamma and network architecture
cr-rl-sweep --timesteps 1000000 --eval-episodes 200

# curriculum: RandomAgent first, then LogicAgent
cr-rl-curriculum --phase1 300000 --phase2 700000

# summary report and plots for the experiments chapter
cr-rl-stats --experiments --plot
```

Short runs with fewer steps are only a smoke test; thesis results should use
the same seeds, step budget and at least 200 evaluation matches per variant.

## Docker and hosting

```bash
docker build -t clash-royale-rl-coach .
docker run --rm -p 8000:8000 clash-royale-rl-coach
```

The multi-stage image builds the frontend and runs the API as a non-root user.
The `models/ppo_cr_best.zip` model is included if it is present in the build
context. Alternatively, set `MODEL_URL`; without a model the app falls back to
the heuristic coach. Publishing instructions: [DEPLOYMENT.md](DEPLOYMENT.md).

## License

Educational / personal project — Clash Royale is a trademark of Supercell.
