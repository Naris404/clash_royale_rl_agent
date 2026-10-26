# Deployment

The production image builds the React/PixiJS frontend, installs the Python API,
and serves both from one Uvicorn process. WebSocket traffic uses the same public
origin as the REST API.

## Build and test locally

Place the trained policies in `models/` before building if they should be bundled
into the image (see the Model catalogue in `docs/rl-agent.md`):

- `models/ppo_cr_best.zip` — the Shipped model, the default RL Opponent and the
  only model the Coach uses. `cr-rl-final` puts it there when the shipped run
  passes the Ship gate.
- `models/ppo_cr_v2_shipped.zip` — the v2 Shipped model, kept as its own choice
  once a v3 model is shipped (`cr-rl-v3` writes it).
- `models/ppo_cr_self_play.zip` — the self-play model, an optional second choice
  for the RL Opponent.

A catalogue model whose file is not bundled is shown as unavailable on the home
screen.

```bash
docker build -t clash-royale-rl-coach .
docker run --rm -p 8000:8000 clash-royale-rl-coach
```

Open `http://localhost:8000`; health information is available at
`http://localhost:8000/api/health`.

If the Shipped model is not bundled, set `MODEL_URL` to a direct HTTPS download
URL. The container downloads it once at startup to `MODEL_PATH`. `MODEL_URL`
covers the Shipped model only. Without either source,
the application remains usable with the deterministic heuristic coach.

## Runtime variables

- `PORT` — HTTP port, default `8000`.
- `MODEL_PATH` — Shipped model path, default `models/ppo_cr_best.zip`.
- `MODEL_URL` — optional direct URL for the Shipped model, used only when `MODEL_PATH` is absent.

## Hosting

Any platform that builds the root `Dockerfile` and routes HTTP + WebSocket
traffic to a single port works (a VPS with Docker, or a PaaS with Dockerfile
support). Make sure WebSocket upgrades are allowed, and provide the model via
`MODEL_URL` when it is not baked into the image.
