# Deployment

The production image builds the React/PixiJS frontend, installs the Python API,
and serves both from one Uvicorn process. WebSocket traffic uses the same public
origin as the REST API.

## Build and test locally

Place the trained policy at `models/ppo_cr_best.zip` before building if it
should be bundled into the image:

```bash
docker build -t clash-royale-rl-coach .
docker run --rm -p 8000:8000 clash-royale-rl-coach
```

Open `http://localhost:8000`; health information is available at
`http://localhost:8000/api/health`.

If a model is not bundled, set `MODEL_URL` to a direct HTTPS download URL. The
container downloads it once at startup to `MODEL_PATH`. Without either source,
the application remains usable with the deterministic heuristic coach.

## Runtime variables

- `PORT` — HTTP port, default `8000`.
- `MODEL_PATH` — policy path, default `models/ppo_cr_best.zip`.
- `MODEL_URL` — optional direct URL used only when `MODEL_PATH` is absent.

## Hosting

Any platform that builds the root `Dockerfile` and routes HTTP + WebSocket
traffic to a single port works (a VPS with Docker, or a PaaS with Dockerfile
support). Make sure WebSocket upgrades are allowed, and provide the model via
`MODEL_URL` when it is not baked into the image.
