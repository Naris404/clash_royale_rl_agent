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

## Railway

1. Create a Railway project from this repository. `railway.json` selects the
   root `Dockerfile` and configures `/api/health`.
2. If the repository does not contain the model artifact, add a `MODEL_URL`
   variable pointing to a private object-storage download URL.
3. Generate a public domain in **Settings → Networking**. Railway supplies
   `PORT` automatically.

CLI alternative:

```bash
railway login
railway link
railway up
```

## Fly.io

Change the globally unique `app` value in `fly.toml`, then deploy from a local
checkout. A local `models/ppo_cr_best.zip` is included in the Docker context.

```bash
fly auth login
fly launch --no-deploy
fly deploy
fly status
```

For a remote model instead, set a secret before deployment:

```bash
fly secrets set MODEL_URL=https://example.invalid/private/ppo_cr_best.zip
```

The default Fly configuration may stop an idle machine. The browser client
sends WebSocket keepalive messages while a match is active; the first request
after an idle period can still incur a cold start.

## Runtime variables

- `PORT` — HTTP port, default `8000`.
- `MODEL_PATH` — policy path, default `models/ppo_cr_best.zip`.
- `MODEL_URL` — optional direct URL used only when `MODEL_PATH` is absent.
