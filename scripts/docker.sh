#!/usr/bin/env bash
# Build and run the full Clash Royale RL Coach app in Docker (Linux/macOS).
# Usage:
#   ./scripts/docker.sh           # build (if needed) and run
#   ./scripts/docker.sh rebuild   # force a fresh image build
#   ./scripts/docker.sh stop      # stop and remove the container
set -euo pipefail

IMAGE="${IMAGE:-clash-royale-rl-coach}"
CONTAINER="${CONTAINER:-cr-rl-coach}"
PORT="${PORT:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v docker >/dev/null 2>&1 || { echo "Docker not found." >&2; exit 1; }

case "${1:-}" in
  stop)
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "Stopped container '$CONTAINER'."
    exit 0
    ;;
  rebuild)
    docker build -t "$IMAGE" "$ROOT"
    ;;
  "")
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      docker build -t "$IMAGE" "$ROOT"
    fi
    ;;
  *)
    echo "Unknown argument: $1 (expected: rebuild | stop)" >&2
    exit 2
    ;;
esac

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run --rm -d --name "$CONTAINER" -e PORT=8000 -p "${PORT}:8000" "$IMAGE" >/dev/null

echo ""
echo "App:     http://localhost:${PORT}"
echo "Health:  http://localhost:${PORT}/api/health"
echo "Logs:    docker logs -f ${CONTAINER}"
echo "Stop:    ./scripts/docker.sh stop"
