# 03: Backend protocol v2 + session resume

**What to build:** The WebSocket protocol grows up: `hello` carries a protocol
version, the deploy/spell zone geometry, and the *effective* opponent
(`bot` | `rl` — never a silent fallback when the model file is missing). Rematch
resends `hello` so the client resets cleanly. One speed clamp (0.25–4) is shared
by protocol, session, and UI. Rejected plays return reason codes the client can
show. Matches survive a page refresh: the URL carries `/game/{id}`, a reconnect
rejoins the live session, and an expired session fails cleanly back to home.
Placement validation lives only on the server — the client renders valid areas
from the `hello` geometry instead of duplicating rules. The `match_end` event
carries post-game stats (damage dealt/taken, elixir efficiency) alongside grades
and the move log.

**Blocked by:** 02 (zone geometry is part of the `hello` contract).

**Status:** ready-for-agent

- [ ] `hello` includes protocol version, zone geometry, and effective opponent; RL fallback is explicit
- [ ] Rematch resends `hello`; speed clamp 0.25–4 identical in protocol, session, and UI
- [ ] Rejected plays return reason codes; client surfaces them
- [ ] Refresh on `/game/{id}` rejoins the live match; expired session id fails cleanly to home
- [ ] Client placement rendering derives from server geometry; server remains sole validator
- [ ] `match_end` payload includes damage dealt/taken and elixir efficiency
- [ ] Server-seam tests (TestClient + WebSocket) cover handshake, rematch, resume, rejections; `python -m pytest` green
