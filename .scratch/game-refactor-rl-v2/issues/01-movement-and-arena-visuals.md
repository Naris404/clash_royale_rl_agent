# 01: Natural movement + Clash Royale arena visuals

**What to build:** Units walk naturally — diagonally to the bridge bank, straight
across the bridge, then diagonally to the target — replacing the L-shaped
"sideways first, then straight" paths. The arena looks like Clash Royale: a
one-time fetch script reads the official API key from `.env` and caches the six
card PNGs plus a manifest into the web assets (committed, so runtime needs no
key); arena units render as icon-in-ring tokens and the hand shows card icons,
with emoji as the offline fallback. Combat feels responsive: damage numbers, hit
flashes, death poofs, projectile arcs.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Units path diagonally via bridge banks; regression test shows the diagonal path is shorter than the old L-path
- [ ] River is crossed only at bridge lanes (Hog Rider jump unchanged); same seed still produces the same match
- [ ] Fetch script caches official card art + manifest into the web assets; app and Docker image need no API key at runtime
- [ ] Arena units and hand cards render official icons; missing art falls back to emoji
- [ ] Damage numbers, hit flashes, death poofs, and projectile arcs are visible in a live match
- [ ] `python -m pytest` and `npm run build --prefix web` are green
