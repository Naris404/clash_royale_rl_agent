# 04: UI layout + match flows

**What to build:** The approved layout lands: top bar (menu, timer, pause, speed,
settings); arena with tower HP bars at its edges; collapsible coach panel on the
right (suggestion with reason, eval bar, key-moments log); bottom hand bar (elixir
bar, four cards, next card). The match flow becomes: home → opponent picker
(rule-based bot / RL policy / coach toggle, with an honest note when RL falls back
to bot) → match → post-game stats screen (result, damage dealt/taken, elixir
efficiency, grades, key moments) → rematch or menu. Match routes are
deep-linkable (`/game/{id}`). All new copy exists in Polish and English; code and
comments stay English.

**Blocked by:** 03 (resume route, stats payload, opponent signaling).

**Status:** ready-for-agent

- [ ] Layout matches the approved mock: top bar / arena + HP bars / collapsible coach panel / hand bar
- [ ] Pre-match opponent picker with honest RL-fallback signaling
- [ ] Post-game stats screen renders the `match_end` payload (result, damage, elixir efficiency, grades, key moments) with rematch/menu actions
- [ ] `/game/{id}` deep link routes into the live match
- [ ] All new strings present in both PL and EN
- [ ] `npm run build --prefix web` green; Playwright smoke passes on the new flow
