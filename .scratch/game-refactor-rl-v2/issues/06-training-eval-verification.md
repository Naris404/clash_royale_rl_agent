# 06: Training runs, v1-vs-v2 report, final verification

**What to build:** The v2 agent is trained and judged. Curriculum training runs
within the 1–3h CPU budget; the evaluation protocol produces fixed-seed results
(200 episodes vs LogicAgent + 200 vs RandomAgent); the ablation matrix (full /
no-shaping / no-curriculum) runs if time allows. A v1-vs-v2 comparison report is
written for the thesis, and the in-game RL opponent is swapped to v2 **only if
v2 beats v1** under the protocol. The project then passes full verification:
complete pytest suite, web build, Playwright smoke on the redesigned `/game`,
Docker image build and run via the helper scripts, and README/docs refreshed to
the final state.

**Blocked by:** 04 (complete frontend for end-to-end verification), 05 (trained
artifact source).

**Status:** ready-for-agent

- [ ] Curriculum training completed within budget; artifacts under `models/`
- [ ] Eval results (200 vs Logic + 200 vs Random, fixed seeds) recorded with metadata
- [ ] Ablation results recorded if time allowed
- [ ] v1-vs-v2 comparison report written; in-game opponent swapped only if v2 wins
- [ ] Full pytest + `npm run build` + Playwright smoke + Docker build/run all green
- [ ] README and docs reflect the final state
