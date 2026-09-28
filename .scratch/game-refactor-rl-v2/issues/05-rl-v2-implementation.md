# 05: RL v2 implementation (no training)

**What to build:** The v2 agent architecture per ADR-0002, implemented and
contract-tested but not yet trained: enriched flat observations (elixir advantage,
per-lane HP-weighted threat mass, card-cycle state, bridge-distance features)
feeding the MLP policy; shaped rewards with per-term toggles (troop trade value,
elixir-leak penalty, kill/loss delta) with tower-HP delta dominant; curriculum
wiring (RandomAgent → LogicAgent) with the self-play hook kept but deferred;
SubprocVecEnv support (~12 envs on the 14-core CPU); and the fixed-seed evaluation
protocol script (200 episodes vs LogicAgent + 200 vs RandomAgent). `OBS_DIM` and
`NUM_ACTIONS` are versioned constants pinned by tests.

**Blocked by:** 01, 02 (final movement dynamics and the zone action space).

**Status:** ready-for-agent

- [ ] Observation includes the new feature groups; `OBS_DIM` versioned and pinned by tests
- [ ] Each shaped reward term toggles independently; tower-HP delta stays dominant
- [ ] Curriculum (Random → Logic) runs from the training entry point; self-play hook present but unused
- [ ] SubprocVecEnv training path works with ~12 envs
- [ ] Eval protocol script runs 200 vs Logic + 200 vs Random with fixed seeds and writes reproducible metadata
- [ ] `python -m pytest` green
