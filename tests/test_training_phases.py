"""Training phases: the three layouts (full, no-curriculum, no-self-play) and their opponents."""

from __future__ import annotations

from collections import Counter

import pytest

from cr_rl.training.phases import PHASE_LAYOUTS, phase_opponents, phase_steps


def layout(name: str, timesteps: int = 1000) -> list[tuple[str, int]]:
    return [(phase.name, steps) for phase, steps in phase_steps(name, timesteps)]


class TestLayouts:
    def test_full_has_warm_up_scripted_mix_and_self_play(self):
        assert layout("full") == [("warm-up", 200), ("scripted-mix", 400), ("self-play-mix", 400)]

    def test_no_curriculum_drops_the_warm_up(self):
        assert layout("no-curriculum") == [("scripted-mix", 600), ("self-play-mix", 400)]

    def test_no_self_play_stretches_the_scripted_mix_to_the_end(self):
        assert layout("no-self-play") == [("warm-up", 200), ("scripted-mix", 800)]

    def test_the_steps_always_add_up_to_the_budget(self):
        for name in PHASE_LAYOUTS:
            assert sum(steps for _, steps in layout(name, 12_345)) == 12_345

    def test_an_unknown_layout_is_rejected(self):
        with pytest.raises(ValueError):
            phase_steps("everything", 1000)


class TestPhaseOpponents:
    def phase(self, name: str):
        return next(p for p, _ in phase_steps("full", 1000) if p.name == name)

    def test_warm_up_is_all_random(self):
        assert phase_opponents(self.phase("warm-up"), 12) == ["random"] * 12

    def test_scripted_mix_assigns_8_2_2(self):
        opponents = phase_opponents(self.phase("scripted-mix"), 12)
        assert Counter(opponents) == {"logic": 8, "logic-eps": 2, "random": 2}

    def test_self_play_mix_assigns_4_6_1_1_with_past_selves_in_pool_rotation(self):
        pool = ["a.zip", "b.zip", "c.zip"]
        opponents = phase_opponents(self.phase("self-play-mix"), 12, pool)
        assert Counter(opponents) == {
            "a.zip": 2, "b.zip": 1, "c.zip": 1, "logic": 6, "logic-eps": 1, "random": 1,
        }
        assert opponents[:4] == ["a.zip", "b.zip", "c.zip", "a.zip"]
