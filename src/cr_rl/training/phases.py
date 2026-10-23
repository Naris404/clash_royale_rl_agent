"""Training phases: which Training opponents each env plays, and for how much of the step budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

PAST_SELF = "past-self"  # placeholder in a mix; replaced by a Past-self pool member per env


@dataclass(frozen=True)
class TrainingPhase:
    """A stage of a run: `mix` is cycled over the env indices; the phase ends at `end` of the budget."""

    name: str
    mix: tuple[str, ...]
    end: float

    @property
    def uses_pool(self) -> bool:
        return PAST_SELF in self.mix


WARM_UP_MIX = ("random",) * 12
SCRIPTED_MIX = ("logic",) * 8 + ("logic-eps",) * 2 + ("random",) * 2
SELF_PLAY_MIX = (PAST_SELF,) * 4 + ("logic",) * 6 + ("logic-eps",) + ("random",)

# One setting selects a layout: "full" is the main configuration, the other two are the
# no-curriculum and no-self-play Ablation arms.
PHASE_LAYOUTS: dict[str, tuple[TrainingPhase, ...]] = {
    "full": (
        TrainingPhase("warm-up", WARM_UP_MIX, 0.2),
        TrainingPhase("scripted-mix", SCRIPTED_MIX, 0.6),
        TrainingPhase("self-play-mix", SELF_PLAY_MIX, 1.0),
    ),
    "no-curriculum": (
        TrainingPhase("scripted-mix", SCRIPTED_MIX, 0.6),
        TrainingPhase("self-play-mix", SELF_PLAY_MIX, 1.0),
    ),
    "no-self-play": (
        TrainingPhase("warm-up", WARM_UP_MIX, 0.2),
        TrainingPhase("scripted-mix", SCRIPTED_MIX, 1.0),
    ),
}


def phase_steps(layout: str, timesteps: int) -> list[tuple[TrainingPhase, int]]:
    """Each phase of the layout with its share of the step budget; the last one takes the rest."""
    if layout not in PHASE_LAYOUTS:
        raise ValueError(f"Unknown phase layout {layout!r}; choose from {sorted(PHASE_LAYOUTS)}")
    result = []
    used = 0
    for phase in PHASE_LAYOUTS[layout]:
        end = timesteps if phase.end == 1.0 else int(timesteps * phase.end)
        result.append((phase, end - used))
        used = end
    return result


def phase_opponents(phase: TrainingPhase, n_envs: int, past_selves: Sequence = ()) -> list:
    """Opponent of each env index; the k-th Past-self env gets pool member k, in rotation."""
    opponents = []
    past_self_envs = 0
    for index in range(n_envs):
        opponent = phase.mix[index % len(phase.mix)]
        if opponent == PAST_SELF:
            opponent = past_selves[past_self_envs % len(past_selves)]
            past_self_envs += 1
        opponents.append(opponent)
    return opponents
