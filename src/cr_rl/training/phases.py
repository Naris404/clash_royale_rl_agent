"""Training phases: which Training opponents each env plays, and for how much of the step budget."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Sequence

PAST_SELF = "past-self"  # placeholder in a mix; replaced by a Past-self pool member per env


@dataclass(frozen=True)
class TrainingPhase:
    """A stage of a run: `mix` is cycled over the env indices; the phase ends at `end` of the budget."""

    name: str
    mix: tuple[str, ...]
    end: float
    pfsp: bool = False  # pool envs are drawn by PFSP instead of taken from the pool in rotation

    @property
    def uses_pool(self) -> bool:
        return PAST_SELF in self.mix

    def env_mix(self, n_envs: int) -> list[str]:
        """The mix entry of each of the first `n_envs` envs."""
        return [self.mix[index % len(self.mix)] for index in range(n_envs)]

    def past_self_envs(self, n_envs: int) -> int:
        """How many of the first `n_envs` envs play a pool member."""
        return self.env_mix(n_envs).count(PAST_SELF)


WARM_UP_MIX = ("random",) * 12
SCRIPTED_MIX = ("logic",) * 8 + ("logic-eps",) * 2 + ("random",) * 2
SELF_PLAY_MIX = (PAST_SELF,) * 4 + ("logic",) * 6 + ("logic-eps",) + ("random",)
FINE_TUNE_MIX = (PAST_SELF,) * 10 + ("logic",) * 2

# One setting selects a layout: FULL is the main configuration, the other two are the
# no-curriculum and no-self-play Ablation arms. FINE_TUNE continues a Final model against the pool.
FULL = "full"
NO_CURRICULUM = "no-curriculum"
NO_SELF_PLAY = "no-self-play"
FINE_TUNE = "fine-tune"

PHASE_LAYOUTS: dict[str, tuple[TrainingPhase, ...]] = {
    FULL: (
        TrainingPhase("warm-up", WARM_UP_MIX, 0.2),
        TrainingPhase("scripted-mix", SCRIPTED_MIX, 0.6),
        TrainingPhase("self-play-mix", SELF_PLAY_MIX, 1.0),
    ),
    NO_CURRICULUM: (
        TrainingPhase("scripted-mix", SCRIPTED_MIX, 0.6),
        TrainingPhase("self-play-mix", SELF_PLAY_MIX, 1.0),
    ),
    NO_SELF_PLAY: (
        TrainingPhase("warm-up", WARM_UP_MIX, 0.2),
        TrainingPhase("scripted-mix", SCRIPTED_MIX, 1.0),
    ),
    FINE_TUNE: (TrainingPhase("fine-tune-mix", FINE_TUNE_MIX, 1.0, pfsp=True),),
}


def phase_steps(layout: str, timesteps: int) -> list[tuple[TrainingPhase, int]]:
    """Each phase of the layout with its share of the step budget; the last one takes the rest."""
    if layout not in PHASE_LAYOUTS:
        raise ValueError(f"Unknown phase layout {layout!r}; choose from {sorted(PHASE_LAYOUTS)}")
    result = []
    used = 0
    for phase in PHASE_LAYOUTS[layout]:
        end = round(timesteps * phase.end)
        result.append((phase, end - used))
        used = end
    return result


def phase_opponents(phase: TrainingPhase, n_envs: int, past_selves: Sequence[Path] = ()) -> list[str | Path]:
    """Opponent of each env index; the k-th Past-self env gets pool member k, in rotation."""
    if phase.uses_pool and not past_selves:
        raise ValueError(f"Phase {phase.name!r} plays Past selves but the Past-self pool is empty")
    members = cycle(past_selves)
    return [next(members) if opponent == PAST_SELF else opponent for opponent in phase.env_mix(n_envs)]
