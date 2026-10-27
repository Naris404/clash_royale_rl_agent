"""Arrows experiment: how a model uses Arrows on a tower and what blocking that play costs (ADR-0007).

Part 1: the model plays the Report seeds against LogicAgent; every decision whose top choice is Arrows
on a tower is recorded with the state it was made in. Part 2 replays the same seeds with that play
masked out, against LogicAgent and against a frozen model, next to the unmasked win rates.

Usage:
  cr-rl-arrows --model models/ppo_cr_best.zip --frozen path/to/ppo_cr_final.zip
"""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from sb3_contrib import MaskablePPO

from cr_rl.coach.inference import PolicyInspector
from cr_rl.game.board import DEPLOY_ZONES, NUM_ZONES, Board
from cr_rl.jsonio import write_json
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.matches import MatchResult, play_matches
from cr_rl.training.report import REPORT_SEEDS, OpponentReport, ReportMetadata, report_matches, report_metadata

ARROWS = "Arrows"
TOWER_ZONES = ("spell-tower-L", "spell-tower-R", "spell-king")
LOGIC = "logic"
OUTCOME_BY_WINNER = {0: "win", 1: "loss", None: "draw"}  # the model is player 0
DEFAULT_OUT = EXPERIMENT_RUNS_DIR / "arrows" / "results.json"


@dataclass(frozen=True)
class ArrowsDecision:
    """One decision whose top choice is Arrows on a tower, with the state it was made in."""

    seed: int
    time: float  # seconds into the match
    zone: str
    own_elixir: float
    enemy_elixir: float
    p_chosen: float
    p_noop: float
    value: float  # V(s) of the state
    outcome: str  # of the match: "win", "loss" or "draw"


@dataclass(frozen=True)
class ArrowsProfile:
    """Every Arrows decision of a model on the seeds, and how they spread over zones."""

    matches: int
    plays_per_zone: dict[str, int]  # all Arrows plays, whatever the zone
    tower_share: float  # of all Arrows plays, the share aimed at a tower
    decisions: tuple[ArrowsDecision, ...]  # the plays on a tower, in match order
    matches_with_tower_play: int
    median_own_elixir: float
    median_enemy_elixir: float


class OpponentRun(NamedTuple):
    """The model against one opponent on the seeds: its Arrows profile and its win rate."""

    profile: ArrowsProfile
    win_rate: OpponentReport


@dataclass(frozen=True)
class WinRates:
    """The same seeds with Arrows on a tower allowed and blocked."""

    allowed: OpponentReport
    blocked: OpponentReport


@dataclass(frozen=True)
class ArrowsResults:
    metadata: ReportMetadata
    profile: ArrowsProfile  # Part 1: the model against LogicAgent, nothing blocked
    blocked_profile: ArrowsProfile  # the same with Arrows on a tower blocked: no tower decisions are left
    win_rates: dict[str, WinRates]  # Part 2: by opponent name, "logic" and the frozen model


@dataclass(frozen=True)
class _Play:
    """An Arrows play before its match is over."""

    time: float
    zone: str
    own_elixir: float
    enemy_elixir: float
    p_chosen: float
    p_noop: float
    value: float


class _ArrowsProbe:
    """Player 0 that picks its policy's top action and notes every Arrows play."""

    def __init__(self, model: Any, *, block_tower_arrows: bool):
        self._inspector = PolicyInspector(model=model)
        self._block = block_tower_arrows
        self.plays: list[_Play] = []

    def choose_action(self, board: Board, player: int) -> int:
        board.set_pending_play(player, None)
        blocked = _tower_arrows_actions(board, player) if self._block else ()
        estimate = self._inspector.estimate(board, player, blocked=blocked)
        card, zone_index = board.action_to_card_zone(estimate.action, player)
        if card == ARROWS:
            self.plays.append(
                _Play(
                    time=board.time,
                    zone=DEPLOY_ZONES[player][zone_index].name,
                    own_elixir=board.elixir[player],
                    enemy_elixir=board.elixir[1 - player],
                    p_chosen=float(estimate.probs[estimate.action]),
                    p_noop=float(estimate.probs[0]),
                    value=estimate.value,
                )
            )
        return estimate.action


def _tower_arrows_actions(board: Board, player: int) -> list[int]:
    """The actions that play Arrows from the hand on a tower."""
    zones = [i for i, zone in enumerate(DEPLOY_ZONES[player]) if zone.name in TOWER_ZONES]
    return [
        1 + slot * NUM_ZONES + zone
        for slot, card in enumerate(board.get_hand(player))
        if card == ARROWS
        for zone in zones
    ]


def _play(
    model: Any, opponent: Any, name: str, seeds: Sequence[int], *, block_tower_arrows: bool
) -> OpponentRun:
    """Play the model against one opponent on the seeds."""
    decisions: list[ArrowsDecision] = []
    zones: Counter[str] = Counter()
    results: list[MatchResult] = []
    for seed in seeds:
        probe = _ArrowsProbe(model, block_tower_arrows=block_tower_arrows)
        result = play_matches(probe, opponent, [seed])[0]
        results.append(result)
        zones.update(play.zone for play in probe.plays)
        decisions += [
            ArrowsDecision(seed=seed, outcome=OUTCOME_BY_WINNER[result.winner], **vars(play))
            for play in probe.plays
            if play.zone in TOWER_ZONES
        ]
    plays = sum(zones.values())
    profile = ArrowsProfile(
        matches=len(seeds),
        plays_per_zone=dict(zones),
        tower_share=len(decisions) / plays if plays else 0.0,
        decisions=tuple(decisions),
        matches_with_tower_play=len({decision.seed for decision in decisions}),
        median_own_elixir=statistics.median(d.own_elixir for d in decisions) if decisions else 0.0,
        median_enemy_elixir=statistics.median(d.enemy_elixir for d in decisions) if decisions else 0.0,
    )
    return OpponentRun(profile, report_matches(name, results))


def run_arrows_experiment(
    model: Any,
    frozen: Any,
    *,
    seeds: range = REPORT_SEEDS,
    model_name: str = "model",
    frozen_name: str = "frozen",
) -> ArrowsResults:
    """Both parts for `model` on `seeds`; `frozen` plays as player 1 in the second win-rate pair."""
    logic_allowed = _play(model, LOGIC, LOGIC, seeds, block_tower_arrows=False)
    logic_blocked = _play(model, LOGIC, LOGIC, seeds, block_tower_arrows=True)
    frozen_allowed = _play(model, frozen, frozen_name, seeds, block_tower_arrows=False)
    frozen_blocked = _play(model, frozen, frozen_name, seeds, block_tower_arrows=True)
    return ArrowsResults(
        metadata=report_metadata(model_name, seeds, config={"frozen": frozen_name}),
        profile=logic_allowed.profile,
        blocked_profile=logic_blocked.profile,
        win_rates={
            LOGIC: WinRates(logic_allowed.win_rate, logic_blocked.win_rate),
            frozen_name: WinRates(frozen_allowed.win_rate, frozen_blocked.win_rate),
        },
    )


def save_arrows_results(results: ArrowsResults, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"experiment": "arrows", **asdict(results)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrows experiment: Arrows on a tower, allowed and blocked")
    parser.add_argument("--model", type=Path, required=True, help="the model to study")
    parser.add_argument("--frozen", type=Path, required=True, help="the frozen opponent of the second win-rate pair")
    parser.add_argument("--episodes", type=int, default=len(REPORT_SEEDS), help="Report seeds to play")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seeds = range(REPORT_SEEDS.start, REPORT_SEEDS.start + args.episodes)
    results = run_arrows_experiment(
        MaskablePPO.load(str(args.model)),
        MaskablePPO.load(str(args.frozen)),
        seeds=seeds,
        model_name=args.model.name,
        frozen_name=args.frozen.name,
    )
    save_arrows_results(results, args.out)
    profile = results.profile
    print(
        f"Arrows on a tower: {len(profile.decisions)} plays in {profile.matches_with_tower_play}/{profile.matches} "
        f"matches, {profile.tower_share:.0%} of all Arrows plays; median elixir "
        f"{profile.median_own_elixir:.2f} own vs {profile.median_enemy_elixir:.2f} enemy"
    )
    for opponent, pair in results.win_rates.items():
        for label, report in (("allowed", pair.allowed), ("blocked", pair.blocked)):
            print(
                f"vs {opponent} {label}: win rate {report.win_rate:.1%} "
                f"(95% CI {report.ci_low:.1%} to {report.ci_high:.1%})"
            )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
