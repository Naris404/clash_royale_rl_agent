"""Reference-pool selection: a run's model is the checkpoint with the best average win rate (ADR-0007).

Every saved checkpoint of a run plays the Selection seeds against each opponent of the reference
pool: LogicAgent plus any frozen models. Plain selection (`selection.py`) scores against LogicAgent
only. The per-opponent scores are kept in `reference_selection.json` in the run directory.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Sequence

from sb3_contrib import MaskablePPO

from cr_rl.jsonio import write_json
from cr_rl.training.matches import MatchResult, play_matches
from cr_rl.training.report import MANIFEST_FILENAME, REPORT_SEEDS, ReportEvaluation, evaluate_report
from cr_rl.training.seeds import SELECTION_SEEDS
from cr_rl.training.selection import CHECKPOINTS_DIR, SELECTION_FILENAME, load_selection

REFERENCE_FILENAME = "reference_selection.json"
LOGIC_OPPONENT = "logic"

# A reference opponent is the bot name "logic" or the path of a frozen model.
Opponent = str | Path
PlayMatches = Callable[[Path, Opponent, Sequence[int]], list[MatchResult]]


@dataclass(frozen=True)
class ReferenceScore:
    """One checkpoint's wins against each reference opponent on the Selection seeds."""

    step: int
    path: str  # file name inside the run's `checkpoints/` directory
    wins: dict[str, int]  # opponent name -> wins
    episodes: int  # matches played against each opponent

    def checkpoint(self, run_dir: Path) -> Path:
        return run_dir / CHECKPOINTS_DIR / self.path

    @property
    def win_rates(self) -> dict[str, float]:
        return {name: wins / self.episodes for name, wins in self.wins.items()}

    @property
    def average(self) -> float:
        rates = self.win_rates.values()
        return sum(rates) / len(rates)


def play_checkpoint(checkpoint: Path, opponent: Opponent, seeds: Sequence[int]) -> list[MatchResult]:
    """Play a saved checkpoint (player 0) against a bot or a frozen model on the seeds."""
    if isinstance(opponent, Path):
        opponent = MaskablePPO.load(str(opponent), device="cpu")
    return play_matches(MaskablePPO.load(str(checkpoint), device="cpu"), opponent, seeds)


def _count_wins(play: PlayMatches, seeds: Sequence[int], task: tuple[Path, Opponent]) -> int:
    checkpoint, opponent = task
    return sum(match.winner == 0 for match in play(checkpoint, opponent, seeds))


def score_checkpoints(
    run_dir: Path,
    opponents: Mapping[str, Opponent],
    seeds: Sequence[int] = SELECTION_SEEDS,
    *,
    min_step: int = 0,
    max_step: int | None = None,
    play: PlayMatches = play_checkpoint,
    workers: int = 1,
) -> list[ReferenceScore]:
    """Score the run's checkpoints with `min_step <= step <= max_step` against every opponent.

    With `workers` > 1 the (checkpoint, opponent) pairs are played in parallel processes.
    """
    checkpoints = sorted(
        (
            result
            for result in load_selection(run_dir / SELECTION_FILENAME)
            if result.step >= min_step and (max_step is None or result.step <= max_step)
        ),
        key=lambda result: result.step,
    )
    tasks = [
        (run_dir / CHECKPOINTS_DIR / checkpoint.path, opponent)
        for checkpoint in checkpoints
        for opponent in opponents.values()
    ]
    count = partial(_count_wins, play, seeds)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            wins = list(pool.map(count, tasks))
    else:
        wins = list(map(count, tasks))
    scores = []
    for i, checkpoint in enumerate(checkpoints):
        counts = dict(zip(opponents, wins[i * len(opponents) : (i + 1) * len(opponents)]))
        scores.append(ReferenceScore(checkpoint.step, checkpoint.path, counts, len(seeds)))
    return scores


def select_reference(scores: Sequence[ReferenceScore]) -> ReferenceScore:
    """Highest average win rate; a tie goes to the later checkpoint."""
    return max(scores, key=lambda score: (score.average, score.step))


def save_scores(scores: Sequence[ReferenceScore], path: Path) -> None:
    write_json(path, [asdict(score) for score in scores])


def load_scores(path: Path) -> list[ReferenceScore]:
    return [ReferenceScore(**record) for record in json.loads(path.read_text(encoding="utf-8"))]


def evaluate_selected(run_dir: Path, selected: ReferenceScore, *, seeds: range = REPORT_SEEDS) -> ReportEvaluation:
    """The selected checkpoint on the Report seeds, with the run's manifest as its configuration."""
    checkpoint = selected.checkpoint(run_dir)
    manifest = run_dir / MANIFEST_FILENAME
    config = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
    return evaluate_report(
        MaskablePPO.load(str(checkpoint), device="cpu"), model=str(checkpoint), config=config, seeds=seeds
    )


def _report_selected(run_dir: Path, selected: ReferenceScore) -> dict:
    evaluation = evaluate_selected(run_dir, selected)
    for opponent in evaluation.opponents:
        print(f"{run_dir.name} Report seeds vs {opponent.opponent}: {opponent.win_rate:.1%} "
              f"(95% CI {opponent.ci_low:.1%} to {opponent.ci_high:.1%})")
    return {"run": run_dir.name, "selected": asdict(selected), "evaluation": asdict(evaluation)}


def _model_argument(text: str) -> tuple[str, Path]:
    name, separator, path = text.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select each run's checkpoint by average win rate against a reference pool"
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories")
    parser.add_argument(
        "--model", action="append", default=[], type=_model_argument, metavar="NAME=PATH",
        help="Frozen model added to the reference pool (LogicAgent is always in it)",
    )
    parser.add_argument("--min-step", type=int, default=0, help="Only score checkpoints from this step on")
    parser.add_argument("--max-step", type=int, default=None, help="Only score checkpoints up to this step")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes playing matches")
    parser.add_argument(
        "--report-out", type=Path, default=None,
        help="Also play each selected checkpoint on the Report seeds and save the results here",
    )
    args = parser.parse_args()

    opponents: dict[str, Opponent] = {LOGIC_OPPONENT: LOGIC_OPPONENT, **dict(args.model)}
    reports = []
    for run_dir in args.runs:
        scores = score_checkpoints(
            run_dir, opponents, min_step=args.min_step, max_step=args.max_step, workers=args.workers
        )
        save_scores(scores, run_dir / REFERENCE_FILENAME)
        for score in scores:
            rates = " ".join(f"{name}={rate:.0%}" for name, rate in score.win_rates.items())
            print(f"{run_dir.name} {score.step:>9,}: average {score.average:.1%} ({rates})")
        selected = select_reference(scores)
        print(f"{run_dir.name} selected: {selected.path} (average {selected.average:.1%})")
        if args.report_out:
            reports.append(_report_selected(run_dir, selected))
    if args.report_out:
        write_json(args.report_out, reports)


if __name__ == "__main__":
    main()
