"""Read-only summary of the run plans' results, from the files a plan already wrote."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cr_rl.experiments.plan_results import main, summarize_plan

# Selection rounds recorded by final_configuration/full-seed0 (1M steps, full layout): the
# self-play phase is planned at 600k but starts with the round at 601,524, after the rollout end.
FULL_1M_ROUNDS = [
    (50004, 33), (200016, 47), (250020, 71), (550044, 95), (600048, 92),
    (601524, 98), (651528, 86), (1001556, 80), (1002540, 80),
]


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def opponent(name: str, win_rate: float, mean_length: float = 30.0, episodes: int = 200) -> dict:
    wins = round(win_rate * episodes)
    return {
        "opponent": name, "episodes": episodes, "wins": wins, "losses": episodes - wins, "draws": 0,
        "win_rate": win_rate, "ci_low": win_rate - 0.05, "ci_high": win_rate + 0.05,
        "mean_return": 0.5, "mean_length": mean_length,
    }


def run_record(name: str, step: int, wins: int, logic: float, random: float = 0.9) -> dict:
    return {
        "name": name,
        "selection": {"step": step, "wins": wins, "episodes": 100, "path": f"ppo_cr_{step}_steps.zip"},
        "evaluation": {
            "metadata": {"model": name},
            "opponents": [opponent("logic", logic), opponent("random", random)],
        },
    }


def write_run(plan_dir: Path, name: str, *, layout: str, timesteps: int, rounds: list[tuple[int, int]]) -> None:
    manifest = {"arm": name.rsplit("-seed", 1)[0], "layout": layout, "timesteps": timesteps}
    write(plan_dir / name / "manifest.json", manifest)
    write(
        plan_dir / name / "selection.json",
        [{"step": step, "wins": wins, "episodes": 100, "path": f"ppo_cr_{step}_steps.zip"} for step, wins in rounds],
    )


def final_plan(tmp_path: Path, rounds: list[tuple[int, int]] = FULL_1M_ROUNDS) -> Path:
    """A final-configuration plan with one run that selected its best round."""
    plan_dir = tmp_path / "final_configuration"
    step, wins = max(rounds, key=lambda point: (point[1], point[0]))
    write_run(plan_dir, "full-seed0", layout="full", timesteps=1_000_000, rounds=rounds)
    write(
        plan_dir / "results.json",
        {
            "experiment": "final_configuration",
            "timesteps": 1_000_000,
            "runs": [run_record("full-seed0", step, wins, logic=0.935)],
            "shipped": "full-seed0",
            "spread": {"logic": {"mean": 0.935, "std": 0.0, "min": 0.935, "max": 0.935}},
            "curriculum": {"full-seed0": {"jumpstart": 0.47, "steps_to_half": 250020, "report_win_rate": 0.935}},
        },
    )
    return plan_dir


def pair(model: str, other: str, win_rate: float) -> dict:
    """A recorded head-to-head of `model` against `other`."""
    report = opponent(other, win_rate, episodes=400)
    return {"metadata": {"model": model}, "overall": report, "as_player_0": report, "as_player_1": report}


def ablation_plan(tmp_path: Path) -> Path:
    """Two arms of two runs each and a 2 x 2 head-to-head matrix between them."""
    plan_dir = tmp_path / "ablations"
    rounds = [(50004, 30), (100008, 60), (300762, 80), (501270, 70)]
    for name in ("full-seed0", "full-seed1"):
        write_run(plan_dir, name, layout="full", timesteps=500_000, rounds=rounds)
    for name in ("no-self-play-seed0", "no-self-play-seed1"):
        write_run(plan_dir, name, layout="no-self-play", timesteps=500_000, rounds=rounds)
    matrix = [
        pair("full-seed0", "no-self-play-seed0", 0.80),
        pair("full-seed0", "no-self-play-seed1", 0.60),
        pair("full-seed1", "no-self-play-seed0", 0.70),
        pair("full-seed1", "no-self-play-seed1", 0.30),
    ]
    write(
        plan_dir / "results.json",
        {
            "experiment": "ablations",
            "timesteps": 500_000,
            "arms": {
                "full": {
                    "runs": [run_record("full-seed0", 300762, 80, 0.76), run_record("full-seed1", 300762, 80, 0.80)],
                    "shipped": "full-seed1",
                    "spread": {"logic": {"mean": 0.78, "std": 0.028, "min": 0.76, "max": 0.80}},
                },
                "no-self-play": {
                    "runs": [
                        run_record("no-self-play-seed0", 300762, 80, 0.86),
                        run_record("no-self-play-seed1", 300762, 80, 0.84),
                    ],
                    "shipped": "no-self-play-seed0",
                    "spread": {"logic": {"mean": 0.85, "std": 0.014, "min": 0.84, "max": 0.86}},
                },
            },
            "curriculum": {
                "full-seed0": {"jumpstart": 0.40, "steps_to_half": 100008, "report_win_rate": 0.76},
                "full-seed1": {"jumpstart": 0.50, "steps_to_half": 200016, "report_win_rate": 0.80},
                "no-self-play-seed0": {"jumpstart": 0.40, "steps_to_half": 100008, "report_win_rate": 0.86},
                "no-self-play-seed1": {"jumpstart": 0.45, "steps_to_half": None, "report_win_rate": 0.84},
            },
            "head_to_head": pair("full-seed1", "no-self-play-seed0", 0.70),
            "head_to_head_matrix": matrix,
        },
    )
    return plan_dir


class TestArms:
    def test_each_arm_has_the_mean_and_sd_of_its_report_win_rate_vs_logic(self, tmp_path):
        arms = summarize_plan(ablation_plan(tmp_path)).arms
        assert [(arm.arm, arm.logic_spread.mean, arm.logic_spread.std) for arm in arms] == [
            ("full", 0.78, 0.028),
            ("no-self-play", 0.85, 0.014),
        ]

    def test_each_arm_has_the_mean_curriculum_metrics_of_its_runs(self, tmp_path):
        full, no_self_play = summarize_plan(ablation_plan(tmp_path)).arms
        assert full.jumpstart == pytest.approx(0.45)
        assert full.steps_to_half == pytest.approx(150012)
        assert no_self_play.steps_to_half is None  # one run never reached 50%

    def test_marks_the_shipped_run_of_each_arm(self, tmp_path):
        arms = summarize_plan(ablation_plan(tmp_path)).arms
        assert [[run.shipped for run in arm.runs] for arm in arms] == [[False, True], [True, False]]


class TestHeadToHead:
    def test_the_matrix_has_each_rows_and_each_columns_mean(self, tmp_path):
        grid = summarize_plan(ablation_plan(tmp_path)).matrix
        assert grid.rows == ("full-seed0", "full-seed1")
        assert grid.columns == ("no-self-play-seed0", "no-self-play-seed1")
        assert grid.win_rates == ((0.80, 0.60), (0.70, 0.30))
        assert grid.row_means == pytest.approx((0.70, 0.50))
        assert grid.column_means == pytest.approx((0.75, 0.45))

    def test_keeps_the_shipped_pair(self, tmp_path):
        shipped = summarize_plan(ablation_plan(tmp_path)).shipped_pair
        assert (shipped.model, shipped.result.opponent, shipped.result.win_rate) == (
            "full-seed1", "no-self-play-seed0", 0.70,
        )

    def test_the_final_plan_has_selected_vs_final_and_no_matrix(self, tmp_path):
        plan_dir = final_plan(tmp_path)
        results = json.loads((plan_dir / "results.json").read_text(encoding="utf-8"))
        results["selected_vs_final"] = [pair("full-seed0 selected", "full-seed0 final", 0.225)]
        write(plan_dir / "results.json", results)
        summary = summarize_plan(plan_dir)
        assert summary.matrix is None and summary.shipped_pair is None
        assert [(p.model, p.result.opponent, p.result.win_rate) for p in summary.selected_vs_final] == [
            ("full-seed0 selected", "full-seed0 final", 0.225)
        ]


class TestSelectedPhase:
    def test_the_round_that_starts_the_self_play_phase_is_labelled_with_the_phase_before_it(self, tmp_path):
        run = summarize_plan(final_plan(tmp_path)).arms[0].runs[0]
        assert run.selection.step == 601524
        assert run.selected_phase == "scripted-mix"
        assert run.selected_before_self_play

    def test_a_round_after_the_self_play_start_is_in_the_self_play_phase(self, tmp_path):
        rounds = [(step, 99 if step == 651528 else wins) for step, wins in FULL_1M_ROUNDS]
        run = summarize_plan(final_plan(tmp_path, rounds)).arms[0].runs[0]
        assert (run.selection.step, run.selected_phase) == (651528, "self-play-mix")
        assert not run.selected_before_self_play

    def test_a_layout_without_self_play_never_warns(self, tmp_path):
        plan_dir = final_plan(tmp_path)
        write_run(plan_dir, "full-seed0", layout="no-self-play", timesteps=1_000_000, rounds=FULL_1M_ROUNDS)
        run = summarize_plan(plan_dir).arms[0].runs[0]
        assert run.selected_phase == "scripted-mix"
        assert not run.selected_before_self_play

    def test_the_selection_curve_marks_each_rounds_phase(self, tmp_path):
        curve = summarize_plan(final_plan(tmp_path)).arms[0].runs[0].curve
        assert [(point.step, point.phase) for point in curve] == [
            (50004, "warm-up"),
            (200016, "scripted-mix"),  # past the planned 200k; no round marks the warm-up's real end
            (250020, "scripted-mix"),
            (550044, "scripted-mix"),
            (600048, "scripted-mix"),
            (601524, "scripted-mix"),
            (651528, "self-play-mix"),
            (1001556, "self-play-mix"),
            (1002540, "self-play-mix"),
        ]

    def test_a_short_run_whose_final_round_comes_soon_after_the_self_play_start(self, tmp_path):
        plan_dir = final_plan(tmp_path)
        write_run(plan_dir, "full-seed0", layout="full", timesteps=20_000, rounds=[(12288, 40), (20480, 38)])
        curve = summarize_plan(plan_dir).arms[0].runs[0].curve
        assert [point.phase for point in curve] == ["scripted-mix", "self-play-mix"]

    def test_without_the_runs_selection_file_the_phase_is_unknown(self, tmp_path):
        plan_dir = final_plan(tmp_path)
        (plan_dir / "full-seed0" / "selection.json").unlink()
        run = summarize_plan(plan_dir).arms[0].runs[0]
        assert run.selection.step == 601524
        assert run.selected_phase is None and run.curve == () and not run.selected_before_self_play


class TestCommand:
    def test_reads_every_plan_directory_that_has_results(self, tmp_path, capsys):
        final_plan(tmp_path)
        ablation_plan(tmp_path)
        (tmp_path / "logs").mkdir()
        main([str(tmp_path), "--json"])
        assert [plan["name"] for plan in json.loads(capsys.readouterr().out)] == ["ablations", "final_configuration"]

    def test_skips_the_results_of_an_experiment_that_is_not_a_run_plan(self, tmp_path, capsys):
        final_plan(tmp_path)
        write(tmp_path / "arrows" / "results.json", {"experiment": "arrows", "profile": {}, "win_rates": {}})
        main([str(tmp_path), "--json"])
        assert [plan["name"] for plan in json.loads(capsys.readouterr().out)] == ["final_configuration"]

    def test_reads_a_single_plan_directory(self, tmp_path, capsys):
        main([str(final_plan(tmp_path)), "--json"])
        (plan,) = json.loads(capsys.readouterr().out)
        run = plan["arms"][0]["runs"][0]
        assert (run["selection"]["step"], run["selected_phase"], run["selected_before_self_play"]) == (
            601524, "scripted-mix", True,
        )

    def test_warns_when_the_selected_checkpoint_has_no_self_play_training(self, tmp_path, capsys):
        main([str(final_plan(tmp_path))])
        out = capsys.readouterr().out
        assert "full-seed0: selected before self-play: the shipped model has no self-play training" in out

    def test_the_warning_names_the_selected_model_of_a_run_that_was_not_shipped(self, tmp_path, capsys):
        main([str(ablation_plan(tmp_path))])
        out = capsys.readouterr().out
        assert "full-seed0: selected before self-play: the selected model has no self-play training" in out
        assert "full-seed1: selected before self-play: the shipped model has no self-play training" in out
        assert "no-self-play-seed0: selected before self-play" not in out

    def test_says_when_a_runs_selection_file_is_missing(self, tmp_path, capsys):
        plan_dir = final_plan(tmp_path)
        (plan_dir / "full-seed0" / "selection.json").unlink()
        main([str(plan_dir)])
        out = capsys.readouterr().out
        assert "full-seed0: no selection.json: phase, selection curve and self-play check unavailable" in out

    def test_the_selection_curve_marks_the_phase_boundaries_and_the_selected_round(self, tmp_path, capsys):
        main([str(final_plan(tmp_path))])
        assert (
            "full-seed0  warm-up 50k:33 | scripted-mix 200k:47 250k:71 550k:95 600k:92 601.5k:98*"
            " | self-play-mix 651.5k:86 1001.6k:80 1002.5k:80"
        ) in capsys.readouterr().out

    def test_the_tables_show_the_numbers_of_the_json(self, tmp_path, capsys):
        plan_dir = ablation_plan(tmp_path)
        main([str(plan_dir), "--json"])
        (plan,) = json.loads(capsys.readouterr().out)
        main([str(plan_dir)])
        text = capsys.readouterr().out
        rates = [
            value
            for arm in plan["arms"]
            for run in arm["runs"]
            for report in run["report"]
            for value in (report["win_rate"], report["ci_low"], report["ci_high"])
        ]
        rates += [arm["logic_spread"][key] for arm in plan["arms"] for key in ("mean", "std")]
        rates += [rate for row in plan["matrix"]["win_rates"] for rate in row]
        rates += plan["matrix"]["row_means"] + plan["matrix"]["column_means"]
        rates.append(plan["shipped_pair"]["result"]["win_rate"])
        missing = [rate for rate in rates if f"{rate:.1%}" not in text]
        assert not missing
