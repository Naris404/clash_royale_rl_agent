"""The v3 plan driver with training and match playing replaced by fakes (existing plan-test style)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cr_rl.experiments import arrows as arrows_module
from cr_rl.experiments import exploiter as exploiter_module
from cr_rl.experiments import fine_tune as fine_tune_module
from cr_rl.experiments import v3
from cr_rl.experiments.plan_results import format_plan, summarize_plan
from cr_rl.stats.reporting import load_experiment_results
from cr_rl.training.head_to_head import HeadToHead
from cr_rl.training.matches import MatchResult
from cr_rl.training.report import REPORT_FILENAME, ReportEvaluation, report_matches, report_metadata
from cr_rl.training import reference_selection
from cr_rl.training.reference_selection import score_checkpoints
from cr_rl.training.selection import CheckpointResult, save_selection

SEEDS = (0, 1)
REPORT_SEEDS = range(2_000_000, 2_000_010)
TRAINING_SEEDS = (0, 1, 2)  # the v2 Final models exist for all three, whatever seeds the plan trains
STEPS = 300  # a fine-tune checkpoints at +0 (the start), +100, +200 and +300
START_STEP = 1_000_000  # a fine-tune continues the step count of the model it starts from


class FakeModel:
    """Stands in for a MaskablePPO loaded from `path`."""

    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def load(cls, path, **_kwargs):
        return cls(path)

    def learn(self, *, callback, **_kwargs):
        for winner in [0] * 72 + [1] * 28:
            callback.rule.record(winner)
        self.num_timesteps = 120_000
        callback.reason = callback.rule.stop_reason(self.num_timesteps)  # stopped by the 70% rule

    def save(self, path):
        Path(path).write_bytes(b"exploiter")


class Fakes:
    """What the faked training and evaluation did, and how strong the models are made to look."""

    def __init__(self):
        self.events: list[tuple] = []
        self.scored: list[tuple[str, dict, int]] = []  # run name, opponents, min_step
        self.arrows: list[tuple[str, str]] = []  # model, frozen model
        self.logic_wins = 9  # of 10 Report seeds, for every checkpoint
        self.h2h_wins = 12  # of 20 Report matches against the v2 Shipped model


@pytest.fixture()
def fakes(monkeypatch):
    fakes = Fakes()

    def fake_train(**kwargs):
        run_dir: Path = kwargs["model_dir"]
        generation = int(run_dir.name[3])
        fakes.events.append(("fine-tune", run_dir.name, kwargs["resume"], list(kwargs["permanent_members"])))
        assert all(member.is_file() for member in kwargs["permanent_members"]), "Exploiters exist before the fine-tune"
        steps = [START_STEP * generation + offset for offset in range(0, STEPS + 1, 100)]
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        for step in steps:
            (run_dir / "checkpoints" / f"ppo_cr_{step}_steps.zip").write_bytes(f"{run_dir.name} {step}".encode())
        save_selection([CheckpointResult(step, 50, 100, f"ppo_cr_{step}_steps.zip") for step in steps], run_dir / "selection.json")
        (run_dir / "ppo_cr_final.zip").write_bytes(f"{run_dir.name} final".encode())

    def fake_exploiter_env(seed, opponents, **_kwargs):
        assert len(set(opponents)) == 1, "every env plays the one frozen target"
        fakes.events.append(("exploiter", Path(opponents[0]), seed))
        return type("Env", (), {"close": lambda self: None})()

    def fake_score(run_dir, opponents, seeds, *, min_step=0, **kwargs):
        fakes.scored.append((run_dir.name, dict(opponents), min_step))
        return real_score(run_dir, opponents, seeds, min_step=min_step, play=fake_play, **kwargs)

    def fake_play(checkpoint, opponent, seeds):
        run_seed = int(checkpoint.parent.parent.name[-1])
        wins = 5 + run_seed  # a later seed is stronger; a later checkpoint ties and so wins the tie
        return [MatchResult(seed, 0 if i < wins else 1, 0.0, 1) for i, seed in enumerate(seeds)]

    def fake_report(player, *, model="", config=None, seeds=REPORT_SEEDS, opponents=("logic", "random")):
        results = {
            "logic": [MatchResult(s, 0 if i < fakes.logic_wins else 1, 0.0, 1) for i, s in enumerate(seeds)],
            "random": [MatchResult(s, 0, 0.0, 1) for s in seeds],
        }
        return ReportEvaluation(report_metadata(model, seeds, config), tuple(report_matches(n, results[n]) for n in opponents))

    def fake_head_to_head(model_a, model_b, *, name_a="a", name_b="b", config=None, seeds=REPORT_SEEDS):
        outcomes = [MatchResult(0, 0 if i < fakes.h2h_wins else 1, 0.0, 1) for i in range(2 * len(seeds))]
        overall = report_matches(name_b, outcomes)
        return HeadToHead(report_metadata(name_a, seeds, config), overall, overall, overall)

    def fake_arrows(model, frozen, **kwargs):
        fakes.arrows.append((model.path.name, frozen.path.name))
        return {"model": kwargs["model_name"]}

    real_score = score_checkpoints
    monkeypatch.setattr(v3, "MaskablePPO", FakeModel)
    monkeypatch.setattr(exploiter_module, "MaskablePPO", FakeModel)
    monkeypatch.setattr(exploiter_module, "make_vec_training_env", fake_exploiter_env)
    monkeypatch.setattr(fine_tune_module, "train", fake_train)
    monkeypatch.setattr(v3, "score_checkpoints", fake_score)
    monkeypatch.setattr(reference_selection, "MaskablePPO", FakeModel)
    monkeypatch.setattr(reference_selection, "evaluate_report", fake_report)
    monkeypatch.setattr(v3, "evaluate_head_to_head", fake_head_to_head)
    monkeypatch.setattr(v3, "run_arrows_experiment", fake_arrows)
    monkeypatch.setattr(v3, "save_arrows_results", lambda results, path: path.write_text(json.dumps(results), encoding="utf-8"))
    return fakes


@pytest.fixture()
def models(tmp_path):
    """The v2 models on disk: three Final models and the Shipped model."""
    v2_dir = tmp_path / "v2"
    for seed in TRAINING_SEEDS:
        final = v2_dir / f"full-seed{seed}" / "ppo_cr_final.zip"
        final.parent.mkdir(parents=True)
        final.write_bytes(f"v2 final {seed}".encode())
    shipped = tmp_path / "models" / "ppo_cr_best.zip"
    shipped.parent.mkdir()
    shipped.write_bytes(b"v2 shipped")
    return v2_dir, shipped


def run_plan(tmp_path, models, **overrides) -> Path:
    v2_dir, shipped = models
    out_dir = tmp_path / "v3"
    options = dict(
        timesteps=STEPS, seeds=SEEDS, out_dir=out_dir, v2_dir=v2_dir, shipped_path=shipped,
        v2_shipped_path=shipped.with_name("ppo_cr_v2_shipped.zip"), report_seeds=REPORT_SEEDS,
        n_envs=2, exploiter_steps=1_000, selection_seeds=range(10), workers=1,
    )
    v3.run_v3(**{**options, **overrides})
    return out_dir


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_each_seed_runs_two_generations_with_every_exploiter_a_permanent_member_before_its_fine_tune(
    tmp_path, models, fakes
):
    out_dir = run_plan(tmp_path, models)
    v2_dir = models[0]

    for seed in SEEDS:
        final_v2 = v2_dir / f"full-seed{seed}" / "ppo_cr_final.zip"
        gen1, gen2 = (out_dir / f"gen{g}-seed{seed}" for g in (1, 2))
        exploiter1, exploiter2 = (out_dir / f"exploiter{g}-seed{seed}" / "exploiter.zip" for g in (1, 2))
        fine_tunes = {event[1]: event for event in fakes.events if event[0] == "fine-tune"}
        assert fine_tunes[gen1.name][2:] == (final_v2, [exploiter1])
        assert fine_tunes[gen2.name][2:] == (gen1 / "ppo_cr_final.zip", [exploiter1, exploiter2])
        # Exploiter g trains against the generation g-1 main agent, ahead of the fine-tune that uses it.
        sequence = [
            (event[0], event[1]) for event in fakes.events
            if (event[0] == "exploiter" and event[1] in (final_v2, gen1 / "ppo_cr_final.zip"))
            or (event[0] == "fine-tune" and event[1] in (gen1.name, gen2.name))
        ]
        assert sequence == [
            ("exploiter", final_v2), ("fine-tune", gen1.name),
            ("exploiter", gen1 / "ppo_cr_final.zip"), ("fine-tune", gen2.name),
        ]
        assert read(gen1 / "manifest.json")["source_model"] == str(final_v2)


def test_only_generation_2_checkpoints_the_run_trained_are_scored_against_the_reference_pool(tmp_path, models, fakes):
    out_dir = run_plan(tmp_path, models)
    v2_dir, shipped = models

    assert [name for name, _, _ in fakes.scored] == ["gen2-seed0", "gen2-seed1"]
    for name, opponents, min_step in fakes.scored:
        assert list(opponents) == ["logic", "v2-shipped", "v2-final-seed0", "v2-final-seed1", "v2-final-seed2"]
        assert opponents["v2-final-seed2"] == v2_dir / "full-seed2" / "ppo_cr_final.zip"
        assert opponents["v2-shipped"] == shipped.with_name("ppo_cr_v2_shipped.zip")
        assert min_step == 2 * START_STEP + 1  # the checkpoint the generation started from is the previous one's
    scored_steps = [score["step"] for score in read(out_dir / "gen2-seed0" / "reference_selection.json")]
    assert scored_steps == [2 * START_STEP + offset for offset in (100, 200, 300)]
    assert not (out_dir / "gen1-seed0" / "reference_selection.json").exists()


def test_the_best_run_by_reference_score_is_the_candidate_and_its_latest_best_checkpoint_the_selected_model(
    tmp_path, models, fakes
):
    out_dir = run_plan(tmp_path, models)

    results = read(out_dir / REPORT_FILENAME)
    assert results["shipped"] == "gen2-seed1"  # the fake play makes seed 1 stronger
    selected = next(run for run in results["runs"] if run["name"] == "gen2-seed1")["selection"]
    assert selected["path"] == f"ppo_cr_{2 * START_STEP + 300}_steps.zip"  # all tie: the later checkpoint
    assert results["reference_pool"] == ["logic", "v2-shipped", "v2-final-seed0", "v2-final-seed1", "v2-final-seed2"]


def test_a_run_that_meets_the_target_is_shipped_and_the_v2_shipped_model_is_kept(tmp_path, models, fakes):
    _, shipped = models
    out_dir = run_plan(tmp_path, models)

    target = read(out_dir / REPORT_FILENAME)["target"]
    assert target["met"] and target["run"] == "gen2-seed1"
    assert shipped.read_bytes() == b"gen2-seed1 " + str(2 * START_STEP + 300).encode()
    assert shipped.with_name("ppo_cr_v2_shipped.zip").read_bytes() == b"v2 shipped"


@pytest.mark.parametrize(
    "logic_wins, h2h_wins",
    [(8, 12), (9, 10), (9, 9)],  # below 90% vs LogicAgent / an even head-to-head / a lost head-to-head
    ids=["weak-vs-logic", "even-head-to-head", "lost-head-to-head"],
)
def test_a_run_that_misses_the_target_is_not_shipped(tmp_path, models, fakes, logic_wins, h2h_wins):
    _, shipped = models
    fakes.logic_wins, fakes.h2h_wins = logic_wins, h2h_wins
    out_dir = run_plan(tmp_path, models)

    assert not read(out_dir / REPORT_FILENAME)["target"]["met"]
    assert shipped.read_bytes() == b"v2 shipped"
    assert shipped.with_name("ppo_cr_v2_shipped.zip").read_bytes() == b"v2 shipped"


def test_reports_the_selected_models_head_to_head_against_the_v2_shipped_model(tmp_path, models, fakes):
    out_dir = run_plan(tmp_path, models)

    results = read(out_dir / REPORT_FILENAME)
    assert [(pair["metadata"]["model"], pair["overall"]["opponent"]) for pair in results["head_to_head_matrix"]] == [
        ("gen2-seed0", "v2-shipped"), ("gen2-seed1", "v2-shipped"),
    ]
    assert results["head_to_head"]["metadata"]["model"] == "gen2-seed1"
    assert {row["opponent"] for row in results["results"]} == {"logic", "random"}


def test_the_arrows_experiment_runs_on_the_v3_selected_model_against_the_v2_shipped_model(tmp_path, models, fakes):
    out_dir = run_plan(tmp_path, models)

    assert fakes.arrows == [(f"ppo_cr_{2 * START_STEP + 300}_steps.zip", "ppo_cr_v2_shipped.zip")]
    assert (out_dir / "arrows.json").is_file()


def test_the_run_plan_summary_and_the_statistics_tooling_read_the_results(tmp_path, models, fakes):
    out_dir = run_plan(tmp_path, models)

    plan = summarize_plan(out_dir)
    assert [run.name for run in plan.arms[0].runs] == ["gen2-seed0", "gen2-seed1"]
    assert [run.shipped for run in plan.arms[0].runs] == [False, True]
    text = format_plan(plan)
    assert "v3 target" in text and "MET" in text
    assert "exploiter1-seed0" in text and "72.0%" in text
    assert len(load_experiment_results(out_dir.parent)) == 4  # 2 runs x 2 opponents


def test_finished_runs_are_not_trained_again_on_restart(tmp_path, models, fakes):
    run_plan(tmp_path, models)
    trained = len(fakes.events)

    run_plan(tmp_path, models)

    assert len(fakes.events) == trained


@pytest.mark.slow
def test_the_smoke_configuration_runs_the_whole_pipeline_with_real_training(tmp_path, untrained_model_path):
    v2_dir = tmp_path / "v2"
    for seed in TRAINING_SEEDS:
        final = v2_dir / f"full-seed{seed}" / "ppo_cr_final.zip"
        final.parent.mkdir(parents=True)
        shutil.copyfile(untrained_model_path, final)
    shipped = tmp_path / "models" / "ppo_cr_best.zip"
    shipped.parent.mkdir()
    shutil.copyfile(untrained_model_path, shipped)
    out_dir = tmp_path / "v3"

    v3.run_v3(
        **{**v3.SMOKE_OPTIONS, "out_dir": out_dir, "v2_dir": v2_dir, "shipped_path": shipped,
           "v2_shipped_path": shipped.with_name("ppo_cr_v2_shipped.zip")}
    )

    results = read(out_dir / REPORT_FILENAME)
    assert results["experiment"] == "v3" and results["shipped"] == "gen2-seed0"
    assert isinstance(results["target"]["met"], bool)
    assert (out_dir / "arrows.json").is_file() and shipped.with_name("ppo_cr_v2_shipped.zip").is_file()
