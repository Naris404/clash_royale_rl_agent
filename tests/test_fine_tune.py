"""Fine-tuning a Final model: the run's manifest records where it started from."""

from __future__ import annotations

import json
from pathlib import Path

from cr_rl.experiments import fine_tune as fine_tune_module
from cr_rl.experiments.fine_tune import fine_tune
from cr_rl.training.report import ENV_VERSION


def test_the_manifest_records_the_source_model_and_the_fine_tuning_setup(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fine_tune_module, "train", lambda **kwargs: calls.append(kwargs))
    source, exploiter, run_dir = Path("source/ppo_cr_final.zip"), Path("exploiter.zip"), tmp_path / "run"

    fine_tune(source, seed=3, timesteps=500, run_dir=run_dir, permanent_members=[exploiter], n_envs=2)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_model"] == str(source)
    assert manifest["permanent_members"] == [str(exploiter)]
    assert (manifest["layout"], manifest["seed"], manifest["timesteps"]) == ("fine-tune", 3, 500)
    assert manifest["env_version"] == ENV_VERSION
    assert manifest["reward_weights"] == {"win": 1.0, "crown": 0.2, "hp": 0.5}
    (call,) = calls
    assert call["resume"] == source and call["layout"] == "fine-tune" and call["model_dir"] == run_dir
    assert call["permanent_members"] == [exploiter] and call["n_envs"] == 2


def test_a_finished_fine_tune_is_not_trained_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fine_tune_module, "train", lambda **kwargs: calls.append(kwargs))
    (tmp_path / "ppo_cr_final.zip").write_bytes(b"")
    fine_tune(Path("source.zip"), seed=0, timesteps=500, run_dir=tmp_path)
    assert calls == []
