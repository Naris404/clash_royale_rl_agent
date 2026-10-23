"""Training diagnostics: win rate per Training opponent, logging cadence, the entropy arm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.experiments.sweep import CONFIGS
from cr_rl.training.train import LOG_INTERVAL, WinRateCallback


class _Logger:
    """Collects what the callback records until the test takes it, like SB3's logger dump."""

    def __init__(self):
        self._pending: dict[str, float] = {}

    def record(self, key, value):
        self._pending[key] = value

    def take(self) -> dict[str, float]:
        taken, self._pending = self._pending, {}
        return taken


def _callback() -> tuple[WinRateCallback, _Logger, SimpleNamespace]:
    logger = _Logger()
    model = SimpleNamespace(num_timesteps=0, get_env=lambda: None, logger=logger)
    callback = WinRateCallback()
    callback.init_callback(model)
    return callback, logger, model


def _step(callback, model, n_envs: int, finished: list[tuple[str, int | None]] = ()) -> None:
    """One vec-env step in which the first envs finish an episode: (opponent, winner)."""
    model.num_timesteps += n_envs
    dones = [False] * n_envs
    infos = [{} for _ in range(n_envs)]
    for i, (opponent, winner) in enumerate(finished):
        dones[i] = True
        infos[i] = {"opponent": opponent, "winner": winner}
    callback.update_locals({"dones": dones, "infos": infos})
    callback.on_step()


class TestWinRatePerOpponent:
    def test_each_training_opponent_gets_its_own_rate(self):
        callback, logger, model = _callback()
        _step(callback, model, 4, [("random", 0), ("logic", 1), ("logic", 0), ("logic-eps", 1)])
        _step(callback, model, 4, [("random", 0), ("logic", 1)])
        model.num_timesteps = LOG_INTERVAL
        _step(callback, model, 1)
        logged = logger.take()
        assert logged["rollout/win_rate_vs_random"] == pytest.approx(1.0)
        assert logged["rollout/win_rate_vs_logic"] == pytest.approx(1 / 3)
        assert logged["rollout/win_rate_vs_logic-eps"] == pytest.approx(0.0)
        assert "rollout/win_rate_vs_past-self" not in logged

    def test_draws_are_skipped(self):
        callback, logger, model = _callback()
        _step(callback, model, 2, [("logic", None), ("logic", 0)])
        model.num_timesteps = LOG_INTERVAL
        _step(callback, model, 1)
        assert logger.take()["rollout/win_rate_vs_logic"] == pytest.approx(1.0)

    def test_env_reports_the_opponent_it_plays(self):
        def opponent_in_step_info(opponent) -> str:
            env = ClashRoyaleEnv(seed=0, opponent=opponent)
            env.reset()
            return env.step(0)[4]["opponent"]

        assert opponent_in_step_info("logic-eps") == "logic-eps"
        assert opponent_in_step_info("random") == "random"
        frozen = SimpleNamespace(predict=lambda *args, **kwargs: (0, None))
        assert opponent_in_step_info(frozen) == "past-self"


@pytest.mark.parametrize("n_envs", [1, 12])
def test_logs_about_every_log_interval_for_any_number_of_envs(n_envs):
    callback, logger, model = _callback()
    logged_at = []
    for _ in range(7 * LOG_INTERVAL // 2 // n_envs):
        _step(callback, model, n_envs, [("logic", 0)])
        if logger.take():
            logged_at.append(model.num_timesteps)
    assert len(logged_at) == 3
    gaps = [b - a for a, b in zip(logged_at, logged_at[1:])]
    assert all(LOG_INTERVAL <= gap < LOG_INTERVAL + n_envs for gap in gaps)


def test_sweep_includes_the_0_01_entropy_arm():
    assert any(overrides.get("ent_coef") == 0.01 for overrides in CONFIGS.values())
