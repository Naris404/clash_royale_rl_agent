"""
Statystyki treningu i jakości modelu PPO vs LogicAgent.

Uruchomienie:
  python statistics.py
  python statistics.py --model models/ppo_cr_best.zip --episodes 100
  python statistics.py --plot
  python statistics.py --compare-checkpoints --episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from train import DEFAULT_BEST_PATH, DEFAULT_MODEL_DIR, evaluate_model

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


STATS_DIR = DEFAULT_MODEL_DIR / "stats"
KNOWN_MODELS = (
    ("Najlepszy (eval)", DEFAULT_BEST_PATH),
    ("Finalny", DEFAULT_MODEL_DIR / "ppo_cr_final.zip"),
    ("Eval callback", DEFAULT_MODEL_DIR / "best_model.zip"),
)


@dataclass
class ScalarSeries:
    tag: str
    steps: list[int]
    values: list[float]
    run: str


@dataclass
class ModelEvalResult:
    label: str
    path: str
    exists: bool
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    mean_reward: float = 0.0
    mean_length: float = 0.0
    error: str | None = None


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def _discover_tb_runs(tb_root: Path) -> list[Path]:
    if not tb_root.is_dir():
        return []
    runs = [p for p in tb_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def load_tensorboard_scalars(tb_root: Path) -> list[ScalarSeries]:
    if not _HAS_TB:
        return []

    series: list[ScalarSeries] = []
    for run_dir in _discover_tb_runs(tb_root):
        try:
            accumulator = EventAccumulator(str(run_dir))
            accumulator.Reload()
        except Exception as error:
            print(f"Nie udalo sie wczytac TensorBoard ({run_dir.name}): {error}")
            continue

        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            if not events:
                continue
            series.append(
                ScalarSeries(
                    tag=tag,
                    steps=[int(e.step) for e in events],
                    values=[float(e.value) for e in events],
                    run=run_dir.name,
                )
            )
    return series


def load_eval_npz(eval_dir: Path) -> dict[str, Any] | None:
    npz_path = eval_dir / "evaluations.npz"
    if not npz_path.is_file():
        return None
    data = np.load(npz_path)
    return {
        "timesteps": data["timesteps"].tolist(),
        "results": data["results"].tolist(),
        "ep_lengths": data["ep_lengths"].tolist(),
    }


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    files = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return files


def discover_models(extra: str | None = None) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    if extra:
        path = Path(extra)
        if path.is_file():
            found.append(("Wybrany", path))
            seen.add(path.resolve())

    for label, path in KNOWN_MODELS:
        resolved = path.resolve()
        if path.is_file() and resolved not in seen:
            found.append((label, path))
            seen.add(resolved)

    for checkpoint in find_checkpoints(DEFAULT_MODEL_DIR / "checkpoints"):
        resolved = checkpoint.resolve()
        if resolved not in seen:
            found.append((f"Checkpoint: {checkpoint.stem}", checkpoint))
            seen.add(resolved)

    return found


def evaluate_random_baseline(episodes: int, seed: int) -> ModelEvalResult:
    from rl_agent import RLAgent

    from gym_env import ClashRoyaleEnv

    env = ClashRoyaleEnv(seed=seed)
    agent = RLAgent(seed=seed, model_path=None, play_chance=0.5)

    wins = losses = draws = 0
    rewards: list[float] = []
    lengths: list[int] = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.choose_action(env.board, player=0)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        winner = info.get("winner")
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

    env.close()
    return ModelEvalResult(
        label="Losowy agent (baseline)",
        path="(brak modelu)",
        exists=True,
        episodes=episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / max(1, episodes),
        mean_reward=float(np.mean(rewards)),
        mean_length=float(np.mean(lengths)),
    )


def evaluate_saved_model(
    label: str,
    path: Path,
    *,
    episodes: int,
    seed: int,
) -> ModelEvalResult:
    if not path.is_file():
        return ModelEvalResult(label=label, path=str(path), exists=False)

    try:
        stats = evaluate_model(path, n_episodes=episodes, seed=seed)
    except Exception as error:
        return ModelEvalResult(
            label=label,
            path=str(path),
            exists=True,
            error=str(error),
        )

    return ModelEvalResult(
        label=label,
        path=str(path),
        exists=True,
        episodes=int(stats["episodes"]),
        wins=int(stats["wins"]),
        losses=int(stats["losses"]),
        draws=int(stats["draws"]),
        win_rate=float(stats["win_rate"]),
        mean_reward=float(stats["mean_reward"]),
        mean_length=float(stats["mean_length"]),
    )


def print_training_curves(series: list[ScalarSeries]) -> None:
    _section("Logi treningu (TensorBoard)")

    if not _HAS_TB:
        print("Brak pakietu tensorboard — zainstaluj: pip install tensorboard")
        return

    if not series:
        print(f"Brak logow w {DEFAULT_MODEL_DIR / 'tb'}")
        print("Uruchom trening: python train.py")
        return

    runs = sorted({s.run for s in series})
    print(f"Znalezione runy: {', '.join(runs)}")

    interesting = [
        "rollout/ep_rew_mean",
        "rollout/win_rate_vs_logic",
        "eval/mean_reward",
        "eval/mean_ep_length",
        "train/explained_variance",
        "train/value_loss",
        "train/entropy_loss",
        "time/fps",
    ]

    for tag in interesting:
        matching = [s for s in series if s.tag == tag]
        if not matching:
            continue
        best = max(matching, key=lambda s: s.steps[-1] if s.steps else -1)
        first = best.values[0]
        last = best.values[-1]
        peak = max(best.values)
        print(
            f"  {tag}: "
            f"pierwsza={_fmt_float(first)} | ostatnia={_fmt_float(last)} | "
            f"szczyt={_fmt_float(peak)} | punkty={len(best.values)}"
        )


def print_eval_npz_summary(eval_data: dict[str, Any] | None) -> None:
    _section("Historia ewaluacji (evaluations.npz)")

    if eval_data is None:
        print(f"Brak pliku {DEFAULT_MODEL_DIR / 'eval' / 'evaluations.npz'}")
        print("(pojawia sie po dluższym treningu — callback co ~25k krokow)")
        return

    timesteps = eval_data["timesteps"]
    results = eval_data["results"]
    lengths = eval_data["ep_lengths"]

    print(f"Liczba ewaluacji: {len(timesteps)}")
    if not timesteps:
        return

    best_idx = int(np.argmax(results))
    print(f"Najlepsza srednia nagroda: {_fmt_float(results[best_idx])} @ krok {timesteps[best_idx]:,}")
    print(f"Ostatnia srednia nagroda: {_fmt_float(results[-1])} @ krok {timesteps[-1]:,}")
    print(f"Ostatnia srednia dlugosc meczu: {_fmt_float(lengths[-1], 0)} tickow")


def print_model_table(results: list[ModelEvalResult]) -> None:
    _section("Ewaluacja na zywo vs LogicAgent")

    if not results:
        print("Brak modeli do oceny.")
        return

    header = f"{'Model':<28} {'Win%':>7} {'W':>5} {'L':>5} {'D':>4} {'Nagr.':>9} {'Ticki':>8}"
    print(header)
    print("-" * len(header))

    for row in results:
        if row.error:
            print(f"{row.label:<28}  BLAD: {row.error}")
            continue
        if not row.exists:
            print(f"{row.label:<28}  (brak pliku)")
            continue
        print(
            f"{row.label:<28} "
            f"{_fmt_pct(row.win_rate):>7} "
            f"{row.wins:>5} "
            f"{row.losses:>5} "
            f"{row.draws:>4} "
            f"{_fmt_float(row.mean_reward):>9} "
            f"{_fmt_float(row.mean_length, 0):>8}"
        )


def print_recommendation(results: list[ModelEvalResult]) -> None:
    _section("Podsumowanie")

    trained = [r for r in results if r.exists and not r.error and r.label != "Losowy agent (baseline)"]
    if not trained:
        print("Brak wytrenowanych modeli. Uruchom: python train.py")
        return

    best = max(trained, key=lambda r: r.win_rate)
    baseline = next((r for r in results if r.label == "Losowy agent (baseline)"), None)

    print(f"Najlepszy model: {best.label}")
    print(f"  Sciezka: {best.path}")
    print(f"  Win rate ({best.episodes} meczy): {_fmt_pct(best.win_rate)}")
    print(f"  Srednia nagroda (HP wiez): {_fmt_float(best.mean_reward)}")

    if baseline and baseline.exists:
        delta = best.win_rate - baseline.win_rate
        sign = "+" if delta >= 0 else ""
        print(
            f"  vs losowy baseline ({_fmt_pct(baseline.win_rate)}): "
            f"{sign}{_fmt_pct(delta)}"
        )

    if best.win_rate >= 0.55:
        print("  Ocena: stabilnie bije LogicAgent — dobry wynik na projekt.")
    elif best.win_rate >= 0.45:
        print("  Ocena: wyrównany poziom — kontynuuj trening (wiecej krokow).")
    else:
        print("  Ocena: slaby wynik — potrzebny dluzszy trening lub tuning hiperparametrow.")


def save_plots(
    series: list[ScalarSeries],
    eval_data: dict[str, Any] | None,
    live_results: list[ModelEvalResult],
    output_dir: Path,
) -> list[Path]:
    if not _HAS_MPL:
        print("\nBrak matplotlib — pomijam wykresy (pip install matplotlib).")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _plot_series(tags: list[str], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 4))
        plotted = False
        for tag in tags:
            for s in series:
                if s.tag != tag:
                    continue
                ax.plot(s.steps, s.values, label=f"{s.run} / {tag}", linewidth=1.8)
                plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("Krok treningu")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    _plot_series(
        ["rollout/ep_rew_mean", "eval/mean_reward"],
        "Srednia nagroda w trakcie treningu",
        "Nagroda",
        "reward_curve.png",
    )
    _plot_series(
        ["rollout/win_rate_vs_logic"],
        "Win rate vs LogicAgent (okno treningowe)",
        "Win rate",
        "win_rate_training.png",
    )
    _plot_series(
        ["train/explained_variance", "train/value_loss"],
        "Stabilnosc uczenia (value function)",
        "Wartosc",
        "training_stability.png",
    )

    if eval_data and eval_data["timesteps"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(eval_data["timesteps"], eval_data["results"], marker="o", linewidth=1.8)
        ax.set_title("Ewaluacja okresowa (eval callback)")
        ax.set_xlabel("Krok treningu")
        ax.set_ylabel("Srednia nagroda")
        ax.grid(True, alpha=0.3)
        path = output_dir / "eval_callback.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    trained_live = [
        r
        for r in live_results
        if r.exists and not r.error and r.label != "Losowy agent (baseline)"
    ]
    if trained_live:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [r.label for r in trained_live]
        rates = [100.0 * r.win_rate for r in trained_live]
        colors = ["#3b82f6" if "Checkpoint" not in l else "#94a3b8" for l in labels]
        ax.barh(labels, rates, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Win rate (%)")
        ax.set_title(f"Porownanie modeli ({trained_live[0].episodes} meczy kazdy)")
        ax.grid(True, axis="x", alpha=0.3)
        path = output_dir / "model_win_rates.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


def save_report_json(
    series: list[ScalarSeries],
    eval_data: dict[str, Any] | None,
    live_results: list[ModelEvalResult],
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tensorboard_tags": sorted({s.tag for s in series}),
        "eval_npz": eval_data,
        "live_evaluation": [asdict(r) for r in live_results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_statistics(
    *,
    model_path: str | None = None,
    episodes: int = 50,
    seed: int = 0,
    plot: bool = False,
    compare_checkpoints: bool = False,
    skip_baseline: bool = False,
    save_json: bool = True,
) -> None:
    print("Clash Royale RL — raport statystyk")
    print(f"Katalog modeli: {DEFAULT_MODEL_DIR.resolve()}")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tb_series = load_tensorboard_scalars(DEFAULT_MODEL_DIR / "tb")
    eval_data = load_eval_npz(DEFAULT_MODEL_DIR / "eval")

    print_training_curves(tb_series)
    print_eval_npz_summary(eval_data)

    models = discover_models(model_path)
    if not compare_checkpoints:
        models = [m for m in models if "Checkpoint:" not in m[0]]

    live_results: list[ModelEvalResult] = []
    if not skip_baseline:
        print("\nEwaluacja losowego baseline...")
        live_results.append(evaluate_random_baseline(episodes=min(episodes, 30), seed=seed))

    for label, path in models:
        print(f"Ewaluacja: {label}...")
        live_results.append(
            evaluate_saved_model(label, path, episodes=episodes, seed=seed)
        )

    print_model_table(live_results)
    print_recommendation(live_results)

    if save_json:
        json_path = STATS_DIR / "report.json"
        save_report_json(tb_series, eval_data, live_results, json_path)
        print(f"\nZapisano JSON: {json_path}")

    if plot:
        figures = save_plots(tb_series, eval_data, live_results, STATS_DIR / "figures")
        if figures:
            _section("Wykresy")
            for fig_path in figures:
                print(f"  {fig_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Statystyki treningu PPO")
    parser.add_argument("--model", type=str, default=None, help="Dodatkowy model .zip do oceny")
    parser.add_argument("--episodes", type=int, default=50, help="Mecze na model (live eval)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="Zapisz wykresy do models/stats/figures/")
    parser.add_argument(
        "--compare-checkpoints",
        action="store_true",
        help="Oceń tez wszystkie checkpointy (wolniejsze)",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    run_statistics(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        plot=args.plot,
        compare_checkpoints=args.compare_checkpoints,
        skip_baseline=args.skip_baseline,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
