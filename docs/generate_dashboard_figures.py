from __future__ import annotations

import csv
from pathlib import Path

from rlevolution.config import SimulationConfig
from rlevolution.simulation import EvolutionEngine, GenerationMetrics
from rlevolution.visualization.plots import Visualizer


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "docs" / "experiments" / "rl_evolution"
FIGURE_DIR = ROOT / "docs" / "figures"


def _without_plot_title(figure):
    figure.update_layout(title=None)
    return figure


def _metrics_from_csv(path: Path) -> list[GenerationMetrics]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metrics: list[GenerationMetrics] = []
    for row in rows:
        metrics.append(
            GenerationMetrics(
                generation=int(row["generation"]),
                steps=int(row["steps"]),
                population_end=int(row["population_end"]),
                population_total=int(row["population_total"]),
                survival_rate=float(row["survival_rate"]),
                avg_fitness=float(row["avg_fitness"]),
                best_fitness=float(row["best_fitness"]),
                median_fitness=float(row["median_fitness"]),
                avg_energy_alive=float(row["avg_energy_alive"]),
                food_eaten=int(row["food_eaten"]),
                reproductions=int(row["reproductions"]),
                hazard_hits=int(row["hazard_hits"]),
                avg_speed=float(row["avg_speed"]),
                avg_endurance=float(row["avg_endurance"]),
                avg_foraging=float(row["avg_foraging"]),
                avg_reproduction=float(row["avg_reproduction"]),
                avg_q_states=float(row["avg_q_states"]),
            )
        )
    return metrics


def _base_config(seed: int = 3211) -> SimulationConfig:
    return SimulationConfig(
        initial_agents=32,
        food_count=75,
        max_generations=12,
        episode_steps=260,
        random_seed=seed,
    )


def _write_dashboard_history_figures() -> None:
    history = _metrics_from_csv(EXPERIMENT_DIR / "baseline_seed3211.csv")
    outputs = {
        "dashboard_fitness_history.png": Visualizer.fitness_history(history),
        "dashboard_population_history.png": Visualizer.population_history(history),
        "dashboard_trait_history.png": Visualizer.trait_history(history),
    }
    for name, figure in outputs.items():
        _without_plot_title(figure)
        figure.write_image(FIGURE_DIR / name, width=920, height=520, scale=2)


def _write_snapshot_figures() -> None:
    class NullHistoryWriter:
        def write(self, history: list[GenerationMetrics]) -> None:
            return None

    engine = EvolutionEngine(_base_config(), history_writer=NullHistoryWriter())

    while engine.current_generation < 5:
        engine.run_generation()
    for _ in range(130):
        engine.step()

    _without_plot_title(Visualizer.environment_figure(
        engine.environment,
        engine.agents,
        engine.current_generation,
        engine.step_in_generation,
    )).write_image(FIGURE_DIR / "dashboard_environment_snapshot.png", width=940, height=720, scale=2)

    _without_plot_title(Visualizer.trait_distribution(engine.agents)).write_image(
        FIGURE_DIR / "dashboard_trait_distribution.png",
        width=920,
        height=520,
        scale=2,
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_dashboard_history_figures()
    _write_snapshot_figures()


if __name__ == "__main__":
    main()
