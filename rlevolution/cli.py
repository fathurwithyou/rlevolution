from __future__ import annotations

import argparse
import sys

from rlevolution.config import HISTORY_CSV, SimulationConfig


def run_headless(args: argparse.Namespace) -> None:
    from rlevolution.simulation import EvolutionEngine

    config = SimulationConfig(
        initial_agents=args.agents,
        food_count=args.food,
        hazard_count=args.hazards,
        mutation_rate=args.mutation_rate,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        max_generations=args.generations,
        episode_steps=args.steps,
        random_seed=args.seed,
    )
    engine = EvolutionEngine(config)
    engine.run_experiment(args.generations)
    last = engine.history[-1]
    print(
        "Experiment complete: "
        f"{args.generations} generations, "
        f"best_fitness={last.best_fitness:.2f}, "
        f"avg_fitness={last.avg_fitness:.2f}, "
        f"csv={HISTORY_CSV}"
    )


def run_dashboard() -> None:
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit belum terpasang. Jalankan `uv sync` atau "
            "`pip install -e .`, lalu gunakan `python main.py` lagi."
        ) from exc

    sys.argv = ["streamlit", "run", "app.py"]
    raise SystemExit(stcli.main())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL Evolution Lab: simulasi evolusi dan seleksi alam berbasis Q-Learning."
    )
    parser.add_argument("--headless", action="store_true", help="Jalankan eksperimen tanpa dashboard.")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--food", type=int, default=75)
    parser.add_argument("--hazards", type=int, default=5)
    parser.add_argument("--mutation-rate", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=0.22)
    parser.add_argument("--discount-factor", type=float, default=0.92)
    parser.add_argument("--exploration-rate", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=3211)
    args = parser.parse_args()

    if args.headless:
        run_headless(args)
    else:
        run_dashboard()
