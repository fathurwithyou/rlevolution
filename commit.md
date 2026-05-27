```
Commit Plan (bottom-up, topological order)
Commit 1 — Project skeleton + config
pyproject.toml, .python-version, .gitignore, LICENSE, README.md
rlevolution/__init__.py
rlevolution/config/__init__.py
rlevolution/config/settings.py
Base: no internal dependencies. SimulationConfig + constants used by everything.
Commit 2 — Pure data primitives
rlevolution/domain/traits.py
rlevolution/domain/snapshot.py
rlevolution/domain/fitness.py
rlevolution/simulation/results.py
Stdlib-only dataclasses: Traits, AgentSnapshot, FitnessCalculator, EngineStepResult, GenerationMetrics.
Commit 3 — RL module
rlevolution/rl/__init__.py
rlevolution/rl/q_learning.py
Depends on: Commit 1 (ACTION_NAMES). Introduces QLearningModel + State type.
Commit 4 — Domain agent
rlevolution/domain/agent.py
Depends on: Commits 2 + 3. AdaptiveAgent ties together Traits, QLearningModel, FitnessCalculator, AgentSnapshot.
Commit 5 — Domain factory
rlevolution/domain/factory.py
rlevolution/domain/__init__.py
Depends on: Commits 1 + 4 + 3. AgentFactory.create_random().
Commit 6 — Evolution
rlevolution/evolution/__init__.py
rlevolution/evolution/selection.py
rlevolution/evolution/genetics.py
Depends on: Commits 1 + 4 + 3. FitnessProportionateSelector + GeneticOperator (crossover/mutation).
Commit 7 — Environment world + types
rlevolution/environment/types.py
rlevolution/environment/world.py
Depends on: Commits 2 + 3 + 4 + 1. NaturalSelectionWorld — the core simulation loop.
Commit 8 — Environment gym env
rlevolution/environment/gym_env.py
rlevolution/environment/__init__.py
Depends on: Commit 7 + 5 + 1. NaturalSelectionGymEnv — Gymnasium wrapper.
Commit 9 — Simulation metrics + persistence
rlevolution/simulation/metrics.py
rlevolution/persistence/__init__.py
rlevolution/persistence/csv_history.py
Depends on: Commits 2 + 4 + 1. GenerationMetricsBuilder + CsvHistoryWriter.
Commit 10 — Simulation engine (orchestrator)
rlevolution/simulation/engine.py
rlevolution/simulation/__init__.py
Depends on: all prior commits. EvolutionEngine wires everything together.
Commit 11 — CLI
rlevolution/cli.py
Depends on: Commit 10 (lazy) + 1. Argparse-based headless runner.
Commit 12 — Visualization
rlevolution/visualization/__init__.py
rlevolution/visualization/plots.py
Depends on: Commits 2 + 4 + 8 + 10. Plotly figures.
Commit 13 — UI Dashboard
rlevolution/ui/__init__.py
rlevolution/ui/dashboard.py
Depends on: Commits 1 + 2 + 10 + 12. Streamlit app.
Commit 14 — Entry points
main.py
app.py
Trivial scripts depending on CLI and Dashboard respectively.
Commit 15 — Docs
docs/
.streamlit/config.toml
Documentation and Streamlit theme config.
```