```
Commit 1 — Project skeleton + config
feat: initial project skeleton with config module
Set up pyproject.toml with hatchling build system, .gitignore, LICENSE (MIT),
and .python-version. Add rlevolution package with version marker and
SimulationConfig dataclass defining all simulation parameters, action
constants, and directory paths used across the entire project.
Commit 2 — Pure data primitives
feat: add core domain primitives and simulation result types
Implement Traits (agent attributes with as_dict), AgentSnapshot (frozen
data transfer object for agent state), FitnessCalculator (computes agent
fitness from traits), and GenerationMetrics/EngineStepResult (structured
simulation output records). These are stdlib-only dataclasses with no
internal dependencies.
Commit 3 — RL module
feat: implement Q-learning model with discrete action space
Add QLearningModel dataclass with epsilon-greedy action selection, Q-value
updates, copy/mutate/crossover operations for neuroevolution, and State
type alias. Depends on ACTION_NAMES from config for action dimension.
Commit 4 — Domain agent
feat: add AdaptiveAgent tying traits, RL model, and fitness together
AdaptiveAgent dataclass composes Traits, QLearningModel, position, energy,
and age into a single evolvable unit. Exposes fitness_score(calculator) and
snapshot(calculator) methods. This is the core individual in the simulation.
Commit 5 — Domain factory
feat: add AgentFactory for random agent generation
AgentFactory.create_random() produces agents with randomized traits, random
Q-learning weights, and random initial positions. Uses SimulationConfig for
parameter bounds.
Commit 6 — Evolution module
feat: add selection and genetic operators for evolution
Implement FitnessProportionateSelector (weighted random parent selection
with configurable selection pressure) and GeneticOperator (crossover of
Q-tables and traits, mutation with configurable rate and strength).
Commit 7 — Environment world and types
feat: add NaturalSelectionWorld simulation core and type definitions
WorldStepInfo, WorldStepResult, GymStepResult, GymInfo, RenderFrame define
the data contracts. NaturalSelectionWorld implements the grid-based
simulation loop: movement, food consumption, hazard damage, reproduction
cost, and state encoding for RL observation.
Commit 8 — Environment gym env
feat: add Gymnasium environment adapter
NaturalSelectionGymEnv wraps NaturalSelectionWorld with the standard
gymnasium.Env interface (reset/step/render), enabling RL training loops.
Handles agent indexing, alive-agent tracking, and observation/action spaces.
Commit 9 — Simulation metrics and persistence
feat: add metrics builder and CSV history writer
GenerationMetricsBuilder aggregates per-generation statistics (fitness,
population, trait distribution). CsvHistoryWriter flattens metrics to
CSV via dataclass asdict, writing to the configured HISTORY_CSV path.
Commit 10 — Simulation engine (orchestrator)
feat: add EvolutionEngine orchestrating the full simulation loop
EvolutionEngine composes AgentFactory, NaturalSelectionGymEnv,
FitnessProportionateSelector, GeneticOperator, GenerationMetricsBuilder,
and CsvHistoryWriter into a single pipeline. Manages agent lifecycle:
creation, RL training per step, reproduction via selection+crossover,
generation boundaries, and experiment orchestration over many generations.
Commit 11 — CLI entry point
feat: add command-line interface for headless and dashboard modes
argparse-based CLI supporting 'headless' (run N generations from terminal)
and 'dashboard' (launch Streamlit UI) subcommands. Registered as console
script 'rlevolution' via pyproject.toml.
Commit 12 — Visualization module
feat: add Plotly-based visualization figures
Visualizer class with static methods for rendering the environment grid
(scatter plot), fitness history, population history, trait distributions,
and trait evolution over time. Uses Plotly graph_objects with consistent
theming.
Commit 13 — Streamlit UI dashboard
feat: add interactive Streamlit dashboard
DashboardApp provides a full Streamlit UI with config sidebar, simulation
controls (step/run/reset), real-time environment view, fitness/population
graphs, trait analysis, and CSV data export. Uses session state for
persistence across reruns.
Commit 14 — Entry points
feat: add top-level entry point scripts
main.py calls rlevolution.cli.main() for 'python main.py' usage.
app.py launches the Streamlit dashboard via 'streamlit run app.py'.
Commit 15 — Documentation and configuration
docs: add LaTeX paper, Streamlit theme, and dashboard figure generation
Add .streamlit/config.toml for dashboard theming, ICML-style LaTeX paper
(docs/), bibliography, and a script to generate dashboard figures for the
paper from simulation output.
```
