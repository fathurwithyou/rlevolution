# RL Evolution Lab

RL Evolution Lab is an interactive reinforcement learning simulation for the
IF3211 Domain-Specific Computation course. The system models a population of
adaptive agents living in a two-dimensional environment, where they must search
for food, avoid hazardous zones, manage energy, reproduce, and pass traits to
future generations.

Research question:

> How can reinforcement learning be used to simulate agent adaptation under
> natural selection pressure, and how do population traits change across
> generations?

## Execution

Install dependencies:

```bash
uv sync
```

Run the dashboard:

```bash
python main.py
```

Alternative:

```bash
streamlit run app.py
```

Run a headless experiment and write CSV output:

```bash
python main.py --headless --generations 12 --steps 260 --agents 32
```

Alternative when using the package script:

```bash
uv run rlevolution --headless --generations 12 --steps 260 --agents 32
```

The experiment output is written to:

```text
data/experiment_history.csv
```

## Project Structure

```text
rlevolution/
  config/          Simulation parameters, action names, and output paths
  domain/          Domain entities: agent, traits, factory, fitness
  rl/              Tabular Q-learning and state typing
  environment/     2D world and Gymnasium adapter
  evolution/       Parent selection and genetic operators
  simulation/      Application service and experiment orchestration
  persistence/     CSV persistence for experiment history
  visualization/   Plotly charts for the dashboard
  ui/              Class-based Streamlit dashboard
app.py             Wrapper for `streamlit run app.py`
main.py            Wrapper for `python main.py`
data/              CSV output directory
```

This structure separates responsibilities to align with SOLID principles:

- domain entities do not manage the dashboard or file output;
- the Gymnasium adapter only translates the world into the RL API;
- `EvolutionEngine` acts as the application service;
- parent selection and genetic operators can evolve independently of the UI;
- CSV persistence is isolated from the simulation logic.

## Biological Model

Each agent has a small set of biological traits:

- `speed`: movement distance per step, at the cost of higher metabolism;
- `endurance`: reduces energy cost and hazard impact;
- `foraging`: increases food detection range and food gain;
- `reproduction`: increases the probability of successful reproduction.

Natural selection is modeled through fitness. Agents with higher fitness are
more likely to become parents in the next generation. Fitness depends on age,
food intake, reproduction success, survival duration, hazard penalties, and
death.

Genetic variation is introduced through small mutations in traits and Q-table
parameters. As a result, the population can shift from one generation to the
next under environmental pressure.

## Reinforcement Learning Model

The RL component uses tabular Q-learning on top of a custom Gymnasium
environment named `NaturalSelectionGymEnv`. The environment exposes the
standard `reset()`, `step()`, `observation_space`, and `action_space` interface
while still being compatible with population-level simulation.

Gymnasium spaces:

- `observation_space = MultiDiscrete([5, 5, 3, 3, 5, 3, 3, 5, 5, 5])`
- `action_space = Discrete(6)`

The state includes:

- the agent position on a discrete grid;
- the direction and distance to the nearest food source;
- the direction and distance to the nearest hazard;
- the current energy level;
- the age of the agent.

The action set includes:

- move up;
- move down;
- move left;
- move right;
- stay in place;
- search for a reproduction partner.

Reward structure:

- positive reward for survival, food acquisition, and successful reproduction;
- negative reward for entering a hazardous zone, failed reproduction, and death.

Q-learning update rule:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
```

The parameters `alpha`, `gamma`, and `epsilon` can be adjusted directly from
the dashboard.

## Experiment Mode

Available parameters:

- initial agent count;
- food count;
- number of hazard zones and hazard radius;
- mutation rate;
- learning rate;
- discount factor;
- exploration rate;
- number of generations;
- episode duration;
- random seed.

Dashboard controls:

- `Start`: run the simulation in real time;
- `Pause`: stop temporarily;
- `Reset`: create a new experiment state;
- `1 generasi`: run one full episode;
- `Run experiment`: execute all selected generations.

## Visualization

The dashboard presents:

- a real-time two-dimensional environment;
- agents colored by energy level;
- food items as green diamonds;
- hazards as translucent red zones;
- mean and best fitness charts;
- population, survival, and reproduction charts;
- trait distribution for the active population;
- a table of the strongest agents;
- a table of generation-level experiment results.

The visualization is designed for presentation and video demonstration. Agent
movement is visible in real time, while the charts provide quantitative evidence
for the results and discussion section of the report.

## Result Interpretation

If average fitness increases across generations, the combination of individual
learning and parent selection is improving the population strategy. If the
`foraging` trait increases, the environment is favoring agents that locate food
more efficiently. If `endurance` increases under heavier hazard pressure, the
population is adapting to risk. If the survival rate drops when hazards are
increased, the selection pressure becomes stronger and only agents with
suitable policies and traits remain viable.

The `experiment_history.csv` file can be used to compare scenarios such as low
mutation versus high mutation, low hazard versus high hazard, or low
exploration versus high exploration.
