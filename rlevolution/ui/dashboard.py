from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import time

import pandas as pd
import streamlit as st

from rlevolution.config import HISTORY_CSV, SimulationConfig
from rlevolution.domain import AgentSnapshot
from rlevolution.simulation import EvolutionEngine
from rlevolution.simulation.results import GenerationMetrics
from rlevolution.visualization import Visualizer


class DashboardApp:
    def render(self) -> None:
        self._configure_page()
        config = self._build_config()
        engine = self._ensure_engine(config)
        signature_changed = self._config_signature_changed(config)
        steps_per_frame = self._render_controls(config, signature_changed)

        if st.session_state.get("running"):
            self._advance_running_engine(engine, config, steps_per_frame)

        self._render_header(engine, config)
        self._render_tabs(engine)

        if st.session_state.get("running"):
            time.sleep(0.05)
            st.rerun()

    def _configure_page(self) -> None:
        st.set_page_config(
            page_title="RL Evolution Lab",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(
            """
            <style>
            :root {
                --rl-bg: #f6f8fb;
                --rl-panel: #ffffff;
                --rl-panel-soft: #f8fafc;
                --rl-text: #111827;
                --rl-muted: #4b5563;
                --rl-border: #d9e2ec;
                --rl-accent: #1971c2;
            }
            .stApp {
                background: var(--rl-bg);
                color: var(--rl-text);
            }
            section[data-testid="stSidebar"] {
                background: var(--rl-panel);
                border-right: 1px solid var(--rl-border);
            }
            .stApp h1,
            .stApp h2,
            .stApp h3,
            .stApp h4,
            .stApp h5,
            .stApp h6,
            .stApp p,
            .stApp label,
            .stApp span,
            .stApp div[data-testid="stMarkdownContainer"],
            .stApp div[data-testid="stCaptionContainer"],
            section[data-testid="stSidebar"] * {
                color: var(--rl-text);
            }
            .stApp small,
            .stApp [data-testid="stCaptionContainer"],
            .stApp [data-testid="stMetricDelta"],
            .stApp [data-testid="stTickBarMin"],
            .stApp [data-testid="stTickBarMax"] {
                color: var(--rl-muted);
            }
            div[data-testid="stMetric"] {
                background: var(--rl-panel);
                border: 1px solid var(--rl-border);
                border-radius: 8px;
                padding: 14px 16px;
                min-height: 104px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            }
            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--rl-text);
            }
            div[data-testid="stAlert"] {
                background: #e8f3ff;
                border: 1px solid #b6dbff;
            }
            .stButton > button,
            .stDownloadButton > button {
                background: var(--rl-panel);
                border: 1px solid var(--rl-border);
                color: var(--rl-text);
                border-radius: 8px;
                font-weight: 650;
            }
            .stButton > button:hover,
            .stDownloadButton > button:hover {
                border-color: var(--rl-accent);
                color: var(--rl-accent);
                background: #eef6ff;
            }
            [data-baseweb="tab-list"] {
                gap: 8px;
            }
            [data-baseweb="tab"] {
                color: var(--rl-muted);
                background: transparent;
                border-radius: 8px 8px 0 0;
            }
            [data-baseweb="tab"][aria-selected="true"] {
                color: var(--rl-text);
                background: var(--rl-panel);
                border-bottom: 3px solid var(--rl-accent);
            }
            [data-baseweb="input"] input,
            [data-baseweb="select"] *,
            [data-baseweb="base-input"] input,
            textarea {
                color: var(--rl-text);
                background: var(--rl-panel);
            }
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                color: var(--rl-text);
                background: var(--rl-panel);
            }
            .block-container {
                padding-top: 1.5rem;
                max-width: 1500px;
            }
            h1, h2, h3 {
                letter-spacing: 0;
            }
            svg text {
                fill: var(--rl-text);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _build_config(self) -> SimulationConfig:
        st.sidebar.header("Parameter eksperimen")
        initial_agents = st.sidebar.slider("Jumlah agen awal", 8, 120, 36, 2)
        food_count = st.sidebar.slider("Jumlah makanan", 10, 180, 85, 5)
        hazard_count = st.sidebar.slider("Jumlah zona bahaya", 0, 14, 5, 1)
        hazard_radius = st.sidebar.slider("Radius bahaya", 2.0, 16.0, 7.5, 0.5)
        mutation_rate = st.sidebar.slider("Tingkat mutasi", 0.0, 0.45, 0.12, 0.01)
        learning_rate = st.sidebar.slider("Learning rate", 0.01, 0.8, 0.22, 0.01)
        discount_factor = st.sidebar.slider("Discount factor", 0.1, 0.99, 0.92, 0.01)
        exploration_rate = st.sidebar.slider("Exploration rate", 0.0, 0.8, 0.18, 0.01)
        max_generations = st.sidebar.slider("Jumlah generasi", 2, 80, 24, 1)
        episode_steps = st.sidebar.slider("Durasi episode", 80, 1200, 420, 20)
        seed = st.sidebar.number_input("Random seed", 1, 999999, 3211, 1)

        return SimulationConfig(
            initial_agents=initial_agents,
            max_population=max(30, initial_agents * 3),
            food_count=food_count,
            hazard_count=hazard_count,
            hazard_radius=hazard_radius,
            mutation_rate=mutation_rate,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            max_generations=max_generations,
            episode_steps=episode_steps,
            random_seed=int(seed),
        )

    def _ensure_engine(self, config: SimulationConfig) -> EvolutionEngine:
        signature = self._config_signature(config)
        if "engine" not in st.session_state:
            st.session_state.engine = EvolutionEngine(config)
            st.session_state.config_signature = signature
            st.session_state.running = False
        return st.session_state.engine

    def _reset_engine(self, config: SimulationConfig) -> EvolutionEngine:
        st.session_state.engine = EvolutionEngine(config)
        st.session_state.config_signature = self._config_signature(config)
        st.session_state.running = False
        return st.session_state.engine

    def _render_controls(
        self,
        config: SimulationConfig,
        signature_changed: bool,
    ) -> int:
        st.sidebar.divider()
        steps_per_frame = st.sidebar.slider("Langkah per frame", 1, 40, 8, 1)
        control_cols = st.sidebar.columns(2)

        if control_cols[0].button("Start", width="stretch"):
            if signature_changed:
                self._reset_engine(config)
            st.session_state.running = True

        if control_cols[1].button("Pause", width="stretch"):
            st.session_state.running = False

        control_cols = st.sidebar.columns(2)
        if control_cols[0].button("Reset", width="stretch"):
            self._reset_engine(config)

        if control_cols[1].button("1 generasi", width="stretch"):
            engine = self._reset_engine(config) if signature_changed else st.session_state.engine
            engine.run_generation()

        if st.sidebar.button("Run experiment", width="stretch"):
            engine = self._reset_engine(config) if signature_changed else st.session_state.engine
            self._run_full_experiment(engine, config)

        if signature_changed:
            st.sidebar.info(
                "Parameter berubah. Tekan Start, Reset, 1 generasi, atau Run experiment "
                "untuk memakai konfigurasi baru."
            )
        return steps_per_frame

    def _run_full_experiment(
        self,
        engine: EvolutionEngine,
        config: SimulationConfig,
    ) -> None:
        progress = st.sidebar.progress(0.0)
        while len(engine.history) < config.max_generations:
            engine.run_generation()
            progress.progress(len(engine.history) / config.max_generations)
        st.session_state.running = False

    def _advance_running_engine(
        self,
        engine: EvolutionEngine,
        config: SimulationConfig,
        steps_per_frame: int,
    ) -> None:
        for _ in range(steps_per_frame):
            if len(engine.history) >= config.max_generations:
                st.session_state.running = False
                break
            engine.step()

    def _render_header(self, engine: EvolutionEngine, config: SimulationConfig) -> None:
        st.title("RL Evolution Lab")
        st.caption(
            "Simulasi adaptasi agen dengan Q-Learning, seleksi alam, "
            "pewarisan trait, dan mutasi."
        )
        metric_cols = st.columns(6)
        metric_cols[0].metric("Generasi", engine.current_generation)
        metric_cols[1].metric("Langkah", f"{engine.step_in_generation}/{config.episode_steps}")
        metric_cols[2].metric("Populasi hidup", len(engine.alive_agents()))
        metric_cols[3].metric("Epsilon", f"{engine.exploration_rate():.3f}")
        metric_cols[4].metric(
            "Fitness terbaik",
            f"{max((agent.fitness_score() for agent in engine.agents), default=0):.1f}",
        )
        metric_cols[5].metric("Reproduksi", engine.births_this_generation)

    def _render_tabs(self, engine: EvolutionEngine) -> None:
        sim_tab, graph_tab, data_tab, concept_tab = st.tabs(
            ["Simulasi", "Grafik hasil", "Data", "Konsep"]
        )
        with sim_tab:
            self._render_simulation_tab(engine)
        with graph_tab:
            self._render_graph_tab(engine)
        with data_tab:
            self._render_data_tab(engine)
        with concept_tab:
            self._render_concept_tab()

    def _render_simulation_tab(self, engine: EvolutionEngine) -> None:
        agent_df = pd.DataFrame(self._agent_records(engine.agent_snapshots()))
        left, right = st.columns([1.8, 1.0])
        with left:
            st.plotly_chart(
                Visualizer.environment_figure(
                    engine.environment,
                    engine.agents,
                    engine.current_generation,
                    engine.step_in_generation,
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                Visualizer.trait_distribution(engine.agents),
                width="stretch",
            )
            if not agent_df.empty:
                st.dataframe(
                    agent_df[
                        [
                            "agent_id",
                            "alive",
                            "fitness",
                            "energy",
                            "age",
                            "speed",
                            "endurance",
                            "foraging",
                            "reproduction",
                        ]
                    ]
                    .sort_values("fitness", ascending=False)
                    .head(12),
                    width="stretch",
                    hide_index=True,
                )

    def _render_graph_tab(self, engine: EvolutionEngine) -> None:
        history_df = pd.DataFrame(self._history_records(engine.history))
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(Visualizer.fitness_history(engine.history), width="stretch")
            st.plotly_chart(Visualizer.trait_history(engine.history), width="stretch")
        with col_b:
            st.plotly_chart(Visualizer.population_history(engine.history), width="stretch")
            if not history_df.empty:
                st.dataframe(history_df.tail(10), width="stretch", hide_index=True)

    def _render_data_tab(self, engine: EvolutionEngine) -> None:
        history_df = pd.DataFrame(self._history_records(engine.history))
        if history_df.empty:
            st.info("Jalankan minimal satu generasi untuk membentuk data CSV.")
            return

        st.dataframe(history_df, width="stretch", hide_index=True)
        st.download_button(
            "Unduh experiment_history.csv",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="experiment_history.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(f"File juga ditulis otomatis ke `{HISTORY_CSV}`.")

    def _render_concept_tab(self) -> None:
        st.markdown(
            """
            **Pertanyaan penelitian:** Bagaimana Reinforcement Learning dapat digunakan
            untuk mensimulasikan proses adaptasi agen dalam lingkungan dengan tekanan
            seleksi alam, dan bagaimana perubahan trait populasi terjadi dari generasi
            ke generasi?

            **Gymnasium environment:** simulasi dibungkus sebagai
            `NaturalSelectionGymEnv` dengan API `reset()`, `step()`,
            `observation_space`, dan `action_space`.

            **State Q-Learning:** `MultiDiscrete([5, 5, 3, 3, 5, 3, 3, 5, 5, 5])`
            yang berisi posisi diskrit agen, arah dan jarak makanan terdekat, arah
            dan jarak bahaya terdekat, energi, dan umur.

            **Action:** `Discrete(6)`, yaitu bergerak ke atas, bawah, kiri, kanan,
            diam, atau mencari pasangan reproduksi.

            **Reward:** makanan, survival, dan reproduksi memberi reward positif;
            bahaya, pencarian reproduksi yang gagal, dan kematian memberi reward
            negatif.

            **Evolusi:** setelah episode selesai, fitness menentukan peluang menjadi
            parent. Trait dan Q-table parent diwariskan ke generasi berikutnya
            melalui crossover sederhana dan mutasi kecil.
            """
        )

    def _config_signature_changed(self, config: SimulationConfig) -> bool:
        return st.session_state.get("config_signature") != self._config_signature(config)

    @staticmethod
    def _config_signature(config: SimulationConfig) -> tuple[tuple[str, int | float], ...]:
        return tuple(sorted(config.as_dict().items()))

    @staticmethod
    def _agent_records(
        snapshots: Sequence[AgentSnapshot],
    ) -> list[dict[str, int | float | bool]]:
        return [asdict(snapshot) for snapshot in snapshots]

    @staticmethod
    def _history_records(
        history: Sequence[GenerationMetrics],
    ) -> list[dict[str, int | float]]:
        return [asdict(metric) for metric in history]


def main() -> None:
    DashboardApp().render()
