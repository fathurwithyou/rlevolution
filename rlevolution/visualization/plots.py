from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.environment import NaturalSelectionGymEnv
from rlevolution.simulation import GenerationMetrics


class Visualizer:
    TEXT_COLOR = "#111827"
    MUTED_COLOR = "#4b5563"
    GRID_COLOR = "#d9e2ec"
    PAPER_COLOR = "#ffffff"
    PLOT_COLOR = "#f8fafc"

    @staticmethod
    def environment_figure(
        environment: NaturalSelectionGymEnv,
        agents: list[AdaptiveAgent],
        generation: int,
        step: int,
    ) -> go.Figure:
        config = environment.config
        alive_agents = [agent for agent in agents if agent.alive]
        fig = go.Figure()

        for center, radius in environment.hazards:
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=float(center[0] - radius),
                y0=float(center[1] - radius),
                x1=float(center[0] + radius),
                y1=float(center[1] + radius),
                fillcolor="rgba(214, 48, 49, 0.24)",
                line=dict(color="rgba(214, 48, 49, 0.75)", width=2),
                layer="below",
            )

        if len(environment.food):
            fig.add_trace(
                go.Scatter(
                    x=environment.food[:, 0],
                    y=environment.food[:, 1],
                    mode="markers",
                    name="Makanan",
                    marker=dict(
                        symbol="diamond",
                        size=9,
                        color="#2fb344",
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate="Makanan<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
                )
            )

        if alive_agents:
            fig.add_trace(
                go.Scatter(
                    x=[agent.position[0] for agent in alive_agents],
                    y=[agent.position[1] for agent in alive_agents],
                    mode="markers",
                    name="Agen hidup",
                    marker=dict(
                        size=[9 + 4 * agent.traits.foraging for agent in alive_agents],
                        color=[agent.energy for agent in alive_agents],
                        colorscale=[
                            [0.0, "#c92a2a"],
                            [0.45, "#f08c00"],
                            [0.72, "#1971c2"],
                            [1.0, "#0ca678"],
                        ],
                        cmin=0,
                        cmax=config.max_energy,
                        showscale=True,
                        colorbar=dict(title="Energi", thickness=12),
                        line=dict(color="white", width=1.2),
                    ),
                    customdata=[
                        [
                            agent.agent_id,
                            agent.energy,
                            agent.fitness_score(),
                            agent.age,
                            agent.traits.speed,
                            agent.traits.endurance,
                            agent.traits.foraging,
                            agent.traits.reproduction,
                        ]
                        for agent in alive_agents
                    ],
                    hovertemplate=(
                        "Agen %{customdata[0]}<br>"
                        "Energi=%{customdata[1]:.1f}<br>"
                        "Fitness=%{customdata[2]:.1f}<br>"
                        "Umur=%{customdata[3]}<br>"
                        "Speed=%{customdata[4]:.2f}<br>"
                        "Endurance=%{customdata[5]:.2f}<br>"
                        "Foraging=%{customdata[6]:.2f}<br>"
                        "Reproduksi=%{customdata[7]:.2f}<extra></extra>"
                    ),
                )
            )
        else:
            fig.add_annotation(
                text="Populasi punah",
                x=config.world_width / 2,
                y=config.world_height / 2,
                showarrow=False,
                font=dict(size=22, color="#c92a2a"),
            )

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Generasi {generation}</b>"
                    f"<span style='font-size:14px;color:{Visualizer.MUTED_COLOR};'>"
                    f" &nbsp; Langkah {step}</span>"
                ),
                x=0.0,
                xanchor="left",
                y=0.985,
                yanchor="top",
            ),
            template="plotly_white",
            height=620,
            margin=dict(l=10, r=10, t=70, b=58),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor=Visualizer.GRID_COLOR,
                borderwidth=1,
            ),
            paper_bgcolor=Visualizer.PAPER_COLOR,
            plot_bgcolor=Visualizer.PLOT_COLOR,
        )
        fig.update_xaxes(
            range=[0, config.world_width],
            showgrid=True,
            gridcolor=Visualizer.GRID_COLOR,
            zeroline=False,
            title=None,
        )
        fig.update_yaxes(
            range=[0, config.world_height],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridcolor=Visualizer.GRID_COLOR,
            zeroline=False,
            title=None,
        )
        return Visualizer._apply_text_theme(fig)

    @staticmethod
    def fitness_history(history: Sequence[GenerationMetrics]) -> go.Figure:
        fig = go.Figure()
        if history:
            df = pd.DataFrame(Visualizer._history_records(history))
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["avg_fitness"],
                    mode="lines+markers",
                    name="Fitness rata-rata",
                    line=dict(color="#1971c2", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["best_fitness"],
                    mode="lines+markers",
                    name="Fitness terbaik",
                    line=dict(color="#0ca678", width=3),
                )
            )
        fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=10, r=10, t=36, b=10),
            title="Perkembangan fitness",
            yaxis_title="Fitness",
            xaxis_title="Generasi",
        )
        return Visualizer._apply_text_theme(fig)

    @staticmethod
    def population_history(history: Sequence[GenerationMetrics]) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if history:
            df = pd.DataFrame(Visualizer._history_records(history))
            fig.add_trace(
                go.Bar(
                    x=df["generation"],
                    y=df["population_end"],
                    name="Populasi akhir",
                    marker_color="#1971c2",
                    opacity=0.72,
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["survival_rate"],
                    mode="lines+markers",
                    name="Survival rate",
                    line=dict(color="#f08c00", width=3),
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["generation"],
                    y=df["reproductions"],
                    mode="lines+markers",
                    name="Reproduksi",
                    line=dict(color="#862e9c", width=3),
                ),
                secondary_y=False,
            )
        fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=10, r=10, t=36, b=10),
            title="Populasi, survival, dan reproduksi",
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_yaxes(title_text="Jumlah", secondary_y=False)
        fig.update_yaxes(title_text="Survival rate", range=[0, 1], secondary_y=True)
        return Visualizer._apply_text_theme(fig)

    @staticmethod
    def trait_distribution(agents: list[AdaptiveAgent]) -> go.Figure:
        alive_agents = [agent for agent in agents if agent.alive]
        source = alive_agents or agents
        fig = go.Figure()
        if source:
            traits = {
                "Speed": [agent.traits.speed for agent in source],
                "Endurance": [agent.traits.endurance for agent in source],
                "Foraging": [agent.traits.foraging for agent in source],
                "Reproduksi": [agent.traits.reproduction for agent in source],
            }
            colors = ["#1971c2", "#0ca678", "#f08c00", "#862e9c"]
            for (name, values), color in zip(traits.items(), colors, strict=True):
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=name,
                        marker_color=color,
                        boxmean=True,
                    )
                )
        fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=10, r=10, t=36, b=10),
            title="Distribusi trait populasi aktif",
            yaxis_title="Nilai trait",
        )
        return Visualizer._apply_text_theme(fig)

    @staticmethod
    def trait_history(history: Sequence[GenerationMetrics]) -> go.Figure:
        fig = go.Figure()
        if history:
            df = pd.DataFrame(Visualizer._history_records(history))
            traces = [
                ("avg_speed", "Speed", "#1971c2"),
                ("avg_endurance", "Endurance", "#0ca678"),
                ("avg_foraging", "Foraging", "#f08c00"),
                ("avg_reproduction", "Reproduksi", "#862e9c"),
            ]
            for column, label, color in traces:
                fig.add_trace(
                    go.Scatter(
                        x=df["generation"],
                        y=df[column],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=3),
                    )
                )
        fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=10, r=10, t=36, b=10),
            title="Perubahan trait antargenerasi",
            yaxis_title="Rata-rata trait",
            xaxis_title="Generasi",
        )
        return Visualizer._apply_text_theme(fig)

    @staticmethod
    def _apply_text_theme(fig: go.Figure) -> go.Figure:
        fig.update_layout(
            font=dict(color=Visualizer.TEXT_COLOR),
            title_font=dict(color=Visualizer.TEXT_COLOR),
            legend=dict(font=dict(color=Visualizer.TEXT_COLOR)),
            paper_bgcolor=Visualizer.PAPER_COLOR,
        )
        fig.update_xaxes(
            color=Visualizer.TEXT_COLOR,
            tickfont=dict(color=Visualizer.TEXT_COLOR),
            title_font=dict(color=Visualizer.TEXT_COLOR),
            gridcolor=Visualizer.GRID_COLOR,
            zerolinecolor=Visualizer.GRID_COLOR,
        )
        fig.update_yaxes(
            color=Visualizer.TEXT_COLOR,
            tickfont=dict(color=Visualizer.TEXT_COLOR),
            title_font=dict(color=Visualizer.TEXT_COLOR),
            gridcolor=Visualizer.GRID_COLOR,
            zerolinecolor=Visualizer.GRID_COLOR,
        )
        return fig

    @staticmethod
    def _history_records(
        history: Sequence[GenerationMetrics],
    ) -> list[dict[str, float | int]]:
        return [asdict(metric) for metric in history]
