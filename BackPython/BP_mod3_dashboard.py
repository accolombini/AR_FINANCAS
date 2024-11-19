# Dashboard para análise de desempenho de modelos de curto prazo usando Dash e Plotly

# BP_mod3_dashboard.py
# Importar bibliotecas necessárias``

# Dashboard para análise de desempenho de modelos de curto prazo usando Dash e Plotly

# Importar bibliotecas necessárias

import pandas as pd
import numpy as np
from dash import Dash, html, dcc
import plotly.graph_objs as go


class Dashboard:
    def __init__(self, simulations, weights, benchmark_data):
        """
        Inicializa o dashboard.

        Args:
            simulations (dict): Simulações de Monte Carlo por ativo.
            weights (np.ndarray): Pesos ótimos dos ativos no portfólio.
            benchmark_data (pd.DataFrame): Dados do índice de benchmark (ex.: BOVESPA).
        """
        self.simulations = simulations
        self.weights = weights
        self.benchmark_data = benchmark_data

    def create_dashboard(self):
        """
        Cria o layout do dashboard com os gráficos necessários.
        """
        app = Dash(__name__)

        # Layout do dashboard
        app.layout = html.Div(style={'backgroundColor': '#1f1f1f'}, children=[
            html.H1("Dashboard de Simulações e Otimização de Portfólio",
                    style={'textAlign': 'center', 'color': '#ffffff'}),

            # Gráfico: Distribuição de Retornos Simulados
            html.Div([
                html.H2("Distribuições de Retornos Simulados por Ativo",
                        style={'textAlign': 'center', 'color': '#ffffff'}),
                dcc.Graph(
                    id="simulation-distribution",
                    figure={
                        "data": [
                            go.Histogram(
                                x=self.simulations[asset][-1, :],
                                name=asset,
                                opacity=0.75
                            )
                            for asset in self.simulations.keys()
                        ],
                        "layout": go.Layout(
                            title="Distribuição de Retornos Simulados",
                            xaxis={"title": "Retorno Simulado",
                                   "color": "#ffffff"},
                            yaxis={"title": "Frequência", "color": "#ffffff"},
                            paper_bgcolor="#1f1f1f",
                            plot_bgcolor="#1f1f1f",
                            font=dict(color="#ffffff"),
                            barmode="overlay"
                        )
                    }
                )
            ]),

            # Gráfico: Pesos Ótimos do Portfólio
            html.Div([
                html.H2("Pesos Ótimos do Portfólio",
                        style={'textAlign': 'center', 'color': '#ffffff'}),
                dcc.Graph(
                    id="optimal-weights",
                    figure={
                        "data": [
                            go.Bar(
                                x=list(self.simulations.keys()),
                                y=self.weights,
                                marker={"color": "blue"}
                            )
                        ],
                        "layout": go.Layout(
                            title="Distribuição de Pesos no Portfólio",
                            xaxis={"title": "Ativos", "color": "#ffffff"},
                            yaxis={"title": "Peso Ótimo", "color": "#ffffff"},
                            paper_bgcolor="#1f1f1f",
                            plot_bgcolor="#1f1f1f",
                            font=dict(color="#ffffff")
                        )
                    }
                )
            ]),

            # Gráfico: Comparativo com Benchmark
            html.Div([
                html.H2("Comparativo com o Benchmark (BOVESPA)",
                        style={'textAlign': 'center', 'color': '#ffffff'}),
                dcc.Graph(
                    id="benchmark-comparison",
                    figure={
                        "data": [
                            go.Scatter(
                                x=self.benchmark_data.index,
                                y=self.benchmark_data["^BVSP"],
                                mode="lines",
                                name="BOVESPA",
                                line={"color": "green"}
                            ),
                            go.Scatter(
                                x=self.benchmark_data.index,
                                y=np.dot(
                                    self.simulations["Portfolio"], self.weights),
                                mode="lines",
                                name="Portfólio",
                                line={"color": "orange"}
                            )
                        ],
                        "layout": go.Layout(
                            title="Retorno do Portfólio vs. Benchmark",
                            xaxis={"title": "Data", "color": "#ffffff"},
                            yaxis={"title": "Retorno Acumulado",
                                   "color": "#ffffff"},
                            paper_bgcolor="#1f1f1f",
                            plot_bgcolor="#1f1f1f",
                            font=dict(color="#ffffff")
                        )
                    }
                )
            ])
        ])

        return app


if __name__ == "__main__":
    # Dados simulados para teste
    simulated_data = {
        "Asset1": np.random.normal(0.1, 0.02, (252, 1000)),
        "Asset2": np.random.normal(0.08, 0.03, (252, 1000)),
        "Asset3": np.random.normal(0.12, 0.04, (252, 1000)),
        "Portfolio": np.random.normal(0.1, 0.02, (252, 1000))
    }
    optimal_weights = [0.4, 0.3, 0.3]
    benchmark_df = pd.DataFrame({
        "^BVSP": np.cumsum(np.random.normal(0.001, 0.01, 252)),
    }, index=pd.date_range(start="2023-01-01", periods=252, freq="B"))

    # Criar e executar o dashboard
    dashboard = Dashboard(simulated_data, optimal_weights, benchmark_df)
    app = dashboard.create_dashboard()
    app.run_server(debug=True, host="127.0.0.1", port=8050)
