# Dashboard para análise de desempenho de modelos de curto prazo usando Dash e Plotly

# BP_mod3_dashboard.py
# Importar bibliotecas necessárias``

# Dashboard para análise de desempenho de modelos de curto prazo usando Dash e Plotly

# Importar bibliotecas necessárias

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc


class Dashboard:
    def __init__(self, simulations_file, percentiles_file, benchmark_file):
        self.simulations_file = simulations_file
        self.percentiles_file = percentiles_file
        self.benchmark_file = benchmark_file

    def load_data(self):
        # Carregar dados necessários
        try:
            simulations = pd.read_csv(
                self.simulations_file, parse_dates=["Date"])
            percentiles = pd.read_csv(
                self.percentiles_file, parse_dates=["Date"])
            benchmark = pd.read_csv(self.benchmark_file, parse_dates=["Date"])
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Erro ao carregar arquivos: {
                    e}. Certifique-se de que os arquivos de simulação e benchmark existam."
            )
        return simulations, percentiles, benchmark

    def create_dashboard(self):
        simulations, percentiles, benchmark = self.load_data()

        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Gráfico 1: Distribuição dos Pesos
        weight_distribution_fig = go.Figure(
            data=[
                go.Pie(
                    labels=simulations.columns[1:-1],  # Colunas dos ativos
                    values=simulations.iloc[-1, 1:-1],  # Últimos pesos
                    hoverinfo="label+percent",
                    textinfo="value+percent",
                )
            ]
        )
        weight_distribution_fig.update_layout(
            title="Distribuição de Pesos do Portfólio")

        # Gráfico 2: Simulações Monte Carlo
        monte_carlo_fig = go.Figure()
        for column in simulations.columns[1:-1]:
            monte_carlo_fig.add_trace(
                go.Scatter(
                    x=simulations["Date"],
                    y=simulations[column],
                    mode="lines",
                    name=column,
                )
            )
        monte_carlo_fig.add_trace(
            go.Scatter(
                x=benchmark["Date"],
                y=benchmark["^BVSP"],
                mode="lines",
                name="BOVESPA",
                line=dict(dash="dot", color="red"),
            )
        )
        monte_carlo_fig.update_layout(
            title="Simulações de Retornos por Ativo e Benchmark",
            xaxis_title="Dias",
            yaxis_title="Retorno Simulado",
        )

        # Gráfico 3: Desempenho Portfólio vs Benchmark
        performance_fig = go.Figure()
        performance_fig.add_trace(
            go.Scatter(
                x=percentiles["Date"],
                y=percentiles["Portfolio"],
                mode="lines",
                name="Portfólio",
            )
        )
        performance_fig.add_trace(
            go.Scatter(
                x=benchmark["Date"],
                y=benchmark["^BVSP"],
                mode="lines",
                name="Índice BOVESPA",
                line=dict(dash="dot", color="green"),
            )
        )
        performance_fig.update_layout(
            title="Desempenho do Portfólio vs Benchmark",
            xaxis_title="Dias",
            yaxis_title="Valor Normalizado",
        )

        app.layout = dbc.Container(
            [
                html.H1("Dashboard do Portfólio Otimizado",
                        style={"textAlign": "center"}),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(figure=weight_distribution_fig), width=6),
                        dbc.Col(dcc.Graph(figure=monte_carlo_fig), width=6),
                    ]
                ),
                dbc.Row(dbc.Col(dcc.Graph(figure=performance_fig))),
            ],
            fluid=True,
        )

        return app


if __name__ == "__main__":
    simulations_file = "BackPython/DADOS/mc_simulations.csv"
    percentiles_file = "BackPython/DADOS/mc_percentiles.csv"
    benchmark_file = "BackPython/DADOS/historical_data_cleaned.csv"

    dashboard = Dashboard(simulations_file, percentiles_file, benchmark_file)
    app = dashboard.create_dashboard()
    app.run_server(debug=True)
