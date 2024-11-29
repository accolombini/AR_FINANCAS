import pandas as pd
import numpy as np
from dash import dcc, html
import dash
import plotly.express as px


def clean_and_normalize_returns(data):
    """
    Limpa e normaliza os retornos dos ativos.
    """
    returns = data.pct_change().dropna()
    returns = returns.clip(lower=-0.5, upper=0.5)  # Limitar outliers a ±50%
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns


def calculate_annual_returns(data):
    """
    Calcula os retornos anuais a partir de uma série temporal.
    """
    annual_returns = data.resample("YE").apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    return annual_returns


def analyze_retrospective(data, weights, benchmark_column):
    """
    Analisa os retornos históricos do portfólio e do benchmark.
    """
    # Limpar e normalizar os retornos
    data_normalized = clean_and_normalize_returns(data)

    # Calcular os retornos anuais do benchmark
    benchmark_returns = calculate_annual_returns(
        data_normalized[benchmark_column])

    # Calcular os retornos anuais do portfólio
    weighted_returns = data_normalized[list(weights.keys())].mul(
        list(weights.values()), axis=1)
    portfolio_returns = calculate_annual_returns(weighted_returns.sum(axis=1))

    retrospective_analysis = pd.DataFrame({
        "Year": benchmark_returns.index.year,
        "Portfolio Return (%)": portfolio_returns.values,
        "Benchmark Return (%)": benchmark_returns.values
    })

    return retrospective_analysis


if __name__ == "__main__":
    # Caminhos para os arquivos
    historical_data_path = "BackPython/DADOS/historical_data_cleaned.csv"
    portfolio_weights_path = "BackPython/DADOS/portfolio_otimizado.csv"

    # Carregar dados históricos
    historical_data = pd.read_csv(historical_data_path, parse_dates=[
                                  "Date"]).set_index("Date")
    portfolio_optimized = pd.read_csv(portfolio_weights_path)

    # Última coluna é o benchmark
    benchmark_column = historical_data.columns[-1]

    # Filtrar ativos relevantes
    weights_dict = dict(
        zip(portfolio_optimized["Ativo"], portfolio_optimized["Peso (%)"]))
    relevant_assets = weights_dict.keys()

    # Análise Retrospectiva
    retrospective_df = analyze_retrospective(
        historical_data, weights_dict, benchmark_column)
    retrospective_df.to_csv(
        "BackPython/DADOS/retrospective_analysis_corrected.csv", index=False)

    # Dashboard com Dash
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Portfólio vs Benchmark: Retornos Históricos Corrigidos"),
        dcc.Graph(
            id="retrospective-graph",
            figure=px.line(retrospective_df, x="Year", y=["Portfolio Return (%)", "Benchmark Return (%)"],
                           title="Retornos Históricos Ano-a-Ano (Corrigidos)")
        )
    ])

    app.run_server(debug=True)
