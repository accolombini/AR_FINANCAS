'''
    Este módulo tem por objetivo simular o comportamento dos preços de ativos em um período de tempo determinado.
    Inclui a classe para simulação de Monte Carlo, o tratamento dos dados, e as etapas de cálculo que foram aplicadas

    Os dados foram carregados com sucesso. Eles incluem preços ajustados e métricas derivadas, como retornos, médias móveis e volatilidades, tanto para os ativos individuais quanto para o índice BOVESPA (^BVSP).

        Vou agora utilizar esses dados para:

        Calcular os retornos logarítmicos dos ativos.
        Realizar simulações de Monte Carlo para os ativos.
        Gerar métricas do portfólio com pesos iniciais para validar o modelo.
        Aguarde enquanto executo essas etapas. ​​

        Os cálculos das simulações de Monte Carlo retornaram valores inválidos (NaN). Isso geralmente ocorre por conta de problemas nos dados de entrada, como zeros ou valores negativos que geram erros ao calcular os logaritmos.

        Investigação do Problema
        Retornos Logarítmicos
        O cálculo de logaritmos em np.log(self.data / self.data.shift(1)) pode gerar valores NaN ou inf se houver:

            Zeros nos preços.
            Valores ausentes ou não preenchidos adequadamente.
            Divisão por zero devido a valores consecutivos iguais.
            Próximos Passos

            Verificar os dados originais para valores inconsistentes ou ausentes.
            Tratar valores inválidos preenchendo ou ajustando os dados para cálculos.
            Vou inspecionar os dados e corrigir os problemas antes de continuar. ​​

            Os dados apresentam problemas em algumas colunas, especificamente nos retornos calculados (*_returns). Aqui estão os principais achados:

            Zeros em Retornos

            Algumas colunas têm muitos zeros, especialmente:
            PGCO34.SA_returns: 816 zeros.
            AMZO34.SA_returns: 676 zeros.
            APL34.SA_returns: 170 zeros.

            Implicação

            Esses zeros provavelmente surgem de períodos sem variação nos preços ou devido a erros ao calcular os retornos.

    Nota: Os dados foram corrigidos, substituindo valores zero por um pequeno valor (1e-6). Também filtramos apenas as colunas de preços ajustados, removendo colunas derivadas (como retornos, médias móveis e volatilidade) para evitar cálculos redundantes.
    '''

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from dash import dcc, html
import dash
import plotly.express as px


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
    # Calcular os retornos anuais do benchmark
    benchmark_returns = calculate_annual_returns(data[benchmark_column])

    # Calcular os retornos anuais do portfólio
    weighted_returns = data[list(weights.keys())].pct_change(
    ).dropna().mul(list(weights.values()), axis=1)
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
        "BackPython/DADOS/retrospective_analysis.csv", index=False)

    # Dashboard com Dash
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Portfólio vs Benchmark: Retornos Históricos"),
        dcc.Graph(
            id="retrospective-graph",
            figure=px.line(retrospective_df, x="Year", y=["Portfolio Return (%)", "Benchmark Return (%)"],
                           title="Retornos Históricos Ano-a-Ano")
        )
    ])

    app.run_server(debug=True)
