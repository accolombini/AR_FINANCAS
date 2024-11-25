''' 
    Estrutura:

    Simular o portfólio em uma janela de 6 meses, considerando o mesmo modelo EGARCH.
    Incorporar retornos mês a mês, além de médias móveis para identificar tendências de curto prazo.
    
    ||> Métricas Adicionais:

        Retorno Mensal Acumulado.
        Volatilidade Mensal.
        Correlação com o Benchmark.
    
    ||> Visualização:

        Gráficos com foco em tendências de curto prazo:
        Retornos mensais acumulados do portfólio vs benchmark.
        Análise de dispersão dos retornos diários no período.
    
    ||> Análise de Ajuste Dinâmico:

        Possibilidade de sugerir ajustes nos pesos do portfólio caso algum ativo apresente desempenho muito descolado da média.

    ||> Flexibilidade para Avaliar Outros Algoritmos:

        Deixar aberta a possibilidade de utilizar outros modelos além do EGARCH é muito importante.
        Isso nos permite testar métodos como GARCH simplificado, SARIMA, ou até mesmo redes neurais LSTM, caso necessário.
        Podemos introduzir uma camada de validação cruzada para comparar diferentes algoritmos e decidir qual se        adapta melhor ao curto prazo.
    
    ||> Portfólio Sempre Superando o Índice:

        Adicionar uma métrica que garanta, na média mensal, que o portfólio supere o benchmark é crucial para validar a eficácia da estratégia.
        Essa abordagem nos ajuda a garantir que o portfólio está gerando "alpha" (retorno acima do mercado) consistentemente.
        Podemos implementar uma lógica para ajustar dinamicamente os pesos caso o portfólio esteja ficando muito próximo ou abaixo do benchmark.

'''

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from arch import arch_model
import dash
from dash import Dash, dcc, html
import plotly.graph_objects as go
from datetime import timedelta


def monte_carlo_simulation(data, weights, num_simulations=1000, days_ahead=180):
    """
    Simula cenários futuros para os próximos 6 meses usando Monte Carlo.
    """
    # Selecionar retornos históricos para cada ativo
    returns = data.pct_change().dropna()
    weighted_returns = returns.dot(
        np.array([weights[col] for col in returns.columns]) / 100)

    # Parâmetros da simulação
    mean_return = weighted_returns.mean()
    volatility = weighted_returns.std()
    last_price = data.iloc[-1].dot(np.array([weights[col]
                                   for col in data.columns]) / 100)

    simulations = []
    for _ in range(num_simulations):
        prices = [last_price]
        for _ in range(days_ahead):
            random_return = np.random.normal(mean_return, volatility)
            prices.append(prices[-1] * (1 + random_return))
        simulations.append(prices)

    simulations = np.array(simulations)
    return simulations


def calculate_percentiles(simulations):
    """
    Calcula os percentis para cada dia da simulação.
    """
    percentiles = {
        "5%": np.percentile(simulations, 5, axis=0),
        "50% (Mediana)": np.percentile(simulations, 50, axis=0),
        "95%": np.percentile(simulations, 95, axis=0),
    }
    return percentiles


if __name__ == "__main__":
    # Configurações e entrada de dados
    historical_data_path = "BackPython/DADOS/historical_data_cleaned.csv"
    portfolio_weights_path = "BackPython/DADOS/portfolio_otimizado.csv"

    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
    portfolio_optimized = pd.read_csv(portfolio_weights_path)

    weights_dict = dict(
        zip(portfolio_optimized["Ativo"], portfolio_optimized["Peso (%)"])
    )

    relevant_assets = [
        col for col in historical_data.columns if col in weights_dict.keys()]
    data = historical_data[relevant_assets].set_index(historical_data["Date"])

    # Simulação de Monte Carlo
    num_simulations = 1000
    days_ahead = 180
    simulations = monte_carlo_simulation(
        data[relevant_assets], weights_dict, num_simulations, days_ahead)
    percentiles = calculate_percentiles(simulations)

    # Gerar datas futuras
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i)
                    for i in range(days_ahead + 1)]

    # Preparar visualização com Dash e Plotly
    app = Dash(__name__)

    # Gráfico de projeção com percentis
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=percentiles["50% (Mediana)"],
        mode="lines",
        name="50% (Mediana)",
        line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=percentiles["5%"],
        mode="lines",
        name="5%",
        line=dict(color="red", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=percentiles["95%"],
        mode="lines",
        name="95%",
        line=dict(color="green", dash="dot"),
    ))
    fig.add_hline(
        # Meta ajustada para 6 meses
        y=data.iloc[-1].mean() * 1.15 ** (1 / 12),
        line_dash="dash",
        line_color="black",
        annotation_text="Meta de Retorno (15% Anual Ajustado)",
    )
    fig.update_layout(
        title="Projeção de Retornos Futuros (6 Meses) com Monte Carlo",
        xaxis_title="Data",
        yaxis_title="Retorno (%)",
        legend_title="Cenários",
    )

    # Layout do Dash
    app.layout = html.Div(
        style={"textAlign": "center", "fontFamily": "Arial"},
        children=[
            html.H1("Projeção de Retornos Futuros (Curto Prazo)"),
            dcc.Graph(id="monte-carlo-projection", figure=fig),
            html.Div(
                [
                    html.H2("Resumo dos Cenários Simulados"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    ["Cenário", "Retorno Inicial (%)", "Retorno Final (%)"])
                            ),
                            html.Tbody(
                                [
                                    html.Tr([
                                        html.Td(percentile),
                                        html.Td(
                                            f"{percentiles[percentile][0]:.2f}%"),
                                        html.Td(
                                            f"{percentiles[percentile][-1]:.2f}%"),
                                    ])
                                    for percentile in percentiles.keys()
                                ]
                            ),
                        ],
                        style={"margin": "auto", "fontSize": "16px",
                               "borderCollapse": "collapse"},
                    ),
                ],
                style={"marginTop": "20px"},
            ),
        ],
    )

    app.run_server(debug=True)
