# Import necessary libraries

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go


def calculate_annual_returns(data, weights=None):
    """
    Calcula os retornos anuais de cada ativo, benchmark e portfólio otimizado.
    """
    annual_returns = data.resample("YE").last().pct_change().dropna()
    if weights is not None:
        portfolio_returns = (
            annual_returns.drop(columns="^BVSP")
            .dot(np.array([weights[col] for col in annual_returns.columns if col in weights.keys()]) / 100)
        )
        annual_returns["Portfolio_Otimo"] = portfolio_returns
    annual_returns.reset_index(inplace=True)
    annual_returns.rename(columns={"Date": "Período"}, inplace=True)
    annual_returns["Período"] = annual_returns["Período"].dt.year
    return annual_returns * 100  # Converter para porcentagem


def simulate_with_egarch(data, weights, benchmark):
    """
    Simula o portfólio e calcula retornos anuais e validação.
    """
    relevant_assets = [col for col in data.columns if col in weights.keys()]
    data = data[["Date"] + relevant_assets + [benchmark]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)

    # Calcular os retornos anuais
    annual_returns = calculate_annual_returns(data, weights)

    # Validar o portfólio nos últimos 60 dias
    validation_data = data[-60:]  # Últimos 60 dias
    validation_returns = (
        validation_data[relevant_assets]
        .pct_change()
        .iloc[1:]
        .dot(np.array([weights[col] for col in relevant_assets]) / 100)
        .values
    )
    benchmark_returns = validation_data[benchmark].pct_change().iloc[1:].values

    # Calcular métricas de erro
    mae = np.abs(validation_returns -
                 benchmark_returns[:len(validation_returns)]).mean()
    rmse = np.sqrt(np.mean(
        (validation_returns - benchmark_returns[:len(validation_returns)]) ** 2))
    precision = (1 - mae) * 100

    # Retorno médio dos últimos 60 dias anualizado
    avg_return_60d = np.mean(validation_returns) * 252  # 252 dias úteis no ano

    # Criar DataFrame de validação
    validation_df = pd.DataFrame(
        {
            "Portfolio Simulated": validation_returns,
            "Benchmark": benchmark_returns[:len(validation_returns)],
        },
        index=validation_data.index[1:len(validation_returns) + 1],
    )

    return validation_df, annual_returns, mae, rmse, precision, avg_return_60d


if __name__ == "__main__":
    historical_data_path = "BackPython/DADOS/historical_data_cleaned.csv"
    portfolio_weights_path = "BackPython/DADOS/portfolio_otimizado.csv"
    benchmark_column = "^BVSP"

    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
    portfolio_optimized = pd.read_csv(portfolio_weights_path)

    weights_dict = dict(
        zip(portfolio_optimized["Ativo"], portfolio_optimized["Peso (%)"])
    )

    validation_df, annual_returns, mae, rmse, precision, avg_return_60d = simulate_with_egarch(
        historical_data, weights_dict, benchmark_column
    )

    # Preparar visualização com Dash e Plotly
    app = Dash(__name__)

    # Resetar índice para facilitar o uso com o gráfico
    validation_df.reset_index(inplace=True)
    validation_df.rename(columns={"index": "Date"}, inplace=True)

    # Gráfico de validação
    fig = px.line(
        validation_df,
        x="Date",
        y=["Portfolio Simulated", "Benchmark"],
        title="Validação do Portfólio vs Benchmark",
        labels={"value": "Retornos (%)", "Date": "Data"},
    )

    # Adicionar linha de referência para o retorno esperado de 15% anual
    fig.add_trace(
        go.Scatter(
            x=validation_df["Date"],
            y=[0.15 / 252] * len(validation_df),  # Dividir por 252 dias úteis
            mode="lines",
            name="Meta de Retorno (15% Anual)",
            line=dict(dash="dash", color="green"),
        )
    )

    fig.update_layout(legend_title_text="Curvas")

    # Layout do Dash
    app.layout = html.Div(
        style={"textAlign": "center", "fontFamily": "Arial"},
        children=[
            html.H1("Validação do Portfólio"),
            dcc.Graph(id="portfolio-validation-graph", figure=fig),
            html.Div(
                [
                    html.H2("Retornos Anuais (em %)"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([html.Th(col)
                                        for col in annual_returns.columns])
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(f"{val:.2f}%" if isinstance(
                                                val, (float, int)) else val)
                                            for val in annual_returns.iloc[i]
                                        ]
                                    )
                                    for i in range(len(annual_returns))
                                ]
                            ),
                        ],
                        style={"margin": "auto", "fontSize": "16px",
                               "borderCollapse": "collapse"},
                    ),
                ],
                style={"marginTop": "20px"},
            ),
            html.Div(
                [
                    html.H3("Métricas de Validação do Portfólio:"),
                    html.P(f"Erro Médio Absoluto (MAE): {mae:.2%}"),
                    html.P(f"Erro Quadrático Médio (RMSE): {rmse:.2%}"),
                    html.P(f"Precisão: {precision:.2f}%"),
                    html.P(f"Retorno Médio Anualizado (Últimos 60 dias): {
                           avg_return_60d:.2%}"),
                ],
                style={"marginTop": "20px", "fontSize": "18px"},
            ),
        ],
    )

    app.run_server(debug=True)
