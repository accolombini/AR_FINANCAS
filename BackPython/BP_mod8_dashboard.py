# BP_mod8_dashboard_updated.py: Dashboard Interativo com Melhorias
# -----------------------------------------------------------
# Este script aprimora o dashboard com:
# - Separação de projeções para 6 meses e 5 anos.
# - Adição de retornos anuais históricos e projetados.
# -----------------------------------------------------------

import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, dash_table
from BP_mod1_config import OUTPUT_DIR

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados():
    """Carrega os dados necessários para o dashboard."""
    caminho_historico = os.path.join(OUTPUT_DIR, "portfolio_comportamento.csv")
    caminho_projecoes = os.path.join(OUTPUT_DIR, "projecoes_portfolio.csv")

    dados_historicos = pd.read_csv(
        caminho_historico, index_col="Date", parse_dates=True)
    dados_projecoes = pd.read_csv(caminho_projecoes)
    dados_projecoes["Date"] = pd.to_datetime(
        dados_projecoes["Date"])  # Garantir formato datetime
    return dados_historicos, dados_projecoes


def calcular_retornos(dados_historicos, dados_projecoes):
    """Calcula retornos anuais históricos e projetados."""
    # Retorno anual histórico
    dados_historicos["Ano"] = dados_historicos.index.year
    retorno_anual_historico = dados_historicos.groupby("Ano")["portfolio_otimo"].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
    )

    # Retornos projetados
    retorno_6m = (dados_projecoes.iloc[:126]["Projecoes"].iloc[-1] - dados_projecoes.iloc[0]["Projecoes"]) / \
        dados_projecoes.iloc[0]["Projecoes"] * 100
    retorno_5y = (dados_projecoes.iloc[-1]["Projecoes"] - dados_projecoes.iloc[0]["Projecoes"]) / \
        dados_projecoes.iloc[0]["Projecoes"] * 100

    return retorno_anual_historico, retorno_6m, retorno_5y


def gerar_grafico_linhas(dados_historicos, dados_projecoes):
    """Gera um gráfico de linhas para o comportamento do portfólio."""
    # Histórico
    trace_historico = go.Scatter(
        x=dados_historicos.index,
        y=dados_historicos["portfolio_otimo"],
        mode="lines",
        name="Histórico do Portfólio",
        line=dict(color="blue")
    )

    # Projeções (6 meses e 5 anos)
    trace_projecoes_6m = go.Scatter(
        x=dados_projecoes["Date"].iloc[:126],
        y=dados_projecoes["Projecoes"].iloc[:126],
        mode="lines",
        name="Projeção 6 Meses",
        line=dict(color="green", dash="dot")
    )

    trace_projecoes_5y = go.Scatter(
        x=dados_projecoes["Date"].iloc[126:],
        y=dados_projecoes["Projecoes"].iloc[126:],
        mode="lines",
        name="Projeção 5 Anos",
        line=dict(color="red", dash="dot")
    )

    layout = go.Layout(
        title="Comportamento do Portfólio",
        xaxis=dict(title="Data"),
        yaxis=dict(title="Valor do Portfólio"),
        legend=dict(x=0, y=1)
    )

    return go.Figure(data=[trace_historico, trace_projecoes_6m, trace_projecoes_5y], layout=layout)


def gerar_tabela_estatisticas(retorno_anual_historico, retorno_6m, retorno_5y):
    """Gera uma tabela com as estatísticas do modelo."""
    estatisticas = {
        "Métrica": ["RMSE", "MAPE (%)", "R²", "Retorno Anual Médio (%)", "Retorno Projetado 6M (%)", "Retorno Projetado 5Y (%)"],
        "Valor": [7.023581873839172, 21.18939912047932, 0.7798900309450476,
                  retorno_anual_historico.mean(), retorno_6m, retorno_5y]
    }
    df_estatisticas = pd.DataFrame(estatisticas)
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_estatisticas.columns],
        data=df_estatisticas.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
    )

# ---------------------------
# Configuração do Dashboard
# ---------------------------


# Carregar os dados
dados_historicos, dados_projecoes = carregar_dados()
retorno_anual_historico, retorno_6m, retorno_5y = calcular_retornos(
    dados_historicos, dados_projecoes)

# Inicializar o app Dash
app = Dash(__name__)

# Layout do Dashboard
app.layout = html.Div([
    html.H1("Dashboard do Portfólio Ótimo"),
    dcc.Graph(
        id="grafico-linhas",
        figure=gerar_grafico_linhas(dados_historicos, dados_projecoes)
    ),
    html.H2("Estatísticas do Modelo"),
    gerar_tabela_estatisticas(retorno_anual_historico, retorno_6m, retorno_5y),
])

# ---------------------------
# Execução do Servidor
# ---------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
