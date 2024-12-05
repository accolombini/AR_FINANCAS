# BP_mod5_monte_carlo_dash.py: Dashboard Interativo com VaR e CVaR
'''
        ||>VaR (Value at Risk):

            Definição: É o valor que representa a maior perda esperada em um portfólio em um determinado intervalo de confiança (neste caso, 5%).
            Interpretação: Para um VaR de 0.46 (como no exemplo de 5 anos), isso significa que, com 95% de confiança, as perdas não excederão 46% em um determinado horizonte de tempo (5 anos no caso).
            Limitação: O VaR não informa o que acontece além desse limite, ou seja, em casos extremos.

        ||> CVaR (Conditional Value at Risk):

            Definição: É o valor médio das perdas que excedem o VaR, ou seja, mede o "rabo" da distribuição de perdas.
            Interpretação: No exemplo, o CVaR de 0.32 mostra que, nos piores 5% dos cenários, as perdas médias são de aproximadamente 32%.
            Uso: É mais conservador e informativo que o VaR, pois considera o impacto de eventos extremos.
'''
# -----------------------------------------------------------
# Este script utiliza Dash e Plotly para criar um dashboard interativo,
# permitindo visualizar os cenários de Monte Carlo para horizontes de 6 meses
# e 5 anos, com percentis (5%, 50%, 95%), VaR, CVaR e melhorias visuais.
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ---------------------------
# Funções Auxiliares
# ---------------------------


def gerar_trajetorias_mc(retornos_medio, volatilidade, dias, n_simulacoes=1000):
    """
    Gera trajetórias de Monte Carlo com base nos retornos e volatilidade histórica.

    Args:
        retornos_medio (float): Retorno médio diário.
        volatilidade (float): Volatilidade diária.
        dias (int): Número de dias a simular.
        n_simulacoes (int): Número de simulações.

    Returns:
        np.ndarray: Trajetórias simuladas.
    """
    trajetorias = np.zeros((dias, n_simulacoes))
    for sim in range(n_simulacoes):
        choques = np.random.normal(retornos_medio, volatilidade, dias)
        trajetorias[:, sim] = np.cumprod(1 + choques)
    return trajetorias


def calcular_metricas_mc(trajetorias):
    """
    Calcula métricas de percentis (5%, 50%, 95%), VaR e CVaR das trajetórias.

    Args:
        trajetorias (np.ndarray): Trajetórias simuladas.

    Returns:
        dict: Métricas calculadas.
    """
    percentis = {
        "5%": np.percentile(trajetorias[-1], 5),
        "50%": np.percentile(trajetorias[-1], 50),
        "95%": np.percentile(trajetorias[-1], 95),
    }
    var = percentis["5%"]
    cvar = trajetorias[-1][trajetorias[-1] <= var].mean()

    percentis["VaR (5%)"] = var
    percentis["CVaR (5%)"] = cvar

    return percentis


def calcular_portfolio_otimo(dias, retorno_medio, volatilidade):
    """
    Calcula o crescimento acumulado do portfólio ótimo.

    Args:
        dias (int): Número de dias a simular.
        retorno_medio (float): Retorno médio diário.
        volatilidade (float): Volatilidade diária.

    Returns:
        np.ndarray: Crescimento acumulado do portfólio ótimo.
    """
    choques = np.random.normal(retorno_medio, volatilidade, dias)
    crescimento_acumulado = np.cumprod(1 + choques)
    return crescimento_acumulado


def criar_grafico(trajetorias, percentis, portfolio_otimo, titulo):
    """
    Cria um gráfico interativo das trajetórias simuladas, dos percentis e do portfólio ótimo.

    Args:
        trajetorias (np.ndarray): Trajetórias simuladas.
        percentis (dict): Percentis calculados.
        portfolio_otimo (np.ndarray): Crescimento acumulado do portfólio ótimo.
        titulo (str): Título do gráfico.

    Returns:
        plotly.graph_objs.Figure: Gráfico interativo.
    """
    fig = go.Figure()

    # Adicionar trajetórias
    for i in range(trajetorias.shape[1]):
        fig.add_trace(go.Scatter(
            x=list(range(trajetorias.shape[0])),
            y=trajetorias[:, i],
            mode='lines',
            line=dict(color='lightgray'),
            name='Simulações' if i == 0 else None,
            showlegend=(i == 0)
        ))

    # Adicionar linhas de percentil
    fig.add_trace(go.Scatter(
        x=[0, trajetorias.shape[0] - 1],
        y=[percentis["5%"], percentis["5%"]],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='5% Percentil'
    ))
    fig.add_trace(go.Scatter(
        x=[0, trajetorias.shape[0] - 1],
        y=[percentis["50%"], percentis["50%"]],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='50% Percentil'
    ))
    fig.add_trace(go.Scatter(
        x=[0, trajetorias.shape[0] - 1],
        y=[percentis["95%"], percentis["95%"]],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='95% Percentil'
    ))

    # Adicionar linha do portfólio ótimo
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_otimo))),
        y=portfolio_otimo,
        mode='lines',
        line=dict(color='black', width=2),
        name='Portfólio Ótimo'
    ))

    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Dias",
        yaxis_title="Crescimento Acumulado",
        legend=dict(font=dict(size=15), x=0, y=-0.2, orientation="h")
    )
    return fig

# ---------------------------
# Configuração do Dashboard
# ---------------------------


app = dash.Dash(__name__)
app.title = "Simulação de Monte Carlo com VaR e CVaR"

# Layout
app.layout = html.Div([
    html.H1("Dashboard de Simulação de Monte Carlo",
            style={'text-align': 'center'}),
    dcc.Dropdown(
        id='horizonte-dropdown',
        options=[
            {'label': '6 Meses', 'value': 126},
            {'label': '5 Anos', 'value': 1260}
        ],
        value=126,
        style={'width': '50%', 'margin': 'auto'}
    ),
    dcc.Graph(id='grafico-monte-carlo'),
    dash_table.DataTable(
        id='tabela-metricas',
        columns=[{"name": col, "id": col} for col in ["Métrica", "Valor"]],
        style_table={'width': '60%', 'margin': 'auto'},
        style_cell={'textAlign': 'center', 'fontSize': 15}
    )
])

# Callbacks


@app.callback(
    [Output('grafico-monte-carlo', 'figure'),
     Output('tabela-metricas', 'data')],
    [Input('horizonte-dropdown', 'value')]
)
def atualizar_grafico_e_tabela(horizonte):
    retorno_medio_diario = 0.0005
    volatilidade_diaria = 0.02
    n_simulacoes = 1000

    trajetorias = gerar_trajetorias_mc(
        retorno_medio_diario, volatilidade_diaria, horizonte, n_simulacoes)
    percentis = calcular_metricas_mc(trajetorias)
    portfolio_otimo = calcular_portfolio_otimo(
        horizonte, retorno_medio_diario, volatilidade_diaria)
    fig = criar_grafico(trajetorias, percentis, portfolio_otimo,
                        f"Simulação de Monte Carlo - {horizonte} Dias")
    tabela = [{"Métrica": k, "Valor": round(v, 2)}
              for k, v in percentis.items()]
    return fig, tabela

# ---------------------------
# Executar o Dashboard
# ---------------------------


if __name__ == '__main__':
    app.run_server(debug=True)
