# Dashboard para análise de desempenho de modelos de curto prazo usando Dash e Plotly

# BP_mod3_dashboard.py
# Importar bibliotecas necessárias``

# Dashboard para análise de desempenho de modelos de curto prazo usando Dash e Plotly

import pandas as pd
import numpy as np
from dash import Dash, html, dcc
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Configuração de diretórios e arquivos
DATA_DIR = "BackPython/DADOS"
validation_file_path = os.path.join(DATA_DIR, "y_random_forest.csv")
predicted_file_path = os.path.join(DATA_DIR, "y_pred_rf.csv")

# Verificar se os arquivos existem antes de carregar
if not os.path.exists(validation_file_path) or not os.path.exists(predicted_file_path):
    raise FileNotFoundError(
        f"Arquivos necessários não encontrados em {
            DATA_DIR}. Verifique os arquivos de validação e previsão."
    )

# Carregar dados de validação e previsões
validation_data = pd.read_csv(
    validation_file_path, index_col=0, parse_dates=True)
predicted_data = pd.read_csv(
    predicted_file_path, index_col=0, parse_dates=True)

# Garantir que os índices sejam do tipo DatetimeIndex
if not isinstance(validation_data.index, pd.DatetimeIndex):
    raise ValueError("O índice de validation_data deve ser um DatetimeIndex.")

# Renomear colunas para consistência
validation_data.columns = ["y_val"]
predicted_data.columns = ["Predicted"]

# Calcular métricas de avaliação
mae = mean_absolute_error(
    validation_data["y_val"], predicted_data["Predicted"])
mse = mean_squared_error(validation_data["y_val"], predicted_data["Predicted"])
r2 = r2_score(validation_data["y_val"], predicted_data["Predicted"])

# Calcular erro percentual
validation_data["Predicted"] = predicted_data["Predicted"]
validation_data["Error %"] = (
    (validation_data["Predicted"] -
     validation_data["y_val"]) / validation_data["y_val"]
) * 100

# Filtrar dados para os últimos dois meses
last_date = validation_data.index.max()
two_months_prior = last_date - pd.DateOffset(months=2)

# Encontrar o intervalo válido no DataFrame
validation_period = validation_data[
    (validation_data.index >= two_months_prior) & (
        validation_data.index <= last_date)
]

# Checar se validation_period contém dados
if validation_period.empty:
    raise ValueError("Os dados para os últimos dois meses estão vazios.")

# Inicializar o aplicativo Dash
app = Dash(__name__)
app.title = "Dashboard de Análise de Modelos - Curto Prazo"

# Layout do Dashboard
app.layout = html.Div(style={'backgroundColor': '#1f1f1f'}, children=[
    # Título do Dashboard
    html.H1(
        "Dashboard de Análise de Modelos - Curto Prazo",
        style={'textAlign': 'center', 'color': '#ffffff'}
    ),

    # Gráfico de desempenho do modelo
    dcc.Graph(
        id="model-performance",
        figure={
            "data": [
                go.Scatter(
                    x=validation_period.index, y=validation_period["y_val"],
                    mode="lines+markers", name="Valores Reais",
                    line={"color": "blue"}
                ),
                go.Scatter(
                    x=validation_period.index, y=validation_period["Predicted"],
                    mode="lines+markers", name="Previsões",
                    line={"color": "red"}
                )
            ],
            "layout": go.Layout(
                title="Desempenho do Modelo no Conjunto de Validação",
                xaxis={"title": "Data", "color": "#ffffff"},
                yaxis={"title": "Retorno", "color": "#ffffff"},
                paper_bgcolor="#1f1f1f",
                plot_bgcolor="#1f1f1f",
                font=dict(color="#ffffff")
            )
        }
    ),

    # Gráfico de erros de previsão
    dcc.Graph(
        id="model-errors",
        figure={
            "data": [
                go.Bar(
                    x=validation_period.index,
                    y=np.abs(validation_period["Error %"]),
                    name="Erro (%)",
                    marker={"color": "blue"}
                )
            ],
            "layout": go.Layout(
                title="Erros de Previsão no Conjunto de Validação",
                xaxis={"title": "Data", "color": "#ffffff"},
                yaxis={"title": "Erro (%)", "color": "#ffffff"},
                paper_bgcolor="#1f1f1f",
                plot_bgcolor="#1f1f1f",
                font=dict(color="#ffffff")
            )
        }
    ),

    # Métricas de avaliação
    html.Div([
        html.H2("Métricas de Avaliação", style={'color': '#ffffff'}),
        html.P(f"Erro Médio Absoluto (MAE): {
               mae:.2f}", style={'color': '#ffffff'}),
        html.P(f"Erro Médio Quadrado (MSE): {
               mse:.2f}", style={'color': '#ffffff'}),
        html.P(f"Coeficiente de Determinação (R²): {
               r2:.2f}", style={'color': '#ffffff'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Tabela de comparação para os últimos dois meses
    html.H2(
        "Comparação de Valores Reais e Previstos (Últimos 2 Meses)",
        style={'color': '#ffffff', 'textAlign': 'center'}
    ),
    html.Div([
        html.Table([
            html.Thead([
                html.Tr([html.Th(col, style={'padding': '10px', 'fontSize': '16px'}) for col in [
                    "Data", "Valores Reais", "Valores Previstos", "Erro (%)"]])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(str(date), style={
                            'padding': '10px', 'fontSize': '14px'}),
                    html.Td(f"{real:.5f}", style={
                            'padding': '10px', 'fontSize': '14px'}),
                    html.Td(f"{predicted:.5f}", style={
                            'padding': '10px', 'fontSize': '14px'}),
                    html.Td(f"{error:.2f}%", style={
                            'padding': '10px', 'fontSize': '14px'})
                ])
                for date, real, predicted, error in zip(
                    validation_period.index,
                    validation_period["y_val"],
                    validation_period["Predicted"],
                    validation_period["Error %"]
                )
            ])
        ], style={
            'width': '80%',
            'margin': '0 auto',
            'color': '#ffffff',
            'backgroundColor': '#2e2e2e',
            'borderCollapse': 'collapse'
        })
    ])
])

# Executar o Dashboard
if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
