# BP_mod3_dashboard.py - Refatorado para corrigir o uso de .last e garantir compatibilidade com DatetimeIndex

import pandas as pd
import numpy as np
from dash import Dash, html, dcc
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar dados de validação e previsões
validation_data = pd.read_csv(
    "BackPython/DADOS/y_val.csv", index_col=0, parse_dates=True)
predicted_data = pd.read_csv(
    "BackPython/DADOS/y_pred.csv", index_col=0, parse_dates=True)

# Verificar o nome correto da coluna de valores reais no arquivo y_val.csv
# Pega o primeiro nome de coluna no arquivo y_val.csv
real_values_column = validation_data.columns[0]
# Pega o primeiro nome de coluna no arquivo y_pred.csv
predicted_column = predicted_data.columns[0]

# Renomear para garantir consistência e evitar conflitos
validation_data = validation_data.rename(columns={real_values_column: "y_val"})
predicted_data = predicted_data.rename(columns={predicted_column: "Predicted"})

# Calcular métricas de avaliação
mae = mean_absolute_error(
    validation_data["y_val"], predicted_data["Predicted"])
mse = mean_squared_error(validation_data["y_val"], predicted_data["Predicted"])
r2 = r2_score(validation_data["y_val"], predicted_data["Predicted"])

# Calcular o erro percentual
validation_data['Predicted'] = predicted_data["Predicted"]
validation_data['Error %'] = (
    (validation_data['Predicted'] - validation_data['y_val']) / validation_data['y_val']) * 100

# Selecionar o período de validação para os últimos dois meses
# Obtém a última data do conjunto e usa `.loc` para filtrar
last_date = validation_data.index.max()
two_months_prior = last_date - pd.DateOffset(months=2)
validation_period = validation_data.loc[two_months_prior:last_date]

# Criação do Dashboard
app = Dash(__name__)
app.title = "Dashboard de Análise de Modelos - Curto Prazo"

# Layout do Dashboard
app.layout = html.Div(style={'backgroundColor': '#1f1f1f'}, children=[
    html.H1(
        "Dashboard de Análise de Modelos - Curto Prazo",
        style={'textAlign': 'center', 'color': '#ffffff'}
    ),
    dcc.Graph(
        id="model-performance",
        figure={
            "data": [
                go.Scatter(
                    x=validation_period.index, y=validation_period['y_val'],
                    mode="lines+markers", name="Valores Reais",
                    line={"color": "blue"}
                ),
                go.Scatter(
                    x=validation_period.index, y=validation_period['Predicted'],
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
    dcc.Graph(
        id="model-errors",
        figure={
            "data": [
                go.Bar(
                    x=validation_period.index,
                    y=np.abs(validation_period['Error %']),
                    name="Erro",
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
    html.H2("Comparação de Valores Reais e Previstos (Últimos 2 Meses)",
            style={'color': '#ffffff', 'textAlign': 'center'}),
    html.Div([
        html.Table([
            html.Thead([
                html.Tr([html.Th(col, style={'padding': '10px', 'fontSize': '16px'}) for col in [
                        "Data", "Valores Reais", "Valores Previstos", "Erro %"]])
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
                    validation_period['y_val'],
                    validation_period['Predicted'],
                    validation_period['Error %']
                )
            ])
        ], style={'width': '80%', 'margin': '0 auto', 'color': '#ffffff', 'backgroundColor': '#2e2e2e', 'borderCollapse': 'collapse'})
    ])
])

# Executar o Dashboard
if __name__ == "__main__":
    app.run_server(debug=True)
