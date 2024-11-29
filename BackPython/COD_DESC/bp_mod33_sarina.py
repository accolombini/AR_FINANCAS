import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dash import Dash, dcc, html
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Carregar dados
data_path = "BackPython/DADOS/historical_data_cleaned.csv"
portfolio_path = "BackPython/DADOS/portfolio_otimizado.csv"
historical_data = pd.read_csv(data_path, parse_dates=["Date"])
portfolio_optimized = pd.read_csv(portfolio_path)

# Configurar frequência explícita para os dados temporais
historical_data.set_index("Date", inplace=True)
historical_data = historical_data.asfreq(
    "B")  # Definir frequência de dias úteis

# Calcular retornos logarítmicos
for column in historical_data.columns:
    historical_data[f"{column}_log_return"] = np.log(
        historical_data[column] / historical_data[column].shift(1))
historical_data = historical_data.dropna()

# Calcular retorno ponderado do portfólio
portfolio_weights = portfolio_optimized.set_index("Ativo")["Peso (%)"] / 100.0
aligned_weights = portfolio_weights.reindex(
    [col.split("_log_return")[0]
     for col in historical_data.columns if "_log_return" in col],
    fill_value=0
)
matching_columns = [f"{asset}_log_return" for asset in aligned_weights.index]
aligned_weights = aligned_weights.loc[aligned_weights.index.intersection(
    [col.split("_log_return")[0] for col in matching_columns])]

if len(matching_columns) != len(aligned_weights):
    raise ValueError(
        "Nomes das colunas de retornos e índices de pesos no portfólio não estão alinhados.")

historical_data["Portfolio_log_return"] = historical_data[matching_columns].dot(
    aligned_weights.values)

# Selecionar série temporal
returns_series = historical_data["Portfolio_log_return"]

# Dividir os dados em treino e teste
split_date = "2023-01-01"
train_series = returns_series[:split_date]
test_series = returns_series[split_date:]

# Ajustar o modelo SARIMA usando auto_arima com mais iterações para explorar sazonalidade maior
auto_model = auto_arima(
    train_series,
    seasonal=True,
    m=90,  # Sazonalidade trimestral para capturar padrões maiores
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
    max_order=None,  # Permitir maior exploração de parâmetros
    n_jobs=-1
)

print(f"Parâmetros selecionados: {
      auto_model.order} e sazonalidade {auto_model.seasonal_order}")

# Treinar modelo SARIMA com os parâmetros encontrados
sarima_model = SARIMAX(
    train_series,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)

# Fazer previsões para o conjunto de teste
test_predictions = sarima_fit.forecast(steps=len(test_series))
mae = mean_absolute_error(test_series, test_predictions)
rmse = mean_squared_error(test_series, test_predictions, squared=False)
r2 = r2_score(test_series, test_predictions)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Previsões futuras para 180 dias úteis
future_predictions = sarima_fit.forecast(steps=180)
future_dates = pd.date_range(
    start=test_series.index[-1], periods=180, freq="B")

# Dashboard com Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Projeção de Retornos com SARIMA - Refatorado"),
    dcc.Graph(
        id="sarima-forecast-graph",
        figure={
            "data": [
                go.Scatter(x=train_series.index, y=train_series,
                           mode="lines", name="Treinamento"),
                go.Scatter(x=test_series.index, y=test_series,
                           mode="lines", name="Teste"),
                go.Scatter(x=test_series.index, y=test_predictions,
                           mode="lines", name="Previsão SARIMA (Teste)"),
                go.Scatter(x=future_dates, y=future_predictions, mode="lines",
                           name="Previsão SARIMA (Futuro)", line=dict(color="red"))
            ],
            "layout": go.Layout(
                title="Previsão de Retornos Futuros com SARIMA - Refatorado",
                xaxis_title="Data",
                yaxis_title="Retorno Logarítmico",
                legend_title="Curvas"
            )
        }
    ),
    html.Div([
        html.H3("Métricas do Modelo"),
        html.Ul([
            html.Li(f"MAE: {mae:.4f}"),
            html.Li(f"RMSE: {rmse:.4f}"),
            html.Li(f"R²: {r2:.4f}")
        ])
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)
