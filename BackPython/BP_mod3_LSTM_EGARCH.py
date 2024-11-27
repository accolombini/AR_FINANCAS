# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from arch import arch_model
from dash import Dash, dcc, html
import plotly.graph_objects as go

# Caminhos dos dados
data_path = "BackPython/DADOS/historical_data_cleaned.csv"
portfolio_path = "BackPython/DADOS/portfolio_otimizado.csv"

# Carregar dados
historical_data = pd.read_csv(data_path, parse_dates=["Date"])
portfolio_optimized = pd.read_csv(portfolio_path)

# Calcular retornos logarítmicos
for column in historical_data.columns[1:]:
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
returns_series = historical_data.set_index("Date")["Portfolio_log_return"]

# Preparar os dados para treinamento LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
returns_scaled = scaler.fit_transform(returns_series.values.reshape(-1, 1))

# Criar sequências para o modelo LSTM


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


SEQ_LENGTH = 60
X, y = create_sequences(returns_scaled, SEQ_LENGTH)

# Dividir em treino (80%), validação (10%) e teste (10%)
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size +
                 val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Construir e treinar o modelo LSTM
model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=50, batch_size=32, verbose=1)

# Previsões com LSTM
y_pred_lstm_scaled = model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
y_test_inverse = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inverse, y_pred_lstm)
rmse = mean_squared_error(y_test_inverse, y_pred_lstm) ** 0.5
r2 = r2_score(y_test_inverse, y_pred_lstm)

print(f"LSTM - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Calcular resíduos para EGARCH e reescalá-los
residuals = (y_test_inverse.flatten() - y_pred_lstm.flatten()) * \
    100  # Reescalar os resíduos

# Ajustar EGARCH nos resíduos
egarch_model = arch_model(
    residuals,
    vol="EGARCH",
    p=1,
    q=1,
    mean="Zero",
    dist="normal",
    rescale=False
)
egarch_fit = egarch_model.fit(disp="off")
print(egarch_fit.summary())

# Previsão de volatilidade iterativa para horizonte de 180 dias
forecast_horizon = 180
volatility_forecast = []

last_residual = residuals[-1]  # Usar o último resíduo como base
for _ in range(forecast_horizon):
    forecast = egarch_fit.forecast(horizon=1, reindex=False)
    next_variance = forecast.variance.values[-1, 0]
    # Raiz quadrada da variância
    volatility_forecast.append(np.sqrt(next_variance))

# Converter lista de volatilidades para array
volatility_forecast = np.array(volatility_forecast)

# Combinar previsões de retornos com volatilidade
future_predictions = y_pred_lstm.flatten(
)[:forecast_horizon] + volatility_forecast

# Criar datas futuras para previsões
future_dates = pd.date_range(
    start=returns_series.index[-1],
    periods=forecast_horizon,
    freq="B"
)

# Dashboard com Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Previsões com LSTM + EGARCH - Refatorado"),
    dcc.Graph(
        id="lstm-egarch-forecast-graph",
        figure={
            "data": [
                go.Scatter(x=returns_series.index[-len(y_test_inverse):], y=y_test_inverse.flatten(),
                           mode="lines", name="Valores Reais"),
                go.Scatter(x=returns_series.index[-len(y_pred_lstm):], y=y_pred_lstm.flatten(),
                           mode="lines", name="Previsões LSTM"),
                go.Scatter(x=future_dates, y=future_predictions, mode="lines",
                           name="Previsões Futuras (LSTM + EGARCH)", line=dict(color="red"))
            ],
            "layout": go.Layout(
                title="Previsões de Retornos com LSTM + EGARCH - Refatorado",
                xaxis_title="Data",
                yaxis_title="Retorno Logarítmico",
                legend_title="Curvas"
            )
        }
    ),
    html.Div([
        html.H3("Métricas do Modelo"),
        html.Ul([
            html.Li(f"LSTM - MAE: {mae:.4f}"),
            html.Li(f"LSTM - RMSE: {rmse:.4f}"),
            html.Li(f"LSTM - R²: {r2:.4f}")
        ])
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)
