# BP_mod6_egarch_lstm_dashboard.py
# -----------------------------------------------------------
# Este módulo utiliza EGARCH para volatilidade e LSTM para retornos do portfólio
# com visualizações no Dashboard incluindo projeções mês a mês.
# -----------------------------------------------------------

# BP_mod6_lstm_egarch_final_dashboard.py
# -----------------------------------------------------------
# Modelo consolidado: LSTM para previsões de retornos e EGARCH para volatilidade.
# -----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from arch import arch_model
import plotly.graph_objects as go
from dash import Dash, dcc, html

# Caminhos dos dados
data_path = "BackPython/DADOS/historical_data_cleaned.csv"
portfolio_path = "BackPython/DADOS/portfolio_otimizado.csv"

# Configurações
SEQ_LENGTH = 120
EPOCHS = 100
FORECAST_HORIZON = 180

# 1. Carregar dados e calcular retornos logarítmicos
historical_data = pd.read_csv(data_path, parse_dates=["Date"])
portfolio_optimized = pd.read_csv(portfolio_path)

for column in historical_data.columns[1:]:
    historical_data[f"{column}_log_return"] = np.log(
        historical_data[column] / historical_data[column].shift(1))
historical_data = historical_data.dropna()

portfolio_weights = portfolio_optimized.set_index("Ativo")["Peso (%)"] / 100.0
aligned_weights = portfolio_weights.reindex(
    [col.split("_log_return")[0]
     for col in historical_data.columns if "_log_return" in col],
    fill_value=0
)
matching_columns = [f"{asset}_log_return" for asset in aligned_weights.index]
historical_data["Portfolio_log_return"] = historical_data[matching_columns].dot(
    aligned_weights.values)
returns_series = historical_data.set_index("Date")["Portfolio_log_return"]

# 2. Preparar dados para LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
returns_scaled = scaler.fit_transform(returns_series.values.reshape(-1, 1))


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


X, y = create_sequences(returns_scaled, SEQ_LENGTH)

train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 3. Construir e treinar o modelo LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=1)

# Previsões com LSTM
y_pred_lstm_scaled = model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
y_test_inverse = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inverse, y_pred_lstm)
rmse = mean_squared_error(y_test_inverse, y_pred_lstm) ** 0.5
r2 = r2_score(y_test_inverse, y_pred_lstm)

# 4. Calcular resíduos e ajustar EGARCH
residuals = (y_test_inverse.flatten() - y_pred_lstm.flatten()) * 100
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

# Previsão de volatilidade com EGARCH
volatility_forecast = []
for _ in range(FORECAST_HORIZON):
    forecast = egarch_fit.forecast(horizon=1, reindex=False)
    next_variance = forecast.variance.values[-1, 0]
    volatility_forecast.append(np.sqrt(next_variance))

# 5. Combinar previsões LSTM e EGARCH
future_predictions = y_pred_lstm.flatten(
)[:FORECAST_HORIZON] + volatility_forecast
future_dates = pd.date_range(
    start=returns_series.index[-1],
    periods=FORECAST_HORIZON,
    freq="B"
)

# Calcular métricas e indicadores
indicadores = pd.DataFrame({
    "Indicador": ["Retorno Médio", "Volatilidade Média", "Retorno Máximo", "Retorno Mínimo"],
    "Valor (%)": [
        np.mean(future_predictions) * 100,
        np.mean(volatility_forecast) * 100,
        np.max(future_predictions) * 100,
        np.min(future_predictions) * 100
    ]
})

# 6. Dashboard interativo
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Projeção de Retornos e Volatilidade com LSTM + EGARCH"),
    dcc.Graph(
        id="forecast-graph",
        figure={
            "data": [
                go.Scatter(x=future_dates, y=future_predictions,
                           mode="lines", name="Retornos Projetados"),
                go.Scatter(x=future_dates, y=volatility_forecast,
                           mode="lines", name="Volatilidade Projetada")
            ],
            "layout": go.Layout(
                title="Projeções Futuras (LSTM + EGARCH)",
                xaxis_title="Data",
                yaxis_title="Valores (%)"
            )
        }
    ),
    html.Table([
        html.Tr([html.Th(col) for col in indicadores.columns]),
        *[html.Tr([html.Td(row[col]) for col in indicadores.columns]) for _, row in indicadores.iterrows()]
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)
