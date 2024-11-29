import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from dash import Dash, dcc, html
import plotly.graph_objects as go

# Carregar dados
data_path = "BackPython/DADOS/historical_data_cleaned.csv"
portfolio_path = "BackPython/DADOS/portfolio_otimizado.csv"
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

# Preparar os dados para treinamento
portfolio_returns = historical_data["Portfolio_log_return"].values.reshape(
    -1, 1)

# Escalar os dados para a faixa [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
portfolio_scaled = scaler.fit_transform(portfolio_returns)

# Criar sequências para o modelo LSTM


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


SEQ_LENGTH = 60  # Usar os últimos 60 dias para prever o próximo
X, y = create_sequences(portfolio_scaled, SEQ_LENGTH)

# Dividir em treino (80%), validação (10%) e teste (10%)
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size +
                 val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Construir o modelo LSTM com mais complexidade
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])

# Usar RMSProp como otimizador
model.compile(optimizer="rmsprop", loss="mean_squared_error")
early_stop = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True)

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Avaliar o modelo
y_pred = model.predict(X_test)
y_pred_inverse = scaler.inverse_transform(y_pred)
y_test_inverse = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
rmse = mean_squared_error(y_test_inverse, y_pred_inverse, squared=False)
r2 = r2_score(y_test_inverse, y_pred_inverse)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Fazer previsões para os próximos 180 dias úteis
last_sequence = portfolio_scaled[-SEQ_LENGTH:]
future_predictions = []
for _ in range(180):
    next_prediction = model.predict(last_sequence.reshape(1, SEQ_LENGTH, 1))[0]
    future_predictions.append(next_prediction)
    # Atualizar a sequência com o valor previsto
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_prediction

# Reverter o escalonamento das previsões
future_predictions = scaler.inverse_transform(future_predictions).flatten()

# Criar datas futuras para as previsões
future_dates = pd.date_range(
    start=historical_data["Date"].iloc[-1], periods=180, freq="B")

# Dashboard com Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Projeção de Retornos com LSTM - Ajustado"),
    dcc.Graph(
        id="lstm-forecast-graph",
        figure={
            "data": [
                go.Scatter(x=historical_data["Date"], y=historical_data["Portfolio_log_return"],
                           mode="lines", name="Histórico"),
                go.Scatter(x=future_dates, y=future_predictions,
                           mode="lines", name="Previsão LSTM", line=dict(color="red"))
            ],
            "layout": go.Layout(
                title="Previsão de Retornos Futuros (180 Dias)",
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
