# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from dash import Dash, dcc, html
import plotly.graph_objects as go

# Caminhos de entrada
data_path = "BackPython/DADOS/historical_data_cleaned.csv"
portfolio_path = "BackPython/DADOS/portfolio_otimizado.csv"

# Função para diagnosticar problemas com NaN


def diagnose_nan(df, etapa):
    if df.isnull().values.any():
        print(f"Diagnóstico: NaNs encontrados após {etapa}")
        print(df.isnull().sum())
        print(df[df.isnull().any(axis=1)])
        raise ValueError(f"NaNs encontrados após {
                         etapa}. Verifique os cálculos.")


# Carregar dados
historical_data = pd.read_csv(data_path, parse_dates=["Date"])
portfolio_optimized = pd.read_csv(portfolio_path)

# Excluir o índice ˆBVSP para evitar valores fora de escala
if "ˆBVSP" in historical_data.columns:
    historical_data = historical_data.drop(columns=["ˆBVSP"])

if "ˆBVSP" in portfolio_optimized["Ativo"].values:
    portfolio_optimized = portfolio_optimized[portfolio_optimized["Ativo"] != "ˆBVSP"]

# Calcular retornos logarítmicos
for column in historical_data.columns[1:]:
    historical_data[f"{column}_log_return"] = np.log(
        historical_data[column] / historical_data[column].shift(1)
        # Substituir infinitos por valores mínimos
    ).replace([np.inf, -np.inf], 1e-6)

# Remover NaNs resultantes do cálculo de retornos
historical_data = historical_data.dropna()
diagnose_nan(historical_data, "cálculo de retornos logarítmicos")

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
        "Nomes das colunas de retornos e índices de pesos no portfólio não estão alinhados."
    )

historical_data["Portfolio_log_return"] = historical_data[matching_columns].dot(
    aligned_weights.values
)

# Calcular variáveis auxiliares (volatilidade, médias móveis)
historical_data["Volatility"] = historical_data["Portfolio_log_return"].rolling(
    window=30).std().fillna(1e-6)
historical_data["MA_30"] = historical_data["Portfolio_log_return"].rolling(
    window=30).mean().fillna(method="bfill")
historical_data["MA_180"] = historical_data["Portfolio_log_return"].rolling(
    window=180).mean().fillna(method="bfill")

diagnose_nan(historical_data, "cálculo de variáveis auxiliares")

# Selecionar as features para treinamento
required_columns = ["Portfolio_log_return", "Volatility", "MA_30", "MA_180"]
historical_data = historical_data[required_columns]

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
returns_scaled = scaler.fit_transform(historical_data)

# Criar sequências para o LSTM


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # Features
        y.append(data[i + seq_length, 0])  # Target é o retorno
    return np.array(X), np.array(y)


# Usar todo o histórico menos os últimos 180 dias
SEQ_LENGTH = len(returns_scaled) - 180
X, y = create_sequences(returns_scaled, SEQ_LENGTH)

# Dividir os dados
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size +
                 val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Construir e treinar o modelo LSTM
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=100, batch_size=32, verbose=1)

# Fazer previsões
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(
    np.hstack([y_pred_scaled, np.zeros((y_pred_scaled.shape[0], 3))]))[:, 0]
y_test_inverse = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 3))]))[:, 0]

# Verificar NaN antes de métricas
diagnose_nan(pd.DataFrame(
    {"y_test_inverse": y_test_inverse, "y_pred": y_pred}), "pipeline de previsão")

# Calcular métricas
mae = mean_absolute_error(y_test_inverse, y_pred)
rmse = mean_squared_error(y_test_inverse, y_pred, squared=False)
r2 = r2_score(y_test_inverse, y_pred)

print(f"LSTM - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Monte Carlo - Simulações de Retorno
np.random.seed(42)
num_simulations = 1000
mc_simulations = []

for _ in range(num_simulations):
    simulated_returns = []
    last_return = y_pred[-1]
    for _ in range(180):
        simulated_return = last_return + np.random.normal(
            historical_data["Portfolio_log_return"].mean(),
            historical_data["Portfolio_log_return"].std()
        )
        simulated_returns.append(simulated_return)
        last_return = simulated_return
    mc_simulations.append(simulated_returns)

mc_simulations = np.array(mc_simulations)

# Criar datas futuras
future_dates = pd.date_range(
    start=historical_data.index[-1], periods=180, freq="B")

# Criar dashboard com Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Projeções Futuras com LSTM e Monte Carlo"),
    dcc.Graph(
        id="forecast-graph",
        figure={
            "data": [
                go.Scatter(x=historical_data.index[-len(y_test_inverse):], y=y_test_inverse,
                           mode="lines", name="Valores Reais"),
                go.Scatter(x=future_dates, y=mc_simulations.mean(axis=0),
                           mode="lines", name="Monte Carlo - Média"),
                go.Scatter(x=future_dates, y=mc_simulations.min(axis=0),
                           mode="lines", name="Monte Carlo - Cenário Min"),
                go.Scatter(x=future_dates, y=mc_simulations.max(axis=0),
                           mode="lines", name="Monte Carlo - Cenário Max")
            ],
            "layout": go.Layout(
                title="Projeções Futuras com LSTM e Monte Carlo",
                xaxis_title="Data",
                yaxis_title="Retornos",
                legend_title="Curvas"
            )
        }
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
