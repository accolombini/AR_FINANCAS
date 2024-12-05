# BP_mod2_forecast_pipeline.py: Pipeline de Previsão com LSTM
# -------------------------------------------------------------------------
# Este script realiza a divisão dos dados, configuração do modelo,
# treinamento, teste, validação e previsões futuras.
# -------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from BP_mod1_config import OUTPUT_DIR, HISTORICAL_DATA_PATH

# Verificar o caminho correto do arquivo
file_path = os.path.join(OUTPUT_DIR, "historical_data_portfolio.csv")

# Verificar se o arquivo correto existe
if not os.path.isfile(file_path):
    raise FileNotFoundError(
        f"[ERRO] Arquivo histórico não encontrado: {file_path}")

print(f"[INFO] Carregando dados históricos de: {file_path}")

# Carregar os dados históricos
df = pd.read_csv(file_path, parse_dates=["Date"])

# Verificar a integridade da base
assert "Portfólio" in df.columns, "[ERRO] Coluna 'Portfólio' não encontrada na base de dados!"
assert df["Date"].is_monotonic_increasing, "[ERRO] As datas não estão ordenadas corretamente!"

print("[INFO] Dados carregados e validados com sucesso.")

# Normalizar a coluna do portfólio
scaler = MinMaxScaler(feature_range=(0, 1))
df["Portfólio"] = scaler.fit_transform(df["Portfólio"].values.reshape(-1, 1))

# Determinar os períodos para treinamento, teste e validação
validation_start_date = df["Date"].iloc[-1] - pd.DateOffset(months=2)
val_start_idx = df[df["Date"] >= validation_start_date].index[0]

train_test_data = df.iloc[:val_start_idx]
train_size = int(len(train_test_data) * 0.8)
train_data = train_test_data.iloc[:train_size]
test_data = train_test_data.iloc[train_size:]
val_data = df.iloc[val_start_idx:]

print("[INFO] Dados divididos:")
print(f"Treinamento: {len(train_data)} registros")
print(f"Teste: {len(test_data)} registros")
print(f"Validação: {len(val_data)} registros")

# Ajustar o look_back dinamicamente
look_back = 60
look_back = min(look_back, len(val_data) - 1)
if look_back < 1:
    raise ValueError(
        "[ERRO] Dados insuficientes para criar sequências com look_back definido!")

print(f"[INFO] Look-back ajustado para: {look_back}")

# Função para criar sequências de dados


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Criar sequências para treinamento, teste e validação
X_train, y_train = create_sequences(
    train_data["Portfólio"].values.reshape(-1, 1), look_back)
X_test, y_test = create_sequences(
    test_data["Portfólio"].values.reshape(-1, 1), look_back)
X_val, y_val = create_sequences(
    val_data["Portfólio"].values.reshape(-1, 1), look_back)

if X_train.size == 0 or X_test.size == 0 or X_val.size == 0:
    raise ValueError("[ERRO] Dados insuficientes após criação das sequências.")

print(f"[INFO] Sequências criadas com sucesso:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"X_val: {X_val.shape}")

# Remodelar para o formato [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Construir o modelo LSTM
model = Sequential([
    Input(shape=(look_back, 1)),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
print("[INFO] Modelo LSTM criado e compilado.")

# Treinamento do modelo
model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))

# Fazer previsões
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
val_predict = model.predict(X_val)

# Reverter a normalização
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
val_predict = scaler.inverse_transform(val_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

# Plotar os resultados
plt.figure(figsize=(14, 7))

# Treinamento
plt.plot(df["Date"][look_back:look_back + len(y_train)],
         y_train, label="Treinamento", color="blue")
plt.plot(df["Date"][look_back:look_back + len(train_predict)],
         train_predict, linestyle="--", label="Previsão Treinamento", color="cyan")

# Teste
plt.plot(df["Date"][train_size + look_back:train_size +
         look_back + len(y_test)], y_test, label="Teste", color="orange")
plt.plot(df["Date"][train_size + look_back:train_size + look_back + len(test_predict)],
         test_predict, linestyle="--", label="Previsão Teste", color="gold")

# Validação
plt.plot(df["Date"][val_start_idx + look_back:],
         y_val, label="Validação", color="green")
plt.plot(df["Date"][val_start_idx + look_back:], val_predict,
         linestyle="--", label="Previsão Validação", color="lime")

plt.xlabel("Data")
plt.ylabel("Preço")
plt.title("Desempenho do Modelo com Validação")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "validation_results_portfolio_v2.png"))
plt.show()
