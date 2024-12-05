import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Dados simulados (reescala para evitar problemas de convergência)
np.random.seed(42)
data = np.cumsum(np.random.randn(2000)) * 100 + 5000
dates = pd.date_range(start="2014-12-01", periods=len(data), freq="B")
df = pd.DataFrame({"Date": dates, "Price": data})

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
df["Normalized"] = scaler.fit_transform(df[["Price"]])

# Função para criar dataset para CNN-LSTM


def criar_dataset_hibrido(dados, passos=60):
    X, y = [], []
    for i in range(len(dados) - passos):
        X.append(dados[i: i + passos])
        y.append(dados[i + passos])
    return np.array(X), np.array(y)


# Preparar o dataset
passos = 60
dados = df["Normalized"].values
X, y = criar_dataset_hibrido(dados, passos)

# Divisão treino e teste
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Ajustar formato para CNN-LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Construção do modelo CNN-LSTM com Dropout para regularização
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=False),
    Dropout(0.3),  # Regularização
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Treinamento do modelo
history = model.fit(X_train_reshaped, y_train,
                    epochs=50, batch_size=32, verbose=1)

# Previsão: Ajustar o horizonte futuro de dezembro/2024 a dezembro/2029
future_steps = 1250  # Aproximadamente 5 anos úteis
last_sequence = dados[-passos:]  # Última sequência conhecida
future_predictions = []

for _ in range(future_steps):
    input_sequence = np.reshape(last_sequence, (1, passos, 1))
    next_prediction = model.predict(input_sequence)[0, 0]
    future_predictions.append(next_prediction)
    last_sequence = np.append(last_sequence[1:], next_prediction)

# Reverter a normalização
future_predictions_inverse = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1))

# Criar as datas futuras
future_dates = pd.date_range(
    start="2024-12-01", periods=future_steps, freq="B")

# Gráfico atualizado
plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_predictions_inverse,
         label="Previsões Longo Prazo (CNN-LSTM)", color="red")
plt.title("Previsões de Longo Prazo com CNN-LSTM (Corrigido)")
plt.xlabel("Data")
plt.ylabel("Preço")
plt.legend()
plt.grid()
plt.show()
