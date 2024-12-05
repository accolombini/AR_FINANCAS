import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


def criar_dataset(dados, passos=60):
    X, y = [], []
    for i in range(len(dados) - passos):
        X.append(dados[i:i + passos])
        y.append(dados[i + passos])
    return np.array(X), np.array(y)


def plotar_resultados(valores_reais, previsoes, scaler, datas):
    if len(previsoes) == 0:
        print("[ERROR] Nenhuma previsão foi gerada.")
        return 0, 0, 0

    previsoes_reais = scaler.inverse_transform(previsoes)
    valores_reais = scaler.inverse_transform(valores_reais.reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(datas, valores_reais, label="Valores Reais", color="blue")
    plt.plot(datas, previsoes_reais, label="Previsões", color="red")
    plt.title("Previsões de Curto Prazo com LSTM")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid()
    plt.show()

    mae = mean_absolute_error(valores_reais, previsoes_reais)
    mse = mean_squared_error(valores_reais, previsoes_reais)
    mape = np.mean(
        np.abs((valores_reais - previsoes_reais) / valores_reais)) * 100

    print("[INFO] Métricas de Desempenho:")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")

    return mae, mse, mape


def main():
    # Dados simulados
    np.random.seed(42)
    data = np.cumsum(np.random.randn(1500)) + 50
    datas = pd.date_range(start="2014-12-01", periods=len(data), freq="B")

    df = pd.DataFrame({"Date": datas, "Price": data})

    scaler = MinMaxScaler(feature_range=(0, 1))
    df["Normalized"] = scaler.fit_transform(df[["Price"]])

    passos = 60
    dados_normalizados = df["Normalized"].values
    X, y = criar_dataset(dados_normalizados, passos)

    tamanho_treino = int(0.8 * len(X))
    X_treino, y_treino = X[:tamanho_treino], y[:tamanho_treino]
    X_teste, y_teste = X[tamanho_treino:], y[tamanho_treino:]

    if X_teste.size == 0:
        print("[ERROR] Conjunto de teste está vazio. Verifique a divisão dos dados.")
        return

    X_treino = X_treino.reshape((X_treino.shape[0], X_treino.shape[1], 1))
    X_teste = X_teste.reshape((X_teste.shape[0], X_teste.shape[1], 1))

    # Modelo LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_treino.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_treino, y_treino, epochs=50, batch_size=32, verbose=1)

    previsoes = model.predict(X_teste)

    # Gerar as datas corretamente para as previsões
    datas_teste = pd.date_range(
    start="2024-12-01", periods=len(previsoes), freq="D")


    mae, mse, mape = plotar_resultados(y_teste, previsoes, scaler, datas_teste)


if __name__ == "__main__":
    main()
