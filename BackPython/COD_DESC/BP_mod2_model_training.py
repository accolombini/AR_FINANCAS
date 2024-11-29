# BP_mod2_model_training.py
# Treinamento de modelos de curto prazo usando Random Forest

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os


def build_lstm_model(input_shape):
    """
    Constrói o modelo LSTM com Dropout para regularização.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # Uso explícito de Input
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_and_evaluate_lstm(data_dir='BackPython/DADOS/', model_output='BackPython/DADOS/lstm_model.keras'):
    """
    Treina e avalia o modelo LSTM, retornando métricas sem imprimir diretamente.
    """
    print("[INFO] Carregando os dados pré-processados...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    # Construir o modelo
    print("[INFO] Construindo o modelo LSTM...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape)

    # Treinar o modelo
    print("[INFO] Iniciando o treinamento...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # Avaliar no conjunto de teste
    print("[INFO] Avaliando o modelo no conjunto de teste...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    # Calcular métricas descritivas
    y_mean = np.mean(y_test)
    mse_percentage = (test_loss / y_mean) * 100
    mae_percentage = (test_mae / y_mean) * 100
    metrics = {
        "Erro Quadrático Médio (MSE)": f"{mse_percentage:.2f}% - Mede a variância dos erros.",
        "Erro Absoluto Médio (MAE)": f"{mae_percentage:.2f}% - Mede o desvio médio entre previsto e real.",
        "MAE como Percentual da Média": f"{mae_percentage:.2f}% - Relaciona o erro absoluto à média dos valores reais."
    }

    # Salvar o modelo no formato recomendado
    print(f"[INFO] Salvando o modelo treinado em: {model_output}")
    model.save(model_output)

    # Retornar as métricas para o pipeline
    return metrics
