'''
    Estrutura do Modelo:

    O modelo configurado tem duas camadas LSTM com 50 unidades cada, seguidas por Dropout para regularização e uma camada Dense final para gerar uma única previsão. Essa é uma boa estrutura para começar, embora possamos experimentar diferentes configurações, como alterar o número de unidades LSTM e a taxa de dropout.
    Compilação e Treinamento:

    O modelo está usando o otimizador adam com mean_squared_error como função de perda. O mse é adequado para regressão, mas podemos também testar com mean_absolute_error (MAE) se quisermos que o modelo foque em minimizar desvios absolutos.
    epochs=20 e batch_size=32 são um ponto de partida, mas podem ser ajustados dependendo do desempenho e da precisão.
    Métricas e Salvamento de Resultados:

    Após a previsão no conjunto de validação, o código calcula o MAE, que é uma boa métrica para verificar o erro médio das previsões.
    Ele salva as previsões (y_pred.csv) e os valores reais (y_val.csv) para que possamos visualizá-los no dashboard.

    Carregar Dados do CSV:

        Nosso pipeline salva os dados em arquivos CSV (X_train.csv, y_train.csv, etc.), então precisamos modificar o código para ler diretamente os arquivos CSV em vez de npy.
        Normalização e Formatação dos Dados:

        O LSTM espera os dados em um formato tridimensional: (samples, timesteps, features). Teremos que garantir que X_train e X_val estejam nesse formato.
        Salvar Modelo Treinado:

        Adicionaremos o código para salvar o modelo LSTM, caso ele tenha um bom desempenho e possa ser reutilizado ou carregado para previsões futuras.
        Salvar Previsões em y_pred.csv para o Dashboard:

        Vamos garantir que as previsões estejam no formato correto para serem integradas ao dashboard de análise.
'''

# BP_mod4_lstm_model.py

# Importar bibliotecas necessárias
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


def load_data():
    """
    Carrega os conjuntos de dados para treino e validação.

    Retorna:
        X_train, y_train, X_val, y_val: Arrays numpy para treinamento e validação.
    """
    try:
        X_train = pd.read_csv(
            "BackPython/DADOS/X_train.csv", index_col=0).values
        y_train = pd.read_csv(
            "BackPython/DADOS/y_train.csv", index_col=0).values
        X_val = pd.read_csv("BackPython/DADOS/X_val.csv", index_col=0).values
        y_val = pd.read_csv("BackPython/DADOS/y_val.csv", index_col=0).values
        return X_train, y_train, X_val, y_val
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Erro ao carregar os dados: {
                e}. Verifique os arquivos de entrada."
        )


def preprocess_data(X_train, X_val):
    """
    Ajusta os dados para o formato esperado pela LSTM.

    Parâmetros:
        X_train: Dados de treino.
        X_val: Dados de validação.

    Retorna:
        X_train_reshaped, X_val_reshaped: Arrays ajustados.
    """
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    return X_train_reshaped, X_val_reshaped


def build_lstm_model(input_shape):
    """
    Configura e retorna o modelo LSTM.

    Parâmetros:
        input_shape: Forma dos dados de entrada (timesteps, features).

    Retorna:
        Modelo LSTM compilado.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Treina e avalia o modelo LSTM.

    Parâmetros:
        model: Modelo LSTM.
        X_train, y_train: Dados de treino.
        X_val, y_val: Dados de validação.

    Retorna:
        y_pred: Previsões no conjunto de validação.
        mae: Erro absoluto médio no conjunto de validação.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_val, y_val), callbacks=[early_stopping])

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return y_pred, mae


def save_results(y_pred, y_val, model):
    """
    Salva as previsões, os valores reais e o modelo treinado.

    Parâmetros:
        y_pred: Previsões.
        y_val: Valores reais.
        model: Modelo treinado.
    """
    os.makedirs("BackPython/DADOS", exist_ok=True)
    os.makedirs("BackPython/MODELS", exist_ok=True)

    # Salvar previsões
    pd.DataFrame(y_pred, columns=['y_pred']).to_csv(
        "BackPython/DADOS/y_pred_lstm.csv", index=False)
    pd.DataFrame(y_val, columns=['y_val']).to_csv(
        "BackPython/DADOS/y_val_lstm.csv", index=False)

    # Salvar o modelo no novo formato recomendado
    model.save("BackPython/MODELS/lstm_model.keras")
    print("Modelo LSTM salvo no formato .keras.")


def main():
    """
    Função principal para executar o treinamento do modelo LSTM.
    """
    try:
        print("Carregando dados...")
        X_train, y_train, X_val, y_val = load_data()

        print("Pré-processando os dados...")
        X_train, X_val = preprocess_data(X_train, X_val)

        print("Construindo o modelo LSTM...")
        model = build_lstm_model(input_shape=(
            X_train.shape[1], X_train.shape[2]))

        print("Treinando o modelo...")
        y_pred, mae = train_and_evaluate_model(
            model, X_train, y_train, X_val, y_val)

        print(f"MAE no conjunto de validação: {mae:.2f}")

        print("Salvando resultados...")
        save_results(y_pred, y_val, model)

    except Exception as e:
        print(f"Erro durante a execução: {e}")


if __name__ == "__main__":
    main()
