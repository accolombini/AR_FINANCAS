# BP_mod7_train_and_project_refactored_v3.py: Treinar Modelo com LSTM e Realizar Projeções
# -----------------------------------------------------------
# Este script refatorado utiliza LSTM para prever o comportamento do portfólio
# e realiza projeções para 6 meses e 5 anos com correções adicionais.
# -----------------------------------------------------------

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input
from BP_mod1_config import OUTPUT_DIR
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados(caminho):
    """Carrega os dados consolidados com o portfólio ótimo."""
    try:
        logging.info(f"Carregando dados consolidados de: {caminho}")
        dados = pd.read_csv(caminho, index_col="Date", parse_dates=True)
        logging.info(
            f"Número de linhas e colunas (dados consolidados): {dados.shape}")
        return dados
    except Exception as e:
        logging.error(f"Erro ao carregar os dados consolidados: {e}")
        raise


def preparar_dados_lstm(dados, look_back=21):
    """Prepara os dados para entrada no modelo LSTM."""
    logging.info("Preparando os dados para o modelo LSTM...")

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_scaled = scaler.fit_transform(dados[["portfolio_otimo"]])

    # Criar janelas de dados
    X, y = [], []
    for i in range(len(dados_scaled) - look_back):
        X.append(dados_scaled[i:i + look_back, 0])
        y.append(dados_scaled[i + look_back, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape para 3D

    logging.info(f"Formato dos dados para LSTM - X: {X.shape}, y: {y.shape}")
    return X, y, scaler


def criar_modelo_lstm(input_shape):
    """Cria o modelo LSTM."""
    logging.info("Criando o modelo LSTM...")
    input_layer = Input(shape=input_shape)
    lstm1 = LSTM(units=50, return_sequences=True)(input_layer)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(units=50, return_sequences=False)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    output_layer = Dense(units=1)(dropout2)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")
    logging.info("Modelo LSTM criado com sucesso.")
    return model


def realizar_projecoes(model, dados_recentes, scaler, steps=126):
    """Realiza projeções futuras com base no modelo treinado."""
    logging.info(f"Realizando projeções para os próximos {steps} períodos...")
    previsoes = []
    dados_input = dados_recentes.flatten()  # Garantir que seja 1D

    for _ in range(steps):
        # Pega os últimos 21 valores
        entrada = dados_input[-21:].reshape(1, 21, 1)
        predicao = model.predict(entrada)
        previsoes.append(predicao[0, 0])
        # Atualiza com a previsão
        dados_input = np.append(dados_input, predicao[0, 0])

    previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1))
    logging.info("Projeções futuras concluídas.")
    return previsoes.flatten()


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    # Início do processo
    inicio_execucao = datetime.now()
    logging.info(f"Início da análise: {inicio_execucao}")

    # Caminhos dos arquivos
    caminho_dados = os.path.join(OUTPUT_DIR, "portfolio_comportamento.csv")

    # Carregar e preparar dados
    dados = carregar_dados(caminho_dados)
    X, y, scaler = preparar_dados_lstm(dados)

    # Dividir em treino e teste
    tamanho_treino = int(len(X) * 0.8)
    X_train, X_test = X[:tamanho_treino], X[tamanho_treino:]
    y_train, y_test = y[:tamanho_treino], y[tamanho_treino:]

    # Criar e treinar o modelo LSTM
    modelo = criar_modelo_lstm((X_train.shape[1], 1))
    modelo.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Avaliar o modelo
    logging.info("Avaliando o modelo...")
    resultados = modelo.evaluate(X_test, y_test)
    logging.info(f"Erro médio quadrático no teste: {resultados}")

    # Projeções futuras
    previsoes_6m = realizar_projecoes(modelo, X_test[-1], scaler, steps=126)
    previsoes_5y = realizar_projecoes(modelo, X_test[-1], scaler, steps=1260)

    # Ajustar comprimentos para criar DataFrame
    tamanho_minimo = min(len(previsoes_6m), len(previsoes_5y))
    previsoes_df = pd.DataFrame({
        "6M_Projecoes": previsoes_6m[:tamanho_minimo],
        "5Y_Projecoes": previsoes_5y[:tamanho_minimo]
    })

    # Salvar projeções
    caminho_saida = os.path.join(OUTPUT_DIR, "projecoes_portfolio_lstm_v3.csv")
    previsoes_df.to_csv(caminho_saida, index=False)
    logging.info(f"Projeções futuras salvas em: {caminho_saida}")

    # Fim do processo
    fim_execucao = datetime.now()
    logging.info(f"Término da análise: {fim_execucao}")
    logging.info(f"Duração total: {fim_execucao - inicio_execucao}")


if __name__ == "__main__":
    main()
