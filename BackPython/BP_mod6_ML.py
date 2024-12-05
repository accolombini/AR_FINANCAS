# BP_mod6_ML.py: Script Final de Machine Learning para Previsão de Retornos
# -----------------------------------------------------------
# Este script utiliza os dados consolidados do portfólio para treinar modelos
# de machine learning, validar a precisão e extrapolar previsões para o futuro.
# -----------------------------------------------------------

import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from BP_mod1_config import OUTPUT_DIR
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados(caminho):
    """Carrega os dados consolidados."""
    try:
        logging.info(f"Carregando dados de: {caminho}")
        dados = pd.read_csv(caminho, index_col="Date", parse_dates=True)
        logging.info(f"Período de dados: {
                     dados.index.min()} a {dados.index.max()}")
        logging.info(f"Número de linhas e colunas: {dados.shape}")
        return dados
    except Exception as e:
        logging.error(f"Erro ao carregar os dados: {e}")
        raise


def preparar_dados(dados):
    """Prepara os dados para o modelo de machine learning."""
    logging.info("Preparando os dados para treinamento...")
    # Calcular retornos futuros como target
    dados["Retorno_Futuro_6M"] = dados["VALE3.SA"].pct_change(
        periods=126).shift(-126)
    dados["Retorno_Futuro_5Y"] = dados["VALE3.SA"].pct_change(
        periods=1260).shift(-1260)

    # Features financeiras
    dados["Retorno_Passado_1M"] = dados["VALE3.SA"].pct_change(periods=21)
    dados["Volatilidade_1M"] = dados["VALE3.SA"].rolling(window=21).std()

    # Remover linhas com NaN
    dados = dados.dropna()
    logging.info(f"Número de linhas após preparação: {dados.shape[0]}")
    return dados


def dividir_dados(dados):
    """Divide os dados em treino, teste e validação."""
    logging.info("Dividindo os dados em treino, teste e validação...")
    dados_treino_teste = dados.iloc[:-60]  # Treinamento e teste
    dados_validacao = dados.iloc[-60:]    # Validação (últimos 60 dias)

    # Separar features e targets
    X = dados_treino_teste[["Retorno_Passado_1M", "Volatilidade_1M"]]
    y_6M = dados_treino_teste["Retorno_Futuro_6M"]
    y_5Y = dados_treino_teste["Retorno_Futuro_5Y"]
    X_val = dados_validacao[["Retorno_Passado_1M", "Volatilidade_1M"]]
    y_val_6M = dados_validacao["Retorno_Futuro_6M"]
    y_val_5Y = dados_validacao["Retorno_Futuro_5Y"]

    X_train, X_test, y_train_6M, y_test_6M = train_test_split(
        X, y_6M, test_size=0.2, random_state=42)
    _, _, y_train_5Y, y_test_5Y = train_test_split(
        X, y_5Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train_6M, y_test_6M, y_train_5Y, y_test_5Y, X_val, y_val_6M, y_val_5Y


def treinar_modelo(X_train, y_train, modelo="xgboost"):
    """Treina um modelo de Machine Learning."""
    logging.info(f"Treinando modelo: {modelo}")
    if modelo == "xgboost":
        regressor = XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42)
    elif modelo == "random_forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Modelo desconhecido: {modelo}")

    regressor.fit(X_train, y_train)
    logging.info(f"Modelo treinado com {len(X_train)} amostras")
    return regressor


def avaliar_modelo(modelo, X_test, y_test, X_val, y_val, periodo):
    """Avalia o desempenho do modelo."""
    logging.info(f"Avaliando o modelo para o período de {periodo}...")
    y_pred_test = modelo.predict(X_test)
    y_pred_val = modelo.predict(X_val)

    # Avaliação em teste
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    logging.info(
        f"Teste - RMSE: {rmse_test}, MAPE: {mape_test}, R²: {r2_test}")

    # Avaliação em validação
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    logging.info(
        f"Validação - RMSE: {rmse_val}, MAPE: {mape_val}, R²: {r2_val}")

    return rmse_val, mape_val, r2_val


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    # Início do processo
    inicio_execucao = datetime.now()
    logging.info(f"Início da análise: {inicio_execucao}")

    # Caminhos dos arquivos
    caminho_dados = os.path.join(
        OUTPUT_DIR, "dados_consolidados_proporcional.csv")

    # Carregar dados
    dados = carregar_dados(caminho_dados)

    # Preparar dados
    dados_preparados = preparar_dados(dados)

    # Dividir dados
    X_train, X_test, y_train_6M, y_test_6M, y_train_5Y, y_test_5Y, X_val, y_val_6M, y_val_5Y = dividir_dados(
        dados_preparados)

    # Treinar modelos
    modelo_6M = treinar_modelo(X_train, y_train_6M, modelo="xgboost")
    modelo_5Y = treinar_modelo(X_train, y_train_5Y, modelo="random_forest")

    # Avaliar modelos
    avaliar_modelo(modelo_6M, X_test, y_test_6M, X_val, y_val_6M, "6 meses")
    avaliar_modelo(modelo_5Y, X_test, y_test_5Y, X_val, y_val_5Y, "5 anos")

    # Fim do processo
    fim_execucao = datetime.now()
    logging.info(f"Término da análise: {fim_execucao}")
    logging.info(f"Duração total: {fim_execucao - inicio_execucao}")


if __name__ == "__main__":
    main()
