# BP_mod2_model_training.py
# Módulo para treinamento de um modelo de Random Forest com dados financeiros de curto prazo

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuração de diretórios e nomes de arquivos
DATA_DIR = "BackPython/DADOS/"
MODEL_DIR = "BackPython/MODELS/"
# Nome do arquivo onde o modelo será salvo
MODEL_NAME = "short_term_rf_model.pkl"
# Arquivo para salvar previsões de validação
PREDICTIONS_FILE = f"{DATA_DIR}/y_pred_rf.csv"


def load_data():
    """
    Carrega os conjuntos de dados para treino e validação a partir dos arquivos CSV.

    Retorna:
        - X_train, y_train: Features e target para o treinamento
        - X_val, y_val: Features e target para a validação
    """
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv", index_col=0)
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv", index_col=0)
    X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv", index_col=0)
    y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv", index_col=0)

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    """
    Treina um modelo de Random Forest e avalia seu desempenho no conjunto de validação.

    Parâmetros:
        - X_train, y_train: Dados de treino
        - X_val, y_val: Dados de validação

    Salva:
        - O modelo treinado em um arquivo
        - As previsões do conjunto de validação em um CSV para análise no dashboard
    """
    print("Iniciando o treinamento do modelo de Random Forest...")

    # Instancia e treina o modelo de Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Realiza previsões no conjunto de validação
    y_pred_val = model.predict(X_val)

    # Calcula métricas de desempenho no conjunto de validação
    mse = mean_squared_error(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    # Exibe as métricas para avaliação do modelo
    print(f"Erro Médio Quadrado (MSE) na validação: {mse}")
    print(f"Erro Médio Absoluto (MAE) na validação: {mae}")
    print(f"Coeficiente de Determinação (R²) na validação: {r2}")

    # Salva o modelo treinado para uso posterior
    joblib.dump(model, f"{MODEL_DIR}/{MODEL_NAME}")
    print(f"Modelo salvo em {MODEL_DIR}/{MODEL_NAME}")

    # Salva as previsões do conjunto de validação em um arquivo CSV
    y_pred_df = pd.DataFrame(y_pred_val, columns=[
                             "Predicted"], index=y_val.index)
    y_pred_df.to_csv(PREDICTIONS_FILE)
    print(f"Previsões do conjunto de validação salvas em {PREDICTIONS_FILE}")


def main():
    """
    Função principal para execução do treinamento do modelo.
    """
    # Carrega os dados de treino e validação
    X_train, y_train, X_val, y_val = load_data()

    # Executa o treinamento e validação do modelo
    train_model(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
