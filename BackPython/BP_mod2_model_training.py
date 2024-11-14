# BP_mod2_model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuração de parâmetros
DATA_DIR = "BackPython/DADOS/"
MODEL_DIR = "BackPython/MODELS/"
MODEL_NAME = "short_term_rf_model.pkl"
PREDICTIONS_FILE = f"{DATA_DIR}/y_pred.csv"


def load_data():
    """
    Carrega os conjuntos de dados para treino, validação e teste.
    """
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv", index_col=0)
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv", index_col=0)
    X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv", index_col=0)
    y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv", index_col=0)

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    """
    Treina um modelo de Random Forest para o curto prazo e valida-o.
    """
    print("Iniciando o treinamento do modelo de curto prazo...")

    # Instancia o modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Treinamento do modelo
    model.fit(X_train, y_train.values.ravel())

    # Predição no conjunto de validação
    y_pred_val = model.predict(X_val)

    # Avaliação do modelo
    mse = mean_squared_error(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"Erro Médio Quadrado (MSE) na validação: {mse}")
    print(f"Erro Médio Absoluto (MAE) na validação: {mae}")
    print(f"R² na validação: {r2}")

    # Salvando o modelo treinado
    joblib.dump(model, f"{MODEL_DIR}/{MODEL_NAME}")
    print(f"Modelo salvo em {MODEL_DIR}/{MODEL_NAME}")

    # Salvando as previsões no conjunto de validação para uso no dashboard
    y_pred_df = pd.DataFrame(y_pred_val, columns=[
                             "Predicted"], index=y_val.index)
    y_pred_df.to_csv(PREDICTIONS_FILE)
    print(f"Previsões do conjunto de validação salvas em {PREDICTIONS_FILE}")


def main():
    """
    Função principal para o treinamento do modelo de curto prazo.
    """
    # Carrega os dados
    X_train, y_train, X_val, y_val = load_data()

    # Treina e valida o modelo
    train_model(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
