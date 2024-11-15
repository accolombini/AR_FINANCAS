# BP_mod2_model_training.py
# Gera modelos para análise utilizando o Algoritmo Random Forest

# Importar bibliotecas necessárias

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Configuração de diretórios e arquivos
DATA_DIR = "BackPython/DADOS/"
MODEL_NAME = "X_random_forest.csv"
PREDICTIONS_FILE = f"{DATA_DIR}/y_random_forest.csv"


def load_and_prepare_data():
    """
    Carrega e prepara os dados para o modelo Random Forest.
    Divide os dados em treino, validação e teste.
    """
    # Carregar o arquivo gerado pelo pipeline 1
    file_path = f"{DATA_DIR}/asset_data_cleaner.csv"
    print(f"Carregando dados de: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Definir a coluna de destino (^BVSP) e as features
    target_column = '^BVSP'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Dividir os dados em treino, validação e teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    """
    Treina um modelo de Random Forest para o curto prazo e valida-o.
    """
    print("Iniciando o treinamento do modelo de curto prazo...")

    # Instanciar o modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Treinamento do modelo
    model.fit(X_train, y_train)

    # Predição no conjunto de validação
    y_pred_val = model.predict(X_val)

    # Avaliação do modelo
    mse = mean_squared_error(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"Erro Médio Quadrado (MSE) na validação: {mse}")
    print(f"Erro Médio Absoluto (MAE) na validação: {mae}")
    print(f"R² na validação: {r2}")

    # Salvar o conjunto de treino em CSV como solicitado
    X_train.to_csv(f"{DATA_DIR}/{MODEL_NAME}")
    y_train.to_csv(f"{DATA_DIR}/y_train_random_forest.csv")
    print(f"Conjunto de treino salvo em {
          DATA_DIR}/{MODEL_NAME} e y_train_random_forest.csv")

    # Salvar as previsões no conjunto de validação para uso no dashboard
    y_pred_df = pd.DataFrame({'Predicted': y_pred_val}, index=y_val.index)
    y_pred_df.to_csv(PREDICTIONS_FILE)
    print(f"Previsões do conjunto de validação salvas em {PREDICTIONS_FILE}")


def main():
    """
    Função principal para o treinamento do modelo de curto prazo.
    """
    # Carrega e prepara os dados
    X_train, y_train, X_val, y_val = load_and_prepare_data()

    # Treina e valida o modelo
    train_model(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
