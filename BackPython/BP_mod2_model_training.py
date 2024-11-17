# BP_mod2_model_training.py
# Treinamento de modelos de curto prazo usando Random Forest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Configuração de diretórios e arquivos
DATA_DIR = "BackPython/DADOS"
OUTPUT_X_FILE = f"{DATA_DIR}/X_random_forest.csv"
OUTPUT_Y_TRAIN_FILE = f"{DATA_DIR}/y_train_random_forest.csv"
PREDICTIONS_FILE = f"{DATA_DIR}/y_pred_rf.csv"
INPUT_FILE = f"{DATA_DIR}/asset_data_cleaner.csv"


def load_and_prepare_data():
    """
    Carrega e prepara os dados para o modelo Random Forest.
    Divide os dados em treino, validação e teste.
    """
    print(f"Carregando dados de: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

    # Verificar se o DataFrame foi carregado corretamente
    if df.empty:
        raise ValueError(
            "O arquivo de entrada está vazio ou não foi carregado corretamente.")

    # Definir a coluna de destino (^BVSP) e as features
    target_column = "^BVSP"
    if target_column not in df.columns:
        raise KeyError(f"A coluna alvo '{
                       target_column}' não foi encontrada nos dados.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Dividir os dados em treino, validação e teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
    )

    return X_train, y_train, X_val, y_val


def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """
    Treina um modelo Random Forest e avalia seu desempenho.
    """
    print("Iniciando o treinamento do modelo de curto prazo...")

    # Instanciar e treinar o modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predição no conjunto de validação
    y_pred_val = model.predict(X_val)

    # Avaliação do modelo
    mse = mean_squared_error(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"Erro Médio Quadrado (MSE) na validação: {mse:.2f}")
    print(f"Erro Médio Absoluto (MAE) na validação: {mae:.2f}")
    print(f"Coeficiente de Determinação (R²): {r2:.2f}")

    # Salvar os conjuntos de treino
    X_train.to_csv(OUTPUT_X_FILE)
    y_train.to_csv(OUTPUT_Y_TRAIN_FILE)
    print(f"Conjunto de treino salvo em {
          OUTPUT_X_FILE} e {OUTPUT_Y_TRAIN_FILE}")

    # Salvar previsões do conjunto de validação
    y_pred_df = pd.DataFrame({'Predicted': y_pred_val}, index=y_val.index)
    y_pred_df.to_csv(PREDICTIONS_FILE)
    print(f"Previsões salvas em {PREDICTIONS_FILE}")


def main():
    """
    Função principal para o treinamento do modelo de curto prazo.
    """
    X_train, y_train, X_val, y_val = load_and_prepare_data()
    train_and_evaluate_model(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
