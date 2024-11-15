# BP_mod2_data_preparation.py
# Módulo para preparação dos dados, incluindo divisão e separação de variáveis

import pandas as pd
from sklearn.model_selection import train_test_split
# Usar o mesmo diretório de saída para consistência
from BP_mod1_config import OUTPUT_DIR

# Proporções dos dados
TRAIN_RATIO = 0.7
TEST_RATIO = 0.25
VALIDATION_RATIO = 0.05  # Aproximadamente os últimos dois meses para validação


def load_data():
    """
    Carrega o arquivo principal de dados 'asset_data_cleaner.csv' gerado pelo módulo 1.
    """
    filepath = f"{OUTPUT_DIR}/asset_data_cleaner.csv"
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def split_data(df, target_column='^BVSP'):
    """
    Divide os dados em conjuntos de treino, validação e teste.

    Parâmetros:
        - df (pd.DataFrame): DataFrame com todos os dados.
        - target_column (str): Nome da coluna de destino/target para prever.

    Retorna:
        - Tupla contendo os conjuntos de treino, validação e teste para X e y.
    """
    # Define as variáveis de entrada (X) e alvo (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Primeira divisão: Treino e Conjunto Temporário (Validação + Teste)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=TRAIN_RATIO, shuffle=False  # Mantemos a ordem temporal
    )

    # Segunda divisão: Validação e Teste
    val_size = VALIDATION_RATIO / (TEST_RATIO + VALIDATION_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False  # Mantemos a ordem temporal
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Salva os conjuntos de dados preparados em CSV no diretório de saída.

    Parâmetros:
        - X_train, X_val, X_test, y_train, y_val, y_test (pd.DataFrame): Conjuntos de dados para salvar.
    """
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv")
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv")
    X_val.to_csv(f"{OUTPUT_DIR}/X_val.csv")
    y_val.to_csv(f"{OUTPUT_DIR}/y_val.csv")
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv")
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv")
    print("Conjuntos de dados de treino, validação e teste salvos em CSV.")


def main():
    # Carregar os dados principais
    df = load_data()

    # Dividir os dados em treino, validação e teste
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Salvar os conjuntos de dados preparados
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
