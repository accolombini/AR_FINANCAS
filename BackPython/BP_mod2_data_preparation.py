# BP_mod2_data_preparation.py
# Módulo para preparação dos dados, incluindo divisão em treino, validação e teste

import pandas as pd
from sklearn.model_selection import train_test_split
from BP_mod1_config import OUTPUT_DIR

# Configuração de proporções
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def load_data():
    """
    Carrega os dados pré-processados salvos pelo Módulo 1.
    """
    filepath = f"{OUTPUT_DIR}/asset_data_cleaner.csv"
    print(f"Carregando dados de: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError("Os dados carregados estão vazios.")
    return df


def split_data(df, target_column='^BVSP'):
    """
    Divide os dados em conjuntos de treino, validação e teste.

    Parâmetros:
        - df (pd.DataFrame): DataFrame com os dados.
        - target_column (str): Nome da coluna de destino.

    Retorna:
        - Tupla contendo os conjuntos (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if target_column not in df.columns:
        raise KeyError(f"A coluna alvo '{
                       target_column}' não foi encontrada nos dados.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Divisão em treino e conjunto temporário (validação + teste)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=TRAIN_RATIO, random_state=42, shuffle=False
    )

    # Divisão do conjunto temporário em validação e teste
    validation_ratio = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-validation_ratio, random_state=42, shuffle=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Salva os conjuntos de dados em arquivos CSV.
    """
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv")
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv")
    X_val.to_csv(f"{OUTPUT_DIR}/X_val.csv")
    y_val.to_csv(f"{OUTPUT_DIR}/y_val.csv")
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv")
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv")
    print("Dados de treino, validação e teste salvos com sucesso.")


def main():
    """
    Função principal para carregar, dividir e salvar os dados.
    """
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
