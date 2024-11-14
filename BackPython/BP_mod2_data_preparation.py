# BP_mod2_data_preparation.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Constantes de proporção dos dados
TRAIN_RATIO = 0.7
TEST_RATIO = 0.25
VALIDATION_RATIO = 0.05  # Aproximadamente os últimos dois meses para validação


def load_data(filepath):
    """
    Carrega os dados do arquivo CSV e retorna o DataFrame.
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def split_data(df):
    """
    Divide os dados em conjuntos de treino, validação e teste com base nas proporções especificadas.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    test_start = train_end
    test_end = int(n * (TRAIN_RATIO + TEST_RATIO))

    train_data = df.iloc[:train_end]
    test_data = df.iloc[test_start:test_end]
    validation_data = df.iloc[test_end:]

    return train_data, validation_data, test_data


def normalize_data(train_data, test_data, validation_data):
    """
    Normaliza os dados de treino, validação e teste usando MinMaxScaler com base nos dados de treino.
    """
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    validation_data_scaled = scaler.transform(validation_data)

    # Convertendo de volta para DataFrame para facilitar o uso
    train_data_scaled = pd.DataFrame(
        train_data_scaled, columns=train_data.columns, index=train_data.index)
    test_data_scaled = pd.DataFrame(
        test_data_scaled, columns=test_data.columns, index=test_data.index)
    validation_data_scaled = pd.DataFrame(
        validation_data_scaled, columns=validation_data.columns, index=validation_data.index)

    return train_data_scaled, test_data_scaled, validation_data_scaled, scaler


def separate_features_target(data, target_column):
    """
    Separa o DataFrame em features (X) e alvo (y), onde y é a coluna target_column.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def main():
    # Caminho para o arquivo CSV dos dados
    filepath = 'BackPython/DADOS/asset_data_cleaner.csv'

    # Carregar os dados
    df = load_data(filepath)

    # Dividir os dados em treino, validação e teste
    train_data, validation_data, test_data = split_data(df)

    # Normalizar os dados
    train_data_scaled, test_data_scaled, validation_data_scaled, scaler = normalize_data(
        train_data, test_data, validation_data)

    # Separar as variáveis X e y para cada conjunto
    target_column = '^BVSP'  # Exemplo de coluna alvo; ajuste conforme necessário
    X_train, y_train = separate_features_target(
        train_data_scaled, target_column)
    X_test, y_test = separate_features_target(test_data_scaled, target_column)
    X_val, y_val = separate_features_target(
        validation_data_scaled, target_column)

    # Salvar os conjuntos preparados em CSV para uso posterior
    X_train.to_csv('BackPython/DADOS/X_train.csv')
    y_train.to_csv('BackPython/DADOS/y_train.csv')
    X_test.to_csv('BackPython/DADOS/X_test.csv')
    y_test.to_csv('BackPython/DADOS/y_test.csv')
    X_val.to_csv('BackPython/DADOS/X_val.csv')
    y_val.to_csv('BackPython/DADOS/y_val.csv')

    print("Dados preparados e salvos em CSV.")


if __name__ == "__main__":
    main()
