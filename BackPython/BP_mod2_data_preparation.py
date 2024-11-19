# BP_mod2_data_preparation.py
# Módulo para preparação dos dados, incluindo divisão em treino, validação e teste

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta


def preprocess_for_lstm_with_paths(
    input_file='BackPython/DADOS/asset_data_cleaner.csv',
    target_column='^BVSP',
    test_months=2,
    sequence_length=30,
    output_dir='BackPython/DADOS/'
):
    print(f"[INFO] Carregando dados de: {input_file}")
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)

    def preprocess_for_lstm(df, target_column, test_months, sequence_length):
        df = df.sort_index()
        last_date = df.index.max()
        test_start_date = last_date - timedelta(days=test_months * 30)
        test_data = df[df.index >= test_start_date]
        train_val_data = df[df.index < test_start_date]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train_val_data)
        scaled_test_data = scaler.transform(test_data)

        def create_sequences(data, sequence_length):
            X, y = [], []
            for i in range(sequence_length, len(data)):
                X.append(data[i - sequence_length:i])
                y.append(data[i, df.columns.get_loc(target_column)])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, sequence_length)
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        X_test, y_test = create_sequences(scaled_test_data, sequence_length)

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_for_lstm(
        df, target_column, test_months, sequence_length
    )
    np.save(f"{output_dir}/X_train.npy", X_train)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/X_val.npy", X_val)
    np.save(f"{output_dir}/y_val.npy", y_val)
    np.save(f"{output_dir}/X_test.npy", X_test)
    np.save(f"{output_dir}/y_test.npy", y_test)
    print(f"[INFO] Dados processados salvos em {output_dir}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


if __name__ == "__main__":
    preprocess_for_lstm_with_paths(
        input_file="BackPython/DADOS/asset_data_cleaner.csv",
        target_column="^BVSP",
        test_months=2,
        sequence_length=30,
        output_dir="BackPython/DADOS/"
    )
