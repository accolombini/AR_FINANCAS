'''Ajustes de hiperparâmetros'''

# train_lstm_multivariado_optimized_manual.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Definir o template do plotly
pio.templates.default = "plotly_white"

# Função para preparar os dados incluindo indicadores macroeconômicos


def prepare_data(df, target_column, sequence_length=60):
    if df.empty:
        raise ValueError("Erro: O DataFrame está vazio.")
    if target_column not in df.columns:
        raise ValueError(f"Erro: A coluna alvo '{
                         target_column}' não foi encontrada no DataFrame.")

    macro_columns = ['inflacao', 'taxa_juros', 'pib']
    for col in macro_columns:
        if col in df.columns:
            df[col] = df[col].ffill()  # Preenchendo valores ausentes

    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    df_scaled[target_column] = target_scaler.fit_transform(df[[target_column]])

    X, y = [], []
    for i in range(sequence_length, len(df_scaled)):
        X.append(df_scaled.iloc[i-sequence_length:i].values)
        y.append(df_scaled.iloc[i][target_column])

    return np.array(X), np.array(y), scaler, target_scaler

# Função para construir o modelo LSTM


def build_lstm_model(input_shape, units_1=100, units_2=50, dropout_rate=0.2):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units_1, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units_2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Função de busca manual de hiperparâmetros


def manual_hyperparameter_search(X, y, input_shape, param_grid):
    best_rmse = float('inf')
    best_params = None

    for units_1 in param_grid['units_1']:
        for units_2 in param_grid['units_2']:
            for dropout_rate in param_grid['dropout_rate']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        print(f"Testando parâmetros: units_1={units_1}, units_2={units_2}, "
                              f"dropout_rate={dropout_rate}, batch_size={batch_size}, epochs={epochs}")
                        model = build_lstm_model(
                            input_shape, units_1=units_1, units_2=units_2, dropout_rate=dropout_rate)

                        early_stop = EarlyStopping(
                            monitor='val_loss', patience=5, restore_best_weights=True)
                        reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

                        tscv = TimeSeriesSplit(n_splits=3)
                        rmse_scores = []

                        for train_index, test_index in tscv.split(X):
                            X_train, X_val = X[train_index], X[test_index]
                            y_train, y_val = y[train_index], y[test_index]

                            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(
                                X_val, y_val), callbacks=[early_stop, reduce_lr], verbose=0)
                            y_pred = model.predict(X_val)
                            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                            rmse_scores.append(rmse)

                        avg_rmse = np.mean(rmse_scores)
                        print(f"Average RMSE: {avg_rmse}")

                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_params = {'units_1': units_1, 'units_2': units_2,
                                           'dropout_rate': dropout_rate, 'batch_size': batch_size, 'epochs': epochs}

    print("Melhores hiperparâmetros encontrados:", best_params)
    return best_params

# Função para plotar as previsões finais e tabela de resumo


def plot_dashboard(y_true, y_pred, rmse_scores, mae_scores, mape_scores):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Previsões com LSTM Multivariado",
                        "Resumo das Métricas"),
        specs=[[{"type": "xy"}, {"type": "table"}]]
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_true))),
            y=y_true.flatten(),
            mode='lines',
            name='Real',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred.flatten(),
            mode='lines',
            name='Previsão',
            line=dict(color='orange')
        ),
        row=1, col=1
    )

    metrics_table = go.Table(
        header=dict(values=["Fold", "RMSE", "MAE", "MAPE (%)"],
                    fill_color='paleturquoise',
                    align='center'),
        cells=dict(values=[
            list(range(1, len(rmse_scores) + 1)),
            np.round(rmse_scores, 4),
            np.round(mae_scores, 4),
            np.round(mape_scores, 2)
        ],
            fill_color='lavender',
            align='center')
    )

    fig.add_trace(metrics_table, row=1, col=2)

    fig.update_layout(
        title_text="Dashboard de Previsões e Métricas com LSTM Multivariado",
        height=600,
        showlegend=True
    )
    fig.show()

# Função principal para execução


def main():
    df = pd.read_csv('TESTES/DADOS/train_data_combined.csv',
                     index_col=0, parse_dates=True)
    target_column = '^BVSP'
    sequence_length = 60

    if df.empty:
        raise ValueError("O DataFrame está vazio após a leitura.")

    X, y, scaler, target_scaler = prepare_data(
        df, target_column, sequence_length)
    input_shape = (X.shape[1], X.shape[2])

    param_grid = {
        'units_1': [50, 100],
        'units_2': [25, 50],
        'dropout_rate': [0.2, 0.3],
        'batch_size': [32],
        'epochs': [50]
    }

    best_params = manual_hyperparameter_search(X, y, input_shape, param_grid)

    model = build_lstm_model(
        input_shape, units_1=best_params['units_1'], units_2=best_params['units_2'], dropout_rate=best_params['dropout_rate'])
    early_stop = EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, min_lr=1e-5)
    model.fit(X, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[
              early_stop, reduce_lr], verbose=1)

    y_pred = model.predict(X[-30:])
    y_pred = target_scaler.inverse_transform(y_pred)
    y_true = target_scaler.inverse_transform(y[-30:].reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred.flatten()) / y_true)) * 100

    plot_dashboard(y_true, y_pred, [rmse], [mae], [mape])


if __name__ == "__main__":
    main()
