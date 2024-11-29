'''
    Módulo 3: Modelos de Previsão (Curto e Longo Prazo)
    Este módulo é central para o sucesso do projeto, portanto deve ser robusto e flexível.

    Ajustes: Separe implementações de modelos para curto e longo prazo em sub-módulos:
    Curto Prazo: Modelos como LSTM, XGBoost e/ou LightGBM, que capturam variações de curto prazo.
    Longo Prazo: Modelos como Prophet, ARIMA e LSTM multivariado, para capturar tendências e sazonalidades amplas.
    Ensemble: Integre as previsões dos modelos em um ensemble, como discutido, combinando os pontos fortes de cada abordagem.
    Validar e Ajustar Intervalos de Confiança: Incluir intervalos de confiança nas previsões de longo prazo para considerar a incerteza.
    Output atualizado: Previsões com intervalos de confiança, tanto para 30 dias quanto para 2 a 5 anos, com foco em interpretar como cada modelo contribui para o resultado final.

    ---------

    1. Estrutura do Módulo 3:
        Arquivo principal: t20_md3_prev_curto_longo.py
        Sub-módulos:
        curto_prazo.py: Implementação de modelos para previsões de curto prazo.
        longo_prazo.py: Implementação de modelos para previsões de longo prazo.
        ensemble.py: Integração e combinação dos modelos.
        2. Curto Prazo:
        Modelos sugeridos:
        LSTM: Para captura de padrões temporais sequenciais.
        XGBoost ou LightGBM: Modelos baseados em árvores de decisão que podem capturar variações mais imediatas nos dados de ativos e indicadores econômicos.
        Sub-módulo: curto_prazo.py:

        Coleta de dados históricos processados (já preparados no Módulo 1).
        Implementação de LSTM com ajuste para capturar variações de curto prazo (30 dias).
        Implementação de XGBoost ou LightGBM como modelo alternativo para previsões mais rápidas e interpretáveis.
        Treinamento e validação usando parte dos dados de treino.
        3. Longo Prazo:
        Modelos sugeridos:
        Prophet: Um modelo robusto para sazonalidades e tendências de longo prazo.
        ARIMA: Para séries temporais que exibem tendências claras e que podem ser modeladas de forma estatística.
        LSTM multivariado: Para capturar padrões temporais com base em múltiplas variáveis, como os diferentes ativos e indicadores econômicos.
        Sub-módulo: longo_prazo.py:

        Coleta de dados históricos processados.
        Implementação de Prophet para previsões de longo prazo (2 a 5 anos).
        Implementação de ARIMA para captura de tendências de longo prazo.
        Implementação de LSTM multivariado, para capturar relações entre múltiplas variáveis ao longo do tempo.
        Validar e ajustar intervalos de confiança.
        4. Ensemble:
        Estratégia:
        Combinar os resultados de diferentes modelos, ponderando cada um conforme seu desempenho ou um método de ensemble (como média ponderada ou stacking).
        Sub-módulo: ensemble.py:

        Implementação de um ensemble para integrar as previsões de modelos de curto e longo prazo.
        Análise de como cada modelo contribui para o resultado final.
        Integração dos intervalos de confiança de modelos de longo prazo para avaliar a incerteza.
        5. Saídas:
        Previsões para os próximos 30 dias e 2 a 5 anos.
        Intervalos de confiança nas previsões de longo prazo.
        Visualização das previsões e de como cada modelo contribui para o ensemble.
        Arquivo CSV para salvar as previsões.

    -----------------------

    Divisão proposta para o Módulo 3: Modelos de Previsão (Curto e Longo Prazo)
        Parte 1: Previsão de Curto Prazo (curto_prazo.py)
        Conteúdo: Modelos de previsão para o curto prazo (ex.: LSTM, XGBoost, LightGBM)
        Foco: Capturar variações rápidas em períodos curtos (30 dias).
        Modelos:
        LSTM para padrões temporais sequenciais.
        XGBoost / LightGBM para captar variações imediatas nos dados dos ativos e indicadores econômicos.
        Validação: Validar as previsões no conjunto de validação de curto prazo (30 dias).
        Parte 2: Previsão de Longo Prazo (longo_prazo.py)
        Conteúdo: Modelos de previsão para o longo prazo (ex.: Prophet, ARIMA, LSTM multivariado)
        Foco: Capturar tendências e sazonalidades de longo prazo (2 a 5 anos).
        Modelos:
        Prophet para tendências de longo prazo e sazonalidades.
        ARIMA para séries temporais estatísticas.
        LSTM multivariado para capturar padrões complexos entre diferentes ativos e indicadores econômicos.
        Validação: Implementar intervalos de confiança para previsões de longo prazo e avaliar o desempenho.
        Parte 3: Ensemble de Modelos (ensemble.py)
        Conteúdo: Combinação dos modelos de curto e longo prazo.
        Foco: Criar um ensemble para aproveitar as forças de cada abordagem.
        Métodos:
        Média ponderada: Simples combinação de previsões ponderadas.
        Stacking: Combinação de modelos com um meta-modelo, ajustando o peso de cada previsão.
        Saídas: Previsões finais com intervalos de confiança ajustados e interpretação da contribuição de cada modelo.
    ------------------------

    Vantagens da divisão
        Modularidade e Reuso: Como fizemos no Módulo 1, cada parte pode ser usada de maneira independente ou integrada. Isso facilita o ajuste de um modelo ou abordagem sem impactar outros módulos.
        Facilidade de Manutenção: Manter código para modelos de curto e longo prazo separados torna o desenvolvimento mais ágil e reduz riscos de interferências entre métodos que trabalham com diferentes horizontes temporais.
        Flexibilidade: Podemos ajustar e experimentar com diferentes modelos em curto e longo prazo sem impactar o conjunto final, ajustando o ensemble separadamente.
        Testes e Validação: Cada módulo pode ser validado individualmente, facilitando a identificação de erros e melhorias.
        Proposta para Início
        Começaremos implementando o sub-módulo de curto prazo, focando nas previsões de 30 dias com modelos como LSTM e XGBoost/LightGBM.
        Depois disso, avançamos para o sub-módulo de longo prazo, onde vamos trabalhar com Prophet, ARIMA, e LSTM multivariado.
        Finalmente, integraremos as previsões usando um ensemble, que poderá ser ajustado separadamente.

        --------------------------------

        Validação Cruzada: O código agora inclui a validação cruzada usando o TimeSeriesSplit. Isso permitirá uma melhor avaliação do desempenho dos modelos ao longo do tempo.

        Ensemble: O código inclui uma função ensemble_predictions para combinar as previsões dos três modelos (LSTM, XGBoost e LightGBM). Neste exemplo, estamos usando uma média simples.

        Plotagem: Os resultados, incluindo as previsões de cada modelo e o ensemble, serão plotados usando Plotly.

'''

# Importando bibliotecas necessárias
# t20_md3_prev_curto_prazo.py

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objs as go
import plotly.io as pio
from tensorflow.keras import Input

# Definir o template do plotly
pio.templates.default = "plotly_white"

# Função para preparar os dados com normalização para a coluna alvo e variáveis financeiras


def prepare_data(df, target_column, sequence_length=60):
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()  # Scaler específico para a coluna alvo

    # Aplicar o scaler para todas as colunas
    df_scaled = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)
    df_scaled[target_column] = target_scaler.fit_transform(df[[target_column]])

    X, y = [], []
    for i in range(sequence_length, len(df_scaled)):
        X.append(df_scaled.iloc[i-sequence_length:i].values)
        y.append(df_scaled.iloc[i][target_column])

    return np.array(X), np.array(y), target_scaler

# Modelo LSTM multivariado com regularização e early stopping


def train_lstm(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), verbose=0, callbacks=[early_stop])

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'LSTM RMSE: {rmse}')

    return model, predictions

# Modelo XGBoost ajustado para séries temporais


def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'XGBoost RMSE: {rmse}')

    return model, predictions

# Modelo LightGBM ajustado para séries temporais


def train_lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(verbosity=-1)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'LightGBM RMSE: {rmse}')

    return model, predictions

# Validação cruzada usando TimeSeriesSplit


def cross_validate_models(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    lstm_rmses, xgb_rmses, lgb_rmses = [], [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        _, lstm_predictions = train_lstm(X_train, y_train, X_test, y_test)
        lstm_rmses.append(
            np.sqrt(mean_squared_error(y_test, lstm_predictions)))

        _, xgb_predictions = train_xgboost(X_train, y_train, X_test, y_test)
        xgb_rmses.append(np.sqrt(mean_squared_error(y_test, xgb_predictions)))

        _, lgb_predictions = train_lightgbm(X_train, y_train, X_test, y_test)
        lgb_rmses.append(np.sqrt(mean_squared_error(y_test, lgb_predictions)))

    print(f'LSTM RMSE médio: {np.mean(lstm_rmses)}')
    print(f'XGBoost RMSE médio: {np.mean(xgb_rmses)}')
    print(f'LightGBM RMSE médio: {np.mean(lgb_rmses)}')

# Ensemble usando média ponderada


def ensemble_predictions(lstm_predictions, xgb_predictions, lgb_predictions, weights=[0.5, 0.25, 0.25]):
    return (weights[0] * lstm_predictions + weights[1] * xgb_predictions + weights[2] * lgb_predictions)

# Função para plotar os resultados com Plotly


def plot_results(y_test, lstm_predictions, xgb_predictions, lgb_predictions, ensemble_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test,
                             mode='lines', name='Real', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=list(range(len(lstm_predictions))), y=lstm_predictions.flatten(),
                             mode='lines', name='LSTM', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=list(range(len(xgb_predictions))), y=xgb_predictions.flatten(),
                             mode='lines', name='XGBoost', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=list(range(len(lgb_predictions))), y=lgb_predictions.flatten(),
                             mode='lines', name='LightGBM', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=list(range(len(ensemble_pred))), y=ensemble_pred.flatten(),
                             mode='lines', name='Ensemble', line=dict(color='purple', dash='dash')))

    fig.update_layout(title='Previsões de Curto Prazo dos Modelos',
                      xaxis_title='Dias', yaxis_title='Preço',
                      legend=dict(x=0, y=1), template='plotly_white')
    fig.show()

# Função principal


def main():
    df = pd.read_csv('TESTES/DADOS/train_data.csv', index_col=0)
    target_column = '^BVSP'

    X, y, target_scaler = prepare_data(df, target_column)

    cross_validate_models(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    lstm_model, lstm_predictions = train_lstm(X_train, y_train, X_test, y_test)
    xgb_model, xgb_predictions = train_xgboost(
        X_train, y_train, X_test, y_test)
    lgb_model, lgb_predictions = train_lightgbm(
        X_train, y_train, X_test, y_test)

    ensemble_pred = ensemble_predictions(
        lstm_predictions, xgb_predictions, lgb_predictions)

    lstm_predictions = target_scaler.inverse_transform(lstm_predictions)
    xgb_predictions = target_scaler.inverse_transform(
        xgb_predictions.reshape(-1, 1))
    lgb_predictions = target_scaler.inverse_transform(
        lgb_predictions.reshape(-1, 1))
    ensemble_pred = target_scaler.inverse_transform(
        ensemble_pred.reshape(-1, 1))
    y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    plot_results(y_test, lstm_predictions, xgb_predictions,
                 lgb_predictions, ensemble_pred)


if __name__ == "__main__":
    main()
