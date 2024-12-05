# BP_mod00_an_curto_prazo_LSTM.py: Modelo LSTM para Previsão de Curto Prazo
# -------------------------------------------------------------------------
# Este script implementa um modelo LSTM para prever valores futuros de uma
# série temporal baseada em dados simulados. O foco está em prever valores
# futuros de forma confiável e robusta a partir de um conjunto de dados de
# treinamento gerado aleatoriamente.
# -------------------------------------------------------------------------

# BP_mod00_portfolio_forecast_LSTM_refactor_v6.py: Previsões do Portfólio com LSTM
# -------------------------------------------------------------------------
# Este script realiza previsões com base no portfólio utilizando um modelo LSTM.
# -------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from BP_mod1_config import OUTPUT_DIR

# Configuração do caminho para o arquivo CSV
file_name = "historical_data_portfolio.csv"
file_path = os.path.join(OUTPUT_DIR, file_name)

# Verificar se o arquivo existe
if not os.path.isfile(file_path):
    raise FileNotFoundError(
        f"[ERRO] O arquivo {file_path} não foi encontrado.")

# Carregar a base de dados
df = pd.read_csv(file_path, parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Verificar a integridade dos dados
assert df.isnull().sum().sum() == 0, "Os dados contêm valores nulos!"
assert df['Date'].is_monotonic_increasing, "As datas não estão ordenadas!"
assert 'Portfólio' in df.columns, "[ERRO] Coluna 'Portfólio' não encontrada!"

# Normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
df['Portfólio'] = scaler.fit_transform(df['Portfólio'].values.reshape(-1, 1))

# Determinar o período de validação (últimos dois meses de pregões úteis)
validation_start_date = df['Date'].iloc[-1] - pd.DateOffset(months=2)
val_start_idx = df[df['Date'] >= validation_start_date].index[0]

# Ajustar tamanhos de treinamento e teste
look_back = 60
train_test_data = df.iloc[:val_start_idx]
train_size = int(len(train_test_data) * 0.8)

train_data = train_test_data.iloc[:train_size]
test_data = train_test_data.iloc[train_size:]
val_data = df.iloc[val_start_idx:]

# Diagnóstico inicial
print(f"Tamanho total do conjunto de dados: {len(df)}")
print(f"Tamanho do conjunto de treino: {len(train_data)}")
print(f"Tamanho do conjunto de teste: {len(test_data)}")
print(f"Tamanho do conjunto de validação: {len(val_data)}")
print(f"Look back inicial: {look_back}")

# Ajustar o look_back dinamicamente baseado no tamanho do conjunto de validação
look_back = min(look_back, len(val_data) - 1)
if look_back < 1:
    raise ValueError(
        "[ERRO] O conjunto de validação é muito pequeno para criar sequências.")

print(f"[AVISO] Ajustando look_back para {
      look_back} devido ao tamanho dos dados.")


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Criar os conjuntos de treinamento, teste e validação
X_train, y_train = create_sequences(
    train_data['Portfólio'].values.reshape(-1, 1), look_back)
X_test, y_test = create_sequences(
    test_data['Portfólio'].values.reshape(-1, 1), look_back)
X_val, y_val = create_sequences(
    val_data['Portfólio'].values.reshape(-1, 1), look_back)

# Verificar se os conjuntos foram criados corretamente
if X_train.size == 0 or X_test.size == 0 or X_val.size == 0:
    raise ValueError(
        "[ERRO] Dados insuficientes após a criação das sequências.")

# Remodelar para o formato LSTM [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Construção do modelo LSTM
model = Sequential([
    Input(shape=(look_back, 1)),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo
model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))

# Previsões
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
val_predict = model.predict(X_val)

# Reverter normalização
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
val_predict = scaler.inverse_transform(val_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

# Plotar os resultados
plt.figure(figsize=(14, 7))

# Treinamento
plt.plot(df['Date'][look_back:look_back + len(y_train)],
         y_train, label="Treinamento", color='blue')
plt.plot(df['Date'][look_back:look_back + len(train_predict)],
         train_predict, label="Previsão Treinamento", linestyle='--', color='cyan')

# Teste
plt.plot(df['Date'][train_size + look_back:train_size +
         look_back + len(y_test)], y_test, label="Teste", color='orange')
plt.plot(df['Date'][train_size + look_back:train_size + look_back + len(test_predict)],
         test_predict, label="Previsão Teste", linestyle='--', color='gold')

# Validação
plt.plot(df['Date'][val_start_idx + look_back:],
         y_val, label="Validação", color='green')
plt.plot(df['Date'][val_start_idx + look_back:], val_predict,
         label="Previsão Validação", linestyle='--', color='lime')

plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.title('Desempenho do Modelo com Validação')
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'validation_results_portfolio_v6.png'))
plt.show()
