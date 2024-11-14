import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Carregar dados de treino e validação
# Ou use pd.read_csv se estiver em CSV
X_train = np.load("BackPython/DADOS/X_train.npy")
y_train = np.load("BackPython/DADOS/y_train.npy")
X_val = np.load("BackPython/DADOS/X_val.npy")
y_val = np.load("BackPython/DADOS/y_val.npy")

# Configurar o modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_val, y_val))

# Fazer previsões no conjunto de validação
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print("MAE no conjunto de validação:", mae)

# Salvar previsões para o dashboard
pd.DataFrame(y_pred, columns=['y_pred']).to_csv("y_pred.csv", index=False)
pd.DataFrame(y_val, columns=['y_val']).to_csv("y_val.csv", index=False)
