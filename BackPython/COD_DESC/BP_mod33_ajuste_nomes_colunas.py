'''
    Corrigindo nome das colunas
'''

# Importar bibliotecas necessárias

import pandas as pd

# Carregar o arquivo mc_simulations.csv
file_path = "BackPython/DADOS/mc_simulations.csv"  # Substitua pelo caminho correto
mc_simulations = pd.read_csv(file_path)

# Ajustar o nome da coluna "Time" para "Date"
mc_simulations.rename(columns={"Time": "Date"}, inplace=True)

# Salvar o arquivo atualizado
mc_simulations.to_csv(
    "BackPython/DADOS/mc_simulations.csv", index=False)
