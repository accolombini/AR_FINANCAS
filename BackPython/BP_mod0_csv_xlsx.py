''' 
    Objetivo: Realizar a leitura de um arquivo CSV e salvar os dados como xlsx.
'''

# Importar a biblioteca pandas

import pandas as pd

# Leitura do arquivo CSV
csv_file = 'BackPython/DADOS/asset_data.csv'  # Substitua pelo nome do seu arquivo CSV
data = pd.read_csv(csv_file)

# Salvando como XLSX
xlsx_file = 'BackPython/DADOS/asset_data.xlsx'  # Nome do arquivo de saída
data.to_excel(xlsx_file, index=False)

print(f"Arquivo salvo como {xlsx_file}")
