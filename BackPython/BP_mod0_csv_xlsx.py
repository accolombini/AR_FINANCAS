"""
Objetivo: Realizar a leitura de um arquivo CSV e salvar os dados como xlsx.
Se a coluna "Date" estiver presente e com timezone, remover o timezone.
"""

import pandas as pd

# Caminho do arquivo CSV e destino XLSX
# Substitua pelo seu arquivo CSV
csv_file = 'BackPython/DADOS/historical_data_cleaned.csv'
xlsx_file = 'BackPython/DADOS/asset_data_cleaner.xlsx'  # Nome do arquivo de saída

# Carregar o CSV no DataFrame
data = pd.read_csv(csv_file)

# Verificar e processar a coluna "Date" (se existir)
if "Date" in data.columns:
    try:
        # Converter para datetime
        data["Date"] = pd.to_datetime(data["Date"], errors='coerce')

        # Remover timezone de todos os valores
        data["Date"] = data["Date"].apply(lambda x: x.tz_localize(
            None) if pd.notnull(x) and x.tzinfo else x)
        print("Timezone removido da coluna 'Date'.")
    except Exception as e:
        print(f"Erro ao processar a coluna 'Date': {e}")

# Salvar o DataFrame como arquivo Excel
data.to_excel(xlsx_file, index=False)

print(f"Arquivo salvo como {xlsx_file}")
