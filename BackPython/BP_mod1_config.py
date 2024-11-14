# BP_mod1_config.py
# Configurações do sistema para o Módulo 1: Coleta e Processamento de Dados

from datetime import datetime

# Lista de ativos para análise
ASSETS = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
          'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']

# Intervalo de tempo para coleta de dados históricos
START_DATE = '2014-01-01'
# Define a data final como a data de hoje
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Diretório de saída para os arquivos processados
OUTPUT_DIR = 'BackPython/DADOS'

# Flag para definir se o dashboard deve ser executado
RUN_DASHBOARD = True  # Defina como False se o dashboard não precisar ser executado
