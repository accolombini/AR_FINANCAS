# BP_mod1_config.py
# Configurações do sistema para o Módulo 1: Coleta e Processamento de Dados

from datetime import datetime

# Lista de ativos para análise
# Defina aqui os símbolos dos ativos e índices que serão analisados.
# Os ativos incluem ações e índices como VALE3.SA, PETR4.SA, etc., e ^BVSP para o índice Bovespa.
ASSETS = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
          'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']

# Intervalo de tempo para coleta de dados históricos
# START_DATE: Data de início para a coleta dos dados históricos dos ativos.
START_DATE = '2014-01-01'

# END_DATE: Data final para a coleta dos dados históricos dos ativos.
# Definida automaticamente como a data atual do sistema.
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Diretório de saída para os arquivos processados
# OUTPUT_DIR: Caminho onde os dados processados serão salvos em formato CSV ou outro desejado.
OUTPUT_DIR = 'BackPython/DADOS'

# Flag para definir se o dashboard deve ser executado automaticamente após o processamento
# RUN_DASHBOARD: Defina como True para executar o dashboard após o processamento dos dados.
# Caso não seja necessário visualizar os dados no dashboard, defina como False.
RUN_DASHBOARD = True
