# BP_mod1_config.py: Configurações do sistema para Coleta e Processamento de Dados
# -----------------------------------------------------------
# Este módulo centraliza as configurações para os scripts de análise financeira.
# Inclui parâmetros fixos, como diretórios e ativos, além de flexibilidade
# para ajustar dinamicamente o período inicial (START_DATE) com base no ativo
# com menor histórico disponível.
# -----------------------------------------------------------

import os
from datetime import datetime

# ---------------------------------------------
# Lista de ativos para análise
# ---------------------------------------------
ASSETS = [
    'VALE3.SA',  # Vale S.A.
    'PETR4.SA',  # Petrobras PN
    'ITUB4.SA',  # Itaú Unibanco PN
    'PGCO34.SA',  # Procter & Gamble (B3)
    'AAPL34.SA',  # Apple (B3)
    'AMZO34.SA',  # Amazon (B3)
    '^BVSP'       # Índice Bovespa
]

# Índice de benchmark (ex.: Bovespa)
BENCHMARK = '^BVSP'

# ---------------------------------------------
# Intervalo de tempo para coleta de dados históricos
# ---------------------------------------------
DEFAULT_START_DATE = (datetime.today().replace(
    year=datetime.today().year - 10)).strftime('%Y-%m-%d')
START_DATE = DEFAULT_START_DATE
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Ativo responsável pelo período inicial (atualizado dinamicamente)
START_DATE_ASSET = None

# ---------------------------------------------
# Diretórios de saída para arquivos processados
# ---------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "DADOS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------
# Configuração de execução do Dashboard
# ---------------------------------------------
RUN_DASHBOARD = True

# ---------------------------------------------
# Funções utilitárias para configuração dinâmica
# ---------------------------------------------


def atualizar_start_date(dates: dict):
    """
    Atualiza a configuração START_DATE dinamicamente com base no menor período histórico disponível.

    Parâmetros:
        - dates (dict): Dicionário no formato {ticker: 'AAAA-MM-DD'}, onde a data representa o início histórico.

    Atualiza:
        - START_DATE com a menor data encontrada.
        - START_DATE_ASSET com o ativo responsável por essa data.
    """
    global START_DATE, START_DATE_ASSET
    if dates:
        START_DATE_ASSET, START_DATE = min(dates.items(), key=lambda x: x[1])
        print(f"[INFO] START_DATE atualizado para {
              START_DATE} com base no ativo {START_DATE_ASSET}.")
    else:
        print("[WARNING] Nenhuma data fornecida para atualização de START_DATE.")
