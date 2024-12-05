# BP_mod1_config.py: Configuração Global do Projeto
# -----------------------------------------------------------
# Este módulo centraliza as configurações e caminhos para os arquivos
# necessários nos demais scripts do projeto.
# -----------------------------------------------------------

import os
import json
from datetime import datetime

# ---------------------------
# Configurações Gerais
# ---------------------------

# Base do diretório do projeto
BASE_DIR = os.path.join(os.getcwd(), "BackPython")

# Diretório de saída para os dados
OUTPUT_DIR = os.path.join(BASE_DIR, "DADOS")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Cria o diretório, se necessário

# Lista de ativos
ATIVOS = [
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "PGCO34.SA",
    "AAPL34.SA", "AMZO34.SA", "^BVSP"
]

# Caminhos de arquivos
PREPARED_DATA_PATH = os.path.join(OUTPUT_DIR, "prepared_data.csv")
HISTORICAL_DATA_PATH = os.path.join(OUTPUT_DIR, "historical_data_cleaned.csv")

# Período histórico
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# ---------------------------
# Funções Auxiliares
# ---------------------------


def atualizar_start_date(start_date):
    """
    Atualiza a data inicial do período histórico.

    Args:
        start_date (str): Nova data inicial no formato 'YYYY-MM-DD'.
    """
    global START_DATE
    START_DATE = start_date
    print(f"[INFO] START_DATE atualizado para {START_DATE}.")


def validar_arquivos_necessarios():
    """
    Valida a existência dos arquivos essenciais e gera alertas, se necessário.
    """
    arquivos = [PREPARED_DATA_PATH, HISTORICAL_DATA_PATH]
    for arquivo in arquivos:
        if not os.path.exists(arquivo):
            print(f"[ALERTA] O arquivo {arquivo} não foi encontrado.")


def salvar_configuracoes_em_log(log_path=None):
    """
    Salva as configurações atuais em um arquivo de log JSON.

    Args:
        log_path (str): Caminho para salvar o log. Se não fornecido, usa o diretório padrão.
    """
    if log_path is None:
        log_path = os.path.join(OUTPUT_DIR, "config_log.json")

    config = {
        "BASE_DIR": BASE_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "ATIVOS": ATIVOS,
        "PREPARED_DATA_PATH": PREPARED_DATA_PATH,
        "HISTORICAL_DATA_PATH": HISTORICAL_DATA_PATH,
        "START_DATE": START_DATE,
        "END_DATE": END_DATE
    }
    with open(log_path, "w") as log_file:
        json.dump(config, log_file, indent=4)
    print(f"[INFO] Configurações salvas em {log_path}.")


# ---------------------------
# Execução para Testes
# ---------------------------
if __name__ == "__main__":
    # Exibe e valida configurações
    print("[INFO] Configurações atuais:")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"ATIVOS: {ATIVOS}")
    print(f"START_DATE: {START_DATE}")
    print(f"END_DATE: {END_DATE}")
    validar_arquivos_necessarios()

    # Salva configurações em log
    salvar_configuracoes_em_log()
