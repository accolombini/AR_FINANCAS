# BP_mod4_valida_scripts_anteriores.py: Validação dos Módulos e Resultados
# -----------------------------------------------------------
# Este script executa e valida cada módulo existente, garantindo que os
# resultados intermediários estão corretos. Ele utiliza os arquivos gerados
# pelos módulos anteriores e verifica sua integridade.
# -----------------------------------------------------------

import os
import logging
import pandas as pd
from BP_mod3_data_collection import main as coleta_dados
from BP_mod3_prepara_dados import main as prepara_dados
from BP_mod2_retorno_ativos_anual import main as retorno_anual
from BP_mod2_retorno_ativos_anual_otimo import main as retorno_anual_otimo
from BP_mod3_portfolio_otimo_mk import main as portfolio_otimo

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "DADOS")


def verificar_arquivo(filepath):
    """
    Verifica se um arquivo existe e tem conteúdo.
    """
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logging.info(f"Arquivo validado: {filepath}")
        return True
    else:
        logging.error(f"Arquivo inválido ou não encontrado: {filepath}")
        return False


def validar_resultados():
    """
    Executa os módulos e valida seus resultados.
    """
    # 1. Coleta de Dados
    logging.info("Validando coleta de dados...")
    coleta_dados()
    if not verificar_arquivo(os.path.join(OUTPUT_DIR, "historical_data_cleaned.csv")):
        return False

    # 2. Preparação de Dados
    logging.info("Validando preparação de dados...")
    prepara_dados()
    if not verificar_arquivo(os.path.join(OUTPUT_DIR, "prepared_data.csv")):
        return False

    # 3. Cálculo de Retornos Anualizados
    logging.info("Validando cálculo de retornos anualizados...")
    retorno_anual()
    if not verificar_arquivo(os.path.join(OUTPUT_DIR, "retorno_anual.csv")):
        return False

    # 4. Filtragem de Retornos Anualizados
    logging.info("Validando filtragem de retornos anualizados...")
    retorno_anual_otimo()
    if not verificar_arquivo(os.path.join(OUTPUT_DIR, "filtered_data.csv")):
        return False
    if not verificar_arquivo(os.path.join(OUTPUT_DIR, "historical_data_filtered.csv")):
        return False

    # 5. Otimização do Portfólio
    logging.info("Validando otimização do portfólio...")
    portfolio_otimo()
    if not verificar_arquivo(os.path.join(OUTPUT_DIR, "portfolio_otimizado.csv")):
        return False

    logging.info("Validação concluída com sucesso!")
    return True


if __name__ == "__main__":
    resultado = validar_resultados()
    if resultado:
        logging.info("Todos os módulos foram validados com êxito.")
    else:
        logging.error("Falha na validação de um ou mais módulos.")
