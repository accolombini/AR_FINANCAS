# BP_mod6_ML_LogarithmicCheck.py: Verificação de Cálculo Logarítmico e NaN
# -----------------------------------------------------------
# Este script rastreia se o cálculo de retornos utiliza logaritmos
# e identifica onde os valores NaN aparecem no pipeline.
# -----------------------------------------------------------

import os
import pandas as pd
import numpy as np
import logging
from BP_mod1_config import OUTPUT_DIR
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados(caminho):
    """Carrega os dados consolidados."""
    try:
        logging.info(f"Carregando dados de: {caminho}")
        dados = pd.read_csv(caminho, index_col="Date", parse_dates=True)
        logging.info(f"Período de dados: {
                     dados.index.min()} a {dados.index.max()}")
        logging.info(
            f"Número de linhas e colunas antes do processamento: {dados.shape}")
        return dados
    except Exception as e:
        logging.error(f"Erro ao carregar os dados: {e}")
        raise


def calcular_retornos_e_verificar(dados):
    """Calcula os retornos e verifica se NaN são introduzidos."""
    logging.info("Calculando retornos e verificando NaN...")

    # Cálculo de Retorno Futuro com logaritmos
    dados["Retorno_Log_6M"] = np.log(
        dados["VALE3.SA"] / dados["VALE3.SA"].shift(126))
    dados["Retorno_Log_5Y"] = np.log(
        dados["VALE3.SA"] / dados["VALE3.SA"].shift(1260))

    # Verificar NaN introduzidos
    na_6M = dados["Retorno_Log_6M"].isna().sum()
    na_5Y = dados["Retorno_Log_5Y"].isna().sum()
    logging.info(f"Valores NaN em Retorno_Log_6M: {na_6M}")
    logging.info(f"Valores NaN em Retorno_Log_5Y: {na_5Y}")

    # Identificar linhas com NaN
    linhas_com_na = dados[dados[["Retorno_Log_6M",
                                 "Retorno_Log_5Y"]].isna().any(axis=1)]
    logging.info(f"Número total de linhas com NaN: {len(linhas_com_na)}")

    # Salvar linhas com NaN para análise
    if len(linhas_com_na) > 0:
        caminho_na = os.path.join(OUTPUT_DIR, "linhas_com_na_logarithmic.csv")
        linhas_com_na.to_csv(caminho_na)
        logging.info(f"Linhas com NaN salvas em: {caminho_na}")

    return dados


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    # Início do processo
    inicio_execucao = datetime.now()
    logging.info(f"Início da análise: {inicio_execucao}")

    # Caminhos dos arquivos
    caminho_dados = os.path.join(
        OUTPUT_DIR, "dados_consolidados_proporcional.csv")

    # Carregar dados
    dados = carregar_dados(caminho_dados)

    # Calcular retornos e verificar NaN
    calcular_retornos_e_verificar(dados)

    # Fim do processo
    fim_execucao = datetime.now()
    logging.info(f"Término da análise: {fim_execucao}")
    logging.info(f"Duração total: {fim_execucao - inicio_execucao}")


if __name__ == "__main__":
    main()
