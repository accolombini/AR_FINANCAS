# BP_mod6_ML_TrackDroppedRows.py: Rastreamento de Linhas Removidas
# -----------------------------------------------------------
# Este script identifica quais linhas estão sendo removidas durante o pipeline.
# Foco em linhas descartadas pelo dropna, verificando inconsistências.
# -----------------------------------------------------------

import os
import pandas as pd
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


def preparar_dados_e_rastrear(dados):
    """Prepara os dados e rastreia linhas removidas."""
    logging.info("Preparando os dados para treinamento...")

    # Cálculo de Retorno Futuro (6M e 5Y)
    dados["Retorno_Futuro_6M"] = dados["VALE3.SA"].pct_change(
        periods=126).shift(-126)
    dados["Retorno_Futuro_5Y"] = dados["VALE3.SA"].pct_change(
        periods=1260).shift(-126)

    # Features financeiras
    dados["Retorno_Passado_1M"] = dados["VALE3.SA"].pct_change(periods=21)
    dados["Volatilidade_1M"] = dados["VALE3.SA"].rolling(window=21).std()

    # Identificar linhas com valores ausentes
    dados_completo = dados.copy()
    linhas_com_na = dados[dados.isna().any(axis=1)]

    # Logando número de linhas removidas e salvando
    logging.info(f"Linhas com valores ausentes: {len(linhas_com_na)}")
    if len(linhas_com_na) > 0:
        caminho_linhas_na = os.path.join(OUTPUT_DIR, "linhas_com_na.csv")
        linhas_com_na.to_csv(caminho_linhas_na)
        logging.info(f"Linhas com valores ausentes salvas em: {
                     caminho_linhas_na}")

    # Remover linhas com NaN
    dados = dados.dropna()
    logging.info(f"Linhas restantes após dropna: {len(dados)}")
    return dados, dados_completo


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

    # Preparar dados e rastrear
    dados_preparados, dados_completo = preparar_dados_e_rastrear(dados)

    # Fim do processo
    fim_execucao = datetime.now()
    logging.info(f"Término da análise: {fim_execucao}")
    logging.info(f"Duração total: {fim_execucao - inicio_execucao}")


if __name__ == "__main__":
    main()
