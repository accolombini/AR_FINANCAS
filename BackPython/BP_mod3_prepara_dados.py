# BP_mod3_prepara_dados.py: Preparação de Dados Históricos para Portfólio Ótimo
# -----------------------------------------------------------
# Este script limpa, organiza e prepara os dados históricos de ativos
# e benchmarks para otimização de portfólio.
# Inclui cálculo de retornos logarítmicos e normalização de períodos.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import os
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega dados históricos de um arquivo CSV ou Excel.

    Parâmetros:
        - caminho (str): Caminho do arquivo.

    Retorna:
        - pd.DataFrame: DataFrame com os dados carregados.
    """
    if not os.path.exists(caminho):
        logging.error(f"Arquivo não encontrado: {caminho}")
        raise FileNotFoundError(
            f"O arquivo especificado não existe: {caminho}")

    try:
        logging.info(f"Carregando dados do arquivo: {caminho}")
        if caminho.endswith('.csv'):
            return pd.read_csv(caminho, index_col="Date", parse_dates=True)
        elif caminho.endswith('.xlsx'):
            return pd.read_excel(caminho, index_col="Date", parse_dates=True)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {caminho}")
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo: {e}")
        raise


def calcular_retorno_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os retornos logarítmicos dos ativos.

    Parâmetros:
        - df (pd.DataFrame): DataFrame com os preços históricos.

    Retorna:
        - pd.DataFrame: DataFrame com os retornos logarítmicos.
    """
    logging.info("Calculando retornos logarítmicos...")
    retornos = df.pct_change().apply(lambda x: np.log(1 + x))
    return retornos


def remover_benchmark(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """
    Remove o benchmark do DataFrame.

    Parâmetros:
        - df (pd.DataFrame): DataFrame com retornos.
        - benchmark (str): Nome da coluna do benchmark.

    Retorna:
        - pd.DataFrame: DataFrame sem o benchmark.
    """
    if benchmark in df.columns:
        logging.info(f"Removendo benchmark '{benchmark}' do DataFrame.")
        return df.drop(columns=[benchmark])
    logging.warning(f"Benchmark '{benchmark}' não encontrado no DataFrame.")
    return df


def alinhar_periodo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove valores ausentes e alinha os períodos históricos.

    Parâmetros:
        - df (pd.DataFrame): DataFrame com retornos logarítmicos.

    Retorna:
        - pd.DataFrame: DataFrame alinhado e sem valores ausentes.
    """
    logging.info(
        "Alinhando períodos históricos e removendo valores ausentes...")
    df = df.dropna()
    # Remove linhas onde todos os retornos são zero
    df = df.loc[(df != 0).all(axis=1)]
    return df


def salvar_arquivo(df: pd.DataFrame, caminho_saida: str):
    """
    Salva o DataFrame consolidado em um arquivo CSV.

    Parâmetros:
        - df (pd.DataFrame): DataFrame consolidado.
        - caminho_saida (str): Caminho para salvar o arquivo.
    """
    try:
        logging.info(f"Salvando arquivo consolidado em: {caminho_saida}")
        df.to_csv(caminho_saida)
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo: {e}")
        raise


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    # Caminho do arquivo de entrada
    input_file = os.path.join(OUTPUT_DIR, "historical_data_cleaned.csv")
    output_file = os.path.join(OUTPUT_DIR, "prepared_data.csv")

    # Carregar dados históricos
    df = carregar_dados(input_file)

    # Calcular retornos logarítmicos
    retornos = calcular_retorno_log(df)

    # Remover benchmark
    benchmark = "^BVSP"
    retornos_sem_benchmark = remover_benchmark(retornos, benchmark)

    # Alinhar períodos e remover valores inválidos
    retornos_alinhados = alinhar_periodo(retornos_sem_benchmark)

    # Salvar dados preparados
    salvar_arquivo(retornos_alinhados, output_file)

    logging.info(f"Processamento concluído. Arquivo salvo em: {output_file}")


if __name__ == "__main__":
    main()
