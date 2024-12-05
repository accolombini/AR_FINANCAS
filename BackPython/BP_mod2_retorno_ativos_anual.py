# BP_mod2_retorno_ativos_anual.py: Análise de Retornos Anuais de Ativos
# -----------------------------------------------------------
# Este script calcula os retornos anualizados de cada ativo
# ao longo do período histórico e exibe os valores em porcentagem,
# incluindo o retorno médio no período.
# -----------------------------------------------------------

import pandas as pd
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def carregar_dados(filepath):
    """
    Carrega os dados históricos dos ativos e do benchmark.
    """
    data = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    benchmark = data.pop("^BVSP") if "^BVSP" in data.columns else None
    return data, benchmark


def calcular_retorno_anual(data):
    """
    Calcula os retornos anualizados para cada ativo e converte para porcentagem.
    """
    retorno_anual = data.resample('YE').last().pct_change()
    retorno_anual *= 100  # Converte para porcentagem
    return retorno_anual


def calcular_retorno_medio(retorno_anual):
    """
    Calcula o retorno médio anualizado no período para cada ativo.
    """
    retorno_medio = retorno_anual.mean(skipna=True)
    retorno_medio.name = "Média (%)"
    return retorno_medio


def exibir_tabela(dataframe, titulo="Retorno Anualizado dos Ativos"):
    """
    Exibe a tabela de retornos anualizados no terminal em formato percentual.
    """
    print(f"\n{titulo}")
    print(dataframe.round(2))  # Exibir com 2 casas decimais


def main():
    # Caminho do arquivo de dados
    filepath = f"{OUTPUT_DIR}/historical_data_cleaned.csv"

    # Carregar dados
    data, benchmark = carregar_dados(filepath)

    # Calcular retornos anualizados
    retorno_anual = calcular_retorno_anual(data)

    # Calcular o retorno médio no período
    retorno_medio = calcular_retorno_medio(retorno_anual)

    # Adicionar a linha de retorno médio à tabela
    retorno_anual.loc["Média (%)"] = retorno_medio

    # **SALVANDO COM O ÍNDICE**
    retorno_anual.to_csv(f"{OUTPUT_DIR}/retorno_anual.csv", index_label="Date")
    logging.info(f"Retorno anual salvo em: {OUTPUT_DIR}/retorno_anual.csv")

    # Exibir tabela no terminal
    exibir_tabela(
        retorno_anual, titulo="Tabela de Retornos Anualizados (em %)")


if __name__ == "__main__":
    main()
