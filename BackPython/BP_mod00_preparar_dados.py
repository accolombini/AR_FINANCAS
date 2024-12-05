# BP_mod00_preparar_dados.py: Script para preparar os dados de séries temporais
# ------------------------------------------------------------------------------
# Este script realiza a preparação de dados financeiros, incluindo:
# - Limpeza e transformação dos dados brutos.
# - Cálculo de retornos logarítmicos e indicadores técnicos.
# - Salvamento dos dados preparados para uso posterior.

import os
import pandas as pd
import numpy as np
from BP_mod1_config import OUTPUT_DIR, HISTORICAL_DATA_PATH, PREPARED_DATA_PATH


def carregar_dados(caminho):
    """
    Carrega os dados históricos do arquivo CSV.

    Args:
        caminho (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: Dados carregados como DataFrame.
    """
    dados = pd.read_csv(caminho)
    # Converter 'Date' para datetime
    dados['Date'] = pd.to_datetime(dados['Date'])
    dados.sort_values('Date', inplace=True)  # Garantir ordenação temporal
    return dados


def calcular_retorno_logaritmico(dados):
    """
    Calcula os retornos logarítmicos para cada ativo.

    Args:
        dados (pd.DataFrame): DataFrame com preços históricos.

    Returns:
        pd.DataFrame: DataFrame com retornos logarítmicos adicionados.
    """
    colunas_ativos = [col for col in dados.columns if col != 'Date']
    for ativo in colunas_ativos:
        # Calcular o retorno logarítmico e preencher valores nulos com zero
        dados[f'{ativo}_log_return'] = np.log(
            dados[ativo] / dados[ativo].shift(1)).fillna(0)
    return dados


def salvar_dados_preparados(dados, caminho):
    """
    Salva os dados preparados em um arquivo CSV.

    Args:
        dados (pd.DataFrame): DataFrame preparado.
        caminho (str): Caminho para salvar o arquivo.
    """
    dados.to_csv(caminho, index=False)
    print(f"[INFO] Dados preparados salvos em: {caminho}")


def main():
    print("[INFO] Iniciando previsão com Prophet...")

    # Carregar os dados preparados
    dados = carregar_dados_preparados(PREPARED_DATA_PATH)

    # Escolher um ativo para análise
    ativo = "VALE3.SA"

    # Preparar os dados para Prophet
    dados_prophet = preparar_dados_prophet(dados, ativo)

    # Verificar os dados preparados
    print("[DEBUG] Primeiros registros dos dados preparados para Prophet:")
    print(dados_prophet.head())  # Verifica se os dados estão corretos

    # Prever com Prophet
    previsao, modelo = prever_com_prophet(dados_prophet, dias=126)

    # Salvar resultados
    salvar_resultados(previsao, os.path.join(
        OUTPUT_DIR, f"{ativo}_prophet_previsao.csv"))

    # Exibir conclusão
    print("[INFO] Previsão concluída. Resultados salvos.")


if __name__ == "__main__":
    main()
