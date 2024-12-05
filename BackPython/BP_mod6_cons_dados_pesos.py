# BP_mod6_cons_dados_pesos.py: Consolidação de Dados com Quantidade Proporcional de Ações
# -----------------------------------------------------------
# Este script combina os dados históricos filtrados com os pesos do portfólio
# otimizado, calculando a quantidade proporcional de ações para cada ativo.
# -----------------------------------------------------------

import os
import pandas as pd
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados(caminho):
    """Carrega dados de um arquivo CSV."""
    try:
        logging.info(f"Carregando dados de: {caminho}")
        return pd.read_csv(caminho, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {caminho}")
        raise
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        raise


def calcular_quantidade_proporcional(dados_filtrados, dados_pesos):
    """
    Calcula a quantidade proporcional de ações para cada ativo com base nos pesos e preços.

    Args:
        dados_filtrados (pd.DataFrame): DataFrame com dados históricos filtrados.
        dados_pesos (pd.DataFrame): DataFrame com pesos atribuídos aos ativos.

    Returns:
        pd.DataFrame: DataFrame com as quantidades proporcionais adicionadas.
    """
    logging.info(
        "Calculando a quantidade proporcional de ações para cada ativo...")
    dados_pesos = dados_pesos.set_index("Ativo")
    dados_quantidade = dados_filtrados.copy()

    for ativo in dados_pesos.index:
        if ativo in dados_quantidade.columns:
            # Convertendo peso para proporção
            peso = dados_pesos.loc[ativo, "Peso (%)"] / 100
            # Calcula a quantidade proporcional de ações
            dados_quantidade[ativo + "_Quantidade_Proporcional"] = peso / \
                dados_quantidade[ativo]
        else:
            logging.warning(
                f"Ativo {ativo} não encontrado nos dados filtrados.")

    return dados_quantidade


def salvar_dados(dados, caminho_saida):
    """Salva o DataFrame consolidado em um arquivo CSV."""
    try:
        logging.info(f"Salvando dados consolidados em: {caminho_saida}")
        dados.to_csv(caminho_saida)
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo: {e}")
        raise


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    # Caminhos dos arquivos
    caminho_dados_filtrados = os.path.join(
        OUTPUT_DIR, "historical_data_filtered.csv")
    caminho_pesos = os.path.join(OUTPUT_DIR, "portfolio_otimizado.csv")
    caminho_saida = os.path.join(
        OUTPUT_DIR, "dados_consolidados_proporcional.csv")

    # Carregar dados
    dados_filtrados = carregar_dados(caminho_dados_filtrados)
    dados_pesos = pd.read_csv(caminho_pesos)

    # Calcular a quantidade proporcional de ações
    dados_consolidados = calcular_quantidade_proporcional(
        dados_filtrados, dados_pesos)

    # Salvar dados consolidados
    salvar_dados(dados_consolidados, caminho_saida)

    logging.info("Processo concluído com sucesso!")


if __name__ == "__main__":
    main()
