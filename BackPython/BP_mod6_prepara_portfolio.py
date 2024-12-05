# BP_mod6_prepare_portfolio.py: Gerar Base Consolidada com Portfólio Ótimo
# -----------------------------------------------------------
# Este script combina os dados históricos e os pesos do portfólio
# para calcular o comportamento do portfólio ótimo ao longo do tempo.
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


def carregar_dados_historicos(caminho):
    """Carrega os dados históricos dos ativos."""
    try:
        logging.info(f"Carregando dados históricos de: {caminho}")
        dados = pd.read_csv(caminho, index_col="Date", parse_dates=True)
        logging.info(
            f"Número de linhas e colunas (dados históricos): {dados.shape}")
        return dados
    except Exception as e:
        logging.error(f"Erro ao carregar os dados históricos: {e}")
        raise


def carregar_pesos_portfolio(caminho):
    """Carrega os pesos do portfólio ótimo."""
    try:
        logging.info(f"Carregando pesos do portfólio de: {caminho}")
        pesos = pd.read_csv(caminho)
        # Converter para fração decimal
        pesos["Peso Decimal"] = pesos["Peso (%)"] / 100
        logging.info(f"Pesos carregados com sucesso: {pesos.shape}")
        return pesos
    except Exception as e:
        logging.error(f"Erro ao carregar os pesos do portfólio: {e}")
        raise


def calcular_portfolio_otimo(dados_historicos, pesos):
    """Calcula o valor do portfólio ótimo para cada linha da base histórica."""
    logging.info("Calculando o valor do portfólio ótimo...")

    # Criar a coluna do portfólio ótimo
    portfolio_otimo = pd.Series(0, index=dados_historicos.index)

    # Iterar pelos ativos e calcular contribuição para o portfólio
    for _, linha in pesos.iterrows():
        ativo = linha["Ativo"]
        peso = linha["Peso Decimal"]
        if ativo in dados_historicos.columns:
            portfolio_otimo += dados_historicos[ativo] * peso
            logging.info(f"Adicionado ativo {ativo} com peso {
                         peso:.4f} ao portfólio.")
        else:
            logging.warning(
                f"Ativo {ativo} não encontrado na base histórica. Ignorando.")

    # Adicionar a nova coluna ao DataFrame histórico
    dados_historicos["portfolio_otimo"] = portfolio_otimo
    logging.info("Cálculo do portfólio ótimo concluído.")

    return dados_historicos


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    # Início do processo
    inicio_execucao = datetime.now()
    logging.info(f"Início da análise: {inicio_execucao}")

    # Caminhos dos arquivos
    caminho_historico = os.path.join(
        OUTPUT_DIR, "historical_data_filtered.csv")
    caminho_pesos = os.path.join(OUTPUT_DIR, "portfolio_otimizado.csv")

    # Carregar dados
    dados_historicos = carregar_dados_historicos(caminho_historico)
    pesos_portfolio = carregar_pesos_portfolio(caminho_pesos)

    # Calcular o portfólio ótimo
    dados_consolidados = calcular_portfolio_otimo(
        dados_historicos, pesos_portfolio)

    # Salvar a base consolidada
    caminho_saida = os.path.join(OUTPUT_DIR, "portfolio_comportamento.csv")
    dados_consolidados.to_csv(caminho_saida)
    logging.info(f"Base consolidada salva em: {caminho_saida}")

    # Fim do processo
    fim_execucao = datetime.now()
    logging.info(f"Término da análise: {fim_execucao}")
    logging.info(f"Duração total: {fim_execucao - inicio_execucao}")


if __name__ == "__main__":
    main()
