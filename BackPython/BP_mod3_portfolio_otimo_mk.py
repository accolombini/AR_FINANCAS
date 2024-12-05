# BP_mod3_portfolio_otimo_mk.py: Otimização de Portfólio com Análise Anual Melhorada
# -----------------------------------------------------------
# Este script otimiza o portfólio e exibe os retornos médios anuais.
# Inclui uma tabela detalhada e formatada com valores percentuais.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def carregar_dados(filepath):
    """
    Carrega os dados do arquivo CSV.
    """
    try:
        data = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        logging.info(f"Dados carregados de {filepath}.")
        return data
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {filepath}")
        raise


def identificar_setores(tickers):
    """
    Identifica os setores dos ativos de forma manual ou usando uma API.
    """
    setores = {
        "PETR4.SA": "Energy",
        "PGCO34.SA": "Consumer Defensive",
        "AAPL34.SA": "Technology",
        "AMZO34.SA": "Consumer Cyclical",
        "VALE3.SA": "Basic Materials"
    }
    return {ticker: setores.get(ticker, "Unknown") for ticker in tickers}


def calcular_retorno_anual(data, pesos):
    """
    Calcula os retornos médios anuais do portfólio otimizado.
    """
    retornos = data.pct_change().dropna()
    retorno_ponderado = retornos.dot(pesos)

    # Calcula os retornos anuais
    retorno_anual = retorno_ponderado.resample(
        'YE').apply(lambda x: (1 + x).prod() - 1)
    retorno_anual.index = retorno_anual.index.year  # Usar apenas os anos no índice

    # Converter para percentuais
    retorno_anual_percentual = retorno_anual * 100
    return retorno_anual_percentual


def otimizar_portfolio(data, setores, pesos_minimos_setor, retorno_minimo=0.15):
    """
    Encontra o portfólio ótimo com restrições setoriais e de retorno mínimo.
    """
    retornos = data.pct_change().dropna()
    media_retornos = retornos.mean()
    cov_matrix = retornos.cov()
    ativos = data.columns.tolist()

    def funcao_objetivo(pesos):
        risco = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        retorno = np.dot(pesos, media_retornos)
        return -retorno / risco

    # Restrições
    constraints = [
        {"type": "eq", "fun": lambda pesos: np.sum(
            pesos) - 1},  # Pesos somam 100%
        {"type": "ineq", "fun": lambda pesos: np.dot(
            pesos, media_retornos) - retorno_minimo / 252}  # Retorno mínimo
    ]

    # Adicionar restrições setoriais dinamicamente
    for setor, peso_minimo in pesos_minimos_setor.items():
        indices_setor = [i for i, ativo in enumerate(
            ativos) if setores.get(ativo) == setor]
        if indices_setor:
            constraints.append({
                "type": "ineq",
                "fun": lambda pesos, indices=indices_setor, peso_minimo=peso_minimo: np.sum(pesos[indices]) - peso_minimo
            })

    # Limites para pesos individuais
    bounds = [(0, 0.3) for _ in ativos]

    # Pesos iniciais
    pesos_iniciais = np.array([1 / len(ativos)] * len(ativos))

    # Otimização
    resultado = minimize(funcao_objetivo, pesos_iniciais,
                         bounds=bounds, constraints=constraints, method='SLSQP')

    if not resultado.success:
        logging.error(f"Detalhes da falha: {resultado.message}")
        raise ValueError("Falha na otimização do portfólio.")

    retorno_esperado = np.dot(resultado.x, media_retornos) * 252
    return {ativos[i]: resultado.x[i] for i in range(len(ativos))}, retorno_esperado

# ---------------------------
# Fluxo Principal
# ---------------------------


def main():
    filepath = f"{OUTPUT_DIR}/historical_data_filtered.csv"
    data = carregar_dados(filepath)

    tickers = data.columns.tolist()
    setores = identificar_setores(tickers)

    logging.info(f"Setores identificados: {setores}")

    pesos_minimos_setor = {
        "Energy": 0.10,
        "Consumer Defensive": 0.10,
        "Technology": 0.20,
        "Basic Materials": 0.10,
        "Consumer Cyclical": 0.10
    }

    pesos_otimos, retorno_esperado_anual = otimizar_portfolio(
        data, setores, pesos_minimos_setor)

    # Calculando retornos anuais do portfólio
    retorno_anual = calcular_retorno_anual(
        data, np.array(list(pesos_otimos.values())))

    # Exibindo os retornos anuais em formato legível
    logging.info("\nRetornos Anuais do Portfólio (em %):")
    print(retorno_anual)

    # Salvar pesos otimizados
    df_pesos = pd.DataFrame({
        "Ativo": list(pesos_otimos.keys()),
        "Peso (%)": [peso * 100 for peso in pesos_otimos.values()],
        "Setor": [setores[ativo] for ativo in pesos_otimos.keys()]
    })

    df_pesos.to_csv(f"{OUTPUT_DIR}/portfolio_otimizado.csv", index=False)
    logging.info(f"Portfólio otimizado salvo em: {
                 OUTPUT_DIR}/portfolio_otimizado.csv")
    logging.info(f"Retorno esperado anual: {retorno_esperado_anual:.2%}")


if __name__ == "__main__":
    main()
