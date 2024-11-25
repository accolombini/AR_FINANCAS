'''
    Objetivos do Script
        Calcular os pesos ótimos dos ativos no portfólio:

        Maximizar o Índice de Sharpe.
        Respeitar restrições como:
        Pesos somando 100%.
        Limitação de 30% para qualquer ativo.
        Representação proporcional de setores.
        Outperformar o índice BOVESPA (opcional como objetivo secundário).
        Salvar os pesos em um arquivo CSV:

        Nome do arquivo: portfolio_otimizado.csv.
        Entrada e saída:

        Entrada: Dados históricos dos ativos (ex.: historical_data_cleaned.csv).
        Saída: Arquivo portfolio_otimizado.csv com os pesos de cada ativo.
        Plano de Desenvolvimento
        Importação de Bibliotecas:

        Usar numpy, pandas para manipulação de dados.
        scipy.optimize para otimização.
        (Opcional) cvxpy ou pyportfolioopt para uma abordagem especializada.
        Funções Principais:

        Função para calcular o Índice de Sharpe:
        Sharpe
        Retorno_Esperado
        Taxa_Livre_de_Risco
        Volatilidade
        Sharpe= Volatilidade Retorno_Esperado-Taxa_Livre_de_Risco
        Função de otimização:
        Utiliza scipy.optimize.minimize com restrições.
        Função para salvar os resultados no CSV.
        Fluxo do Script:

        Carregar os dados históricos.
        Calcular os retornos esperados e a matriz de covariância.
        Configurar e executar a otimização.
        Salvar os pesos resultantes.
'''

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf


def carregar_dados(filepath):
    """
    Carrega os dados do arquivo CSV e remove colunas irrelevantes.
    """
    data = pd.read_csv(filepath, index_col="Date", parse_dates=True)

    # Remover colunas irrelevantes (como o benchmark ^BVSP)
    if "^BVSP" in data.columns:
        data = data.drop(columns=["^BVSP"])

    return data


def identificar_setores(tickers):
    """
    Identifica os setores dos ativos usando yfinance.

    Args:
        tickers (list): Lista de tickers.

    Returns:
        dict: Mapeamento de setores para cada ativo.
    """
    setores = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            setor = info.get("sector", "Desconhecido")
            setores[ticker] = setor
        except Exception as e:
            print(f"[WARNING] Falha ao obter setor para {ticker}: {e}")
            setores[ticker] = "Desconhecido"

    return setores


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
        return risco  # Minimizar o risco para um dado retorno

    # Restrições
    constraints = [
        {"type": "eq", "fun": lambda pesos: np.sum(
            pesos) - 1},  # Pesos somam 100%
        {"type": "ineq", "fun": lambda pesos: np.dot(
            # Retorno mínimo diário equivalente
            pesos, media_retornos) - retorno_minimo / 252}
    ]

    # Adicionar restrições setoriais
    for setor, peso_minimo in pesos_minimos_setor.items():
        indices_setor = [i for i, ativo in enumerate(
            ativos) if setores.get(ativo) == setor]
        if indices_setor:
            constraints.append({
                "type": "ineq",
                "fun": lambda pesos, indices=indices_setor, peso_minimo=peso_minimo: np.sum(pesos[indices]) - peso_minimo
            })

    # Limites para pesos individuais
    bounds = [(0, 0.3) for _ in ativos]  # Pesos entre 0% e 30%

    # Pesos iniciais
    pesos_iniciais = np.array([1 / len(ativos)] * len(ativos))

    # Otimização
    resultado = minimize(funcao_objetivo, pesos_iniciais,
                         bounds=bounds, constraints=constraints, method='SLSQP')

    if not resultado.success:
        raise ValueError("Falha na otimização do portfólio.")

    # Retorna os pesos em porcentagem
    return {ativos[i]: resultado.x[i] * 100 for i in range(len(ativos))}, np.dot(resultado.x, media_retornos) * 252


def main():
    # Definir tickers do portfólio
    tickers = ["PETR4.SA", "ITUB4.SA", "PGCO34.SA",
               "AAPL34.SA", "AMZO34.SA", "VALE3.SA"]

    # Identificar setores automaticamente
    setores = identificar_setores(tickers)
    print(f"[INFO] Setores identificados: {setores}")

    # Pesos mínimos por setor
    pesos_minimos_setor = {
        "Energy": 0.10,
        "Financial Services": 0.10,
        "Consumer Cyclical": 0.10,
        "Technology": 0.20,
        "Basic Materials": 0.10  # Setor para VALE3.SA (Mineração)
    }

    # Retorno mínimo anual (em proporção: ex. 0.15 para 15%)
    retorno_minimo = 0.15

    # Caminho para o arquivo CSV
    filepath = "BackPython/DADOS/historical_data_cleaned.csv"

    # Carregar dados
    data = carregar_dados(filepath)

    # Otimizar portfólio
    pesos_otimos, retorno_esperado_anual = otimizar_portfolio(
        data, setores, pesos_minimos_setor, retorno_minimo=retorno_minimo)

    # Exibir informações no console
    print("\n[INFO] Resultados do Portfólio Otimizado:")
    for setor in set(setores.values()):
        ativos_setor = [ativo for ativo, s in setores.items() if s == setor]
        peso_setor = sum(pesos_otimos.get(ativo, 0) for ativo in ativos_setor)
        print(f"Setor: {setor}")
        for ativo in ativos_setor:
            print(f"  Ativo: {ativo}, Peso: {pesos_otimos.get(ativo, 0):.2f}%")
        print(f"  Peso Total do Setor: {peso_setor:.2f}%\n")

    print(f"Retorno Esperado Anual: {retorno_esperado_anual:.2%}")
    if retorno_esperado_anual < retorno_minimo:
        print("[WARNING] O retorno esperado está abaixo do mínimo desejado!")

    # Salvar portfólio otimizado
    df_pesos = pd.DataFrame(pesos_otimos.items(),
                            columns=["Ativo", "Peso (%)"])
    df_pesos.to_csv("BackPython/DADOS/portfolio_otimizado.csv", index=False)

    print("Portfólio otimizado salvo em 'portfolio_otimizado.csv'.")


if __name__ == "__main__":
    main()
