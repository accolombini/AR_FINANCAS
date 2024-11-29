'''
    Este módulo tem por objetivo otimizar os resultados elevando a qualidade das previsões e potencializando ainda mais os algoritmos.
    Balancear os Pesos do Portfólio:

        Encontrar a alocação ótima para maximizar o retorno ajustado ao risco (Sharpe Ratio ou outra métrica).
        Respeitar as restrições de pesos: mínimo de 15% e máximo de 65% por ativo.
        Ferramentas Utilizadas:

        cvxpy: Resolver o problema de otimização com restrições.
        Entradas: Resultados das simulações de Monte Carlo (retornos médios e volatilidades).
        Saída: Pesos ideais para cada ativo no portfólio.
        Flexibilidade:

        Incorporar diferentes funções objetivo, como maximizar o retorno esperado ou minimizar a volatilidade.

        O módulo terá:

        Função Principal: Resolver o problema de otimização.
        Entrada de Dados: Retornos esperados, matriz de covariância (riscos), restrições.
        Saída: Pesos ótimos dos ativos.
    
    '''

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from tabulate import tabulate


def clean_data(data):
    """
    Limpa os dados históricos substituindo valores consecutivos iguais por um pequeno incremento.
    """
    if "Date" in data.columns:
        data = data.set_index("Date")
    adjusted_data = data.copy()
    adjusted_data[adjusted_data == adjusted_data.shift(
        1)] += 1e-6  # Incremento mínimo
    return adjusted_data


def calculate_annual_returns(data, index_column=None):
    """
    Calcula os retornos anuais por ano para cada ativo e o índice BOVESPA.
    """
    data["Year"] = pd.to_datetime(data.index).year
    numeric_data = data.drop(columns=["Year"], errors="ignore")

    # Calcular retornos logarítmicos
    log_returns = np.log(numeric_data / numeric_data.shift(1)).dropna()
    log_returns["Year"] = data["Year"]
    annual_returns = log_returns.groupby("Year").mean() * 252  # Anualizado

    if index_column and index_column in annual_returns.columns:
        index_returns = annual_returns[index_column]
    else:
        index_returns = None

    return annual_returns, index_returns


def format_table(df, title="Tabela"):
    """
    Formata os dados em uma tabela visualmente agradável.
    """
    df = df.fillna(0).replace([np.inf, -np.inf],
                              "-")  # Substituir NaN/Infinito
    headers = df.columns.tolist()
    rows = df.reset_index().values.tolist()
    print(f"\n{title}\n" + tabulate(rows, headers=headers, tablefmt="pretty"))


def main():
    input_file = "BackPython/DADOS/historical_data_cleaned.csv"
    index_column = "^BVSP"

    print("[INFO] Carregando dados históricos...")
    data = pd.read_csv(input_file, parse_dates=["Date"])
    data = clean_data(data)

    print("[INFO] Calculando retornos anuais históricos...")
    annual_returns, index_returns = calculate_annual_returns(
        data, index_column=index_column)

    # Simular pesos do portfólio
    portfolio_weights = {
        "VALE3.SA": 0.2,
        "PETR4.SA": 0.2,
        "ITUB4.SA": 0.2,
        "PGCO34.SA": 0.2,
        "AAPL34.SA": 0.1,
        "AMZO34.SA": 0.1,
    }

    # Tabela 1: Retornos anuais por ativo e índice
    relevant_columns = list(portfolio_weights.keys()) + [index_column]
    annual_returns_filtered = annual_returns[relevant_columns] * 100
    annual_returns_filtered = annual_returns_filtered.round(2)
    format_table(annual_returns_filtered,
                 title="Retornos Anuais por Ativo e Índice BOVESPA (%)")

    # Tabela 2: Pesos dos ativos no portfólio
    weights_table = pd.DataFrame(
        {"Pesos (%)": [
            f"{weight * 100:.2f}" for weight in portfolio_weights.values()]},
        index=portfolio_weights.keys(),
    )
    format_table(weights_table, title="Pesos dos Ativos no Portfólio")

    # Tabela 3: Comparação Portfólio x Índice
    print("[INFO] Calculando retornos do portfólio ajustado...")
    portfolio_returns = (annual_returns[list(
        portfolio_weights.keys())] * pd.Series(portfolio_weights)).sum(axis=1)
    portfolio_comparison = pd.DataFrame({
        "Year": annual_returns.index,
        "Portfolio Return (%)": (portfolio_returns * 100).round(2),
        "^BVSP (%)": (index_returns * 100).round(2) if index_returns is not None else None
    }).reset_index(drop=True)

    format_table(portfolio_comparison,
                 title="Retornos Anuais: Portfólio vs Índice (%)")


if __name__ == "__main__":
    main()
