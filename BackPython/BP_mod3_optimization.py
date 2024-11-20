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

import numpy as np
import cvxpy as cp


class PortfolioOptimization:
    def __init__(self, expected_returns, cov_matrix, asset_names):
        """
        Inicializa a classe de otimização de portfólio.

        Args:
            expected_returns (np.ndarray): Retornos esperados dos ativos.
            cov_matrix (np.ndarray): Matriz de covariância dos ativos.
            asset_names (list): Lista com os nomes dos ativos.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.num_assets = len(asset_names)

    def optimize(self, target_return=0.001):
        """
        Minimiza a variância do portfólio dado um nível mínimo de retorno esperado.

        Args:
            target_return (float): Retorno esperado mínimo do portfólio.

        Returns:
            dict: Pesos ótimos dos ativos no portfólio em formato percentual.
        """
        weights = cp.Variable(self.num_assets)
        portfolio_risk = cp.quad_form(weights, self.cov_matrix)
        portfolio_return = self.expected_returns @ weights

        # Minimizar a variância
        objective = cp.Minimize(portfolio_risk)

        # Restrições
        constraints = [
            # Somatório dos pesos deve ser 100%
            cp.sum(weights) == 1,
            portfolio_return >= target_return,  # Retorno esperado mínimo
            weights >= 0.10,                   # Peso mínimo de 10% por ativo
            weights <= 0.75                    # Peso máximo de 75% por ativo
        ]

        # Resolver o problema de otimização
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(
                "O problema de otimização não encontrou uma solução viável.")

        # Retornar os pesos em formato percentual
        weights_percent = {name: round(
            weight * 100, 2) for name, weight in zip(self.asset_names, weights.value)}
        return weights_percent

    def diagnose_inputs(self):
        """
        Diagnósticos dos dados de entrada.

        Returns:
            dict: Estatísticas dos retornos esperados e da matriz de covariância.
        """
        return {
            "expected_returns_mean": self.expected_returns.mean(),
            "expected_returns_min": self.expected_returns.min(),
            "expected_returns_max": self.expected_returns.max(),
            "cov_matrix_positive_definite": np.all(np.linalg.eigvals(self.cov_matrix) > 0)
        }


if __name__ == "__main__":
    # Exemplo de uso
    import pandas as pd
    input_file = "BackPython/DADOS/mc_simulations.csv"

    print("[INFO] Carregando simulações...")
    simulations = pd.read_csv(input_file, index_col="Time")

    print("[INFO] Removendo índice BOVESPA dos dados...")
    if "^BVSP" in simulations.columns:
        simulations = simulations.drop(columns=["^BVSP"])

    print("[INFO] Calculando retornos esperados e matriz de covariância...")
    log_returns = np.log(simulations / simulations.shift(1)).dropna()
    expected_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    asset_names = simulations.columns.tolist()

    print("[INFO] Iniciando otimização do portfólio...")
    optimizer = PortfolioOptimization(
        expected_returns, cov_matrix, asset_names)
    diagnostics = optimizer.diagnose_inputs()
    print("Diagnósticos:", diagnostics)

    try:
        optimal_weights = optimizer.optimize(target_return=0.001)
        print("Pesos Ótimos (em %):", optimal_weights)
    except ValueError as e:
        print("Erro na otimização:", e)
