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
    def __init__(self, expected_returns, cov_matrix, asset_names, risk_aversion=0.1, min_weight=0.15, max_weight=0.65):
        """
        Inicializa a classe de otimização de portfólio.

        Args:
            expected_returns (np.ndarray): Retornos esperados dos ativos.
            cov_matrix (np.ndarray): Matriz de covariância dos ativos.
            asset_names (list): Lista de nomes dos ativos.
            risk_aversion (float): Parâmetro de aversão ao risco (λ).
            min_weight (float): Peso mínimo permitido para cada ativo.
            max_weight (float): Peso máximo permitido para cada ativo.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight

    def diagnose_inputs(self):
        """
        Diagnostica os dados de entrada para a otimização.

        Returns:
            dict: Resumo dos diagnósticos.
        """
        diagnostics = {
            "expected_returns_mean": self.expected_returns.mean(),
            "expected_returns_min": self.expected_returns.min(),
            "expected_returns_max": self.expected_returns.max(),
            "cov_matrix_positive_definite": np.all(np.linalg.eigvals(self.cov_matrix) > 0),
        }

        # Identificar ativos problemáticos
        problematic_assets = []
        if diagnostics["expected_returns_min"] <= 0:
            for i, ret in enumerate(self.expected_returns):
                if ret <= 0:
                    problematic_assets.append(self.asset_names[i])

        diagnostics["problematic_assets"] = problematic_assets
        return diagnostics

    def optimize(self):
        """
        Resolve o problema de otimização para maximizar retorno ajustado ao risco.

        Returns:
            np.ndarray: Pesos ótimos para os ativos.
        """
        n = len(self.expected_returns)
        weights = cp.Variable(n)

        # Função objetivo: Maximizar retorno esperado menos aversão ao risco * volatilidade
        portfolio_return = self.expected_returns @ weights
        portfolio_volatility = cp.quad_form(weights, self.cov_matrix)
        objective = cp.Maximize(
            portfolio_return - self.risk_aversion * portfolio_volatility)

        # Restrições
        constraints = [
            cp.sum(weights) == 1,  # A soma dos pesos deve ser 1
            weights >= self.min_weight,  # Peso mínimo
            weights <= self.max_weight  # Peso máximo
        ]

        # Resolver o problema
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(
                "O problema de otimização não encontrou uma solução viável.")

        return weights.value


if __name__ == "__main__":
    # Dados de exemplo para teste
    expected_returns = np.array([0.1, 0.12, -0.02])  # Retornos esperados
    cov_matrix = np.array([  # Matriz de covariância
        [0.1, 0.02, 0.04],
        [0.02, 0.08, 0.03],
        [0.04, 0.03, 0.12]
    ])
    asset_names = ["Asset1", "Asset2", "Asset3"]

    # Inicializar a otimização
    optimizer = PortfolioOptimization(
        expected_returns, cov_matrix, asset_names, risk_aversion=0.1)

    # Diagnóstico dos dados
    diagnostics = optimizer.diagnose_inputs()
    print("Diagnósticos dos Dados:", diagnostics)

    # Calcular pesos ótimos
    try:
        optimal_weights = optimizer.optimize()
        print("Pesos Ótimos:", optimal_weights)
    except ValueError as e:
        print("Erro na otimização:", e)
        print("Ativos Problemáticos:", diagnostics["problematic_assets"])
