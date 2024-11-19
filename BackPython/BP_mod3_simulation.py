'''
    Este módulo tem por objetivo simular o comportamento dos preços de ativos em um período de tempo determinado.
    Inclui a classe para simulação de Monte Carlo, o tratamento dos dados, e as etapas de cálculo que foram aplicadas

    Os dados foram carregados com sucesso. Eles incluem preços ajustados e métricas derivadas, como retornos, médias móveis e volatilidades, tanto para os ativos individuais quanto para o índice BOVESPA (^BVSP).

        Vou agora utilizar esses dados para:

        Calcular os retornos logarítmicos dos ativos.
        Realizar simulações de Monte Carlo para os ativos.
        Gerar métricas do portfólio com pesos iniciais para validar o modelo.
        Aguarde enquanto executo essas etapas. ​​

        Os cálculos das simulações de Monte Carlo retornaram valores inválidos (NaN). Isso geralmente ocorre por conta de problemas nos dados de entrada, como zeros ou valores negativos que geram erros ao calcular os logaritmos.

        Investigação do Problema
        Retornos Logarítmicos
        O cálculo de logaritmos em np.log(self.data / self.data.shift(1)) pode gerar valores NaN ou inf se houver:

            Zeros nos preços.
            Valores ausentes ou não preenchidos adequadamente.
            Divisão por zero devido a valores consecutivos iguais.
            Próximos Passos

            Verificar os dados originais para valores inconsistentes ou ausentes.
            Tratar valores inválidos preenchendo ou ajustando os dados para cálculos.
            Vou inspecionar os dados e corrigir os problemas antes de continuar. ​​

            Os dados apresentam problemas em algumas colunas, especificamente nos retornos calculados (*_returns). Aqui estão os principais achados:

            Zeros em Retornos

            Algumas colunas têm muitos zeros, especialmente:
            PGCO34.SA_returns: 816 zeros.
            AMZO34.SA_returns: 676 zeros.
            APL34.SA_returns: 170 zeros.

            Implicação

            Esses zeros provavelmente surgem de períodos sem variação nos preços ou devido a erros ao calcular os retornos.

    Nota: Os dados foram corrigidos, substituindo valores zero por um pequeno valor (1e-6). Também filtramos apenas as colunas de preços ajustados, removendo colunas derivadas (como retornos, médias móveis e volatilidade) para evitar cálculos redundantes.
    '''

# Importar bibliotecas necessárias

import numpy as np
import pandas as pd


class MonteCarloSimulation:
    def __init__(self, data, num_simulations=1000, time_horizon=252):
        """
        Inicializa a classe de simulação Monte Carlo.

        Args:
            data (pd.DataFrame): Dados históricos dos ativos, contendo preços ajustados.
            num_simulations (int): Número de simulações.
            time_horizon (int): Horizonte de tempo para as simulações (em dias úteis).
        """
        self.data = data
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon

    def calculate_log_returns(self):
        """
        Calcula os retornos logarítmicos dos ativos.

        Returns:
            pd.DataFrame: DataFrame com os retornos logarítmicos.
        """
        return np.log(self.data / self.data.shift(1)).dropna()

    def simulate(self):
        """
        Realiza as simulações de Monte Carlo para os ativos.

        Returns:
            dict: Um dicionário contendo as simulações para cada ativo.
        """
        log_returns = self.calculate_log_returns()
        mean_returns = log_returns.mean()
        std_devs = log_returns.std()

        simulations = {}
        for asset in self.data.columns:
            asset_simulations = np.zeros(
                (self.time_horizon, self.num_simulations))
            for sim in range(self.num_simulations):
                random_walk = np.random.normal(
                    mean_returns[asset], std_devs[asset], self.time_horizon
                )
                asset_simulations[:, sim] = np.cumprod(1 + random_walk)

            simulations[asset] = asset_simulations

        return simulations

    def calculate_portfolio_metrics(self, weights, simulations):
        """
        Calcula os retornos esperados e a volatilidade do portfólio com base nas simulações.

        Args:
            weights (list): Pesos dos ativos no portfólio.
            simulations (dict): Simulações de Monte Carlo dos ativos.

        Returns:
            tuple: Retorno esperado e volatilidade do portfólio.
        """
        portfolio_returns = np.zeros((self.time_horizon, self.num_simulations))

        for asset, asset_simulations in simulations.items():
            asset_index = list(self.data.columns).index(asset)
            portfolio_returns += asset_simulations * weights[asset_index]

        expected_return = portfolio_returns[-1, :].mean()
        volatility = portfolio_returns[-1, :].std()

        return expected_return, volatility


if __name__ == "__main__":
    # Carregar os dados processados
    file_path = "BackPython/DADOS/asset_data_cleaner.csv"
    asset_data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Substituir zeros nos dados por um pequeno valor para evitar problemas
    asset_data_cleaned = asset_data.replace(0, 1e-6)

    # Filtrar apenas os preços ajustados para simulações
    price_columns = [
        col for col in asset_data_cleaned.columns
        if not ("returns" in col or "ma" in col or "volatility" in col)
    ]
    price_data = asset_data_cleaned[price_columns]

    # Inicializar o simulador de Monte Carlo
    mc_simulator = MonteCarloSimulation(price_data)

    # Executar as simulações de Monte Carlo
    simulations = mc_simulator.simulate()

    # Definir pesos iniciais hipotéticos para os ativos
    portfolio_weights = [0.15] * len(price_data.columns)

    # Calcular métricas do portfólio
    expected_return, volatility = mc_simulator.calculate_portfolio_metrics(
        portfolio_weights, simulations
    )

    # Exibir os resultados
    print(f"Expected Return: {expected_return}")
    print(f"Portfolio Volatility: {volatility}")
