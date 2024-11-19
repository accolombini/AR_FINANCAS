''' 
    Objetivo: orquestrar os Módulos

        Executar o BP_mod3_simulation.py para gerar as simulações de Monte Carlo.
        Passar os resultados para o BP_mod3_optimization.py para calcular os pesos ótimos.
        Utilizar os dados gerados para alimentar o BP_mod3_dashboard.py e exibir os resultados.
        Manter a Modularidade

        Permitir que cada módulo funcione independentemente, mas seja chamado e integrado por este pipeline principal.
        Saídas Consolidadas

        Salvar os resultados intermediários (simulações e pesos) para uso futuro.
        Garantir que o dashboard seja inicializado com os dados corretos.

        Estrutura do BP_mod3_main_pipeline.py
            Etapa 1: Executar Simulações
            Chamar o módulo de simulação para calcular retornos futuros dos ativos.
            Etapa 2: Otimizar Portfólio
            Chamar o módulo de otimização com os dados simulados para obter os pesos ótimos.
            Etapa 3: Inicializar o Dashboard
            Passar os dados gerados para o dashboard e inicializá-lo.
'''

# Importar os módulos necessários

from BP_mod3_simulation import MonteCarloSimulation
from BP_mod3_optimization import PortfolioOptimization
from BP_mod3_dashboard import Dashboard
import pandas as pd
import numpy as np


def main_pipeline(data_file="BackPython/DADOS/asset_data_cleaner.csv", benchmark_file="BackPython/DADOS/asset_data_raw.csv"):
    """
    Pipeline principal para o Módulo 3: Simulação, Otimização e Visualização.

    Args:
        data_file (str): Caminho para o arquivo de dados processados.
        benchmark_file (str): Caminho para o arquivo com dados do benchmark.
    """
    # Etapa 1: Carregar dados e executar simulações
    print("[INFO] Carregando dados dos ativos...")
    asset_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    asset_data_cleaned = asset_data.replace(0, 1e-6)

    # Filtrar apenas os preços ajustados
    price_columns = [
        col for col in asset_data_cleaned.columns
        if not ("returns" in col or "ma" in col or "volatility" in col)
    ]
    price_data = asset_data_cleaned[price_columns]

    print("[INFO] Iniciando simulações de Monte Carlo...")
    mc_simulator = MonteCarloSimulation(price_data)
    simulations = mc_simulator.simulate()

    # Etapa 2: Otimizar portfólio
    print("[INFO] Calculando retornos esperados e matriz de covariância...")
    log_returns = mc_simulator.calculate_log_returns()
    expected_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    asset_names = price_data.columns.tolist()  # Nomes dos ativos

    print("[INFO] Iniciando otimização de portfólio...")
    optimizer = PortfolioOptimization(
        expected_returns, cov_matrix, asset_names)
    diagnostics = optimizer.diagnose_inputs()
    print("Diagnósticos dos Dados:", diagnostics)

    try:
        optimal_weights = optimizer.optimize()
        print("Pesos Ótimos:", optimal_weights)
    except ValueError as e:
        print("Erro na otimização:", e)
        print("Ativos Problemáticos:", diagnostics["problematic_assets"])
        return

    # Etapa 3: Preparar dados do benchmark
    print("[INFO] Carregando dados do benchmark...")
    benchmark_data = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
    benchmark_data = benchmark_data[["^BVSP"]]

    # Etapa 4: Inicializar o dashboard
    print("[INFO] Inicializando o dashboard...")
    dashboard = Dashboard(simulations, optimal_weights, benchmark_data)
    app = dashboard.create_dashboard()
    app.run_server(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main_pipeline()
