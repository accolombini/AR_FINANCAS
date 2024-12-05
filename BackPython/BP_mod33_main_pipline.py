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

import pandas as pd
from BP_mod3_simulation import MonteCarloSimulation
from BP_mod3_optimization import PortfolioOptimization
from BP_mod3_dashboard import Dashboard


def main_pipeline(data_file="BackPython/DADOS/historical_data_cleaned.csv",
                  simulations_file="BackPython/DADOS/mc_simulations.csv",
                  benchmark_file="BackPython/DADOS/historical_data_cleaned.csv"):
    """
    Pipeline principal para o módulo 3.

    Args:
        data_file (str): Caminho para o arquivo de dados históricos limpos.
        simulations_file (str): Caminho para salvar as simulações de Monte Carlo.
        benchmark_file (str): Caminho para o arquivo de dados do benchmark.
    """

    # 1. Carregar os dados históricos limpos
    print("[INFO] Carregando dados históricos limpos...")
    historical_data = pd.read_csv(
        data_file, index_col="Date", parse_dates=True)

    # 2. Executar simulações de Monte Carlo
    print("[INFO] Iniciando simulações de Monte Carlo...")
    mc_simulation = MonteCarloSimulation(historical_data)
    mc_results = mc_simulation.run_simulations()
    mc_results.to_csv(simulations_file)
    print("[INFO] Simulações de Monte Carlo concluídas e salvas.")

    # 3. Otimizar o portfólio
    print("[INFO] Carregando simulações para otimização...")
    log_returns = pd.read_csv(simulations_file, index_col="Time")
    expected_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    asset_names = log_returns.columns.tolist()
    optimizer = PortfolioOptimization(
        expected_returns, cov_matrix, asset_names)

    print("[INFO] Executando otimização do portfólio...")
    try:
        optimal_weights = optimizer.optimize(target_return=0.001)
        print("Pesos Ótimos (em %):", optimal_weights)
    except ValueError as e:
        print("Erro na otimização:", e)
        return

    # 4. Inicializar o dashboard
    print("[INFO] Inicializando o dashboard...")
    benchmark_data = pd.read_csv(
        benchmark_file, index_col="Date", parse_dates=True)
    benchmark_data["Normalized"] = benchmark_data["^BVSP"] / \
        benchmark_data["^BVSP"].iloc[0]
    dashboard = Dashboard(mc_results, optimal_weights, benchmark_data)
    app = dashboard.create_dashboard()
    app.run_server(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main_pipeline()
