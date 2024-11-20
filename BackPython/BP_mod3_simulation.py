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

import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm


def simulate_garch(data, num_simulations=1000):
    """
    Simula retornos futuros usando o modelo GARCH(1,1) para o intervalo completo do histórico.

    Args:
        data (pd.DataFrame): Dados históricos contendo os retornos logarítmicos.
        num_simulations (int): Número de simulações a serem realizadas.

    Returns:
        pd.DataFrame: Simulações de Monte Carlo para o período histórico completo.
    """
    simulations = {}
    historical_dates = data["Date"]  # Datas históricas
    num_days = len(historical_dates)  # Usar todo o período histórico

    for column in tqdm(data.columns[1:], desc="Simulando retornos"):
        # Ajustar o modelo GARCH
        returns = data[column].pct_change().dropna()  # Calcular retornos
        returns_scaled = returns * 100  # Ajustar escala para o GARCH

        model = arch_model(returns_scaled, vol='Garch',
                           p=1, q=1, dist='normal')
        fitted_model = model.fit(disp="off")

        # Recuperar os parâmetros ajustados
        omega = fitted_model.params['omega']
        alpha = fitted_model.params['alpha[1]']
        beta = fitted_model.params['beta[1]']
        variance = fitted_model.conditional_volatility.iloc[-1] ** 2
        last_return = returns_scaled.iloc[-1]

        # Simular caminhos futuros
        simulated_paths = []
        for _ in range(num_simulations):
            simulated_returns = []
            current_variance = variance

            for _ in range(num_days):
                innovation = np.random.normal(0, np.sqrt(current_variance))
                simulated_return = last_return + innovation
                current_variance = omega + alpha * \
                    (innovation ** 2) + beta * current_variance
                simulated_returns.append(simulated_return)

            simulated_paths.append(simulated_returns)

        # Combinar dados históricos com a média dos simulados
        simulations[column] = np.concatenate([
            returns_scaled / 100,  # Reverter escala histórica
            np.mean(simulated_paths, axis=0) / 100  # Reverter escala simulada
        ])

    # Criar DataFrame com datas e dados combinados
    simulation_df = pd.DataFrame(simulations)
    simulation_df["Date"] = historical_dates  # Usar o intervalo completo
    return simulation_df


def main():
    # Caminho para os dados históricos
    historical_data_path = "BackPython/DADOS/historical_data_cleaned.csv"
    mc_simulations_path = "BackPython/DADOS/mc_simulations.csv"

    # Carregar os dados históricos
    print("[INFO] Carregando dados históricos...")
    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])

    # Executar simulações de Monte Carlo usando GARCH
    print("[INFO] Iniciando simulações de Monte Carlo usando GARCH...")
    mc_simulations = simulate_garch(historical_data)

    # Salvar as simulações no arquivo
    print("[INFO] Salvando simulações em arquivo...")
    mc_simulations.to_csv(mc_simulations_path, index=False)
    print(f"[INFO] Simulações salvas em {mc_simulations_path}")


if __name__ == "__main__":
    main()
