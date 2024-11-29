'''
    Objetivo desse algoritmo: realizar a simulação de Monte Carlo.

    ||>Estrutura do Algoritmo
        Aqui está o que será incluído no código:

        |> Carregamento das Previsões do Ensemble:

            Importar as previsões do ensemble que foram geradas pelo código anterior. Vamos considerar as previsões de médio e longo prazo como a linha de base para a simulação.
            Simulação de Monte Carlo:

            Implementar o método de Monte Carlo, que simula várias trajetórias futuras usando uma distribuição de retornos com base na média e desvio padrão observados nos dados históricos.
            Definir um número de simulações (ex.: 1000) e um horizonte de tempo de 2 a 5 anos.
            Para cada simulação, usar a previsão do ensemble como base e adicionar variação de acordo com a volatilidade projetada.
RefifnRe
        |> Visualização dos Resultados:

            Plotar as trajetórias simuladas sobre a previsão do ensemble.
            Gerar uma tabela que exiba os percentis das projeções para ajudar na análise (exemplo: percentis 5%, 50%, e 95%).
'''

# t20_script_monte_carlo_simulation.py

# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
import time

# Função para carregar previsões do ensemble existente


def load_ensemble_forecast(filepath):
    df = pd.read_csv(filepath, parse_dates=['Data'])
    df.set_index('Data', inplace=True)
    return df

# Função para calcular volatilidade dinâmica com GARCH


def calculate_garch_volatility(returns, horizon):
    """
    Estima a volatilidade dinâmica usando o modelo GARCH.

    Args:
        returns (pd.Series): Série de retornos históricos.
        horizon (int): Período de previsão em meses.

    Returns:
        np.array: Volatilidades previstas para o horizonte.
    """
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    forecast = model_fit.forecast(horizon=horizon)
    garch_volatility = np.sqrt(
        forecast.variance.values[-1]) / 100  # Ajuste de escala
    return garch_volatility

# Função de Simulação de Monte Carlo usando volatilidade dinâmica


def monte_carlo_simulation_with_garch(ensemble_forecast, returns, num_simulations=1000, upper_limit=1.2, lower_limit=0.8):
    """
    Realiza a simulação de Monte Carlo com volatilidade dinâmica baseada no modelo GARCH.
    """
    np.random.seed(42)  # Para resultados reproduzíveis

    # Usar o último valor do ensemble como ponto de partida
    last_value = ensemble_forecast.iloc[-1]
    horizon = len(ensemble_forecast)  # Usar o mesmo horizonte do ensemble
    simulations = np.zeros((horizon, num_simulations))

    # Obter volatilidade dinâmica com o modelo GARCH
    garch_volatility = calculate_garch_volatility(returns, horizon)

    # Gerar simulações para o período e número de trajetórias definido
    for i in range(num_simulations):
        path = [last_value]

        for step in range(horizon):
            shock = np.random.normal(0, garch_volatility[step])
            next_value = path[-1] * (1 + shock)
            next_value = max(
                min(next_value, path[-1] * upper_limit), path[-1] * lower_limit)
            path.append(next_value)

        simulations[:, i] = path[1:]

    # Definir o índice de datas para coincidir exatamente com o ensemble
    simulation_df = pd.DataFrame(simulations)
    simulation_df.index = ensemble_forecast.index  # Alinhar com o índice do ensemble
    return simulation_df

# Função para visualizar os resultados finais


def plot_monte_carlo_with_ensemble(ensemble_forecast, simulation_df):
    forecast_dates = simulation_df.index

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Simulação de Monte Carlo e Ensemble", "Análise Estatística"],
                        row_heights=[0.7, 0.3], specs=[[{"type": "xy"}], [{"type": "table"}]])

    # Adicionar a linha do ensemble base
    fig.add_trace(go.Scatter(x=ensemble_forecast.index, y=ensemble_forecast,
                  mode='lines', name='Ensemble Base', line=dict(color='red')), row=1, col=1)

    # Adicionar todas as trajetórias de Monte Carlo com opacidade para diferenciar
    for col in simulation_df.columns:
        fig.add_trace(go.Scatter(x=forecast_dates, y=simulation_df[col], mode='lines', line=dict(
            color='rgba(0,0,255,0.02)'), showlegend=False), row=1, col=1)

    # Adicionar os percentis 5%, 50% (mediana) e 95%
    fig.add_trace(go.Scatter(x=forecast_dates, y=simulation_df.quantile(0.05, axis=1),
                  mode='lines', name='5% Percentil', line=dict(color='green', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=simulation_df.quantile(0.50, axis=1),
                  mode='lines', name='Mediana (50%)', line=dict(color='blue', dash='dashdot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=simulation_df.quantile(0.95, axis=1),
                  mode='lines', name='95% Percentil', line=dict(color='purple', dash='dot')), row=1, col=1)

    # Tabela com as estatísticas principais das trajetórias
    statistics_df = pd.DataFrame({
        "Data": forecast_dates,
        "5% Percentil": simulation_df.quantile(0.05, axis=1).values,
        "Mediana (50%)": simulation_df.quantile(0.50, axis=1).values,
        "95% Percentil": simulation_df.quantile(0.95, axis=1).values,
        "Erro Médio Absoluto (MAE)": abs(simulation_df.quantile(0.50, axis=1) - ensemble_forecast.values),
        "Desvio Padrão": simulation_df.std(axis=1)
    })

    fig.add_trace(go.Table(header=dict(values=["Data", "5% Percentil", "Mediana (50%)", "95% Percentil", "Erro Médio Absoluto (MAE)", "Desvio Padrão"], align='center', fill_color='lightgrey'),
                           cells=dict(values=[statistics_df[col] for col in statistics_df.columns], align='center')), row=2, col=1)

    fig.update_layout(
        title="Simulação de Monte Carlo e Ensemble para Médio e Longo Prazo", showlegend=True, height=800)
    fig.show()

# Função principal para execução do fluxo completo


def main():
    filepath = 'TESTES/DADOS/ensemble_forecast.csv'
    ensemble_forecast = load_ensemble_forecast(filepath)['Ensemble']

    # Calcular retornos históricos para o modelo GARCH
    returns = ensemble_forecast.pct_change().dropna() * 100  # Retornos percentuais

    num_simulations = 1000

    start_time = time.time()
    simulation_df = monte_carlo_simulation_with_garch(
        ensemble_forecast, returns, num_simulations)
    print(f"Tempo para execução da Simulação de Monte Carlo: {
          time.time() - start_time:.2f} segundos")

    plot_monte_carlo_with_ensemble(ensemble_forecast, simulation_df)


if __name__ == "__main__":
    main()
