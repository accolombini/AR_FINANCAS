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
from dash import dcc, html
import dash
import plotly.express as px


def simulate_future_returns(returns, num_simulations, forecast_horizon):
    """
    Simula retornos futuros para um ativo usando Monte Carlo.
    """
    scale = returns.std()
    rescale_factor = 10 / scale if scale < 1 else 10
    returns_scaled = returns * rescale_factor

    # Ajustar modelo GARCH
    model = arch_model(returns_scaled, vol="Garch", p=1, q=1, dist="normal")
    fitted_model = model.fit(disp="off")

    omega = fitted_model.params["omega"]
    alpha = fitted_model.params["alpha[1]"]
    beta = fitted_model.params["beta[1]"]
    variance = fitted_model.conditional_volatility.iloc[-1] ** 2
    last_return = returns_scaled.iloc[-1]

    simulated_paths = []
    for _ in range(num_simulations):
        current_variance = variance
        path = []
        for _ in range(forecast_horizon):
            innovation = np.random.normal(0, np.sqrt(current_variance))
            simulated_return = last_return + innovation
            current_variance = omega + alpha * \
                (innovation ** 2) + beta * current_variance
            path.append(simulated_return)
        simulated_paths.append(path)

    simulated_paths = np.array(simulated_paths) / rescale_factor
    percentiles = {
        "5%": np.percentile(simulated_paths, 5, axis=0),
        "50%": np.percentile(simulated_paths, 50, axis=0),
        "95%": np.percentile(simulated_paths, 95, axis=0),
    }
    return simulated_paths, percentiles


def consolidate_portfolio_forecast(forecast_results, weights):
    """
    Consolida as previsões Monte Carlo para o portfólio com base nos pesos.
    """
    portfolio_simulated = {"5%": [], "50%": [], "95%": []}
    for scenario in ["5%", "50%", "95%"]:
        for day in range(len(next(iter(forecast_results.values()))[1][scenario])):
            weighted_sum = sum(
                forecast_results[asset][1][scenario][day] * weights[asset]
                for asset in weights
            )
            portfolio_simulated[scenario].append(weighted_sum * 100)
    return portfolio_simulated


if __name__ == "__main__":
    # Caminhos para os arquivos
    historical_data_path = "BackPython/DADOS/historical_data_cleaned.csv"
    portfolio_weights_path = "BackPython/DADOS/portfolio_otimizado.csv"

    # Carregar dados históricos
    historical_data = pd.read_csv(historical_data_path, parse_dates=[
                                  "Date"]).set_index("Date")
    portfolio_optimized = pd.read_csv(portfolio_weights_path)

    # Última coluna é o benchmark
    benchmark_column = historical_data.columns[-1]
    benchmark_annual_return = 8  # Taxa média do índice BOVESPA (8% ao ano)
    benchmark_days = 1200
    benchmark_daily_return = (
        1 + benchmark_annual_return / 100) ** (1 / 252) - 1
    benchmark_cumulative = [(1 + benchmark_daily_return)
                            ** day for day in range(benchmark_days)]

    # Filtrar ativos relevantes
    weights_dict = dict(
        zip(portfolio_optimized["Ativo"], portfolio_optimized["Peso (%)"]))
    relevant_assets = weights_dict.keys()

    # Previsões simuladas
    forecast_results = {
        asset: simulate_future_returns(
            np.log(historical_data[asset] /
                   historical_data[asset].shift(1)).dropna(),
            num_simulations=1000,
            forecast_horizon=1200  # 5 anos (~252 dias úteis por ano)
        )
        for asset in relevant_assets
    }

    # Consolidar previsões para o portfólio
    portfolio_forecast = consolidate_portfolio_forecast(
        forecast_results, weights_dict)

    # Criar DataFrame para o gráfico
    forecast_df = pd.DataFrame({
        "Day": list(range(1, benchmark_days + 1)),
        "Portfolio (5%)": portfolio_forecast["5%"],
        "Portfolio (50%)": portfolio_forecast["50%"],
        "Portfolio (95%)": portfolio_forecast["95%"],
        "Benchmark": [value * 100 - 100 for value in benchmark_cumulative],
    })

    # Dashboard com Dash
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Monte Carlo: Cenários Prospectivos (5 anos)"),
        dcc.Graph(
            id="portfolio-forecast",
            figure=px.line(
                forecast_df,
                x="Day",
                y=["Portfolio (5%)", "Portfolio (50%)",
                   "Portfolio (95%)", "Benchmark"],
                title="Cenários Prospectivos: Portfólio vs Benchmark"
            ).update_traces(mode="lines", line=dict(dash="dot", width=2))
        )
    ])

    app.run_server(debug=True)
