''' 
    Estrutura:

    Simular o portfólio em uma janela de 6 meses, considerando o mesmo modelo EGARCH.
    Incorporar retornos mês a mês, além de médias móveis para identificar tendências de curto prazo.
    
    ||> Métricas Adicionais:

        Retorno Mensal Acumulado.
        Volatilidade Mensal.
        Correlação com o Benchmark.
    
    ||> Visualização:

        Gráficos com foco em tendências de curto prazo:
        Retornos mensais acumulados do portfólio vs benchmark.
        Análise de dispersão dos retornos diários no período.
    
    ||> Análise de Ajuste Dinâmico:

        Possibilidade de sugerir ajustes nos pesos do portfólio caso algum ativo apresente desempenho muito descolado da média.

    ||> Flexibilidade para Avaliar Outros Algoritmos:

        Deixar aberta a possibilidade de utilizar outros modelos além do EGARCH é muito importante.
        Isso nos permite testar métodos como GARCH simplificado, SARIMA, ou até mesmo redes neurais LSTM, caso necessário.
        Podemos introduzir uma camada de validação cruzada para comparar diferentes algoritmos e decidir qual se        adapta melhor ao curto prazo.
    
    ||> Portfólio Sempre Superando o Índice:

        Adicionar uma métrica que garanta, na média mensal, que o portfólio supere o benchmark é crucial para validar a eficácia da estratégia.
        Essa abordagem nos ajuda a garantir que o portfólio está gerando "alpha" (retorno acima do mercado) consistentemente.
        Podemos implementar uma lógica para ajustar dinamicamente os pesos caso o portfólio esteja ficando muito próximo ou abaixo do benchmark.

'''

# Importar bibliotecas necessárias
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from dash import Dash, dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Função para simulação de Monte Carlo baseada em distribuições empíricas


def monte_carlo_simulation_empirical(returns, num_simulations=1000, days_ahead=180):
    '''
    Realiza simulação de Monte Carlo baseada em distribuições empíricas.

    Parâmetros:
        returns (pd.Series): Retornos logarítmicos históricos.
        num_simulations (int): Número de simulações a serem realizadas.
        days_ahead (int): Número de dias úteis no horizonte de projeção.

    Retorno:
        np.ndarray: Cenários simulados com shape (num_simulations, days_ahead).
    '''
    returns_sample = returns.dropna().values
    simulations = np.zeros((num_simulations, days_ahead))
    for i in range(num_simulations):
        simulated_path = [returns_sample[-1]]
        for _ in range(days_ahead - 1):
            simulated_path.append(np.random.choice(returns_sample))
        simulations[i, :] = simulated_path
    return simulations


# Caminhos corretos para os dados
data_path = "BackPython/DADOS/historical_data_cleaned.csv"
portfolio_path = "BackPython/DADOS/portfolio_otimizado.csv"

# Carregar dados
historical_data = pd.read_csv(data_path, parse_dates=["Date"])
portfolio_optimized = pd.read_csv(portfolio_path)

# Calcular retornos logarítmicos
for column in historical_data.columns[1:]:
    historical_data[f"{column}_log_return"] = np.log(
        historical_data[column] / historical_data[column].shift(1))
historical_data = historical_data.dropna()

# Calcular retorno logarítmico ponderado do portfólio
portfolio_weights = portfolio_optimized.set_index("Ativo")["Peso (%)"] / 100.0
aligned_weights = portfolio_weights.reindex(
    [col.split("_log_return")[0]
     for col in historical_data.columns if "_log_return" in col],
    fill_value=0
)

# Garantir que nomes das colunas e índices de pesos estão alinhados
matching_columns = [f"{asset}_log_return" for asset in aligned_weights.index]
aligned_weights = aligned_weights.loc[aligned_weights.index.intersection(
    [col.split("_log_return")[0] for col in matching_columns])]

# Validar alinhamento antes de prosseguir
if len(matching_columns) != len(aligned_weights):
    raise ValueError(
        "Nomes das colunas de retornos e índices de pesos no portfólio não estão alinhados.")

# Calcular retorno ponderado do portfólio
historical_data["Portfolio_log_return"] = historical_data[matching_columns].dot(
    aligned_weights.values)

# Preparar dados para o modelo
X = historical_data[matching_columns]
y = historical_data["Portfolio_log_return"]

# Configurar validação cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)
params = {"n_estimators": 300, "max_depth": 6,
          "learning_rate": 0.03, "reg_lambda": 1.0, "reg_alpha": 0.5}
model = XGBRegressor(**params)

# Validar modelo com validação cruzada
mae_list, rmse_list, r2_list = [], [], []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_list.append(mean_absolute_error(y_test, y_pred))
    rmse_list.append(root_mean_squared_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

avg_mae = np.mean(mae_list)
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)

print(f"MAE Médio: {avg_mae:.4f}")
print(f"RMSE Médio: {avg_rmse:.4f}")
print(f"R² Médio: {avg_r2:.4f}")

# Previsões iterativas para os próximos 180 dias úteis
future_dates = pd.date_range(
    start=historical_data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=180, freq="B"
)
last_features = X.iloc[-1].values.reshape(1, -1)
future_returns = []

# Introduzir variabilidade nas previsões iterativas para refletir volatilidade
for _ in range(180):
    next_return = model.predict(last_features)[0]
    # Adicionar um ruído baseado na variabilidade histórica
    next_return += np.random.normal(0, y.std())
    future_returns.append(next_return)
    # Atualizar as features para a próxima previsão
    last_features = np.roll(last_features, -1)  # Deslocar as features
    last_features[0, -1] = next_return  # Adicionar a nova previsão

# Agregar previsões por mês
future_df = pd.DataFrame(
    {"Date": future_dates, "Predicted_Returns": future_returns})
future_df["Month"] = future_df["Date"].dt.to_period("M")
monthly_returns = future_df.groupby("Month")["Predicted_Returns"].sum()

# Simulação de Monte Carlo
mc_simulations = monte_carlo_simulation_empirical(
    historical_data["Portfolio_log_return"], days_ahead=180
)

# Garantir que os percentis são calculados corretamente
try:
    mc_percentiles = {
        "5%": np.percentile(mc_simulations, 5, axis=0),
        "50%": np.percentile(mc_simulations, 50, axis=0),
        "95%": np.percentile(mc_simulations, 95, axis=0),
    }
except Exception as e:
    raise ValueError(f"Erro ao calcular percentis de Monte Carlo: {e}")

# Dashboard com Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Projeção de Retornos - Previsões e Monte Carlo"),
    dcc.Graph(
        id="performance-graph",
        figure={
            "data": [
                go.Scatter(x=historical_data["Date"], y=historical_data["Portfolio_log_return"],
                           mode="lines", name="Real"),
                go.Scatter(x=historical_data["Date"][-len(y_test):], y=y_pred,
                           mode="lines", name="Previsto"),
            ],
            "layout": go.Layout(
                title="Desempenho do Modelo",
                xaxis_title="Data",
                yaxis_title="Retorno Logarítmico",
                legend_title="Curvas",
            ),
        },
    ),
    dcc.Graph(
        id="forecast-graph",
        figure={
            "data": [
                go.Scatter(x=future_dates, y=mc_percentiles["50%"],
                           mode="lines", name="Monte Carlo (Mediana)"),
                go.Scatter(x=future_dates, y=mc_percentiles["5%"],
                           mode="lines", name="Monte Carlo (5%)", line=dict(dash="dot")),
                go.Scatter(x=future_dates, y=mc_percentiles["95%"],
                           mode="lines", name="Monte Carlo (95%)", line=dict(dash="dot")),
                go.Scatter(x=future_dates, y=future_returns,
                           mode="lines", name="Previsão do Modelo", line=dict(color="red")),
            ],
            "layout": go.Layout(
                title="Projeção de Retornos Futuros (180 Dias)",
                xaxis_title="Data",
                yaxis_title="Retorno Logarítmico",
                legend_title="Cenários",
            ),
        },
    ),
    html.Div([
        html.H3("Taxa de Retorno Mensal"),
        html.Ul([html.Li(f"{month}: {return_val:.4%}")
                for month, return_val in monthly_returns.items()]),
    ]),
])

if __name__ == "__main__":
    app.run_server(debug=True)
