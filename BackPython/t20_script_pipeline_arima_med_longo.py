'''
    ||> Objetivo: Nesta etapa, será a base para cada modelo, sem entrar em detalhes de ajuste fino. Isso nos permitirá conectar os módulos e realizar os primeiros testes de integração.

Estrutura Comum para Todos os Algoritmos:

Este código fornece uma estrutura comum para o pipeline, contendo:
Carregamento e preparação dos dados(prepare_data), com a preparação de frequências mensais e preenchimento de valores ausentes.
Módulos para cada tipo de modelo(Prophet, ARIMA, GARCH, e LSTM Seq2Seq), onde cada modelo pode ser executado e avaliado individualmente.
Ensemble Final(ensemble_forecast), combinando as previsões de diferentes modelos para obter uma previsão mais robusta.
Ponto de Partida para Cada Algoritmo:

Usaremos este código como "base" para cada modelo de machine learning que planejamos testar. Cada modelo tem seu próprio bloco de código específico, mas a estrutura geral permanece a mesma.
Quando testarmos um novo algoritmo ou ajustarmos hiperparâmetros, faremos isso dentro desta estrutura, substituindo ou ajustando o módulo específico para aquele modelo.
Modularidade e Reusabilidade:

Como o código é modular, ele facilita a adição e a troca de algoritmos. Podemos desenvolver e testar cada modelo(Prophet, ARIMA, GARCH, LSTM Seq2Seq) separadamente e, depois, integrar facilmente ao pipeline final.
Cada componente é encapsulado em funções(prophet_model, arima_model, garch_model, lstm_seq2seq_model, etc.), permitindo que ajustemos um modelo sem impactar os outros.
Refinamento Iterativo e Ensemble:

Após testar e validar cada modelo individualmente, refinaremos o ensemble. A função ensemble_forecast agora usa uma média simples das previsões. Em etapas futuras, podemos ajustar o ensemble com combinações ponderadas ou algoritmos de regressão, usando os resultados individuais dos modelos.
Isso também nos permite adicionar ou remover modelos do ensemble facilmente, dependendo de seu desempenho.
Exemplo de Iteração no Pipeline
Se decidirmos testar um novo modelo, como XGBoost para prever a tendência de longo prazo, por exemplo, podemos:

Adicionar uma função xgboost_model ao código para treinar e prever usando XGBoost.
Inserir as previsões de XGBoost no ensemble junto com os outros modelos.
Testar se o ensemble melhora com esse novo modelo.

'''

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Função para carregar e preparar os dados


def prepare_data(filepath, target_column):
    # Carregar o arquivo
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print("Amostra dos dados:")
    print(df.head())
    print("Colunas dos dados:", df.columns)

    # Resampling para frequência mensal (usando 'ME' para evitar warnings)
    df = df.resample('ME').mean()
    df = df.ffill()  # Preenchendo valores ausentes
    return df

# Modelo Prophet ajustado para capturar sazonalidade


def prophet_model(df, target_column, periods=60):
    prophet_df = df[[target_column]].reset_index()
    prophet_df.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].iloc[-periods:].reset_index(drop=True)

# Modelo ARIMA com parâmetros ajustados


def arima_model(df, target_column, order=(5, 1, 2), steps=60):
    model = ARIMA(df[target_column], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast = np.clip(forecast, 0, None)
    return pd.Series(forecast, name='ARIMA_Forecast').reset_index(drop=True)

# Modelo GARCH para prever volatilidade dinâmica mês a mês


def garch_dynamic_model(df, target_column, steps=60):
    returns = df[target_column].pct_change().dropna() * 100
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=steps)
    volatility_forecast = np.sqrt(forecast.variance.values).flatten()
    return pd.Series(volatility_forecast, name='GARCH_Volatility')

# Função para ensemble de previsões com volatilidade dinâmica


def ensemble_forecast(prophet_forecast, arima_forecast, garch_volatility):
    min_length = min(len(prophet_forecast), len(
        arima_forecast), len(garch_volatility))
    prophet_forecast = prophet_forecast['yhat'].iloc[:min_length].reset_index(
        drop=True)
    arima_forecast = arima_forecast.iloc[:min_length].reset_index(drop=True)
    garch_volatility = garch_volatility.iloc[:min_length].reset_index(
        drop=True)

    # Cálculo do ensemble e ajuste da volatilidade
    combined_forecast = (prophet_forecast + arima_forecast) / 2
    combined_forecast *= (1 + garch_volatility / 100)
    return combined_forecast

# Função para exibir e plotar resultados


def display_forecast_results(df, combined_forecast, prophet_forecast, arima_forecast):
    forecast_dates = pd.date_range(
        df.index[-1], periods=len(combined_forecast) + 1, freq='ME')[1:]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Ensemble de Previsão", "Componentes do Ensemble"],
                        column_widths=[0.7, 0.3], specs=[[{"type": "xy"}, {"type": "table"}]])

    # Gráfico de previsão e tabela
    fig.add_trace(go.Scatter(x=df.index, y=df['^BVSP'], mode='lines', name='Histórico', line=dict(
        color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=combined_forecast, mode='lines',
                  name='Ensemble', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=prophet_forecast['yhat'], mode='lines', name='Prophet', line=dict(
        color='green', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast, mode='lines',
                  name='ARIMA', line=dict(color='purple', dash='dashdot')), row=1, col=1)

    fig.add_trace(go.Table(header=dict(values=["Data", "Ensemble"], align='center', fill_color='lightgrey'),
                           cells=dict(values=[forecast_dates, combined_forecast], align='center')), row=1, col=2)

    fig.update_layout(
        title="Previsões de Médio e Longo Prazo (2 a 5 anos)", showlegend=True, height=600)
    fig.show()

# Função principal para execução


def main():
    filepath = 'TESTES/DADOS/train_data_combined.csv'
    target_column = '^BVSP'

    start_time = time.time()

    # Carregar e preparar os dados
    df = prepare_data(filepath, target_column)
    print(f"Tempo para carregar e preparar os dados: {
          time.time() - start_time:.2f} segundos")

    # Executar os modelos e medir o tempo de cada um
    start_prophet = time.time()
    prophet_forecast = prophet_model(df, target_column, periods=60)
    print(f"Tempo para execução do modelo Prophet: {
          time.time() - start_prophet:.2f} segundos")

    start_arima = time.time()
    arima_forecast = arima_model(df, target_column, order=(5, 1, 2), steps=60)
    print(f"Tempo para execução do modelo ARIMA: {
          time.time() - start_arima:.2f} segundos")

    start_garch = time.time()
    garch_volatility = garch_dynamic_model(df, target_column, steps=60)
    print(f"Tempo para execução do modelo GARCH: {
          time.time() - start_garch:.2f} segundos")

    # Ensemble de previsões com volatilidade dinâmica
    start_ensemble = time.time()
    combined_forecast = ensemble_forecast(
        prophet_forecast, arima_forecast, garch_volatility)
    print(f"Tempo para execução do ensemble: {
          time.time() - start_ensemble:.2f} segundos")

    # Exibir e plotar os resultados
    display_forecast_results(df, combined_forecast,
                             prophet_forecast, arima_forecast)
    print(f"Tempo total de execução: {time.time() - start_time:.2f} segundos")


if __name__ == "__main__":
    main()
