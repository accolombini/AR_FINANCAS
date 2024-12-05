# BP_mod6_ML_curto_prazo.py: Previsão de Retornos/Preços no Curto Prazo
# -----------------------------------------------------------
# Este módulo utiliza ARIMA/SARIMA para prever preços ou retornos
# de ativos no curto prazo (6 meses) e usa Plotly para visualizações.
# Agora restaura o gráfico e exibe a tabela de métricas.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import BP_mod1_config as config  # Importa configurações globais

# ---------------------------
# Funções Auxiliares
# ---------------------------


def verificar_estacionariedade(serie):
    """
    Verifica se a série temporal é estacionária usando o teste ADF.

    Args:
        serie (pd.Series): Série temporal.

    Returns:
        bool: True se estacionária, False caso contrário.
    """
    resultado = adfuller(serie)
    p_valor = resultado[1]
    return p_valor < 0.05


def ajustar_modelo_arima(data, ordem, sazonal=None):
    """
    Ajusta o modelo ARIMA ou SARIMA aos dados fornecidos.

    Args:
        data (pd.Series): Série temporal para ajuste.
        ordem (tuple): Parâmetros (p, d, q) do modelo ARIMA.
        sazonal (tuple): Parâmetros sazonais (P, D, Q, s) para SARIMA (opcional).

    Returns:
        SARIMAX: Modelo ajustado.
    """
    modelo = SARIMAX(data, order=ordem, seasonal_order=sazonal,
                     enforce_stationarity=False, enforce_invertibility=False)
    return modelo.fit(disp=False)


def avaliar_modelo(y_real, y_prev):
    """
    Avalia o desempenho do modelo usando métricas como RMSE e MAE.

    Args:
        y_real (pd.Series): Valores reais.
        y_prev (pd.Series): Valores previstos.

    Returns:
        dict: Métricas calculadas.
    """
    rmse = np.sqrt(mean_squared_error(y_real, y_prev))
    mae = mean_absolute_error(y_real, y_prev)
    return {"RMSE": rmse, "MAE": mae}


def exibir_dashboard(data_real, data_prevista, metricas, titulo):
    """
    Exibe o gráfico de previsões e uma tabela de métricas usando Plotly.

    Args:
        data_real (pd.Series): Dados reais.
        data_prevista (pd.Series): Dados previstos.
        metricas (dict): Métricas de avaliação do modelo.
        titulo (str): Título do gráfico.
    """
    # Gráfico de previsões
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data_real.index,
        y=data_real,
        mode='lines',
        name='Real',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=data_real.index,
        y=data_prevista,
        mode='lines',
        name='Previsto',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Tempo",
        yaxis_title="Valor",
        legend=dict(x=0, y=-0.2, orientation="h"),
        height=600
    )

    # Tabela de métricas
    tabela = go.Figure(data=[go.Table(
        header=dict(values=["Métrica", "Valor"],
                    fill_color="lightgrey",
                    align="center"),
        cells=dict(values=[list(metricas.keys()), list(metricas.values())],
                   fill_color="white",
                   align="center")
    )])

    tabela.update_layout(title="Métricas de Desempenho", height=300)

    # Exibir ambos
    fig.show()
    tabela.show()

# ---------------------------
# Fluxo Principal
# ---------------------------


def main():
    # Configurações
    input_file = config.PREPARED_DATA_PATH  # Caminho dinâmico do arquivo
    target_col = "VALE3.SA"  # Exemplo de ativo
    horizonte_previsao = 126  # Aproximadamente 6 meses (21 dias úteis/mês)

    # Carregar dados
    print("Carregando dados...")
    df = pd.read_csv(input_file, index_col="Date",
                     parse_dates=True).asfreq("B")
    data = df[target_col].dropna()

    # Verificar estacionariedade
    print("Verificando estacionariedade...")
    estacionaria = verificar_estacionariedade(data)
    if not estacionaria:
        print("[INFO] A série não é estacionária. Aplicando diferenciação.")
        data = data.diff().dropna()

    # Ajustar modelo
    print("Ajustando modelo ARIMA...")
    ordem = (1, 1, 1)  # Parâmetros iniciais para ARIMA
    modelo = ajustar_modelo_arima(data, ordem)

    # Previsões
    print("Gerando previsões para o curto prazo...")
    previsoes = modelo.get_forecast(steps=horizonte_previsao)
    previsoes_valores = previsoes.predicted_mean

    # Avaliação
    print("Avaliando modelo...")
    y_real = data[-horizonte_previsao:]
    metricas = avaliar_modelo(y_real, previsoes_valores)
    print("Métricas de Desempenho:", metricas)

    # Exibir dashboard
    print("Exibindo gráfico e tabela de métricas...")
    exibir_dashboard(data[-horizonte_previsao:], previsoes_valores,
                     metricas, "Previsão de Curto Prazo com ARIMA")


if __name__ == "__main__":
    main()
