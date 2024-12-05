import os
import pandas as pd
from prophet import Prophet
from BP_mod1_config import PREPARED_DATA_PATH, OUTPUT_DIR


def carregar_dados_preparados(caminho):
    """
    Carrega os dados preparados do arquivo CSV, sem transformações nos preços.
    """
    dados = pd.read_csv(caminho)
    dados['Date'] = pd.to_datetime(dados['Date'])
    return dados


def preparar_dados_prophet(dados, ativo):
    """
    Prepara os dados para o Prophet, garantindo que os valores originais sejam usados.
    """
    df = dados[['Date', ativo]].rename(columns={'Date': 'ds', ativo: 'y'})
    return df


def prever_com_prophet(dados, dias=1260):
    """
    Configura e treina o modelo Prophet para previsão de longo prazo.
    """
    modelo = Prophet()
    modelo.fit(dados)

    # Criar dataframe de datas futuras
    futuro = modelo.make_future_dataframe(periods=dias, freq='B')
    previsao = modelo.predict(futuro)

    return previsao, modelo


def salvar_resultados(resultados, caminho):
    """
    Salva os resultados da previsão em um arquivo CSV.
    """
    resultados.to_csv(caminho, index=False)
    print(f"[INFO] Resultados salvos em: {caminho}")


def plotar_previsao(previsao, horizonte=1260):
    """
    Plota a previsão do Prophet, limitada ao horizonte de 5 anos (1260 dias úteis).
    """
    import matplotlib.pyplot as plt

    # Limitar os dados ao horizonte de 5 anos
    previsao['ds'] = pd.to_datetime(previsao['ds'])
    previsao_horizonte = previsao.tail(horizonte)

    # Plotar a previsão
    plt.figure(figsize=(14, 7))
    plt.plot(previsao_horizonte['ds'], previsao_horizonte['yhat'],
             label='Previsão (yhat)', color='blue')
    plt.fill_between(previsao_horizonte['ds'], previsao_horizonte['yhat_lower'], previsao_horizonte['yhat_upper'],
                     color='blue', alpha=0.2, label='Intervalo de Confiança (95%)')
    plt.title('Previsão com Prophet - Horizonte de 5 Anos')
    plt.xlabel('Data')
    plt.ylabel('Preço Estimado')
    plt.legend()
    plt.grid()
    plt.show()


def calcular_retorno_anual(previsao, preco_inicial):
    """
    Calcula o retorno médio anual esperado com base na previsão.
    """
    preco_final = previsao['yhat'].iloc[-1]
    anos = 5  # Horizonte de previsão
    retorno_anual = ((preco_final / preco_inicial) ** (1 / anos)) - 1
    return retorno_anual * 100  # Retorno em porcentagem


def main():
    """
    Função principal para previsão com Prophet e cálculo de retorno médio anual.
    """
    print("[INFO] Iniciando previsão de longo prazo com Prophet...")

    # Carregar os dados preparados
    dados = carregar_dados_preparados(PREPARED_DATA_PATH)

    # Escolher um ativo para análise
    ativo = "VALE3.SA"

    # Preparar os dados para Prophet
    dados_prophet = preparar_dados_prophet(dados, ativo)

    # Prever com Prophet
    previsao, modelo = prever_com_prophet(dados_prophet, dias=1260)

    # Salvar os resultados
    salvar_resultados(previsao, os.path.join(
        OUTPUT_DIR, f"{ativo}_prophet_5anos_previsao.csv"))

    # Plotar a previsão
    plotar_previsao(previsao, horizonte=1260)

    # Calcular retorno médio anual
    preco_inicial = dados_prophet['y'].iloc[-1]
    retorno_anual = calcular_retorno_anual(previsao, preco_inicial)
    print(f"[INFO] Retorno médio anual esperado: {retorno_anual:.2f}%")

    print("[INFO] Previsão concluída. Resultados salvos e gráfico gerado.")


if __name__ == "__main__":
    main()
