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


def prever_com_prophet(dados, dias=126):
    """
    Configura e treina o modelo Prophet para previsão de curto prazo.
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


def verificar_entrada(dados_prophet):
    """
    Verifica os dados de entrada para o Prophet.
    """
    print("[DEBUG] Verificando os dados preparados para Prophet:")
    print(dados_prophet.head())
    print("[DEBUG] Últimos registros dos dados preparados para Prophet:")
    print(dados_prophet.tail())


def plotar_previsao(previsao, horizonte=126):
    """
    Plota a previsão do Prophet, limitada ao horizonte de 6 meses (126 dias úteis).
    """
    import matplotlib.pyplot as plt

    # Limitar os dados ao horizonte de 6 meses
    previsao['ds'] = pd.to_datetime(previsao['ds'])
    previsao_horizonte = previsao.tail(horizonte)

    # Plotar a previsão
    plt.figure(figsize=(14, 7))
    plt.plot(previsao_horizonte['ds'], previsao_horizonte['yhat'],
             label='Previsão (yhat)', color='blue')
    plt.fill_between(previsao_horizonte['ds'], previsao_horizonte['yhat_lower'], previsao_horizonte['yhat_upper'],
                     color='blue', alpha=0.2, label='Intervalo de Confiança (95%)')
    plt.title('Previsão com Prophet - Horizonte de 6 Meses')
    plt.xlabel('Data')
    plt.ylabel('Preço Estimado')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    """
    Função principal para previsão com Prophet.
    """
    print("[INFO] Iniciando previsão com Prophet...")

    # Carregar os dados preparados
    dados = carregar_dados_preparados(PREPARED_DATA_PATH)

    # Escolher um ativo para análise
    ativo = "VALE3.SA"

    # Preparar os dados para Prophet
    dados_prophet = preparar_dados_prophet(dados, ativo)

    # Verificar os dados preparados
    verificar_entrada(dados_prophet)

    # Prever com Prophet
    previsao, modelo = prever_com_prophet(dados_prophet, dias=126)

    # Salvar os resultados
    salvar_resultados(previsao, os.path.join(
        OUTPUT_DIR, f"{ativo}_prophet_previsao.csv"))

    # Plotar a previsão
    plotar_previsao(previsao, horizonte=126)

    print("[INFO] Previsão concluída. Resultados salvos e gráfico gerado.")


if __name__ == "__main__":
    main()
