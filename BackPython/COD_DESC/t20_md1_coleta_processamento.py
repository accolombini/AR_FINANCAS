'''
    |||> Objetivo: este código automatiza a coleta e processamento de dados históricos de ativos financeiros. Após baixar os dados, ele aplica uma série de cálculos técnicos, facilitando análises de tendências e volatilidade. O resultado é salvo em um arquivo CSV, pronto para visualização ou análise adicional.
    
    ||> Módulo 1: Coleta e Prepara os Dados

        |> Ajustes: feature Engineering Avançado: Adicione features que capturam sazonalidades de longo prazo (indicadores específicos, médias móveis de 6 a 12 meses) para apoiar previsões de longo prazo.
    Output atualizado: Dados históricos preparados, com features de curto e longo prazo e indicadores econômicos adicionais.
    
    |||> Your registered API key is: 8e5112d9a5cdfb3cff60e7486f2bec09 Documentation is available on the St. Louis Fed web services website. 8e5112d9a5cdfb3cff60e7486f2bec09 -> teste realizado para verificar acesso a API do governo americano
'''

# t20_md1_coleta_processamento.py
# Este módulo realiza a coleta e processamento de dados financeiros para uma série de ativos,
# calculando indicadores técnicos para análise de mercado.

# Importando as bibliotecas necessárias
import yfinance as yf  # Biblioteca para coleta de dados financeiros
import pandas as pd    # Biblioteca para manipulação de dados em formato de tabela
# Biblioteca para fazer requisições HTTP (não usada neste código)
import requests

# Função para baixar os dados dos ativos
# Parâmetros:
#   - assets: lista de ativos financeiros (e.g., ['VALE3.SA', 'PETR4.SA'])
#   - start_date: data inicial para coleta no formato 'AAAA-MM-DD'
#   - end_date: data final para coleta no formato 'AAAA-MM-DD'
# Retorna: DataFrame com os preços ajustados de fechamento para os ativos no período especificado


def get_asset_data(assets, start_date, end_date):
    df = yf.download(assets, start=start_date, end=end_date)
    return df['Adj Close']

# Função para adicionar features (médias móveis, volatilidade, RSI, etc.)
# Parâmetros:
#   - df: DataFrame com os dados de preços dos ativos
#   - assets: lista de ativos para os quais serão calculadas as features
# Retorna: DataFrame com novas colunas para cada indicador técnico calculado


def add_features(df, assets):
    for asset in assets:
        # Calcula o retorno diário do ativo
        df.loc[:, f'{asset}_returns'] = df[asset].pct_change(fill_method=None)

        # Calcula médias móveis para 30 e 180 dias
        df.loc[:, f'{asset}_ma_30'] = df[asset].rolling(window=30).mean()
        df.loc[:, f'{asset}_ma_180'] = df[asset].rolling(window=180).mean()

        # Calcula a volatilidade para 30 e 180 dias
        df.loc[:, f'{asset}_volatility_30'] = df[asset].rolling(
            window=30).std()
        df.loc[:, f'{asset}_volatility_180'] = df[asset].rolling(
            window=180).std()

        # Calcula o Índice de Força Relativa (RSI) com janela de 14 dias
        # RSI = 100 - (100 / (1 + média dos ganhos / média das perdas))
        df.loc[:, f'{asset}_rsi'] = df[asset].pct_change(fill_method=None).rolling(window=14).apply(
            lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean())))) if len(x) > 0 else 50, raw=False)

        # Calcula as Bandas de Bollinger para 20 dias (hband: limite superior, lband: limite inferior)
        df.loc[:, f'{asset}_bb_hband'] = df[asset].rolling(
            window=20).mean() + (df[asset].rolling(window=20).std() * 2)
        df.loc[:, f'{asset}_bb_lband'] = df[asset].rolling(
            window=20).mean() - (df[asset].rolling(window=20).std() * 2)

    # Tratamento de valores ausentes (preenchimento para frente e para trás)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

# Função principal para processar e salvar os dados


def main():
    # Define os ativos a serem analisados e o período de coleta
    assets = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
              'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']
    start_date = '2014-09-25'
    end_date = '2024-09-24'

    # Coleta os dados de preços dos ativos e adiciona os indicadores técnicos
    asset_data = get_asset_data(assets, start_date, end_date)
    asset_data = add_features(asset_data, assets)

    # Define o diretório de saída e salva os dados processados em um arquivo CSV
    output_dir = 'BackPython/DADOS'
    asset_data.to_csv(f'{output_dir}/asset_data.csv')

    print("Dados coletados e processados, salvos em CSV.")


# Executa a função principal quando o script é rodado diretamente
if __name__ == "__main__":
    main()
