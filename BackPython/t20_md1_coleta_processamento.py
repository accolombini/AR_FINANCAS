'''
    Módulo 1: Coleta e Preparação dos Dados
    Este módulo está claro e bem definido. Para alinhar com nossa discussão:

    Ajustes: Inclua indicadores econômicos macro e setoriais como variáveis auxiliares (ex.: inflação, taxa de câmbio, PIB).
    Feature Engineering Avançado: Adicione features que capturam sazonalidades de longo prazo (indicadores específicos, médias móveis de 6 a 12 meses) para apoiar previsões de longo prazo.
    Output atualizado: Dados históricos preparados, com features de curto e longo prazo e indicadores econômicos adicionais.
    
    |||> Your registered API key is: 8e5112d9a5cdfb3cff60e7486f2bec09 Documentation is available on the St. Louis Fed web services website.
    8e5112d9a5cdfb3cff60e7486f2bec09
'''

# t20_md1_coleta_processamento.py

# Importando bibliotecas necessárias
import yfinance as yf
import pandas as pd
import requests

# Função para baixar os dados dos ativos


def get_asset_data(assets, start_date, end_date):
    df = yf.download(assets, start=start_date, end=end_date)
    return df['Adj Close']

# Função para adicionar features (médias móveis, volatilidade, RSI, etc.)


def add_features(df, assets):
    for asset in assets:
        df.loc[:, f'{asset}_returns'] = df[asset].pct_change(fill_method=None)
        df.loc[:, f'{asset}_ma_30'] = df[asset].rolling(window=30).mean()
        df.loc[:, f'{asset}_volatility_30'] = df[asset].rolling(
            window=30).std()
        df.loc[:, f'{asset}_ma_180'] = df[asset].rolling(window=180).mean()
        df.loc[:, f'{asset}_volatility_180'] = df[asset].rolling(
            window=180).std()

        # RSI
        df.loc[:, f'{asset}_rsi'] = df[asset].pct_change(fill_method=None).rolling(window=14).apply(
            lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean())))) if len(x) > 0 else 50, raw=False)

        # Bandas de Bollinger
        df.loc[:, f'{asset}_bb_hband'] = df[asset].rolling(
            window=20).mean() + (df[asset].rolling(window=20).std() * 2)
        df.loc[:, f'{asset}_bb_lband'] = df[asset].rolling(
            window=20).mean() - (df[asset].rolling(window=20).std() * 2)

    # Tratamento de valores ausentes
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

# Função para obter indicadores econômicos brasileiros via API do Banco Central


def get_brazil_economic_indicators(start_date, end_date):
    # Converter datas para o formato 'dd/mm/yyyy'
    start_date = pd.to_datetime(start_date).strftime('%d/%m/%Y')
    end_date = pd.to_datetime(end_date).strftime('%d/%m/%Y')

    # Endpoints do Banco Central para IPCA, Selic e PIB
    series = {
        'inflacao': 433,  # IPCA
        'taxa_juros': 432,  # Selic
        'pib': 1208  # PIB
    }

    indicators = pd.DataFrame()

    for name, code in series.items():
        url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{
            code}/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}'
        response = requests.get(url)
        data = response.json()

        # Verificar se o dado retornado é uma lista antes de tentar acessá-lo
        if isinstance(data, list):
            # Ver os primeiros 5 elementos
            print(f"Dados retornados para {name}: {data[:5]}")
        else:
            print(f"Dados retornados para {
                  name} não estão no formato esperado de lista.")
            # Mostrar todo o conteúdo para inspeção
            print(f"Conteúdo retornado: {data}")

        # Verificar se a chave 'data' está no JSON e processar corretamente
        df = pd.DataFrame(data)
        if 'data' in df.columns and 'valor' in df.columns:
            df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            df.set_index('data', inplace=True)

            # Resample para frequências mensais (IPCA e Selic) e trimestrais (PIB)
            if name == 'pib':
                df = df.resample('QE').mean()  # Trimestral (Quarter End)
            else:
                df = df.resample('ME').mean()  # Mensal (Month End)

            indicators[name] = df['valor']
        else:
            print(f"Erro nos dados retornados para {
                  name}. Verifique a estrutura.")

    return indicators

# Função principal para processar e salvar os dados


def main():
    # Definir os ativos e o período
    assets = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
              'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']
    start_date = '2014-09-25'
    end_date = '2024-09-24'

    # Coletar e processar dados dos ativos
    asset_data = get_asset_data(assets, start_date, end_date)
    asset_data = add_features(asset_data, assets)

    # Coletar dados econômicos brasileiros
    econ_data = get_brazil_economic_indicators(start_date, end_date)

    # Salvar os dados em CSV
    output_dir = 'TESTES/DADOS'
    asset_data.to_csv(f'{output_dir}/asset_data.csv')
    econ_data.to_csv(f'{output_dir}/economic_data.csv')

    print("Dados coletados e processados, salvos em CSV.")


if __name__ == "__main__":
    main()
