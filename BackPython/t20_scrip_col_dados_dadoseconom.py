'''
    ||> Buscar e Normalizar os Dados:
        Buscar os dados de inflação, taxa de juros e PIB para o período de 2014-2024.
        Normalizar e interpolar esses dados para que fiquem com frequência mensal ou diária, conforme necessário.
        Feature Engineering com os Indicadores:

        Incluir médias móveis e volatilidade desses indicadores para capturar mudanças recentes e tendências.
        Incorporar esses dados ao conjunto de treino como variáveis independentes.
'''

# Importar as bibliotecas necessárias
import yfinance as yf
import pandas as pd
import requests

# Lista de ativos
tickers = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
           'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']

# Período de interesse
start_date = '2014-09-01'  # formato compatível com a API do BCB
end_date = '2024-10-31'

# Função para baixar e organizar dados dos ativos


def download_and_prepare_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=pd.to_datetime(
        start_date), end=pd.to_datetime(end_date))['Adj Close']
    data = data.dropna()  # Remove linhas com valores faltantes
    # Remover informações de fuso horário
    data.index = data.index.tz_localize(None)
    return data

# Função para baixar dados macroeconômicos do Banco Central usando a API


def download_macro_data(start_date, end_date):
    # Garantir que as datas estejam no formato correto para a API do BCB
    start_date = pd.to_datetime(start_date).strftime('%d/%m/%Y')
    end_date = pd.to_datetime(end_date).strftime('%d/%m/%Y')

    urls = {
        'inflacao': f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}',
        'taxa_juros': f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}',
        'pib': f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4380/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}'
    }

    # Cria um DataFrame diário para o período desejado
    macro_data = pd.DataFrame(index=pd.date_range(
        start=start_date, end=end_date, freq='D'))

    # Requisição de dados para cada indicador
    for name, url in urls.items():
        print(f"Baixando dados para {name} (URL: {url})...")
        response = requests.get(url)
        data = response.json()

        # Verificar se a resposta é uma lista válida com dados
        if isinstance(data, list) and len(data) > 0 and 'data' in data[0]:
            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df.set_index('data', inplace=True)
            df[name] = pd.to_numeric(df['valor'], errors='coerce')
            df = df[[name]]

            # Resample para base diária com forward-fill para preencher os valores mensais/trimestrais
            if name != 'pib':
                # Mensal para inflação e juros, preencher diariamente
                df = df.resample('D').ffill()
            else:
                # Trimestral para PIB, preencher diariamente
                df = df.resample('D').ffill()

            # Adicionar o indicador ao DataFrame macroeconômico principal
            macro_data = macro_data.join(df, how='left')
        else:
            print(f"Aviso: Dados ausentes ou formato incorreto para '{
                  name}'. Resposta da API: {data}")

    macro_data.index = macro_data.index.tz_localize(
        None)  # Remover informações de fuso horário
    macro_data = macro_data.dropna()  # Remover qualquer linha com valores faltantes
    return macro_data


# Baixar e preparar os dados dos ativos
df_assets = download_and_prepare_data(tickers, start_date, end_date)

# Baixar e preparar os dados macroeconômicos
df_macro = download_macro_data(start_date, end_date)

# Combinar os dados em um único DataFrame com a mesma base temporal
df_combined = df_assets.join(df_macro, how='inner')
df_combined = df_combined.dropna()

# Salvar dados prontos para treinamento
output_path = 'TESTES/DADOS/train_data_combined.csv'
df_combined.to_csv(output_path)
print(f"Dados combinados de ativos e macroeconômicos salvos em: {output_path}")
