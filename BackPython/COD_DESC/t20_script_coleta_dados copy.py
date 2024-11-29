'''
    Objetivo: 
        ||> Baixar os Dados do Yahoo Finance:
                Usaremos a biblioteca yfinance para buscar dados dos ativos mencionados.
                Os dados serão ajustados para o período de 09/2014 a 10/2024.
                Preparar os Dados:

                Organizaremos o dataset de modo que inclua:
                Colunas de preços ajustados para cada ativo.
                Transformação em um único DataFrame para fácil manipulação nos modelos.
                Divisão automática entre treino (09/2014 a 09/2024) e validação (30 dias de 10/2024).
                Salvar em train_data.csv:

                Salvaremos os dados formatados em TESTES/DADOS/train_data.csv, prontos para serem usados diretamente nos scripts de modelagem.
'''

# Importar as bibliotecas necessárias

import yfinance as yf
import pandas as pd

# Lista de ativos
tickers = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
           'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']

# Período de interesse
start_date = '2014-09-01'
end_date = '2024-10-31'

# Função para baixar e organizar dados


def download_and_prepare_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.dropna()  # Remove linhas com valores faltantes
    return data


# Baixar e preparar os dados
df = download_and_prepare_data(tickers, start_date, end_date)

# Salvar dados prontos para treinamento
# Altere o caminho conforme necessário
output_path = 'TESTES/DADOS/train_data.csv'
df.to_csv(output_path)
print(f"Dados salvos em: {output_path}")
