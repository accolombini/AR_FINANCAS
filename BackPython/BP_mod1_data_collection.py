# BP_mod1_data_collection.py
# Módulo para a coleta de dados financeiros de ativos usando a biblioteca yfinance

import yfinance as yf
import pandas as pd


class DataCollector:
    """
    Classe para coletar dados financeiros de ativos. Utiliza a API yfinance para
    baixar dados históricos de fechamento ajustado de uma lista de ativos.
    """

    @staticmethod
    def get_asset_data(assets, start_date, end_date):
        """
        Baixa os dados históricos de fechamento ajustado para uma lista de ativos.

        Parâmetros:
            - assets (list): Lista de ativos (ex: ['AAPL', 'MSFT']). Cada ativo deve ser uma string que representa o símbolo do ativo na bolsa.
            - start_date (str): Data inicial no formato 'AAAA-MM-DD' para a coleta dos dados.
            - end_date (str): Data final no formato 'AAAA-MM-DD' para a coleta dos dados.

        Retorna:
            - pd.DataFrame: DataFrame com os preços ajustados de fechamento dos ativos no intervalo de tempo especificado.
              O índice do DataFrame é a data, e cada coluna representa um ativo da lista `assets`.
        """
        # Baixa os dados dos ativos para o intervalo especificado
        df = yf.download(assets, start=start_date, end=end_date)

        # Retorna apenas a coluna 'Adj Close' (preço de fechamento ajustado) para análise
        return df['Adj Close']
