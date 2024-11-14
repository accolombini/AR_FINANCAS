# BP_mod1_data_collection.py
# Módulo para a coleta de dados financeiros de ativos usando yfinance

import yfinance as yf
import pandas as pd


class DataCollector:
    """Classe para coletar dados financeiros de ativos."""

    @staticmethod
    def get_asset_data(assets, start_date, end_date):
        """
        Baixa os dados históricos de fechamento ajustado para uma lista de ativos.

        Parâmetros:
            - assets (list): lista de ativos (ex: ['AAPL', 'MSFT'])
            - start_date (str): data inicial no formato 'AAAA-MM-DD'
            - end_date (str): data final no formato 'AAAA-MM-DD'

        Retorna:
            - DataFrame com os preços ajustados de fechamento.
        """
        df = yf.download(assets, start=start_date, end=end_date)
        return df['Adj Close']
