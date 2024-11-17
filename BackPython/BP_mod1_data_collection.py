# Módulo para a coleta de dados financeiros de ativos usando a biblioteca yfinance

# BP_mod1_data_collection.py
# Importa as bibliotecas necessárias

import yfinance as yf
import pandas as pd
import os


class DataCollector:
    """
    Classe para coletar dados financeiros de ativos usando a API yfinance.
    """

    @staticmethod
    def get_asset_data(assets, start_date, end_date, save_to_csv=False, output_path="BackPython/DADOS/asset_data_raw.csv"):
        """
        Baixa os dados históricos de fechamento ajustado para uma lista de ativos.

        Parâmetros:
            - assets (list): Lista de ativos (ex: ['VALE3.SA', 'PETR4.SA']).
            - start_date (str): Data inicial no formato 'AAAA-MM-DD'.
            - end_date (str): Data final no formato 'AAAA-MM-DD'.
            - save_to_csv (bool): Se True, salva os dados coletados em CSV.
            - output_path (str): Caminho para salvar os dados (se save_to_csv=True).

        Retorna:
            - pd.DataFrame: DataFrame com os preços ajustados de fechamento dos ativos.
        """
        try:
            # Baixar os dados para todos os ativos
            print(f"Coletando dados de {len(assets)} ativos...")
            df = yf.download(assets, start=start_date,
                             end=end_date, group_by='ticker')

            # Garantir que todos os ativos estão no DataFrame
            missing_assets = [
                asset for asset in assets if asset not in df.columns.levels[0]]
            if missing_assets:
                print(f"Aviso: Não foi possível coletar dados para os seguintes ativos: {
                      missing_assets}")

            # Filtrar apenas a coluna 'Adj Close'
            adj_close_data = {}
            for asset in assets:
                try:
                    adj_close_data[asset] = df[asset]['Adj Close']
                except KeyError:
                    adj_close_data[asset] = None

            # Criar DataFrame consolidado
            final_df = pd.DataFrame(adj_close_data)

            # Salvar os dados, se solicitado
            if save_to_csv:
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                final_df.to_csv(output_path)
                print(f"Dados salvos em {output_path}")

            return final_df

        except Exception as e:
            print(f"Erro ao coletar dados dos ativos: {e}")
            return pd.DataFrame()
