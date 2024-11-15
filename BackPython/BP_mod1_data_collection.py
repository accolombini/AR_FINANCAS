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
    def get_asset_data(assets, start_date, end_date, save_to_csv=False, output_path="BackPython/DADOS/asset_data_raw.csv"):
        """
        Baixa os dados históricos de fechamento ajustado para uma lista de ativos.

        Parâmetros:
            - assets (list): Lista de ativos (ex: ['AAPL', 'MSFT']). Cada ativo deve ser uma string que representa o símbolo do ativo na bolsa.
            - start_date (str): Data inicial no formato 'AAAA-MM-DD' para a coleta dos dados.
            - end_date (str): Data final no formato 'AAAA-MM-DD' para a coleta dos dados.
            - save_to_csv (bool): Se True, salva os dados coletados em um arquivo CSV. O caminho é especificado em `output_path`.
            - output_path (str): Caminho para o arquivo CSV onde os dados serão salvos (usado apenas se `save_to_csv=True`).

        Retorna:
            - pd.DataFrame: DataFrame com os preços ajustados de fechamento dos ativos no intervalo de tempo especificado.
              O índice do DataFrame é a data, e cada coluna representa um ativo da lista `assets`.
        """
        try:
            # Baixa os dados dos ativos para o intervalo especificado
            df = yf.download(assets, start=start_date, end=end_date)

            # Verificar se a coluna 'Adj Close' está presente nos dados retornados
            if 'Adj Close' not in df:
                raise ValueError(
                    "Dados retornados não contêm a coluna 'Adj Close' esperada.")

            # Seleciona apenas a coluna 'Adj Close' para análise
            df = df['Adj Close']

            # Salvar os dados em CSV se a opção estiver ativada
            if save_to_csv:
                # Garante que o diretório de saída exista
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                df.to_csv(output_path)
                print(f"Dados salvos em: {output_path}")

            return df

        except Exception as e:
            print(f"Erro ao coletar dados dos ativos: {e}")
            return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro
