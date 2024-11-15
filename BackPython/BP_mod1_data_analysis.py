# BP_mod1_data_analysis.py
# Módulo para análise preliminar dos dados, incluindo estatísticas descritivas e cálculo de retornos anualizados

# BP_mod1_data_analysis.py
# Módulo para análise preliminar dos dados, incluindo estatísticas descritivas e cálculo de retornos anualizados

import os
import pandas as pd
import numpy as np
from BP_mod1_config import OUTPUT_DIR


class DataAnalysis:
    """
    Classe para realizar análise preliminar dos dados de ativos financeiros. Inclui métodos para
    calcular estatísticas descritivas, identificar dados faltantes, detectar outliers e calcular
    retornos anualizados.
    """

    @staticmethod
    def load_data(file_path):
        """
        Carrega os dados de ativos de um arquivo CSV especificado.

        Parâmetros:
            - file_path (str): Caminho do arquivo CSV a ser carregado.

        Retorna:
            - pd.DataFrame: DataFrame com os dados dos ativos.
        """
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    @staticmethod
    def data_dimensions(df):
        """
        Retorna as dimensões do DataFrame, ou seja, o número de linhas e colunas.

        Parâmetros:
            - df (pd.DataFrame): DataFrame cujas dimensões serão retornadas.

        Retorna:
            - tuple: Uma tupla com o número de linhas e colunas.
        """
        return df.shape

    @staticmethod
    def descriptive_statistics(df):
        """
        Calcula estatísticas descritivas básicas do DataFrame, como média, desvio padrão, mínimo,
        máximo e quartis.

        Parâmetros:
            - df (pd.DataFrame): DataFrame cujas estatísticas descritivas serão calculadas.

        Retorna:
            - pd.DataFrame: DataFrame com as estatísticas descritivas.
        """
        return df.describe()

    @staticmethod
    def missing_data(df):
        """
        Calcula o percentual de dados faltantes por coluna no DataFrame.

        Parâmetros:
            - df (pd.DataFrame): DataFrame cujos dados faltantes serão analisados.

        Retorna:
            - pd.Series: Série com o percentual de dados faltantes por coluna.
        """
        return df.isnull().mean() * 100

    @staticmethod
    def identify_outliers_iqr(df, threshold=1.5):
        """
        Identifica outliers em cada coluna do DataFrame usando o método do Intervalo Interquartil (IQR).

        Parâmetros:
            - df (pd.DataFrame): DataFrame com os dados dos ativos.
            - threshold (float): Multiplicador para definir o intervalo de detecção de outliers (1.5 por padrão).

        Retorna:
            - pd.DataFrame: DataFrame com a contagem de outliers para cada coluna.
        """
        outliers = pd.DataFrame(index=df.columns, columns=['Outliers'])

        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers_count = df[(df[column] < Q1 - threshold * IQR)
                                | (df[column] > Q3 + threshold * IQR)].shape[0]
            outliers.loc[column] = outliers_count

        return outliers

    @staticmethod
    def annual_returns(df):
        """
        Calcula o retorno anual para cada ativo com base nos preços ajustados.

        Parâmetros:
            - df (pd.DataFrame): DataFrame com os preços ajustados dos ativos.

        Retorna:
            - pd.DataFrame: DataFrame com os retornos anuais em porcentagem.
        """
        return df.resample('YE').last().pct_change() * 100

    @staticmethod
    def save_clean_data(df, file_name='asset_data_cleaner.csv'):
        """
        Salva o DataFrame limpo no diretório de saída especificado no arquivo de configuração.

        Parâmetros:
            - df (pd.DataFrame): DataFrame limpo a ser salvo.
            - file_name (str): Nome do arquivo de saída (padrão: 'asset_data_cleaner.csv').
        """
        # Garante que o diretório de saída exista
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        df.to_csv(f'{OUTPUT_DIR}/{file_name}')

    @staticmethod
    def analyze_and_clean_data(file_path):
        """
        Realiza uma análise completa e limpeza dos dados dos ativos financeiros.

        Este método realiza as seguintes operações:
        - Carrega os dados de ativos de um arquivo CSV.
        - Calcula as dimensões do DataFrame.
        - Gera estatísticas descritivas básicas.
        - Identifica dados faltantes.
        - Detecta outliers usando o método IQR.
        - Calcula retornos anuais.
        - Realiza interpolação para preencher dados faltantes.
        - Salva o DataFrame limpo em um arquivo CSV.

        Parâmetros:
            - file_path (str): Caminho do arquivo CSV de entrada.

        Retorna:
            - dict: Dicionário com as métricas de análise preliminar e a tabela de retornos anuais.
        """
        # Carregar dados
        df = DataAnalysis.load_data(file_path)

        # Análise preliminar
        analysis = {
            'Dimensions': DataAnalysis.data_dimensions(df),
            'Descriptive Statistics': DataAnalysis.descriptive_statistics(df),
            'Missing Data (%)': DataAnalysis.missing_data(df),
            'Outliers Count': DataAnalysis.identify_outliers_iqr(df),
            'Annual Returns (%)': DataAnalysis.annual_returns(df)
        }

        # Limpeza de dados (exemplo: preenchimento de dados faltantes)
        df.ffill(inplace=True)  # Preenchimento forward
        df.bfill(inplace=True)  # Preenchimento backward para dados restantes

        # Salvar dados limpos
        DataAnalysis.save_clean_data(df)

        return analysis  # Retorna o dicionário de análise para uso posterior
