# BP_mod1_data_analysis.py
# Módulo para análise preliminar dos dados, incluindo estatísticas descritivas e cálculo de retornos anualizados

# BP_mod1_data_analysis.py
# Módulo para análise preliminar dos dados, incluindo estatísticas descritivas e cálculo de retornos anualizados

import pandas as pd
import numpy as np
from BP_mod1_config import OUTPUT_DIR


class DataAnalysis:
    """Classe para realizar análise preliminar dos dados de ativos."""

    @staticmethod
    def load_data(file_path):
        """Carrega os dados de ativos do arquivo CSV especificado."""
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    @staticmethod
    def data_dimensions(df):
        """Retorna a dimensão do DataFrame."""
        return df.shape

    @staticmethod
    def descriptive_statistics(df):
        """Calcula estatísticas descritivas básicas do DataFrame."""
        return df.describe()

    @staticmethod
    def missing_data(df):
        """Calcula o percentual de dados faltantes por coluna."""
        return df.isnull().mean() * 100

    @staticmethod
    def identify_outliers_iqr(df, threshold=1.5):
        """
        Identifica outliers usando o método do intervalo interquartil (IQR).

        Parâmetros:
            - df: DataFrame com os dados dos ativos.
            - threshold: Multiplicador para definir o intervalo de detecção (1.5 por padrão).

        Retorna:
            - Um DataFrame com a contagem de outliers para cada coluna.
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
        Calcula o retorno anual para cada ativo.

        Retorna um DataFrame com os retornos anuais em porcentagem.
        """
        return df.resample('YE').last().pct_change() * 100

    @staticmethod
    def save_clean_data(df, file_name='asset_data_cleaner.csv'):
        """Salva o DataFrame limpo no diretório de saída."""
        df.to_csv(f'{OUTPUT_DIR}/{file_name}')

    @staticmethod
    def analyze_and_clean_data(file_path):
        """
        Realiza análise completa e limpeza dos dados.

        - Carrega os dados.
        - Realiza análise preliminar (dimensões, estatísticas, dados faltantes, outliers).
        - Calcula os retornos anuais.
        - Salva os dados limpos.

        Retorna:
            - Um dicionário com os resultados da análise preliminar e tabela de retornos anuais.
        """
        # Carregar dados
        df = DataAnalysis.load_data(file_path)

        # Análise preliminar
        analysis = {
            'Dimensions': DataAnalysis.data_dimensions(df),
            'Descriptive Statistics': DataAnalysis.descriptive_statistics(df),
            'Missing Data (%)': DataAnalysis.missing_data(df),
            # Usando o método IQR para outliers
            'Outliers Count': DataAnalysis.identify_outliers_iqr(df),
            'Annual Returns (%)': DataAnalysis.annual_returns(df)
        }

        # Limpeza de dados (exemplo: interpolação para dados faltantes)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Salvar dados limpos
        DataAnalysis.save_clean_data(df)

        return analysis  # Retorna o dicionário de análise, mas não imprime no terminal
