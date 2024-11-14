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
    def identify_outliers(df, threshold=3):
        """
        Identifica outliers com base em um número de desvios padrão (por padrão, 3 desvios).

        Retorna um DataFrame com a contagem de outliers para cada coluna.
        """
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > threshold
        return outliers.sum()

    @staticmethod
    def annual_returns(df):
        """
        Calcula o retorno anualizado para cada ativo.

        Retorna um DataFrame com os retornos anualizados em porcentagem.
        """
        annual_returns = (df.pct_change().resample(
            'Y').agg(lambda x: (1 + x).prod() - 1)) * 100
        return annual_returns.mean()

    @staticmethod
    def save_clean_data(df, file_name='asset_data_cleaner.csv'):
        """Salva o DataFrame limpo no diretório de saída."""
        df.to_csv(f'{OUTPUT_DIR}/{file_name}')
        print(f"Dados limpos salvos em: {OUTPUT_DIR}/{file_name}")

    @staticmethod
    def analyze_and_clean_data(file_path):
        """
        Realiza análise completa e limpeza dos dados.

        - Carrega os dados.
        - Realiza análise preliminar (dimensões, estatísticas, dados faltantes, outliers).
        - Calcula os retornos anualizados.
        - Salva os dados limpos.

        Retorna:
            - Um dicionário com os resultados da análise preliminar.
        """
        # Carregar dados
        df = DataAnalysis.load_data(file_path)

        # Análise preliminar
        analysis = {
            'Dimensions': DataAnalysis.data_dimensions(df),
            'Descriptive Statistics': DataAnalysis.descriptive_statistics(df),
            'Missing Data (%)': DataAnalysis.missing_data(df),
            'Outliers Count': DataAnalysis.identify_outliers(df),
            'Annual Returns (%)': DataAnalysis.annual_returns(df)
        }

        # Limpeza de dados (exemplo: interpolação para dados faltantes)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Salvar dados limpos
        DataAnalysis.save_clean_data(df)

        return analysis
