# Módulo para análise preliminar dos dados, incluindo estatísticas descritivas e cálculo de retornos anualizados

# BP_mod1_data_analysis.py - Correção para lidar com Timestamps no relatório de análise

import os
import pandas as pd
import numpy as np
import json
from BP_mod1_config import OUTPUT_DIR


class DataAnalysis:
    @staticmethod
    def annual_returns(df):
        """
        Calcula o retorno anual para cada ativo com base nos preços ajustados.

        Parâmetros:
            - df (pd.DataFrame): DataFrame com os preços ajustados dos ativos.

        Retorna:
            - pd.DataFrame: DataFrame com os retornos anuais em porcentagem, com índices convertidos para strings.
        """
        annual_returns_df = df.resample('YE').last().pct_change() * 100
        annual_returns_df.index = annual_returns_df.index.strftime(
            '%Y')  # Converte índices para strings
        return annual_returns_df

    @staticmethod
    def save_analysis_report(analysis, file_name="analysis_report.json"):
        """
        Salva o relatório de análise em um arquivo JSON.

        Parâmetros:
            - analysis (dict): Dicionário contendo os resultados da análise.
            - file_name (str): Nome do arquivo de saída.
        """
        output_path = os.path.join(OUTPUT_DIR, file_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Serializa os dados para JSON
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=4)
        print(f"Relatório de análise salvo em {output_path}")

    @staticmethod
    def analyze_and_clean_data(file_path):
        """
        Realiza uma análise completa e limpeza dos dados dos ativos financeiros.

        Parâmetros:
            - file_path (str): Caminho do arquivo CSV de entrada.

        Retorna:
            - dict: Dicionário com as métricas de análise preliminar.
        """
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        if df.empty:
            print("Erro: DataFrame vazio. Verifique o arquivo de entrada.")
            return {}

        # Geração do relatório de análise
        analysis = {
            'Dimensions': df.shape,
            'Descriptive Statistics': df.describe().to_dict(),
            'Missing Data (%)': df.isnull().mean().to_dict(),
            'Outliers Count': DataAnalysis.identify_outliers_iqr(df).to_dict(),
            'Annual Returns (%)': DataAnalysis.annual_returns(df).to_dict()
        }

        # Limpeza dos dados
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Salva os dados limpos
        clean_file_name = "asset_data_cleaner.csv"
        df.to_csv(os.path.join(OUTPUT_DIR, clean_file_name))
        print(f"Dados limpos salvos em {
              os.path.join(OUTPUT_DIR, clean_file_name)}.")

        # Salva o relatório de análise
        DataAnalysis.save_analysis_report(analysis)

        return analysis

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
            outliers_count = df[
                (df[column] < Q1 - threshold * IQR) |
                (df[column] > Q3 + threshold * IQR)
            ].shape[0]
            outliers.loc[column] = outliers_count

        return outliers
