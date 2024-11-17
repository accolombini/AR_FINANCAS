# BP_mod1_feature_engineering.py
# Módulo para aplicação de cálculos de indicadores essenciais para previsões financeiras

import pandas as pd


class FeatureEngineering:
    """
    Classe para calcular indicadores essenciais em dados de ativos financeiros.
    Inclui métodos para calcular retornos diários, médias móveis e volatilidade.
    """

    @staticmethod
    def add_essential_features(df, assets, short_window=30, long_window=180):
        """
        Adiciona retornos diários, médias móveis e volatilidade ao DataFrame de preços de ativos.

        Parâmetros:
            - df (pd.DataFrame): DataFrame contendo os preços dos ativos financeiros.
            - assets (list): Lista de ativos (colunas do DataFrame) para os quais os indicadores serão calculados.
            - short_window (int): Tamanho da janela para cálculos de curto prazo (ex: 30 dias).
            - long_window (int): Tamanho da janela para cálculos de longo prazo (ex: 180 dias).

        Retorna:
            - pd.DataFrame: DataFrame original com os novos indicadores essenciais adicionados.
        """
        for asset in assets:
            if asset not in df.columns:
                print(f"Aviso: O ativo '{
                      asset}' não foi encontrado no DataFrame.")
                continue

            # Calcula o retorno diário do ativo
            df[f'{asset}_returns'] = df[asset].pct_change(fill_method=None)

            # Indicadores de curto prazo (ex: média móvel e volatilidade de 30 dias)
            df[f'{asset}_ma_{short_window}'] = df[asset].rolling(
                window=short_window).mean()
            df[f'{asset}_volatility_{short_window}'] = df[asset].rolling(
                window=short_window).std()

            # Indicadores de longo prazo (ex: média móvel e volatilidade de 180 dias)
            df[f'{asset}_ma_{long_window}'] = df[asset].rolling(
                window=long_window).mean()
            df[f'{asset}_volatility_{long_window}'] = df[asset].rolling(
                window=long_window).std()

        # Tratamento de valores ausentes
        df.ffill(inplace=True)  # Preenchimento forward
        df.bfill(inplace=True)  # Preenchimento backward para dados restantes

        return df
