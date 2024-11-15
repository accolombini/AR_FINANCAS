# BP_mod1_feature_engineering.py
# Módulo para aplicação de cálculos de indicadores essenciais para previsões financeiras

import pandas as pd


class FeatureEngineering:
    """
    Classe para calcular indicadores essenciais em dados de ativos financeiros.
    Essa classe fornece métodos para calcular retornos diários, médias móveis e volatilidade.
    """

    @staticmethod
    def add_essential_features(df, assets):
        """
        Adiciona retornos diários, médias móveis e volatilidade ao DataFrame de preços de ativos.

        Parâmetros:
            - df (pd.DataFrame): DataFrame contendo os preços dos ativos financeiros.
            - assets (list): Lista de ativos (colunas do DataFrame) para os quais os indicadores serão calculados.

        Retorna:
            - pd.DataFrame: DataFrame original com os novos indicadores essenciais adicionados.
        """
        for asset in assets:
            # Calcula o retorno diário do ativo
            df[f'{asset}_returns'] = df[asset].pct_change(fill_method=None)

            # Calcula a média móvel de 30 dias (curto prazo)
            df[f'{asset}_ma_30'] = df[asset].rolling(window=30).mean()
            # Calcula a volatilidade de 30 dias
            df[f'{asset}_volatility_30'] = df[asset].rolling(window=30).std()

            # Calcula a média móvel de 180 dias (longo prazo)
            df[f'{asset}_ma_180'] = df[asset].rolling(window=180).mean()
            # Calcula a volatilidade de 180 dias
            df[f'{asset}_volatility_180'] = df[asset].rolling(window=180).std()

        # Tratamento de valores ausentes (resultantes dos cálculos de indicadores)
        df.ffill(inplace=True)  # Preenchimento forward para valores ausentes
        # Preenchimento backward para quaisquer valores restantes
        df.bfill(inplace=True)

        return df
