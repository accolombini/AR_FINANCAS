# BP_mod1_feature_engineering.py
# Módulo para aplicação de cálculos de indicadores essenciais para previsões financeiras

import pandas as pd


class FeatureEngineering:
    """Classe para calcular indicadores essenciais em dados de ativos."""

    @staticmethod
    def add_essential_features(df, assets):
        """
        Adiciona retornos diários, médias móveis e volatilidade ao DataFrame de preços de ativos.

        Parâmetros:
            - df (DataFrame): Dados de preços dos ativos.
            - assets (list): Lista de ativos para calcular indicadores.

        Retorna:
            - DataFrame com os indicadores essenciais adicionados.
        """
        for asset in assets:
            # Calcula o retorno diário do ativo
            df[f'{asset}_returns'] = df[asset].pct_change(fill_method=None)

            # Médias móveis de 30 e 180 dias
            df[f'{asset}_ma_30'] = df[asset].rolling(window=30).mean()
            df[f'{asset}_volatility_30'] = df[asset].rolling(window=30).std()
            df[f'{asset}_ma_180'] = df[asset].rolling(window=180).mean()
            df[f'{asset}_volatility_180'] = df[asset].rolling(window=180).std()

        # Tratamento de valores ausentes
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df
