# BP_mod1_feature_engineering.py
# Módulo para aplicação de cálculos de indicadores técnicos nos dados de ativos

import pandas as pd


class FeatureEngineering:
    """Classe para aplicar cálculos de indicadores técnicos em dados de ativos."""

    @staticmethod
    def add_features(df, assets):
        """
        Adiciona indicadores técnicos ao DataFrame de preços de ativos.

        Parâmetros:
            - df (DataFrame): Dados de preços dos ativos.
            - assets (list): lista de ativos para calcular indicadores.

        Retorna:
            - DataFrame com indicadores técnicos adicionados.
        """
        # Checando se o DataFrame inicial está vazio
        if df.empty:
            print("O DataFrame de entrada está vazio. Verifique os dados coletados.")
            return None

        for asset in assets:
            # Calcula o retorno diário do ativo
            df[f'{asset}_returns'] = df[asset].pct_change(fill_method=None)

            # Médias móveis e volatilidade
            df[f'{asset}_ma_30'] = df[asset].rolling(window=30).mean()
            df[f'{asset}_volatility_30'] = df[asset].rolling(window=30).std()
            df[f'{asset}_ma_180'] = df[asset].rolling(window=180).mean()
            df[f'{asset}_volatility_180'] = df[asset].rolling(window=180).std()

            # Índice de Força Relativa (RSI)
            df[f'{asset}_rsi'] = df[asset].pct_change(fill_method=None).rolling(window=14).apply(
                lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean())))) if len(x) > 0 else 50, raw=False)

            # Bandas de Bollinger
            df[f'{asset}_bb_hband'] = df[asset].rolling(
                window=20).mean() + (df[asset].rolling(window=20).std() * 2)
            df[f'{asset}_bb_lband'] = df[asset].rolling(
                window=20).mean() - (df[asset].rolling(window=20).std() * 2)

        # Tratamento de valores ausentes
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df  # Garantindo que o DataFrame processado seja retornado
