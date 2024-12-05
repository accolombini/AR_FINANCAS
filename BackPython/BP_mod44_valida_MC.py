# BP_mod4_valida_MC.py: Validação de Simulação de Monte Carlo
# -----------------------------------------------------------
# Este script valida o simulador de Monte Carlo (MC) usando dados históricos e o compara
# com o portfólio otimizado e o índice BOVESPA. Ele também avalia o quão próximo
# estamos dos percentis 5%, 50% e 95% para os dois últimos meses conhecidos.
# -----------------------------------------------------------
# BP_mod4_valida_MC.py: Validação de Simulação de Monte Carlo

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import os
from BP_mod1_config import OUTPUT_DIR, START_DATE, END_DATE

# Configuração do logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Parâmetros do script
N_SIMULATIONS = 10000
VALIDATION_PERIOD_DAYS = 60
META_RETORNO_ANUAL = 0.15  # Meta de 15% ao ano


def carregar_dados(filepath: str, index_col: str = None) -> pd.DataFrame:
    """Carrega dados de um arquivo CSV, valida e preenche ausências."""
    try:
        df = pd.read_csv(filepath, index_col=index_col, parse_dates=True)
        if df.isnull().any().any():
            logging.warning(f"Dados ausentes encontrados em {
                            filepath}. Preenchendo com interpolação linear.")
            df = df.interpolate(method='time').fillna(
                method='bfill').fillna(method='ffill')
        logging.info(f"Dados carregados com sucesso de {filepath}.")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar dados de {filepath}: {e}")
        raise


def revisar_periodos_corrigido_sem_dependencias(df: pd.DataFrame, validation_days: int) -> tuple:
    """Revisa e corrige os períodos de treinamento e validação com base nos dados simulados."""
    try:
        data_final = df.index.max()
        data_inicio_validacao = data_final - pd.Timedelta(days=validation_days)

        df = df.sort_index().asfreq('B')  # Frequência de dias úteis

        validacao = df.loc[data_inicio_validacao:data_final]
        if len(validacao) < validation_days:
            logging.warning(
                f"Dados insuficientes para cobrir os últimos {
                    validation_days} dias completos. "
                f"Usando apenas {len(validacao)} dias disponíveis."
            )

        treinamento = df.loc[:validacao.index.min() - pd.Timedelta(days=1)]
        logging.info(f"Períodos ajustados: Treinamento até {treinamento.index.max(
        )}, Validação de {validacao.index.min()} a {validacao.index.max()}.")
        return treinamento, validacao
    except Exception as e:
        logging.error(f"Erro ao revisar períodos corrigidos: {e}")
        raise


def gerar_simulacoes(training_data: pd.DataFrame, n_simulations: int, n_days: int) -> np.ndarray:
    """Gera simulações de Monte Carlo baseadas nos dados de treinamento."""
    try:
        daily_returns = training_data.pct_change().dropna()
        mean = daily_returns.mean().values
        std = daily_returns.std().values
        simulated_returns = np.random.normal(
            mean, std, size=(n_days, n_simulations, len(mean)))
        logging.info("Simulações de Monte Carlo geradas com sucesso.")
        return simulated_returns
    except Exception as e:
        logging.error(f"Erro ao gerar simulações: {e}")
        raise


def ajustar_precos_e_percentis(last_price: pd.Series, simulated_returns: np.ndarray) -> np.ndarray:
    """Ajusta os preços simulados para refletir a escala do último preço real do período de treinamento."""
    try:
        initial_prices = last_price.values[np.newaxis, np.newaxis, :]
        prices_simulated = initial_prices * \
            np.cumprod(1 + simulated_returns, axis=0)

        if prices_simulated.shape[0] == 0:
            raise ValueError("Nenhum preço simulado foi gerado.")

        logging.info("Preços simulados ajustados com sucesso.")
        return prices_simulated
    except Exception as e:
        logging.error(f"Erro ao ajustar preços simulados: {e}")
        raise


def recalcular_retornos_corrigido(validation_data: pd.DataFrame, pesos: np.ndarray) -> pd.Series:
    """Recalcula os retornos do portfólio, corrigindo escalonamento e alinhamento."""
    try:
        if len(pesos) != validation_data.shape[1]:
            raise ValueError(
                "O número de pesos do portfólio não corresponde ao número de ativos.")
        portfolio_returns = (
            validation_data.pct_change().fillna(0) * pesos).sum(axis=1)
        logging.info(f"Retornos do portfólio recalculados: {
                     portfolio_returns.head()}")
        return portfolio_returns
    except Exception as e:
        logging.error(f"Erro ao recalcular retornos do portfólio: {e}")
        raise


def plotar_resultados_corrigido(percentiles: np.ndarray, validation_data: pd.DataFrame, portfolio_returns: pd.Series, benchmark_returns: pd.Series):
    """Plota resultados com visualização corrigida e simplificada."""
    try:
        dates = validation_data.index
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=dates, y=percentiles[0], mode='lines', line=dict(
            color='gray', dash='dash'), name='5%'))
        fig.add_trace(go.Scatter(
            x=dates, y=percentiles[2], fill='tonexty', mode='lines', line=dict(color='gray'), name='95%'))
        fig.add_trace(go.Scatter(x=dates, y=percentiles[1], mode='lines', line=dict(
            color='blue'), name='Cenário Mediano (50%)'))

        fig.add_trace(go.Scatter(x=dates, y=validation_data.mean(
            axis=1), mode='lines', line=dict(color='green'), name='Preços Reais'))
        fig.add_trace(go.Scatter(x=dates, y=portfolio_returns, mode='lines', line=dict(
            color='red'), name='Portfólio Otimizado'))
        fig.add_trace(go.Scatter(x=dates, y=benchmark_returns,
                      mode='lines', line=dict(color='blue'), name='Benchmark'))

        meta_retorno = [META_RETORNO_ANUAL / 252] * len(dates)
        fig.add_trace(go.Scatter(x=dates, y=meta_retorno, mode='lines', line=dict(
            color='green', dash='dash'), name='Meta de Retorno (15% Anual)'))

        fig.update_layout(title="Validação de Simulação de Monte Carlo",
                          xaxis_title="Datas",
                          yaxis_title="Retornos (%)",
                          template="plotly_dark")
        fig.show()
        logging.info("Gráfico corrigido plotado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao plotar resultados corrigidos: {e}")
        raise


def main_corrigida_com_dados_reais():
    """Função principal utilizando dados reais e corrigindo todas as inconsistências."""
    try:
        filtered_data_filepath = os.path.join(
            OUTPUT_DIR, "historical_data_filtered.csv")
        portfolio_filepath = os.path.join(
            OUTPUT_DIR, "portfolio_otimizado.csv")

        if not os.path.exists(filtered_data_filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {
                                    filtered_data_filepath}")
        if not os.path.exists(portfolio_filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {
                                    portfolio_filepath}")

        data = carregar_dados(filtered_data_filepath, index_col="Date")
        portfolio = carregar_dados(portfolio_filepath)
        portfolio_weights = portfolio['Peso (%)'].values / 100

        training_data, validation_data = revisar_periodos_corrigido_sem_dependencias(
            data, VALIDATION_PERIOD_DAYS)
        last_price = training_data.iloc[-1]

        simulated_returns = gerar_simulacoes(
            training_data, N_SIMULATIONS, VALIDATION_PERIOD_DAYS)
        simulated_prices = ajustar_precos_e_percentis(
            last_price, simulated_returns)

        portfolio_returns = recalcular_retornos_corrigido(
            validation_data, portfolio_weights)
        benchmark_returns = validation_data.mean(axis=1)

        percentiles = np.percentile(simulated_prices, [5, 50, 95], axis=1)
        plotar_resultados_corrigido(
            percentiles, validation_data, portfolio_returns, benchmark_returns)
    except Exception as e:
        logging.error(f"Erro durante a execução do script corrigido: {e}")
        raise


if __name__ == "__main__":
    main_corrigida_com_dados_reais()
