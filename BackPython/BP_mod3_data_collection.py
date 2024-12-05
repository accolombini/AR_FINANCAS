# BP_mod3_data_collection.py: Coleta e Processamento de Dados Históricos da B3
# -----------------------------------------------------------
# Este script realiza a coleta de dados históricos de ativos da B3
# utilizando a biblioteca yfinance. Ele integra-se ao módulo de configuração
# BP_mod1_config.py para acessar parâmetros como lista de ativos, período de análise,
# e diretórios de saída. Inclui:
# - Download de dados históricos de preços ajustados.
# - Processamento do campo 'Date' para remover timezones.
# - Alinhamento ao menor período histórico disponível.
# - Salvamento dos dados limpos em formato CSV.
# -----------------------------------------------------------
import yfinance as yf
import pandas as pd
import os
import time
import logging
from BP_mod1_config import ASSETS, START_DATE, END_DATE, OUTPUT_DIR, atualizar_start_date

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_and_clean_data(tickers, start_date, end_date, max_retries=3, timeout=20):
    """
    Coleta e limpeza de dados históricos para ativos e benchmarks.

    Args:
        tickers (list): Lista de tickers para baixar os dados.
        start_date (str): Data inicial no formato 'YYYY-MM-DD'.
        end_date (str): Data final no formato 'YYYY-MM-DD' (opcional).
        max_retries (int): Número máximo de tentativas em caso de falha.
        timeout (int): Tempo limite para cada requisição em segundos.

    Returns:
        pd.DataFrame: Dados consolidados com preços ajustados de fechamento.
        list: Lista de tickers que falharam.
    """
    logging.info("Iniciando download de dados históricos...")
    failed_tickers = []
    all_data = []
    min_dates = {}

    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                logging.info(f"Baixando dados para {
                             ticker} (tentativa {attempt + 1})...")
                data = yf.download(
                    ticker, start=start_date, end=end_date, group_by="ticker", auto_adjust=True, timeout=timeout
                )

                if "Close" in data and not data["Close"].empty:
                    temp_df = data[["Close"]].rename(columns={"Close": ticker})
                    all_data.append(temp_df)
                    min_dates[ticker] = temp_df.first_valid_index().strftime(
                        '%Y-%m-%d')
                    logging.info(f"Dados para {ticker} baixados com sucesso.")
                    break
                else:
                    raise ValueError(
                        f"Dados inválidos recebidos para {ticker}.")
            except Exception as e:
                logging.warning(f"Falha ao baixar dados para {ticker}: {e}")
                time.sleep(2)
                if attempt == max_retries - 1:
                    logging.error(
                        f"Máximo de tentativas atingido para {ticker}.")
                    failed_tickers.append(ticker)

    if all_data:
        consolidated_df = pd.concat(all_data, axis=1)
    else:
        consolidated_df = pd.DataFrame()

    return consolidated_df, failed_tickers, min_dates


def align_to_min_available_date(df):
    """
    Alinha os dados ao menor período histórico disponível, baseado no primeiro registro de cada ticker.

    Args:
        df (pd.DataFrame): DataFrame consolidado com dados de vários tickers.

    Returns:
        pd.DataFrame: DataFrame alinhado ao menor período histórico comum.
    """
    min_dates = df.apply(lambda x: x.first_valid_index(), axis=0)
    min_date = max(min_dates)
    logging.info(f"Alinhando os dados ao menor período disponível: {min_date}")
    return df.loc[min_date:]


def process_date_column(df):
    """
    Processa o índice 'Date' do DataFrame, removendo timezones.

    Args:
        df (pd.DataFrame): DataFrame com dados históricos.

    Returns:
        pd.DataFrame: DataFrame com o índice 'Date' sem timezone.
    """
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df


def main():
    """
    Executa o fluxo completo de coleta, limpeza e salvamento dos dados históricos.
    """
    output_file = os.path.join(OUTPUT_DIR, "historical_data_cleaned.csv")

    logging.info("Coletando e processando dados históricos...")
    historical_data, failed_tickers, min_dates = fetch_and_clean_data(
        ASSETS, start_date=START_DATE, end_date=END_DATE
    )

    if not historical_data.empty:
        logging.info("Processando campo 'Date' para remover timezone...")
        historical_data = process_date_column(historical_data)

        # Atualizar START_DATE dinamicamente
        atualizar_start_date(min_dates)

        logging.info(
            "Alinhando os dados ao menor período histórico disponível...")
        historical_data = align_to_min_available_date(historical_data)

        logging.info("Imputando valores faltantes...")
        historical_data.ffill(inplace=True)

        logging.info(f"Salvando dados limpos em {output_file}...")
        historical_data.sort_index(inplace=True)
        historical_data.to_csv(output_file, index_label="Date")
        logging.info("Processo concluído com sucesso!")

    if failed_tickers:
        logging.warning(f"Os seguintes tickers falharam: {failed_tickers}")


if __name__ == "__main__":
    main()
