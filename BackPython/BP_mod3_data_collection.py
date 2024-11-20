'''Baixar dados da B3 e salvar em arquivo .csv, manter o campo Date mas sem o timezone'''

# Incluir bibliotecas necessárias
import yfinance as yf
import pandas as pd
import time


def fetch_and_clean_data(tickers, start_date="2010-01-01", end_date=None, max_retries=3, timeout=20):
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
    print("[INFO] Baixando dados históricos...")
    failed_tickers = []
    all_data = []

    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                # Baixar os dados do ticker
                print(f"[INFO] Baixando dados para {
                      ticker} (tentativa {attempt + 1})...")
                data = yf.download(ticker, start=start_date, end=end_date,
                                   group_by="ticker", auto_adjust=True, timeout=timeout)

                # Verificar se o ticker retornou dados válidos
                if "Close" in data and not data["Close"].empty:
                    temp_df = data[["Close"]].rename(columns={"Close": ticker})
                    all_data.append(temp_df)
                    print(f"[INFO] Dados para {ticker} baixados com sucesso.")
                    break  # Sucesso, sair do loop de tentativas
                else:
                    raise ValueError(
                        f"Dados inválidos recebidos para {ticker}.")
            except Exception as e:
                print(f"[WARNING] Falha ao baixar dados para {ticker}: {e}")
                time.sleep(2)  # Esperar antes de tentar novamente
                if attempt == max_retries - 1:
                    print(
                        f"[ERROR] Máximo de tentativas atingido para {ticker}.")
                    failed_tickers.append(ticker)

    # Consolidar os dados em um único DataFrame
    if all_data:
        consolidated_df = pd.concat(all_data, axis=1)
    else:
        consolidated_df = pd.DataFrame()

    print("[INFO] Dados coletados e limpos com sucesso.")
    return consolidated_df, failed_tickers


def process_date_column(df):
    """
    Processa o índice 'Date' do DataFrame, removendo timezones.

    Args:
        df (pd.DataFrame): DataFrame com dados históricos.

    Returns:
        pd.DataFrame: DataFrame com o índice 'Date' sem timezone.
    """
    df.index = pd.to_datetime(
        df.index)  # Certificar-se de que o índice é datetime
    df.index = df.index.tz_localize(None)  # Remover timezone, se presente
    return df


if __name__ == "__main__":
    # Definir ativos de interesse e benchmark
    tickers = ["VALE3.SA", "PETR4.SA", "ITUB4.SA",
               "PGCO34.SA", "AAPL34.SA", "AMZO34.SA", "^BVSP"]
    start_date = "2010-01-01"
    output_file = "BackPython/DADOS/historical_data_cleaned.csv"

    # Coletar e limpar os dados
    historical_data, failed_tickers = fetch_and_clean_data(
        tickers, start_date=start_date)

    # Verificar se dados foram coletados
    if not historical_data.empty:
        # Processar o índice 'Date' para remover timezone
        print("[INFO] Processando campo 'Date' para remover timezone...")
        historical_data = process_date_column(historical_data)

        # Remover linhas com dados faltantes
        print("[INFO] Removendo linhas com dados faltantes...")
        historical_data.dropna(inplace=True)

        # Salvar em CSV
        print(f"[INFO] Salvando dados limpos em {output_file}...")
        historical_data.to_csv(output_file, index_label="Date")
        print("[INFO] Processo concluído com sucesso!")
    else:
        print("[WARNING] Nenhum dado foi baixado. Arquivo CSV não será gerado.")

    # Exibir tickers que falharam
    if failed_tickers:
        print(f"[WARNING] Os seguintes tickers falharam: {failed_tickers}")
