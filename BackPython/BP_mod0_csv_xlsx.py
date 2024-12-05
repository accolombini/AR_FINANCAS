# BP_mod0_csv_xlsx.py: Conversão de CSV para XLSX com seleção interativa
# -----------------------------------------------------------
# Este script lista os arquivos CSV disponíveis na pasta `DADOS` e permite
# que o usuário escolha quais deseja converter para XLSX.
# Inclui o processamento da coluna 'Date', removendo timezones.
# -----------------------------------------------------------

import pandas as pd
import os
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging para monitoramento
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def listar_arquivos_csv(diretorio: str) -> list:
    """
    Lista todos os arquivos .csv no diretório especificado.

    Parâmetros:
        - diretorio (str): Caminho do diretório a ser analisado.

    Retorna:
        - list: Lista de caminhos completos dos arquivos CSV encontrados.
    """
    return [f for f in os.listdir(diretorio) if f.endswith('.csv')]


def exibir_menu_arquivos(arquivos: list) -> list:
    """
    Exibe um menu interativo para o usuário selecionar arquivos.

    Parâmetros:
        - arquivos (list): Lista de arquivos CSV.

    Retorna:
        - list: Lista de arquivos selecionados pelo usuário.
    """
    print("\n[INFO] Arquivos disponíveis para conversão:")
    for idx, arquivo in enumerate(arquivos):
        print(f"{idx + 1}: {arquivo}")
    print("0: Cancelar")

    escolhas = input(
        "\nDigite os números dos arquivos que deseja converter (separados por vírgula): ")
    if escolhas.strip() == "0":
        logging.info("Operação cancelada pelo usuário.")
        return []

    indices = [int(i.strip()) - 1 for i in escolhas.split(",")
               if i.strip().isdigit()]
    return [arquivos[i] for i in indices if 0 <= i < len(arquivos)]


def carregar_csv(caminho: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.

    Parâmetros:
        - caminho (str): Caminho do arquivo CSV.

    Retorna:
        - pd.DataFrame: DataFrame carregado com os dados.
    """
    try:
        logging.info(f"Carregando arquivo CSV: {caminho}")
        return pd.read_csv(caminho)
    except FileNotFoundError:
        logging.error(f"Arquivo CSV não encontrado: {caminho}")
        raise
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo CSV: {e}")
        raise


def processar_coluna_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa a coluna 'Date', convertendo-a para datetime e removendo timezones.

    Parâmetros:
        - df (pd.DataFrame): DataFrame contendo os dados.

    Retorna:
        - pd.DataFrame: DataFrame com a coluna 'Date' processada, se presente.
    """
    if "Date" in df.columns:
        try:
            logging.info("Processando a coluna 'Date'.")
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df["Date"] = df["Date"].apply(
                lambda x: x.tz_localize(None) if pd.notnull(
                    x) and x.tzinfo else x
            )
            logging.info("Timezone removido da coluna 'Date'.")
        except Exception as e:
            logging.error(f"Erro ao processar a coluna 'Date': {e}")
            raise
    return df


def salvar_xlsx(df: pd.DataFrame, caminho: str):
    """
    Salva o DataFrame em formato XLSX.

    Parâmetros:
        - df (pd.DataFrame): DataFrame a ser salvo.
        - caminho (str): Caminho do arquivo XLSX de saída.
    """
    try:
        logging.info(f"Salvando arquivo XLSX: {caminho}")
        df.to_excel(caminho, index=False)
        logging.info("Arquivo XLSX salvo com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo XLSX: {e}")
        raise


def main():
    """
    Executa o fluxo de conversão de arquivos CSV para XLSX.
    """
    arquivos_csv = listar_arquivos_csv(OUTPUT_DIR)
    if not arquivos_csv:
        logging.warning("Nenhum arquivo CSV encontrado no diretório.")
        return

    arquivos_selecionados = exibir_menu_arquivos(arquivos_csv)
    if not arquivos_selecionados:
        logging.info("Nenhum arquivo selecionado para conversão.")
        return

    for arquivo in arquivos_selecionados:
        csv_path = os.path.join(OUTPUT_DIR, arquivo)
        xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"

        # Carregar, processar e salvar
        df = carregar_csv(csv_path)
        df = processar_coluna_date(df)
        salvar_xlsx(df, xlsx_path)


if __name__ == "__main__":
    main()
