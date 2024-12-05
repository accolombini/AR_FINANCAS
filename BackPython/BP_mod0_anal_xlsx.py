# BP_mod0_anal_xlsx.py: Análise Simplificada de Arquivos XLSX
# -----------------------------------------------------------
# Este script lista os arquivos XLSX disponíveis na pasta `DADOS` e permite
# que o usuário escolha quais deseja analisar.
# Após a análise, os resultados são exibidos no terminal e salvos em arquivos XLSX.
# -----------------------------------------------------------

import os
import logging
import pandas as pd
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Funções Auxiliares
# ---------------------------


def listar_arquivos_xlsx(diretorio: str) -> list:
    """
    Lista todos os arquivos .xlsx no diretório especificado.

    Parâmetros:
        - diretorio (str): Caminho do diretório a ser analisado.

    Retorna:
        - list: Lista de nomes de arquivos XLSX encontrados.
    """
    if not os.path.exists(diretorio):
        logging.error(f"Diretório especificado não existe: {diretorio}")
        return []
    return [f for f in os.listdir(diretorio) if f.endswith('.xlsx')]


def exibir_menu_arquivos(arquivos: list) -> list:
    """
    Exibe um menu interativo para o usuário selecionar arquivos.

    Parâmetros:
        - arquivos (list): Lista de arquivos XLSX.

    Retorna:
        - list: Lista de arquivos selecionados pelo usuário.
    """
    print("\n[INFO] Arquivos disponíveis para análise:")
    for idx, arquivo in enumerate(arquivos):
        print(f"{idx + 1}: {arquivo}")
    print("0: Cancelar")

    escolhas = input(
        "\nDigite os números dos arquivos que deseja analisar (separados por vírgula): ")
    if escolhas.strip() == "0":
        logging.info("Operação cancelada pelo usuário.")
        return []

    indices = [int(i.strip()) - 1 for i in escolhas.split(",")
               if i.strip().isdigit()]
    return [arquivos[i] for i in indices if 0 <= i < len(arquivos)]


def gerar_nome_arquivo_saida(nome_arquivo: str) -> str:
    """
    Gera um nome de arquivo para salvar resultados de análise, evitando redundâncias no sufixo.

    Parâmetros:
        - nome_arquivo (str): Nome do arquivo de entrada.

    Retorna:
        - str: Nome do arquivo com o sufixo '_analysis'.
    """
    if "_analysis" not in nome_arquivo:
        return os.path.splitext(nome_arquivo)[0] + "_analysis.xlsx"
    return nome_arquivo


def salvar_resultados(df: pd.DataFrame, caminho_saida: str):
    """
    Salva os resultados da análise em um arquivo Excel.

    Parâmetros:
        - df (pd.DataFrame): DataFrame contendo os resultados.
        - caminho_saida (str): Caminho para salvar o arquivo Excel.
    """
    try:
        df.to_excel(caminho_saida, index=False)
    except Exception as e:
        logging.error(f"Erro ao salvar os resultados: {e}")
        raise


def exibir_resultados_terminal(df: pd.DataFrame, nome_arquivo: str):
    """
    Exibe as estatísticas descritivas no terminal.

    Parâmetros:
        - df (pd.DataFrame): DataFrame com as estatísticas.
        - nome_arquivo (str): Nome do arquivo analisado.
    """
    print(f"\n[ESTATÍSTICAS] Resultados da análise para '{nome_arquivo}':")
    print(df.to_string(index=False))


# ---------------------------
# Fluxo Principal
# ---------------------------

def main():
    logging.info("Listando arquivos XLSX disponíveis...")
    arquivos_xlsx = listar_arquivos_xlsx(OUTPUT_DIR)
    if not arquivos_xlsx:
        logging.warning("Nenhum arquivo XLSX encontrado no diretório.")
        return

    print("\n[INFO] Arquivos listados com sucesso.")
    arquivos_selecionados = exibir_menu_arquivos(arquivos_xlsx)
    if not arquivos_selecionados:
        print("[INFO] Nenhum arquivo selecionado para análise. Encerrando.")
        return

    arquivos_processados = []
    for arquivo in arquivos_selecionados:
        input_file = os.path.join(OUTPUT_DIR, arquivo)
        output_file = gerar_nome_arquivo_saida(input_file)

        df = pd.read_excel(input_file)
        # Estatísticas descritivas completas
        resultado = df.describe(include='all')
        salvar_resultados(resultado, output_file)
        exibir_resultados_terminal(resultado, arquivo)

        arquivos_processados.append(output_file)

    # Exibe um resumo ao final
    print("\n[INFO] Processamento concluído. Arquivos de análise gerados:")
    for arquivo in arquivos_processados:
        print(f"- {arquivo}")


if __name__ == "__main__":
    main()
