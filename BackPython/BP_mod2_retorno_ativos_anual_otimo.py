# BP_mod2_retorno_ativos_anual_otimo.py: Filtragem de Ativos e Histórico de Dados
# -----------------------------------------------------------
# Este script filtra automaticamente os ativos que não atendem
# ao critério de retorno mínimo estabelecido (ex.: 15%).
# Além disso, remove as colunas correspondentes aos tickets
# reprovados do histórico de dados.
# -----------------------------------------------------------

import pandas as pd
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def carregar_dados(filepath):
    """
    Carrega os retornos anualizados salvos no BP_mod2_retorno_ativos_anual.py.
    """
    try:
        data = pd.read_csv(filepath, index_col="Date")
        logging.info(f"Dados carregados com sucesso de {filepath}.")
        return data
    except FileNotFoundError:
        logging.error(f"O arquivo {filepath} não foi encontrado.")
        raise
    except ValueError as e:
        logging.error(f"Erro ao carregar os dados: {e}")
        raise


def carregar_historico(filepath):
    """
    Carrega o histórico de dados diários dos ativos.
    """
    try:
        data = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        logging.info(
            f"Histórico de dados carregado com sucesso de {filepath}.")
        return data
    except FileNotFoundError:
        logging.error(f"O arquivo {filepath} não foi encontrado.")
        raise


def filtrar_tickets(retorno_anual, limite_minimo):
    """
    Filtra os ativos com base no retorno médio mínimo no período.
    """
    retorno_medio = retorno_anual.mean(skipna=True)
    tickets_aprovados = retorno_medio[retorno_medio >= limite_minimo].index
    tickets_reprovados = retorno_medio[retorno_medio < limite_minimo].index

    logging.info(f"Tickets aprovados: {list(tickets_aprovados)}")
    logging.warning(f"Tickets reprovados: {list(tickets_reprovados)}")

    # Retornar apenas os ativos aprovados
    return retorno_anual[tickets_aprovados], list(tickets_aprovados), list(tickets_reprovados)


def salvar_dados_filtrados(data, filepath):
    """
    Salva os dados filtrados em um arquivo CSV.
    """
    data.to_csv(filepath)
    logging.info(f"Dados filtrados salvos em: {filepath}")


def filtrar_historico(historico, tickets_aprovados):
    """
    Filtra o histórico de dados diários para incluir apenas os tickets aprovados.
    """
    return historico[tickets_aprovados]


def main():
    # Caminhos dos arquivos
    input_retornos = f"{OUTPUT_DIR}/retorno_anual.csv"
    input_historico = f"{OUTPUT_DIR}/historical_data_cleaned.csv"
    output_retornos = f"{OUTPUT_DIR}/filtered_data.csv"
    output_historico = f"{OUTPUT_DIR}/historical_data_filtered.csv"

    # Carregar retornos anualizados e histórico de dados
    retorno_anual = carregar_dados(input_retornos)
    historico = carregar_historico(input_historico)

    # Definir critério de exclusão
    limite_minimo = 15.0  # Retorno médio mínimo em %

    # Filtrar tickets
    retorno_anual_filtrado, tickets_aprovados, tickets_reprovados = filtrar_tickets(
        retorno_anual, limite_minimo)

    # Filtrar histórico de dados
    historico_filtrado = filtrar_historico(historico, tickets_aprovados)

    # Salvar dados filtrados
    salvar_dados_filtrados(retorno_anual_filtrado, output_retornos)
    salvar_dados_filtrados(historico_filtrado, output_historico)

    # Registrar os tickets excluídos em um log
    if tickets_reprovados:
        logging.warning(f"Os seguintes tickets foram excluídos: {
                        tickets_reprovados}")


if __name__ == "__main__":
    main()
