# Limpeza de Diretórios e Arquivos com Python
# -----------------------------------------------------------
# Este script realiza a limpeza de diretórios especificados,
# removendo todo o conteúdo de forma segura e gerenciável.
# Útil para reinicializar o estado de um projeto ou remover dados temporários.
# -----------------------------------------------------------

import os
import shutil
import logging
from BP_mod1_config import OUTPUT_DIR

# Configuração de logging para monitoramento
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def limpar_diretorios(diretorios: list):
    """
    Remove os diretórios especificados e todo o seu conteúdo.

    Parâmetros:
        - diretorios (list): Lista de caminhos de diretórios a serem limpos.
    """
    for directory in diretorios:
        if os.path.exists(directory):
            try:
                # Remove o diretório e todo o conteúdo de forma recursiva
                shutil.rmtree(directory)
                logging.info(f"Diretório {directory} apagado com sucesso.")
            except Exception as e:
                logging.error(f"Erro ao apagar o diretório {directory}: {e}")
        else:
            # Exibe uma mensagem se o diretório não existir
            logging.warning(f"Diretório {directory} não encontrado. Pulando.")


def main():
    """
    Executa a limpeza dos diretórios configurados.
    """
    # Lista de diretórios a serem limpos
    dirs_to_clean = [OUTPUT_DIR, os.path.join("BackPython", "MODELS")]

    logging.info("Iniciando o processo de limpeza de diretórios.")
    limpar_diretorios(dirs_to_clean)
    logging.info("Processo de limpeza concluído.")


if __name__ == "__main__":
    main()
