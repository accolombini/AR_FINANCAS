# BP_mod3_main_pipeline.py

import os
import subprocess

# Diretório onde o módulo 3 está localizado
MODULE_DIR = "BackPython/"
DATA_DIR = os.path.join(MODULE_DIR, "DADOS")

# Verificar se os arquivos necessários para o dashboard existem
REQUIRED_FILES = ["y_random_forest.csv", "y_pred_rf.csv"]


def check_required_files():
    """
    Verifica se todos os arquivos necessários para o dashboard estão presentes no diretório de dados.
    """
    missing_files = [file for file in REQUIRED_FILES if not os.path.exists(
        os.path.join(DATA_DIR, file))]
    if missing_files:
        raise FileNotFoundError(f"Os seguintes arquivos estão ausentes em {
                                DATA_DIR}: {', '.join(missing_files)}")


def run_dashboard():
    """
    Executa o dashboard do Módulo 3 usando subprocess.

    Lança uma exceção se o script do dashboard falhar.
    """
    dashboard_script = os.path.join(MODULE_DIR, "BP_mod3_dashboard.py")
    print("Iniciando o Dashboard do Módulo 3...")

    try:
        # Executar o script do dashboard
        subprocess.run(["python", dashboard_script], check=True)
        print("Dashboard encerrado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o Dashboard do Módulo 3: {e}")
        raise


def main():
    """
    Pipeline principal para o Módulo 3.

    Executa o dashboard e gerencia mensagens de status.
    """
    print("Executando o Pipeline do Módulo 3...")

    # Verificar arquivos necessários antes de executar o dashboard
    try:
        check_required_files()
        run_dashboard()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Erro no Pipeline do Módulo 3: {e}")
    else:
        print("Pipeline do Módulo 3 executado com sucesso.")


if __name__ == "__main__":
    main()
