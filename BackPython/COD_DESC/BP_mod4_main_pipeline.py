# BP_mod4_main_pipeline.py

import os
import subprocess


# Função para verificar se um script existe
def check_script(script_path):
    """
    Verifica se o script Python especificado existe.

    Parâmetros:
        script_path (str): Caminho do script.

    Retorna:
        bool: True se o script existir, False caso contrário.
    """
    if not os.path.isfile(script_path):
        print(f"Erro: O script {script_path} não foi encontrado.")
        return False
    return True


# Função para executar um script Python
def run_script(script_name, description):
    """
    Executa um script Python e fornece mensagens informativas.

    Parâmetros:
        script_name (str): Caminho do script Python.
        description (str): Descrição do script sendo executado.
    """
    try:
        print(f"Iniciando {description}...")
        subprocess.run(["python", script_name], check=True)
        print(f"{description} concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {description}: {e}")
    except Exception as e:
        print(f"Erro inesperado durante {description}: {e}")


# Diretório onde o módulo 4 está localizado
MODULE_DIR = "BackPython/"

# Caminhos dos scripts
LSTM_SCRIPT = os.path.join(MODULE_DIR, "BP_mod4_lstm_model.py")
OTHER_MODEL_SCRIPT = os.path.join(MODULE_DIR, "BP_mod4_other_model.py")


def run_lstm_training():
    """
    Executa o treinamento do modelo LSTM.
    """
    if check_script(LSTM_SCRIPT):
        run_script(LSTM_SCRIPT, "Treinamento do Modelo LSTM")


def run_other_model_training():
    """
    Executa o treinamento de outro modelo (futuro).
    """
    if check_script(OTHER_MODEL_SCRIPT):
        run_script(OTHER_MODEL_SCRIPT, "Treinamento de Outro Modelo")


def main():
    """
    Pipeline principal para o Módulo 4 - Treinamento de Modelos.
    """
    print("Executando o Pipeline do Módulo 4...")

    # Treinamento do modelo LSTM
    run_lstm_training()

    # Futuramente, adicionar o treinamento de outros modelos
    # run_other_model_training()


if __name__ == "__main__":
    main()
