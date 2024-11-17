# BP_mod2_main_pipeline.py
# Pipeline principal para o Módulo 2 - Modelagem de Curto Prazo

import os
# Importa o módulo de treinamento do modelo de curto prazo
import BP_mod2_model_training

# Configuração de diretórios
MODEL_DIR = "BackPython/MODELS/"


def setup_directories():
    """
    Configura os diretórios necessários para o pipeline.
    Cria o diretório para salvar os modelos, se não existir.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Diretório criado: {MODEL_DIR}")
    else:
        print(f"Diretório já existente: {MODEL_DIR}")


def train_short_term_model():
    """
    Treina o modelo de curto prazo usando o módulo de treinamento.
    """
    print("Iniciando o treinamento do modelo de curto prazo (Random Forest)...")
    try:
        BP_mod2_model_training.main()
        print("Treinamento e validação do modelo de curto prazo concluídos.")
    except Exception as e:
        print(f"Erro durante o treinamento do modelo de curto prazo: {e}")
        raise


def main():
    """
    Pipeline principal para o Módulo 2 - Modelagem e Treinamento de Modelos.
    """
    print("Executando o Pipeline do Módulo 2 - Modelagem de Curto Prazo...")

    # Configurar diretórios
    setup_directories()

    # Treinamento do modelo de curto prazo
    try:
        train_short_term_model()
    except Exception as e:
        print(f"Erro no Pipeline do Módulo 2: {e}")
    else:
        print("Pipeline do Módulo 2 executado com sucesso.")

    # Placeholder para modelagem de longo prazo
    # print("Modelagem de longo prazo será adicionada futuramente.")


if __name__ == "__main__":
    main()
