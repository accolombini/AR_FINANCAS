# BP_mod2_main_pipeline.py
# Pipeline principal para o Módulo 2 - Modelagem de Curto Prazo

import os
# Importa o módulo de treinamento do modelo de curto prazo
import BP_mod2_model_training


def main():
    """
    Pipeline principal para o Módulo 2 - Modelagem e Treinamento de Modelos.
    Este pipeline organiza o fluxo de trabalho para criar, treinar e validar modelos de previsão.
    """
    # Diretório para salvar os modelos treinados
    model_dir = "BackPython/MODELS/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Cria o diretório se não existir
        print(f"Diretório criado: {model_dir}")

    # Treinamento do modelo de curto prazo (Random Forest)
    print("Iniciando o treinamento do modelo de curto prazo (Random Forest)...")
    BP_mod2_model_training.main()  # Executa o treinamento do modelo

    # Mensagem indicando conclusão do pipeline de curto prazo
    print("Treinamento e validação do modelo de curto prazo concluídos.")

    # Espaço reservado para modelagem de longo prazo
    # print("Modelagem de longo prazo será adicionada futuramente.")


if __name__ == "__main__":
    main()
