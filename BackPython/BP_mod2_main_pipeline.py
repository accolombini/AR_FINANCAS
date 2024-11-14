# BP_mod2_main_pipeline.py

import os
# Importando o módulo de treinamento de modelo de curto prazo
import BP_mod2_model_training


def main():
    """
    Pipeline principal para o Módulo 2 - Modelagem de Curto e Longo Prazo.
    """
    # Verificar se o diretório de modelos existe, se não, cria-lo
    model_dir = "BackPython/MODELS/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Executar o treinamento e validação do modelo de curto prazo
    print("Iniciando o treinamento do modelo de curto prazo...")
    BP_mod2_model_training.main()  # Chama a função principal do módulo de treinamento

    # Aqui, futuramente, vamos adicionar a modelagem de longo prazo
    print("Treinamento e validação do modelo de curto prazo concluídos.")


if __name__ == "__main__":
    main()
