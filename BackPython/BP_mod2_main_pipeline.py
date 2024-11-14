# BP_mod2_main_pipeline.py

from BP_mod2_data_preparation import main as prepare_data_main


def main():
    """
    Pipeline principal para preparação dos dados para modelagem.
    Executa o módulo de preparação de dados para dividir, normalizar e salvar os dados.
    """
    print("Iniciando a preparação dos dados para modelagem...")

    # Executa a preparação dos dados, incluindo divisão, normalização e separação de variáveis
    prepare_data_main()

    print("Preparação de dados concluída. Arquivos salvos em 'BackPython/DADOS/'.")


if __name__ == "__main__":
    main()
