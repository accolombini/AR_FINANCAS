# BP_mod1_main_pipeline.py
# Pipeline principal para execução das etapas de coleta e processamento de dados essenciais para previsão

import os
from BP_mod1_data_collection import DataCollector
from BP_mod1_feature_engineering import FeatureEngineering
from BP_mod1_data_analysis import DataAnalysis
from BP_mod1_dashboard import main as dashboard_main
from BP_mod1_config import ASSETS, START_DATE, END_DATE, OUTPUT_DIR, RUN_DASHBOARD


def main():
    """Função principal que executa o pipeline de coleta e processamento de dados."""

    # 1. Coleta os dados dos ativos financeiros
    asset_data = DataCollector.get_asset_data(ASSETS, START_DATE, END_DATE)

    # 2. Aplica a engenharia de features essenciais nos dados coletados
    #    Calcula retornos, médias móveis e volatilidade para cada ativo
    processed_data = FeatureEngineering.add_essential_features(
        asset_data, ASSETS)

    # 3. Salva os dados processados em CSV no diretório especificado
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Cria o diretório se não existir
    processed_data.to_csv(f'{OUTPUT_DIR}/asset_data_cleaner.csv')
    print("Dados coletados e processados, salvos em CSV.")

    # 4. Realiza análise preliminar dos dados, como estatísticas descritivas e identificação de outliers
    analysis_results = DataAnalysis.analyze_and_clean_data(
        f'{OUTPUT_DIR}/asset_data_cleaner.csv')
    print("Análise preliminar dos dados:")
    for key, value in analysis_results.items():
        print(f"{key}:")
        print(value)
        print("\n")

    # 5. Executa o dashboard interativo para visualização, caso a flag RUN_DASHBOARD esteja configurada como True
    if RUN_DASHBOARD:
        print("Iniciando o dashboard...")
        dashboard_main()  # Executa o módulo do dashboard


# Executa o pipeline se o script for executado diretamente
if __name__ == "__main__":
    main()
