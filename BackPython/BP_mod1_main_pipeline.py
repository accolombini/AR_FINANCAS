# BP_mod1_main_pipeline.py
# Pipeline principal para execução das etapas de coleta e processamento de dados de ativos

import os
from BP_mod1_data_collection import DataCollector
from BP_mod1_feature_engineering import FeatureEngineering
from BP_mod1_config import ASSETS, START_DATE, END_DATE, OUTPUT_DIR


def main():
    """Função principal que executa o pipeline de coleta e processamento de dados."""

    # Coleta os dados dos ativos
    asset_data = DataCollector.get_asset_data(ASSETS, START_DATE, END_DATE)

    # Aplica a engenharia de features nos dados coletados
    processed_data = FeatureEngineering.add_features(asset_data, ASSETS)

    # Salva os dados processados em CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_data.to_csv(f'{OUTPUT_DIR}/asset_data.csv')

    print("Dados coletados e processados, salvos em CSV.")


if __name__ == "__main__":
    main()
