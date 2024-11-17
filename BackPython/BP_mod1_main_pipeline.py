# BP_mod1_feature_engineering.py
# Módulo para aplicação de cálculos de indicadores essenciais para previsões financeiras

# Pipeline principal para o Módulo 1 - Coleta, Processamento, Análise e Visualização de Dados

import os
from BP_mod1_data_collection import DataCollector
from BP_mod1_feature_engineering import FeatureEngineering
from BP_mod1_data_analysis import DataAnalysis
from BP_mod1_dashboard import start_dashboard
from BP_mod1_config import ASSETS, START_DATE, END_DATE, OUTPUT_DIR, RUN_DASHBOARD


def collect_data():
    try:
        print("Iniciando a coleta de dados...")
        asset_data = DataCollector.get_asset_data(
            ASSETS, START_DATE, END_DATE, save_to_csv=True, output_path=os.path.join(OUTPUT_DIR, 'asset_data_raw.csv')
        )
        print("Coleta de dados concluída.")
        return asset_data
    except Exception as e:
        print(f"Erro na coleta de dados: {e}")
        return None


def process_data(asset_data):
    try:
        print("Iniciando a engenharia de features...")
        processed_data = FeatureEngineering.add_essential_features(
            asset_data, ASSETS)
        print("Engenharia de features concluída.")
        return processed_data
    except Exception as e:
        print(f"Erro na engenharia de features: {e}")
        return None


def save_data(processed_data):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, 'asset_data_cleaner.csv')
        processed_data.to_csv(output_path)
        print(f"Dados processados salvos em {output_path}.")
        return output_path
    except Exception as e:
        print(f"Erro ao salvar dados: {e}")
        return None


def analyze_data(file_path):
    try:
        print("Iniciando análise preliminar dos dados...")
        analysis_results = DataAnalysis.analyze_and_clean_data(file_path)
        print("Análise preliminar dos dados concluída.")
        return analysis_results
    except Exception as e:
        print(f"Erro na análise dos dados: {e}")
        return None


def main():
    print("Executando o Pipeline do Módulo 1...")

    # 1. Coleta dos dados
    asset_data = collect_data()
    if asset_data is None:
        print("Pipeline interrompido: erro na coleta de dados.")
        return

    # 2. Processamento dos dados
    processed_data = process_data(asset_data)
    if processed_data is None:
        print("Pipeline interrompido: erro na engenharia de features.")
        return

    # 3. Salvar dados processados
    file_path = save_data(processed_data)
    if file_path is None:
        print("Pipeline interrompido: erro ao salvar os dados processados.")
        return

    # 4. Análise dos dados
    analysis_results = analyze_data(file_path)
    if analysis_results is None:
        print("Pipeline interrompido: erro na análise dos dados.")
        return

    # 5. Iniciar o dashboard
    if RUN_DASHBOARD:
        print("Iniciando o dashboard...")
        start_dashboard()


if __name__ == "__main__":
    print("Executando o Dashboard...")
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("Dashboard encerrado pelo usuário.")
