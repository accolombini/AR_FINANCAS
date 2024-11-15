# BP_mod1_main_pipeline.py
# Pipeline principal para execução das etapas de coleta e processamento de dados essenciais para previsão

import os
from BP_mod1_data_collection import DataCollector
from BP_mod1_feature_engineering import FeatureEngineering
from BP_mod1_data_analysis import DataAnalysis
from BP_mod1_dashboard import main as dashboard_main
from BP_mod1_config import ASSETS, START_DATE, END_DATE, OUTPUT_DIR, RUN_DASHBOARD


def collect_data():
    """Coleta dados dos ativos financeiros especificados no arquivo de configuração."""
    try:
        print("Iniciando a coleta de dados...")
        asset_data = DataCollector.get_asset_data(ASSETS, START_DATE, END_DATE)
        print("Coleta de dados concluída.")
        return asset_data
    except Exception as e:
        print(f"Erro na coleta de dados: {e}")
        return None


def process_data(asset_data):
    """Aplica engenharia de features nos dados coletados."""
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
    """Salva o DataFrame processado no diretório de saída especificado."""
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
    """Realiza análise preliminar nos dados processados e salva resultados no console."""
    try:
        print("Iniciando análise preliminar dos dados...")
        analysis_results = DataAnalysis.analyze_and_clean_data(file_path)
        print("Análise preliminar dos dados concluída:")
        for key, value in analysis_results.items():
            print(f"{key}:")
            print(value)
            print("\n")
        return analysis_results
    except Exception as e:
        print(f"Erro na análise dos dados: {e}")
        return None


def run_dashboard():
    """Executa o dashboard interativo, se configurado para execução."""
    try:
        if RUN_DASHBOARD:
            print("Iniciando o dashboard...")
            dashboard_main()  # Executa o módulo do dashboard
            print("Dashboard encerrado.")
    except Exception as e:
        print(f"Erro ao iniciar o dashboard: {e}")


def main():
    """Função principal que executa o pipeline de coleta, processamento, análise e visualização de dados."""

    # 1. Coleta os dados dos ativos financeiros
    asset_data = collect_data()
    if asset_data is None:
        print("Pipeline interrompido: erro na coleta de dados.")
        return

    # 2. Aplica engenharia de features nos dados coletados
    processed_data = process_data(asset_data)
    if processed_data is None:
        print("Pipeline interrompido: erro na engenharia de features.")
        return

    # 3. Salva os dados processados em CSV no diretório especificado
    file_path = save_data(processed_data)
    if file_path is None:
        print("Pipeline interrompido: erro ao salvar os dados processados.")
        return

    # 4. Realiza análise preliminar dos dados
    analysis_results = analyze_data(file_path)
    if analysis_results is None:
        print("Pipeline interrompido: erro na análise dos dados.")
        return

    # 5. Executa o dashboard interativo para visualização
    run_dashboard()


# Executa o pipeline se o script for executado diretamente
if __name__ == "__main__":
    main()
