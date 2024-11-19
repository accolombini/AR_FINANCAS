'''
        MSE (Erro Quadrático Médio): Mede a média dos quadrados das diferenças entre valores previstos e reais.
        MAE (Erro Absoluto Médio): Mede o erro médio absoluto entre os valores previstos e reais.
        MAE como Percentual da Média: Relaciona o erro absoluto médio à média dos valores reais.
'''
# BP_mod2_main_pipeline.py
# Pipeline principal para o Módulo 2 - Modelagem de Curto Prazo

from BP_mod2_data_preparation import preprocess_for_lstm_with_paths
from BP_mod2_model_training import train_and_evaluate_lstm


def run_pipeline(data_dir='BackPython/DADOS/', target_column='^BVSP', sequence_length=30, test_months=2):
    """
    Executa o pipeline principal com saídas informativas consolidadas.
    """
    print("[INFO] Iniciando o pipeline principal...")

    # Etapa 1: Pré-processamento
    print("[INFO] Etapa 1: Pré-processamento dos dados...")
    preprocess_for_lstm_with_paths(
        input_file=f"{data_dir}/asset_data_cleaner.csv",
        target_column=target_column,
        test_months=test_months,
        sequence_length=sequence_length,
        output_dir=data_dir
    )

    # Etapa 2: Treinamento e Avaliação
    print("[INFO] Etapa 2: Treinamento e avaliação do modelo LSTM...")
    metrics = train_and_evaluate_lstm(
        data_dir=data_dir,
        model_output=f"{data_dir}/lstm_model.keras"
    )

    # Imprimir métricas consolidadas
    print("[INFO] Pipeline concluído. Métricas finais:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return metrics


if __name__ == "__main__":
    metrics = run_pipeline(
        data_dir="BackPython/DADOS/",
        target_column="^BVSP",
        sequence_length=30,
        test_months=2
    )
    print("[INFO] Execução completa.")
