'''
    ||> Integridade dos Dados: Se há valores nulos ou ausentes em qualquer coluna.
    Consistência dos Dados Temporais: Se os dados de todos os ativos e indicadores macroeconômicos estão alinhados temporalmente.
    Distribuição e Estatísticas Básicas: Para garantir que os dados estão dentro dos intervalos esperados.
    Detecção de Outliers: Identificar valores que possam indicar problemas ou anomalias.
    Tipos de Dados: Verificação para assegurar que cada coluna tem o tipo de dado correto para futuras operações de análise e modelagem.'''

# Importar as bibliotecas necessárias

import pandas as pd

# Carregar o dataset
df = pd.read_csv('TESTES/DADOS/train_data_combined.csv',
                 index_col=0, parse_dates=True)


def validate_dataset(df):
    # 1. Verificar valores ausentes
    print("1. Verificação de valores ausentes:")
    missing_values = df.isnull().sum()
    if missing_values.any():
        print(missing_values[missing_values > 0])
    else:
        print("Nenhum valor ausente encontrado.")

    # 2. Verificar alinhamento temporal dos dados
    print("\n2. Verificação de consistência temporal:")
    if df.index.is_monotonic_increasing:
        print("Os dados estão ordenados temporalmente.")
    else:
        print("Problema: os dados não estão ordenados temporalmente.")

    # 3. Estatísticas descritivas e análise de distribuição
    print("\n3. Estatísticas descritivas:")
    print(df.describe())

    # 4. Detecção de outliers com base em desvio padrão
    print("\n4. Detecção de outliers:")
    for column in df.columns:
        mean = df[column].mean()
        std_dev = df[column].std()
        outliers = df[(df[column] < mean - 3 * std_dev) |
                      (df[column] > mean + 3 * std_dev)]
        if not outliers.empty:
            print(f"{column}: {len(outliers)} outliers detectados.")
        else:
            print(f"{column}: Nenhum outlier detectado.")

    # 5. Verificação de tipos de dados
    print("\n5. Verificação dos tipos de dados:")
    print(df.dtypes)


# Executar a função de validação
validate_dataset(df)
