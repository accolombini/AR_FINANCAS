'''Análise de dados com Python'''

# Importar bibliotecas necessárias

import pandas as pd

# Substitua pelo caminho do seu arquivo
file_path = 'BackPython/DADOS/asset_data_cleaner.xlsx'

# Carregar a planilha no pandas
df = pd.read_excel(file_path)

# Função para verificar dados inválidos (valores não numéricos em colunas numéricas)


def calcular_estatisticas(coluna):
    if coluna.dtype in ['float64', 'int64']:
        return {
            'num_registros': coluna.size,
            'dados_faltantes': coluna.isnull().sum(),
            'dados_inválidos': (~coluna.apply(lambda x: isinstance(x, (int, float)) or pd.isnull(x))).sum(),
            'valor_minimo': coluna.min(),
            'valor_maximo': coluna.max(),
            'valor_medio': coluna.mean(),
            'desvio_padrao': coluna.std(),
        }
    elif pd.api.types.is_datetime64_any_dtype(coluna) or coluna.name == "Date":
        coluna_datetime = pd.to_datetime(coluna, errors='coerce')
        return {
            'num_registros': coluna.size,
            'dados_faltantes': coluna.isnull().sum(),
            'dados_inválidos': coluna_datetime.isnull().sum() - coluna.isnull().sum(),
            'data_inicio': coluna_datetime.min(),
            'data_fim': coluna_datetime.max(),
        }
    else:
        return {
            'num_registros': coluna.size,
            'dados_faltantes': coluna.isnull().sum(),
            'dados_inválidos': 0,
        }


# Converter colunas com datas para timezone unaware e corrigir possíveis problemas
if "Date" in df.columns:
    # Converter para datetime
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Date"] = df["Date"].dt.tz_localize(None)  # Remover timezone, se houver

# Criar uma tabela de estatísticas para cada coluna
estatisticas = []
for col in df.columns:
    stats = calcular_estatisticas(df[col])
    stats['coluna'] = col
    stats['tipo_dado'] = df[col].dtype
    estatisticas.append(stats)

# Converter as estatísticas em um DataFrame
resultado = pd.DataFrame(estatisticas)

# Reorganizar as colunas para uma ordem mais lógica e fluída
resultado = resultado[
    ['coluna', 'tipo_dado', 'num_registros', 'dados_faltantes', 'dados_inválidos',
     'valor_minimo', 'valor_maximo', 'valor_medio', 'desvio_padrao',
     'data_inicio', 'data_fim']
]

# Salvar os resultados em um arquivo Excel para visualização
output_file = 'BackPython/DADOS/resultado_analise.xlsx'
resultado.to_excel(output_file, index=False)

print(f"Análise completa salva em: {output_file}")
