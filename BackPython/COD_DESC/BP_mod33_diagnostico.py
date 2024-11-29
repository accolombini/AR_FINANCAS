import pandas as pd

# Caminho para o arquivo mc_simulations.csv
simulations_file = "BackPython/DADOS/mc_simulations.csv"

# Carregar as simulações
mc_simulations = pd.read_csv(simulations_file)

# Identificar ativos com valores zero
zero_values = (mc_simulations == 0).sum()
print("Valores zero por ativo:")
print(zero_values[zero_values > 0])

# Visualizar exemplos de zeros
ativos_com_zeros = zero_values[zero_values > 0].index.tolist()
print("\nExemplo de valores zero:")
for ativo in ativos_com_zeros:
    print(f"\nAtivo: {ativo}")
    print(mc_simulations[mc_simulations[ativo] == 0])
