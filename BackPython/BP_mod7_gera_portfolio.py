# BP_mod7_gera_base_portfolio_refatorado.py
# -----------------------------------------------------------------------------
# Este script gera a base consolidada 'historical_data_portfolio.csv'
# utilizando os dados históricos filtrados e o portfólio otimizado.
# -----------------------------------------------------------------------------

import os
import pandas as pd
from BP_mod1_config import OUTPUT_DIR

# Caminhos para os arquivos
historical_data_file = os.path.join(OUTPUT_DIR, "historical_data_filtered.csv")
portfolio_file = os.path.join(OUTPUT_DIR, "portfolio_otimizado.csv")
output_file = os.path.join(OUTPUT_DIR, "historical_data_portfolio.csv")


def main():
    # Verificar e carregar os dados históricos
    if not os.path.exists(historical_data_file):
        raise FileNotFoundError(f"[ERRO] Arquivo de dados históricos não encontrado: {
                                historical_data_file}")
    historical_data = pd.read_csv(historical_data_file, parse_dates=['Date'])
    print(f"[INFO] Dados históricos carregados. Linhas: {
          historical_data.shape[0]}")

    # Verificar e carregar os dados do portfólio
    if not os.path.exists(portfolio_file):
        raise FileNotFoundError(
            f"[ERRO] Arquivo de portfólio não encontrado: {portfolio_file}")
    portfolio_data = pd.read_csv(portfolio_file)
    print(f"[INFO] Dados do portfólio carregados. Ativos: {
          portfolio_data.shape[0]}")

    # Validar se todos os ativos do portfólio estão na base histórica
    portfolio_assets = portfolio_data['Ativo'].values
    missing_assets = [
        asset for asset in portfolio_assets if asset not in historical_data.columns]
    if missing_assets:
        raise ValueError(f"[ERRO] Ativos ausentes na base histórica: {
                         ', '.join(missing_assets)}")
    print("[INFO] Todos os ativos do portfólio estão presentes na base histórica.")

    # Converter pesos de porcentagem para proporção
    portfolio_data['Peso (%)'] = portfolio_data['Peso (%)'] / 100

    # Filtrar apenas as colunas necessárias (Date + ativos do portfólio)
    relevant_columns = ['Date'] + list(portfolio_assets)
    historical_data = historical_data[relevant_columns]

    # Criar a coluna 'Portfólio'
    print("[INFO] Calculando a coluna 'Portfólio'...")
    historical_data['Portfólio'] = 0.0
    for _, row in portfolio_data.iterrows():
        ativo = row['Ativo']
        peso = row['Peso (%)']
        historical_data['Portfólio'] += historical_data[ativo] * peso

    # Verificar se há valores nulos na coluna 'Portfólio'
    if historical_data['Portfólio'].isnull().sum() > 0:
        raise ValueError(
            "[ERRO] Foram encontrados valores nulos na coluna 'Portfólio'. Verifique os dados de entrada.")
    print("[INFO] Coluna 'Portfólio' calculada com sucesso.")

    # Salvar a base consolidada
    print(f"[INFO] Salvando a base consolidada em: {output_file}")
    historical_data.to_csv(output_file, index=False)
    print("[INFO] Base consolidada salva com sucesso!")


if __name__ == "__main__":
    main()
