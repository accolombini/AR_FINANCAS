# BP_mod1_dashboard.py
# Módulo para visualização dos resultados em um dashboard interativo com Dash

import os
import pandas as pd
from dash import Dash, dcc, html, dash_table
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from BP_mod1_data_analysis import DataAnalysis
from BP_mod1_config import OUTPUT_DIR

# Função para ler os dados salvos


def load_data(file_path):
    """Carrega o arquivo de dados dos ativos."""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

# Função para normalizar os dados


def normalize_data(df):
    """Normaliza os dados para o intervalo [0, 1]."""
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)
    return df_normalized

# Função para criar gráficos de ativos normalizados com destaque no IBOVESPA


def create_asset_graph_with_benchmark(asset_data, assets):
    """Cria gráficos de linha para cada ativo, destacando o índice IBOVESPA."""
    graphs = []
    for asset in assets:
        trace = go.Scatter(
            x=asset_data.index,
            y=asset_data[asset],
            mode='lines',
            name=asset,
            line=dict(width=4 if asset == '^BVSP' else 2,
                      dash='dash' if asset == '^BVSP' else 'solid')
        )
        graphs.append(trace)
    return graphs

# Função principal para criar e rodar o dashboard


def main():
    """Inicializa o aplicativo Dash e exibe o dashboard de visualização de ativos."""

    # Inicializando o aplicativo Dash
    app = Dash(__name__)

    # Carregar e analisar os dados
    file_path = f'{OUTPUT_DIR}/asset_data_cleaner.csv'
    asset_data = load_data(file_path)
    analysis_results = DataAnalysis.analyze_and_clean_data(file_path)
    asset_data_normalized = normalize_data(asset_data)

    # Criar os gráficos normalizados com destaque no IBOVESPA
    assets = list(asset_data.columns)
    asset_graphs = create_asset_graph_with_benchmark(
        asset_data_normalized, assets)

    # Layout do Dash com os elementos de tabela e gráficos
    app.layout = html.Div([
        html.H1("Visualização dos Ativos e Indicadores de Pré-processamento",
                style={'textAlign': 'center'}),

        # Tabela: Dimensões dos Dados
        html.H2("Dimensões dos Dados"),
        dash_table.DataTable(data=[{'Dimensões': f"{analysis_results['Dimensions'][0]} linhas, {
                             analysis_results['Dimensions'][1]} colunas"}]),

        # Tabela: Estatísticas Descritivas
        html.H2("Estatísticas Descritivas"),
        dash_table.DataTable(
            data=analysis_results['Descriptive Statistics'].reset_index().to_dict('records')),

        # Tabela: Dados Faltantes
        html.H2("Dados Faltantes (%)"),
        dash_table.DataTable(
            data=analysis_results['Missing Data (%)'].reset_index().to_dict('records')),

        # Tabela: Outliers
        html.H2("Contagem de Outliers"),
        dash_table.DataTable(
            data=analysis_results['Outliers Count'].reset_index().to_dict('records')),

        # Tabela: Retornos Anuais
        html.H2("Retornos Anuais (%)"),
        dash_table.DataTable(
            data=analysis_results['Annual Returns (%)'].reset_index().to_dict('records')),

        # Gráfico de ativos normalizados
        html.H2("Preços dos Ativos ao Longo do Tempo (Normalizados)"),
        dcc.Graph(
            id='ativos-grafico',
            figure={
                'data': asset_graphs,
                'layout': go.Layout(
                    title={'text': 'Preços dos Ativos Normalizados', 'x': 0.5},
                    xaxis={'title': 'Data'},
                    yaxis={'title': 'Valor Normalizado'}
                )
            }
        )
    ])

    # Rodar o aplicativo
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
