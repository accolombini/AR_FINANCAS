# BP_mod1_dashboard.py
# Módulo para visualização dos resultados em um dashboard interativo com Dash

import os
import pandas as pd
from dash import Dash, dcc, html
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

# Função para ler os dados salvos


def load_data():
    """
    Carrega o arquivo de dados dos ativos.

    Retorna:
        - DataFrame com dados dos ativos.

    Exceção:
        - FileNotFoundError se o arquivo de dados dos ativos não for encontrado.
    """
    asset_file = 'BackPython/DADOS/asset_data.csv'

    # Verificar se o arquivo de dados de ativos existe
    if not os.path.exists(asset_file):
        raise FileNotFoundError(
            "O arquivo de dados de ativos não foi encontrado.")

    # Carregar os dados dos ativos
    asset_data = pd.read_csv(asset_file, index_col=0, parse_dates=True)
    return asset_data

# Função para normalizar os dados


def normalize_data(df):
    """
    Normaliza os dados para que fiquem no intervalo [0, 1].

    Parâmetro:
        - df (DataFrame): DataFrame com os dados a serem normalizados.

    Retorna:
        - DataFrame normalizado.
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)
    return df_normalized

# Função para criar gráficos de ativos normalizados com destaque no IBOVESPA


def create_asset_graph_with_benchmark(asset_data, assets):
    """
    Cria gráficos de linha para cada ativo, destacando o índice IBOVESPA.

    Parâmetros:
        - asset_data (DataFrame): Dados normalizados dos ativos.
        - assets (list): Lista de ativos a serem plotados.

    Retorna:
        - Lista de traces (gráficos) para visualização.
    """
    graphs = []
    for asset in assets:
        if asset == '^BVSP':  # Destacar o índice BOVESPA
            trace = go.Scatter(
                x=asset_data.index,
                y=asset_data[asset],
                mode='lines',
                name=asset,
                # Linha vermelha, espessa, tracejada
                line=dict(color='red', width=4, dash='dash')
            )
        else:
            trace = go.Scatter(
                x=asset_data.index,
                y=asset_data[asset],
                mode='lines',
                name=asset
            )
        graphs.append(trace)
    return graphs

# Função principal para criar e rodar o dashboard


def main():
    """
    Inicializa o aplicativo Dash e exibe o dashboard de visualização de ativos.
    """
    # Inicializando o aplicativo Dash
    app = Dash(__name__)

    # Carregar os dados
    try:
        asset_data = load_data()  # Carregar dados dos ativos
    except FileNotFoundError as e:
        print(str(e))
        return  # Encerrar o programa se os dados não forem encontrados

    # Lista de ativos a serem exibidos
    assets = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
              'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']

    # Normalizar os dados dos ativos
    asset_data_normalized = normalize_data(asset_data)

    # Criar os gráficos normalizados com destaque no IBOVESPA
    asset_graphs = create_asset_graph_with_benchmark(
        asset_data_normalized, assets)

    # Layout do Dash com títulos principais centralizados
    app.layout = html.Div([
        # Centralizar o título principal
        html.H1("Visualização dos Ativos", style={'textAlign': 'center'}),
        dcc.Graph(
            id='ativos-grafico',
            figure={
                'data': asset_graphs,
                'layout': go.Layout(
                    title={
                        'text': 'Preços dos Ativos ao Longo do Tempo (Normalizados)', 'x': 0.5},
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
