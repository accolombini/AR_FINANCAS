'''
    Módulo 1: Foi quebrado em duas partes, esta a segunda parte do módulo 1 responsável pelo Dashboard
    
    Coleta e Preparação dos Dados
    Este módulo está claro e bem definido. Para alinhar com nossa discussão:

    Ajustes: Inclua indicadores econômicos macro e setoriais como variáveis auxiliares(ex.: inflação, taxa de câmbio, PIB).
    Feature Engineering Avançado: Adicione features que capturam sazonalidades de longo prazo(indicadores específicos, médias móveis de 6 a 12 meses) para apoiar previsões de longo prazo.
    Output atualizado: Dados históricos preparados, com features de curto e longo prazo e indicadores econômicos adicionais.

    ||| > Your registered API key is : 8e5112d9a5cdfb3cff60e7486f2bec09 Documentation is available on the St. Louis Fed web services website.
    8e5112d9a5cdfb3cff60e7486f2bec09

'''

# Importando bibliotecas necessárias
# t20_md2_visualizacao_dashboard.py

import os
import pandas as pd
from dash import Dash, dcc, html
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

# Função para ler os dados salvos


def load_data():
    asset_file = 'TESTES/DADOS/asset_data.csv'
    econ_file = 'TESTES/DADOS/economic_data.csv'

    # Verificar se os arquivos existem
    if not os.path.exists(asset_file) or not os.path.exists(econ_file):
        raise FileNotFoundError("Os arquivos de dados não foram encontrados.")

    # Carregar os dados
    asset_data = pd.read_csv(asset_file, index_col=0, parse_dates=True)
    econ_data = pd.read_csv(econ_file, index_col=0, parse_dates=True)

    return asset_data, econ_data

# Função para normalizar os dados


def normalize_data(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)
    return df_normalized

# Função para criar os gráficos de ativos normalizados


# Função para criar os gráficos de ativos normalizados com destaque no IBOVESPA
def create_asset_graph_with_benchmark(asset_data, assets):
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


# Função para interpolar dados trimestrais do PIB


def interpolate_pib(econ_data):
    # Interpolando para preencher os valores mensais
    econ_data['pib'] = econ_data['pib'].interpolate(method='time')
    return econ_data

# Função principal para criar e rodar o dashboard


def main():
    # Inicializando o aplicativo Dash
    app = Dash(__name__)

    # Carregar os dados
    try:
        asset_data, econ_data = load_data()
    except FileNotFoundError as e:
        print(str(e))
        return  # Encerrar o programa se os dados não forem encontrados

    assets = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA',
              'PGCO34.SA', 'AAPL34.SA', 'AMZO34.SA', '^BVSP']

    # Normalizar os dados
    asset_data_normalized = normalize_data(asset_data)

    # Interpolar e normalizar os dados econômicos
    econ_data = interpolate_pib(econ_data)
    econ_data_normalized = normalize_data(econ_data)

    # Criar os gráficos normalizados com destaque no IBOVESPA
    asset_graphs = create_asset_graph_with_benchmark(
        asset_data_normalized, assets)

    # Layout do Dash com títulos principais centralizados
    app.layout = html.Div([
        html.H1("Visualização dos Ativos e Indicadores Econômicos", style={
                'textAlign': 'center'}),  # Centralizar o título principal
        dcc.Graph(
            id='ativos-grafico',
            figure={
                'data': asset_graphs,
                'layout': go.Layout(
                    # Centralizando o título do gráfico
                    title={
                        'text': 'Preços dos Ativos ao Longo do Tempo (Normalizados)', 'x': 0.5},
                    xaxis={'title': 'Data'},
                    yaxis={'title': 'Valor Normalizado'}
                )
            }
        ),
        # Centralizar o subtítulo de indicadores econômicos
        html.H2("Indicadores Econômicos", style={'textAlign': 'center'}),
        dcc.Graph(
            id='indicadores-grafico',
            figure={
                'data': [
                    go.Scatter(x=econ_data_normalized.index,
                               y=econ_data_normalized['inflacao'], mode='lines', name='Inflação'),
                    go.Scatter(x=econ_data_normalized.index,
                               y=econ_data_normalized['taxa_juros'], mode='lines', name='Taxa de Juros'),
                    go.Scatter(x=econ_data_normalized.index,
                               y=econ_data_normalized['pib'], mode='lines', name='PIB')
                ],
                'layout': go.Layout(
                    # Centralizando o título do gráfico
                    title={
                        'text': 'Indicadores Econômicos ao Longo do Tempo (Normalizados)', 'x': 0.5},
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
