# BP_mod1_dashboard.py
# Dashboard para visualização interativa de dados financeiros processados usando Dash

# BP_mod1_dashboard.py
# Dashboard para visualização interativa de dados financeiros processados usando Dash

from dash import Dash, html, dcc, dash_table
import plotly.graph_objs as go
import pandas as pd
import os
from BP_mod1_config import OUTPUT_DIR
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    """Carrega o arquivo de dados dos ativos financeiros a partir de um caminho especificado."""
    try:
        print(f"[LOG] Tentando carregar o arquivo: {file_path}")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return pd.DataFrame()


def normalize_data(df):
    """Normaliza os dados para o intervalo [0, 1] usando MinMaxScaler."""
    if df.empty:
        print("[LOG] DataFrame está vazio. Não foi possível normalizar os dados.")
        return df
    scaler = MinMaxScaler()
    try:
        normalized_df = pd.DataFrame(
            scaler.fit_transform(df), columns=df.columns, index=df.index
        )
        print("[LOG] Dados normalizados com sucesso.")
        return normalized_df
    except ValueError as e:
        print(f"Erro ao normalizar os dados: {e}")
        return pd.DataFrame()


def create_asset_graph_with_benchmark(asset_data, assets):
    """Cria gráficos de linha para cada ativo, com destaque no índice IBOVESPA."""
    graphs = []
    try:
        for asset in assets:
            trace = go.Scatter(
                x=asset_data.index,
                y=asset_data[asset],
                mode="lines",
                name=asset,
                line=dict(
                    width=4 if asset == "^BVSP" else 2,
                    dash="dash" if asset == "^BVSP" else "solid"
                )
            )
            graphs.append(trace)
        print("[LOG] Gráficos criados com sucesso.")
        return graphs
    except Exception as e:
        print(f"Erro ao criar gráficos: {e}")
        return []


def start_dashboard():
    """Inicializa e exibe o dashboard."""
    print("[LOG] Inicializando o dashboard...")

    # Caminhos para os arquivos de dados
    raw_data_path = os.path.join(OUTPUT_DIR, 'asset_data_raw.csv')
    processed_data_path = os.path.join(OUTPUT_DIR, 'asset_data_cleaner.csv')

    # Carregar os dados
    raw_data = load_data(raw_data_path)
    processed_data = load_data(processed_data_path)

    # Verificação de integridade dos dados
    if raw_data.empty or processed_data.empty:
        print("Erro: Dados não encontrados ou inválidos. O dashboard não será iniciado.")
        return

    # Normalizar os dados processados
    print("[LOG] Normalizando os dados.")
    normalized_data = normalize_data(processed_data)

    # Configurar o aplicativo Dash
    print("[LOG] Criando gráficos para os ativos.")
    app = Dash(__name__)
    assets = list(processed_data.columns)
    asset_graphs = create_asset_graph_with_benchmark(normalized_data, assets)

    # Layout do Dash
    app.layout = html.Div(style={'backgroundColor': '#1f1f1f'}, children=[
        html.H1("Dashboard de Visualização de Ativos Financeiros",
                style={'textAlign': 'center', 'color': 'white'}),

        # Tabela: Estatísticas Descritivas dos Dados Brutos
        html.H2("Estatísticas Descritivas - Dados Brutos",
                style={'textAlign': 'center', 'color': 'white'}),
        dash_table.DataTable(
            data=raw_data.describe().reset_index().to_dict('records'),
            columns=[{"name": col, "id": col}
                     for col in raw_data.describe().reset_index().columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={
                'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        ),

        # Tabela: Estatísticas Descritivas dos Dados Processados
        html.H2("Estatísticas Descritivas - Dados Processados",
                style={'textAlign': 'center', 'color': 'white'}),
        dash_table.DataTable(
            data=processed_data.describe().reset_index().to_dict('records'),
            columns=[{"name": col, "id": col}
                     for col in processed_data.describe().reset_index().columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={
                'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        ),

        # Gráfico: Preços dos Ativos Normalizados
        html.H2("Preços dos Ativos Normalizados ao Longo do Tempo",
                style={'textAlign': 'center', 'color': 'white'}),
        dcc.Graph(
            id='ativos-grafico',
            figure={
                'data': asset_graphs,
                'layout': go.Layout(
                    title="Preços Normalizados ao Longo do Tempo",
                    xaxis={'title': 'Data'},
                    yaxis={'title': 'Valor Normalizado'},
                    paper_bgcolor='#1f1f1f',
                    plot_bgcolor='#1f1f1f',
                    font=dict(color='white')
                )
            }
        )
    ])

    print("[LOG] Dashboard disponível em http://127.0.0.1:8050/")
    app.run_server(debug=False, host='127.0.0.1', port=8050)


# GARANTIA: Nenhum código fora de funções
# Certifique-se de que este módulo serve apenas para importar funções
