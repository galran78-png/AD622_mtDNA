"""
Haplogroup clustering.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import plotly.graph_objects as go
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from dash import Dash, html, dcc, Input, Output

# ============================================================
# === 1. GENERAL VARIABLES ===
# ============================================================
FILE_IN      = 'mt_major_3.csv'
CLUSTER_FILE = 'mt_clusters.csv'
OUTPUT_IMG = 'img_1'
CLADE_FILTER = 'клад'
RANDOM_STATE = 42
N_INIT=10
CMAP_NAME    = 'hsv'


REGION_SYMBOLS = [
    'circle', 'square', 'diamond', 'triangle-up',
    'triangle-down', 'pentagon', 'hexagon', 'star',
    'cross', 'x', 'hourglass', 'bowtie',
    'triangle-left', 'triangle-right', 'octagon', 'hexagon2', 'plus', 'asterisk'
]

PLOTLY_LAYOUT = dict(
    template='simple_white',
    font=dict(family='Georgia, "Times New Roman", serif',
              size=11, color='#1a1a1a'),
    title_font=dict(size=14, color='#1a1a1a'),
    margin=dict(l=70, r=30, t=60, b=60),
)

S = {
    'container': {
        'fontFamily': 'Georgia, "Times New Roman", serif',
        'maxWidth': 'auto', 'margin': '0 auto',
        'padding': '24px', 'color': '#1a1a1a',
    },
    'title': {
        'textAlign': 'center', 'fontWeight': 'normal',
        'fontSize': '22px', 'letterSpacing': '0.5px',
        'paddingBottom': '12px',
    },
    'divider': {
        'height': '3px', 'backgroundColor': '#333333',
        'margin': '16px 0',
    },
    'row': {
        'display': 'flex', 'gap': '28px', 'marginTop': '20px',
    },
    'columnLeft': {'flex': '3'},
    'columnRight': {'flex': '1'},
    'sectionTitle': {
        'fontSize': '16px', 'fontWeight': 'bold',
        'marginTop': '0',
    },
    'label': {
        'fontSize': '12px', 'color': '#444444',
    },
    'radioLabel': {
        'marginRight': '16px', 'fontSize': '13px',
    },
    'graph': {'marginTop': '24px'},
}

# ============================================================
# === 2. MODEL ===
# ============================================================
def load_raw(path: str) -> TextFileReader | DataFrame:
    return pd.read_csv(path, sep=';', encoding='cp1251')

def build_matrix(df_raw: pd.DataFrame):
    df = df_raw[df_raw[CLADE_FILTER] != 'Прочие'].copy()
    mat = df.pivot_table(
        index='страна', columns=CLADE_FILTER, values='доля',
        aggfunc='sum', fill_value=0,
    )
    country_region = df.groupby('страна')['регион'].first().reindex(mat.index)
    return mat, country_region, df


def fit_pca(mat: pd.DataFrame):
    n_comp = min(2, mat.shape[0], mat.shape[1])
    if n_comp < 2:
        raise ValueError(f'PCA requires >=2 dimensions, got {mat.shape}')
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    scores = pca.fit_transform(mat.values)
    return pca, scores, pca.explained_variance_ratio_ * 100


def fit_model(mat: pd.DataFrame, k: int, name: str):
    if name == 'kmeans':
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels_old = model.fit_predict(mat.values)
    elif name == 'spectral':
        model = SpectralClustering(
            n_clusters=k, affinity='nearest_neighbors',
            assign_labels='kmeans', random_state=RANDOM_STATE,
        )
        labels_old = model.fit_predict(mat.values)
    elif name == 'gauss':
        model = GaussianMixture(n_components=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels_old = model.fit_predict(mat.values)
    else:
        raise ValueError(f'Invalid model name: {name!r}')

    sizes   = np.bincount(labels_old, minlength=k)
    order   = np.argsort(-sizes)
    mapping = {int(old): int(new) + 1 for new, old in enumerate(order)}
    new_labels = np.array([mapping[int(l)] for l in labels_old])
    return model, new_labels, order


# ============================================================
# === 3. VIEW ===
# ============================================================
def build_scatter_fig(df_plot, expl, k, palette):
    cluster_order = [f'К{i}' for i in range(1, k + 1)]
    df_plot = df_plot.copy()
    df_plot['регион'] = df_plot['регион'].fillna('—')
    regions = sorted(df_plot['регион'].unique().tolist())
    region_symbols = {r: REGION_SYMBOLS[i % len(REGION_SYMBOLS)]
                      for i, r in enumerate(regions)}

    fig = go.Figure()

    # [1] points: country labels
    for cl in cluster_order:
        for reg in regions:
            sub = df_plot[(df_plot['кластер'] == cl) &
                          (df_plot['регион'] == reg)]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub['PC1'].tolist(),
                y=sub['PC2'].tolist(),
                mode='markers+text',
                marker=dict(
                    color=palette[cl], size=10,
                    symbol=region_symbols[reg],
                    line=dict(width=1, color='gray'),
                    opacity=0.85,
                ),
                text=sub['страна'].tolist(),
                textposition='top center',
                textfont=dict(size=12, color='#1a1a1a'),
                showlegend=False,
                hoverinfo='skip',
            ))

    # [2] legend — cluster colors
    for cl in cluster_order:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=palette[cl], size=12,
                        line=dict(width=1, color='gray')),
            name=cl, legendgroup='cluster',
            legendgrouptitle=dict(text='Кластер (цвет)'),
            showlegend=True, hoverinfo='skip',
        ))

    # [3] legend — region shapes
    for reg in regions:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color='#b0b0b0', size=12,
                        symbol=region_symbols[reg],
                        line=dict(width=1, color='gray')),
            name=reg, legendgroup='region',
            legendgrouptitle=dict(text='Регион (форма)'),
            showlegend=True, hoverinfo='skip',
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f'Кластеры стран (цвет, k={k}) и регионы (форма)',
        xaxis_title=f'PC1 ({expl[0]:.1f}%)',
        yaxis_title=f'PC2 ({expl[1]:.1f}%)',
        height=650,
        hovermode=False,
        legend=dict(title='', bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#888', borderwidth=1),
    )
    return fig

def build_figures(df_raw: pd.DataFrame, k: int, model_name: str):
    mat, country_region, df = build_matrix(df_raw)
    _, scores, expl = fit_pca(mat)
    _, clusters, _  = fit_model(mat, k, model_name)

    cmap = plt.get_cmap(CMAP_NAME, k)
    palette = {f'К{i}': to_hex(cmap(i - 1)) for i in range(1, k + 1)}

    df_plot = pd.DataFrame(
        {
            'PC1':     scores[:, 0],
            'PC2':     scores[:, 1],
            'кластер': [f'К{c}' for c in clusters],
            'регион':  country_region.values,
        },
        index=mat.index,
    ).reset_index(names='страна')

    df = df.copy()
    df['cluster'] = df['страна'].map(pd.Series(clusters, index=mat.index))
    df.to_csv(CLUSTER_FILE, index=False, sep=';', encoding='cp1251')

    return build_scatter_fig(df_plot, expl, k, palette)


# ============================================================
# === 4. APP ===
# ============================================================
DF_RAW = load_raw(FILE_IN)

app = Dash(__name__)
server = app.server

app.layout = html.Div(style=S['container'],
    children=[
        html.H1('Кластеризация профилей гаплогрупп мтДНК', style=S['title']),
        html.Div(style=S['divider']),
        html.Div(style=S['row'],
            children=[
                html.Div(style=S['columnLeft'], children=[
                    html.H2('Кластеры стран', style=S['sectionTitle']),
                    html.Label('Подбор числа кластеров', style=S['label']),
                    dcc.Slider(id='n-clusters', min=1, max=10, step=1,
                               value=5,
                               marks={m: str(m) for m in range(1, 11)},
                               tooltip={'placement': 'bottom'}),
                ]),
                html.Div(style=S['columnRight'], children=[
                    html.H2('Выбор модели кластеризации', style=S['sectionTitle']),
                    dcc.RadioItems(
                        id='model-radio',
                        options=[
                            {'label': 'K-средние', 'value': 'kmeans'},
                            {'label': 'Спектр',    'value': 'spectral'},
                            {'label': 'Гаусс',     'value': 'gauss'},
                        ],
                        value='kmeans',
                        inline=True,
                        labelStyle=S['radioLabel']),
                ]),
            ]),
        dcc.Graph(id='scatter-plot', style=S['graph'],
		                config={'responsive': True,
                        'displaylogo': False,
                        'toImageButtonOptions': {
                            'format': 'jpeg',
                            'filename': OUTPUT_IMG,
                            'scale': 2}},
							),
    ])

# ============================================================
# === 5. CONTROLLER ===
# ============================================================
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('n-clusters', 'value'),
    Input('model-radio', 'value'),
)
def update(k: int, model_name: str):
    return build_figures(DF_RAW, int(k), model_name)


if __name__ == '__main__':
    app.run(debug=True, port=8051)