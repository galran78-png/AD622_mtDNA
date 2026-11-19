import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from scipy.stats import binomtest

from dash import Dash, html, dcc, Input, Output


# ============================================================
# === 1. GENERAL VARIABLES ===
# ============================================================
CLUSTER_FILE = 'mt_clusters.csv'
OUTPUT_IMG = 'img_1'
OUTPUT_BAR_IMG = 'img_2'

# --- statistical constants ---
N_DISEASE_GROUPS = 9
P0               = 1 / N_DISEASE_GROUPS
ALPHA            = 0.05
ALPHA_BONF       = ALPHA / N_DISEASE_GROUPS

# --- typography / palette (single source) ---
_FONT   = 'Georgia, "Times New Roman", serif'
_INK    = '#1a1a1a'
_MUTED  = '#555555'
_BORDER = '1px solid #333333'

# --- group-dropdown options ---
GROUP_OPTIONS = [
	{'label': 'Все', 'value': 'all'},
	{'label': 'CD', 'value': 'CD'},
	{'label': 'MD', 'value': 'MD'},
	{'label': 'NDD', 'value': 'NDD'},
	{'label': 'ID', 'value': 'ID'},
	{'label': 'BRM', 'value': 'BRM'},
    {'label': 'OD', 'value': 'OD'},
	{'label': 'PD', 'value': 'PD'},
	{'label': 'AD', 'value': 'AD'},
    {'label': 'SD', 'value': 'SD'}
]

# --- page and graph style ---
S = {
    'plotly': {
        'template': 'simple_white',
        'font':       {'family': _FONT, 'size': 11, 'color': _INK},
        'title_font': {'size': 14, 'color': _INK},
        'margin':     {'l': 70, 'r': 30, 't': 110, 'b': 60},
    },
    'page': {
        'fontFamily': _FONT, 'color': _INK, 'backgroundColor': '#ffffff',
        'maxWidth': 'auto', 'margin': '0 auto', 'padding': '24px',
        'fontSize': '13px', 'lineHeight': '1.5',
    },
    'h1': {
        'textAlign': 'center', 'fontWeight': 'normal', 'fontSize': '22px',
        'letterSpacing': '0.5px', 'margin': '0 0 12px 0',
        'paddingBottom': '12px', 'borderBottom': '3px solid #333333',
    },
    'h2': {
        'fontSize': '15px', 'fontWeight': 'bold', 'letterSpacing': '0.3px',
        'margin': '0 0 8px 0',
    },
    'caption': {
        'fontSize': '11px', 'textTransform': 'uppercase',
        'letterSpacing': '0.8px', 'color': _MUTED, 'marginBottom': '6px',
    },
	'columns': {
        'display': 'flex', 'gap': '16px',
        'marginTop': '20px', 'alignItems': 'stretch',
    },
    'column': {
        'flex': '1', 'minWidth': '0', 'boxSizing': 'border-box',
        'border': _BORDER, 'padding': '16px',
    },
    'bar': {
        'display': 'flex', 'alignItems': 'stretch', 'gap': '16px',
        'padding': '12px 0', 'borderTop': _BORDER, 'borderBottom': _BORDER,
    },
	'abbr_in_bar': {
        'fontSize': '12px', 'color': _MUTED, 'lineHeight': '1.5',
        'flex': '1', 'minWidth': '220px','textAlign': 'justify',
    },
    'vline': {
        'width': '1px', 'backgroundColor': '#333333',
        'alignSelf': 'stretch', 'minHeight': '32px',
    },
    'btn': {
        'fontFamily': 'inherit', 'fontSize': '13px', 'padding': '6px 14px',
        'cursor': 'pointer', 'backgroundColor': '#ffffff', 'color': _INK,
        'border': _BORDER, 'borderRadius': '0',
    },
    'radio_label': {'display': 'block', 'fontSize': '13px', 'marginBottom': '4px'},
    'status':      {'fontSize': '11px', 'color': _MUTED, 'marginTop': '6px'},
    'dropdown':    {'minWidth': '220px', 'fontSize': '13px'},
    'graph':       {'marginTop': '8px'},
}


# ============================================================
# === 2. MODEL ===
# ============================================================
def load_clusters() -> None | TextFileReader | DataFrame:
    if not os.path.exists(CLUSTER_FILE):
        return None
    return pd.read_csv(CLUSTER_FILE, sep=';', encoding='cp1251')


def filter_no_data(df: pd.DataFrame, risk: str) -> pd.DataFrame:
    s = df[risk].astype(str).str.strip().str.lower()
    return df[df[risk].notna() & s.ne('no data')]


def filter_europe(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df['регион'].fillna('').astype(str)
        .str.contains('Европа', case=False, regex=False)
    )
    return df[~mask]


def filter_by_group(df: pd.DataFrame, risk: str,
                    group_value: Optional[str]) -> pd.DataFrame:
    if not group_value or group_value == 'all':
        return df
    mask = (
        df[risk].fillna('').astype(str)
        .str.contains(group_value, case=False, regex=False)
    )
    return df[mask]


def apply_filters(df: pd.DataFrame, bias: str, risk: str,
                  group_value: Optional[str] = None) -> pd.DataFrame:
    df = filter_no_data(df, risk)
    if bias == 'no_europe':
        df = filter_europe(df)
    if group_value is not None:
        df = filter_by_group(df, risk, group_value)
    return df


# ============================================================
# === 3. VIEW ===
# ============================================================
def build_heatmap_fig(df: pd.DataFrame, k: int, risk: str) -> go.Figure:
    counts = (
        df.groupby(['cluster', risk])['страна']
        .nunique()
        .unstack(level='cluster', fill_value=0)
    )
    counts = counts.reindex(columns=range(1, k + 1), fill_value=0)
    counts.columns = [f'К{c}' for c in counts.columns]
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(go.Heatmap(
        z=counts.values,
        x=counts.columns.tolist(),
        y=counts.index.tolist(),
        colorscale='YlOrRd',
        text=counts.values,
        texttemplate='%{text}',
        colorbar=dict(title='Стран')
    ))
    fig.update_layout(
        **S['plotly'],
		title=f'Эффект: {risk}',
        xaxis_title='Кластер',
        yaxis_title='Классы заболеваний',
        height=max(400, 40 * len(counts) + 120),
    )
    return fig


def build_bar_fig(df: pd.DataFrame, k: int, risk: str,
                  group_value: Optional[str], n_test: int,
                  alpha: float = ALPHA_BONF) -> go.Figure:
    counts = df.groupby('cluster')['страна'].nunique()
    counts = counts.reindex(range(1, k + 1), fill_value=0)
    x_arr = [f'К{c}' for c in counts.index]

    # pvalue by clusters
    if 'pvalue' in df.columns:
        pv_raw = df.groupby('cluster')['pvalue'].first()
        # noinspection PyTypeChecker
        pv_dict = {int(kk): float(v) for kk, v in pv_raw.items()}
    else:
        pv_dict = {}
    pvals = pd.Series({c: pv_dict.get(c, 1.0) for c in range(1, k + 1)})

    sig_mask = (pvals <= alpha).values

    texts = []
    for i, c in enumerate(counts.index):
        v = int(counts.iloc[i])
        p = float(pvals.loc[c])
        star = ' *' if sig_mask[i] else ''
        texts.append(f'{v}<br><sub>p={p:.4f}{star}</sub>')

    y_arr = [int(v) for v in counts.values]
    idx_sig = [i for i in range(len(x_arr)) if sig_mask[i]]
    idx_ns  = [i for i in range(len(x_arr)) if not sig_mask[i]]

    fig = go.Figure()
    if idx_ns:
        # noinspection PyTypeChecker
        fig.add_trace(go.Bar(
            x=[x_arr[i] for i in idx_ns],
            y=[y_arr[i] for i in idx_ns],
            text=[texts[i] for i in idx_ns],
            textposition='outside',
            marker_color='#A9A9A9',
            name=f'Не значимо (p > {alpha:.4f})',
            hoverinfo='x+y+text',
        ))
    if idx_sig:
        # noinspection PyTypeChecker
        fig.add_trace(go.Bar(
            x=[x_arr[i] for i in idx_sig],
            y=[y_arr[i] for i in idx_sig],
            text=[texts[i] for i in idx_sig],
            textposition='outside',
            marker_color='#8B0000',
            name=f'Значимо (p ≤ {alpha:.4f})',
            hoverinfo='x+y+text',
        ))

    fig.update_layout(
        **S['plotly'],
        title=dict(
            text=(f'Эффект: {risk}, класс: {group_value} '
                  f'(n = {n_test}, p₀ = 1/{N_DISEASE_GROUPS} ≈ {P0:.4f}, '
                  f'α_Bonf = {alpha:.4f})'),
            x=0.5, xanchor='center',
            y=0.98, yanchor='top',
        ),
        xaxis_title='Кластер',
        yaxis_title='Стран',
        height=480,
        barmode='overlay',
        bargap=0.25,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.0,
            xanchor='center', x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#333333', borderwidth=0,
        ),
        xaxis=dict(
            categoryorder='array',
            categoryarray=x_arr,
            type='category',
            tickmode='array',
            tickvals=x_arr,
            ticktext=x_arr,
        ),
    )
    return fig


def empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color='#888'),
    )
    fig.update_layout(
        **S['plotly'], height=400,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig



abbr = dcc.Markdown('''
**CD** — сердечно-сосудистые, **MD** — метаболические,

**NDD** — нейродегенеративные, **ID** — инфекционные,

**OD** — онкологические, **PD** — психиатрические,

**AD** — аутоиммунные, **SD** — органов чувств,

**BRM** — митохондриально-опосредованные (рассеянный склероз)
''')

# ============================================================
# === 4. APP ===
# ============================================================
app = Dash(__name__)
server = app.server

# noinspection PyTypeChecker
app.layout = html.Div(style=S['page'], children=[


    # ---------- Line 1 ----------
    html.H1('Профиль заболеваний по кластерам стран', style=S['h1']),

    # ---------- Line 2: control board ----------
    html.Div(style=S['bar'], children=[
        # button + status
        html.Div(children=[
            html.Button('Обновить', id='refresh-btn', n_clicks=0, style=S['btn']),
            html.Div(id='status-msg', style=S['status']),
        ]),
        html.Div(style=S['vline']),

        # bias
        html.Div(children=[
            html.Div('Поправка', style=S['caption']),
            dcc.RadioItems(
                id='bias-radio',
                options=[
                    {'label': 'С Европой',  'value': 'europe'},
                    {'label': 'Без Европы', 'value': 'no_europe'},
                ],
                value='europe',
                labelStyle=S['radio_label'],
            ),
        ]),
        html.Div(style=S['vline']),

        # effect
        html.Div(children=[
            html.Div('Эффект', style=S['caption']),
            dcc.RadioItems(
                id='risk-radio',
                options=[
                    {'label': 'Риск',   'value': 'риск'},
                    {'label': 'Защита', 'value': 'защита'},
                ],
                value='риск',
                labelStyle=S['radio_label'],
            ),
        ]),
        html.Div(style=S['vline']),

        # disease class
        html.Div(children=[
            html.Div('Класс заболеваний', style=S['caption']),
            dcc.Dropdown(
                id='group-dropdown',
                options=GROUP_OPTIONS,
				searchable=False,
				clearable=False,
                style=S['dropdown'],
				value='all',
            ),
        ]),
        html.Div(style=S['vline']),

        # abbreviations
        html.Div(children=[
            html.Div('Сокращения для классов заболеваний', style=S['caption']),
            html.Div(id='abbr-text', style=S['abbr_in_bar'],
                     children=abbr),
        ]),
    ]),

    # ---------- Lines 3–6: two columns ----------
    html.Div(style=S['columns'], children=[
        html.Div(style=S['column'], children=[
            html.H2('Общий профиль по числу стран на кластер', style=S['h2']),
            dcc.Graph(id='heatmap-plot', style=S['graph'],
                      config={
                              'displaylogo': False,
                              'toImageButtonOptions': {
                                  'format': 'jpeg',
                                  'filename': OUTPUT_IMG,
                                  'scale': 2}},
                      ),
        ]),
        html.Div(style=S['column'], children=[
            html.H2('Конкретный профиль по числу стран на кластер', style=S['h2']),
            dcc.Graph(id='bar-plot', style=S['graph'],
                      config={
                              'displaylogo': False,
                              'toImageButtonOptions': {
                                  'format': 'jpeg',
                                  'filename': OUTPUT_BAR_IMG,
                                  'scale': 2}},
                      ),
        ]),
    ]),
])

# ============================================================
# === 5. CONTROLLER ===
# ============================================================
# noinspection PyUnresolvedReferences
@app.callback(
    Output('heatmap-plot', 'figure'),
    Output('bar-plot', 'figure'),
    Output('status-msg', 'children'),
    Input('refresh-btn', 'n_clicks'),
    Input('bias-radio', 'value'),
    Input('risk-radio', 'value'),
    Input('group-dropdown', 'value'),
)
def update(_n_clicks, bias: str, risk: str, group_value: Optional[str]):
    df = load_clusters()
    if df is None:
        msg = f'Файл {CLUSTER_FILE} не найден'
        return empty_fig(msg), empty_fig(msg), msg

    df = df.dropna(subset=['cluster'])
    if df.empty:
        msg = 'Файл пуст'
        return empty_fig(msg), empty_fig(msg), msg

    df['cluster'] = df['cluster'].astype(int)
    k = int(df['cluster'].max())

    df_heatmap = apply_filters(df, bias, risk)
    heatmap_fig = (
        empty_fig('После фильтра данных нет') if df_heatmap.empty
        else build_heatmap_fig(df_heatmap, k, risk)
    )

    df_bar = apply_filters(df, bias, risk, group_value)
    if df_bar.empty:
        msg = 'После фильтра данных нет'
        return heatmap_fig, empty_fig(msg), f'k = {k} кластеров · стран: 0'

    n_test = int(df_bar['страна'].nunique())
    counts_dict = (
        df_bar.groupby('cluster')['страна'].nunique().astype(int).to_dict()
    )
    pvalue_map = {
        c: (float(binomtest(counts_dict[c], n=n_test, p=P0,
                            alternative='greater').pvalue)
            if counts_dict.get(c, 0) > 0 else 1.0)
        for c in range(1, k + 1)
    }
    df_bar = df_bar.copy()
    df_bar['pvalue'] = df_bar['cluster'].map(pvalue_map)

    bar_fig = build_bar_fig(df_bar, k, risk, group_value,
                            n_test=n_test, alpha=ALPHA_BONF)

    return heatmap_fig, bar_fig, f'k = {k} кластеров · стран: {n_test}'


if __name__ == '__main__':
    app.run(debug=True, port=8052)