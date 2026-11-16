"""
Statistical processing and haplogroup frequency analysis with interactive threshold selection.

"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# === 1. GENERAL VARIABLES ===
# ============================================================
INPUT_FILE = 'core_mt.csv'
OUTPUT_FILE = 'mt_major.csv'
OUTPUT_IMG = 'img_1'

WILSON_MIN_N     = 50     # Wilson for n >= 50, Clopper-Pearson otherwise
MAX_ABS_WIDTH_PP = 30     # absolute CI width cutoff, pp
MAX_REL_WIDTH_PC = 130    # relative CI width cutoff, %

THRESHOLD_GRID    = (0.03, 0.05, 0.10)   # dropdown options
DEFAULT_THRESHOLD = 0.05                 # initial value
TOP_N_SUBCLADES   = 2                    # top-N l2 subclades per major clade

# --- typography / palette ---
_FONT   = 'Georgia, "Times New Roman", serif'
_INK    = '#1a1a1a'
_MUTED  = '#555555'
_BORDER = '1px solid #333333'

S = {
    'page': {
        'fontFamily': _FONT, 'color': _INK, 'backgroundColor': '#ffffff',
        'maxWidth': '1100px', 'margin': '0 auto', 'padding': '24px',
        'fontSize': '13px', 'lineHeight': '1.5',
    },
    'h1': {
        'textAlign': 'center', 'fontWeight': 'normal', 'fontSize': '22px',
        'letterSpacing': '0.5px', 'margin': '0 0 12px 0',
        'paddingBottom': '12px', 'borderBottom': '3px solid #333333',
    },
    'caption': {
        'fontSize': '11px', 'textTransform': 'uppercase',
        'letterSpacing': '0.8px', 'color': _MUTED, 'marginBottom': '6px',
    },
    'dropdown': {'width': '220px', 'fontSize': '13px', 'searchable': 'False', 'clearable': 'False'},
    'report': {
        'marginTop': '20px', 'fontFamily': 'monospace', 'fontSize': '12px',
        'lineHeight': '1.6', 'whiteSpace': 'pre-wrap',
        'background': '#f7f7f7', 'padding': '12px',
        'border': '1px solid #cccccc', 'borderRadius': '0',
    },
    'graph': {'marginTop': '20px', 'height': '75vh','display': 'flex'},
    'plotly': {
        'font':       {'family': _FONT, 'size': 11, 'color': _INK},
        'title_font': {'size': 14, 'color': _INK},
    },
}


# ============================================================
# === 2. MODEL ===
# ============================================================

# --- 2a. threshold-independent ---
def load_raw(path: str = INPUT_FILE) -> TextFileReader | DataFrame:
    return pd.read_csv(path, sep=';', encoding='cp1251')


def reconstruct_sample_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_est = df['китов'] / df['доля'].replace(0, np.nan)
    df['выборка'] = (n_est.groupby(df['страна'])
                        .transform(lambda s: round(s.median())))
    return df


def add_confidence_intervals(
    df: pd.DataFrame, wilson_min_n: int = WILSON_MIN_N
) -> pd.DataFrame:
    df = df.copy()
    ci_lo = np.full(len(df), np.nan)
    ci_hi = np.full(len(df), np.nan)

    n = pd.to_numeric(df['выборка'], errors='coerce').to_numpy()
    k = pd.to_numeric(df['китов'], errors='coerce').to_numpy()

    valid = np.isfinite(n) & (n > 0)
    k = np.where(valid, np.minimum(k, n), 0)

    for method, mask in (
        ('wilson', valid & (n >= wilson_min_n)),
        ('beta',   valid & (n <  wilson_min_n)),
    ):
        if mask.any():
            lo, hi = proportion_confint(
                k[mask].astype(int), n[mask].astype(int),
                alpha=0.05, method=method,
            )
            ci_lo[mask], ci_hi[mask] = lo, hi

    df['ci_lo'] = ci_lo
    df['ci_hi'] = ci_hi
    df['abs_width_pp'] = (df['ci_hi'] - df['ci_lo']) * 100
    df['rel_width_pc'] = np.where(
        (df['abs_width_pp'] >= 5)
        & (df['abs_width_pp'] <= 30)
        & (df['доля'] > 0),
        df['abs_width_pp'] / df['доля'],
        np.nan,
    )
    return df


def filter_by_ci(
    df: pd.DataFrame,
    max_abs: float = MAX_ABS_WIDTH_PP,
    max_rel: float = MAX_REL_WIDTH_PC,
) -> pd.DataFrame:
    keep = (
        (df['abs_width_pp'] < max_abs)
        & (df['rel_width_pc'].isna() | (df['rel_width_pc'] <= max_rel))
    )
    return df[keep].copy()


# --- 2b. threshold-dependent ---
def aggregate_major_clades(
    df: pd.DataFrame,
    min_share: float,
    top_n: int = TOP_N_SUBCLADES,
) -> pd.DataFrame:
    """Per country: keep l1 clades with share >= *min_share*; within each,
    keep top-*n* l2 subclades; lump the rest under 'Прочие' (Other)."""
    records: list[dict] = []

    for country, g in df.groupby('страна', sort=False):
        n = g['выборка'].iloc[0]
        if pd.isna(n) or n <= 0:
            continue

        g = g.assign(ветка=g['ветка'].astype(str).str.strip())
        l1 = g[g['уровень'] == 'l1']
        l2 = g[g['уровень'] == 'l2']
        used = 0

        for _, clade in l1[l1['доля'] >= min_share].iterrows():
            name, k_clade, region = clade['ветка'], int(clade['китов']), clade['регион']
            used += k_clade

            top = l2[l2['ветка'].str.startswith(name)].nlargest(top_n, 'китов')
            k_top = int(top['китов'].sum())

            for _, sub in top.iterrows():
                records.append(dict(
                    страна=country, регион=region,
                    клад=name, субклада=sub['ветка'],
                    китов=int(sub['китов']), выборка=int(n),
                ))
            if k_clade > k_top:
                records.append(dict(
                    страна=country, регион=region,
                    клад=name, субклада='Прочие',
                    китов=k_clade - k_top, выборка=int(n),
                ))

        if n - used > 0:
            records.append(dict(
                страна=country,
                регион=', '.join(g['регион'].dropna().unique()),
                клад='Прочие', субклада='Прочие',
                китов=int(n - used), выборка=int(n),
            ))

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result['доля'] = (result['китов'] / result['выборка'] * 100).round().astype(int)
    return result


# --- 2c. startup precomputation (once at import) ---
df_raw  = load_raw()
# noinspection PyTypeChecker
df_ci   = add_confidence_intervals(reconstruct_sample_size(df_raw))
df_true = filter_by_ci(df_ci)


# ============================================================
# === 3. VIEW ===
# ============================================================
def build_report(df_raw_: pd.DataFrame, df_true_: pd.DataFrame,
                 threshold: float) -> str:
    l1    = df_true_[df_true_['уровень'] == 'l1']
    major = l1[l1['доля'] >= threshold]
    return '\n'.join([
        f'Исходно стран/регионов:              {df_raw_["страна"].nunique()}',
        f'Исходно гаплогрупп:                  {df_raw_["ветка"].nunique()}',
        f'Стран/регионов после применения CI:  {df_true_["страна"].nunique()}',
        f'Гаплогрупп после применения CI:      {df_true_["ветка"].nunique()}',
        f'Порог мажорных гаплогрупп:           {threshold:.2f}',
        f'Стран/регионов выше порога:          {major["страна"].nunique()}',
        f'Мажорных гаплогрупп выше порога:     {major["ветка"].nunique()}',
    ])


def build_sunburst(df_major: pd.DataFrame, threshold: float) -> go.Figure:
    """World-population sunburst: clade → subclade, values = total carriers."""
    if df_major.empty:
        return go.Figure()

    grouped = (
        df_major.groupby(['клад', 'субклада'], as_index=False)['китов'].sum()
    )

    fig = px.sunburst(
        grouped,
        path=['клад', 'субклада'],
        values='китов',
        color='клад',
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(
        textinfo='label+percent root',
        insidetextorientation='radial',
    )
    fig.update_layout(
        **S['plotly'],
        title=(
            f'Гаплогруппы мтДНК в мировой популяции: клад → субклада '
            f'(порог {threshold:.2f})'
        ),
        margin=dict(t=70, l=0, r=0, b=0),
        hovermode=False,
    )
    return fig


# ============================================================
# === 4. APP ===
# ============================================================
app = Dash(__name__)
server = app.server

# noinspection PyTypeChecker
app.layout = html.Div(style=S['page'], children=[

    # ---- Header ----
    html.H1('Анализ частот гаплогрупп мтДНК', style=S['h1']),

    # ---- threshold selection ----
    html.Div(style={
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'marginTop': '4px',
    }, children=[
        html.Div('Выберите порог для мажорной гаплогруппы',
                 style={**S['caption'], 'marginBottom': 0}),
        dcc.Dropdown(
            id='threshold-dropdown',
            options=[{'label': f'{t:.2f}', 'value': t}
                     for t in THRESHOLD_GRID],
            value=DEFAULT_THRESHOLD,
            clearable=False,
            searchable=False,
            style={**S['dropdown'], 'width': '120px'},
        ),
    ]),

    # ---- Report (left) | Sunburst (right) ----
    html.Div(style={
        'display': 'flex',
        'gap': '24px',
        'marginTop': '16px',
        'alignItems': 'flex-start',
    }, children=[
        # report
        html.Div(style={'flex': '1 1 0', 'minWidth': 0}, children=[
            html.Pre(id='report-output', style=S['report']),
        ]),
        # sunburst
        html.Div(style={'flex': '2 1 0', 'minWidth': 0}, children=[
            dcc.Graph(
                id='sunburst-graph',
                style=S['graph'],
                config={'responsive': True,
                        'displaylogo': False,
                        'toImageButtonOptions': {
                            'format': 'jpeg',
                            'filename': OUTPUT_IMG,
                            'scale': 2}}
            ),
        ]),
    ]),
])


# ============================================================
# === 5. CONTROLLER ===
# ============================================================
@app.callback(
    Output('report-output', 'children'),
    Output('sunburst-graph', 'figure'),
    Input('threshold-dropdown', 'value'),
)
def update_analysis(threshold: float):
    df_major = aggregate_major_clades(df_true, min_share=threshold)
    df_major.to_csv(OUTPUT_FILE, sep=';', index=False, encoding='cp1251')
    # noinspection PyTypeChecker
    return (
        build_report(df_raw, df_true, threshold),
        build_sunburst(df_major, threshold),
    )


if __name__ == '__main__':
    app.run(debug=True)