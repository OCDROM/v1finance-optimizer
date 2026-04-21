import base64
import io
import json
import os
import time
import requests
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update, ALL
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Colour palette (matches v1finance.fr) ─────────────────────────────────────
NAVY    = "#0E3254"
CYAN    = "#00D4E5"
BG      = "#F7F8FA"
WHITE   = "#FFFFFF"
LGRAY   = "#E4E8EE"
RED     = "#e63946"
EMERALD = "#059669"
AMBER   = "#D97706"
FONT    = "'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# ── Card / badge design tokens ─────────────────────────────────────────────────
CARD_STYLE = {
    "background": WHITE,
    "borderRadius": "16px",
    "boxShadow": "0 1px 8px rgba(14,50,84,0.08)",
    "padding": "1.5em 1.75em",
}

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="V1 Finance – Portfolio Optimizer",
    suppress_callback_exceptions=True,
)
server = app.server  # exposed for gunicorn

@server.route("/ping")
def ping():
    return "ok", 200

# ── Google Finance → Yahoo Finance ticker mapping ─────────────────────────────
_GF_TO_YF = {
    "PARRO":      "PARRO.PA",
    "BIT:ARIS":   "ARIS.MI",
    "FRA:LC4":    "LOUP.PA",    # confirmed: LOUP.PA on Yahoo Finance
    "EPA:VU":     "VU.PA",
    "FRA:IR5B":   "IR5B.IR",
    "EPA:COFA":   "COFA.PA",
    "AMS:HEIJM":  "HEIJM.AS",
    "BIT:MAIRE":  "MAIRE.MI",
    "BIT:SOL":    "SOL.MI",
    "EPA:ASY":    "ASY.PA",
    "BIT:SPM":    "SPM.MI",
    "AMS:FUR":    "FUR.AS",
    "EPA:MRN":    "MRN.PA",
    "AMS:BAMNB":  "BAMNB.AS",
    "EPA:EXENS":  "EXENS.PA",
    "EPA:FII":    "FII.PA",
    "EPA:RBT":    "RBT.PA",
    "EPA:VIRP":   "VIRP.PA",
    "EPA:AUB":    "AUB.PA",
    "EBR:CAMB":   "CAMB.BR",
    "EPA:VLTSA":  "VLTSA.PA",   # sold — filtered out when cumulative units = 0
}

_TRADES_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vT-6dbEcY7oBpu4pMlHQp7iWTKw_VTrf4M3uBo2nTUBo-FaQlcLO5LoAq3V-IuTGJXBWh0NqZuJGa0b"
    "/pub?gid=1007960933&single=true&output=csv"
)

_live_cache: dict = {"payload": None, "ts": 0.0}
_LIVE_TTL = 300  # seconds (5 minutes)


def _holdings_from_trades(csv_text: str) -> dict:
    """Parse Trades CSV → {yf_ticker: cumulative_units} for open positions only."""
    latest: dict = {}
    for line in csv_text.strip().splitlines()[1:]:   # skip header row
        cols = line.split(",")
        if len(cols) < 11:
            continue
        direction = cols[1].strip()
        if direction not in ("BUY", "SELL"):
            continue
        gf_ticker = cols[2].strip()
        try:
            cum_units = float(cols[10].replace(",", "").strip())
        except ValueError:
            continue
        latest[gf_ticker] = cum_units   # last row per ticker = current position

    holdings = {}
    for gf_ticker, units in latest.items():
        if units <= 0:
            continue
        yf_ticker = _GF_TO_YF.get(gf_ticker)
        if not yf_ticker:
            continue
        holdings[yf_ticker] = units
    return holdings


def _fetch_live_value() -> tuple:
    """Return (total_eur, errors, holdings_count).
    Uses yf.download for a single batch request (much faster than per-ticker calls).
    """
    resp = requests.get(_TRADES_CSV_URL, timeout=15)
    resp.raise_for_status()
    holdings = _holdings_from_trades(resp.text)
    if not holdings:
        return 0.0, [], 0

    tickers = list(holdings.keys())
    prices: dict = {}

    # ── Batch download: one HTTP round-trip for all tickers ───────────────────
    try:
        raw = yf.download(
            tickers if len(tickers) > 1 else tickers[0],
            period="2d", progress=False, auto_adjust=True,
        )
        if not raw.empty:
            close = raw["Close"]
            if hasattr(close, "columns"):
                # Multi-ticker: Close is a DataFrame keyed by ticker symbol
                for t in tickers:
                    if t in close.columns:
                        col = close[t].dropna()
                        if not col.empty:
                            prices[t] = float(col.iloc[-1])
            else:
                # Single-ticker: Close is a Series
                if len(tickers) == 1 and not close.dropna().empty:
                    prices[tickers[0]] = float(close.dropna().iloc[-1])
    except Exception:
        pass

    # ── Per-ticker fallback for anything the batch missed ─────────────────────
    missed = [t for t in tickers if t not in prices]
    if missed:
        def _single(t):
            try:
                p = getattr(yf.Ticker(t).fast_info, "last_price", None)
                return t, float(p) if p else None
            except Exception:
                return t, None
        with ThreadPoolExecutor(max_workers=4) as ex:
            for t, p in ex.map(_single, missed):
                if p:
                    prices[t] = p

    errors = [t for t in tickers if t not in prices]
    total = sum(prices.get(t, 0) * holdings[t] for t in tickers)
    return round(total, 2), errors, len(holdings)


@server.route("/live-value")
def live_value():
    from flask import jsonify
    now = time.time()
    if _live_cache["payload"] and (now - _live_cache["ts"]) < _LIVE_TTL:
        resp = jsonify(_live_cache["payload"])
        resp.headers["Access-Control-Allow-Origin"] = "https://v1finance.fr"
        return resp
    try:
        value, errors, n = _fetch_live_value()
        payload = {"value": value, "errors": errors, "holdings_count": n, "cached_at": int(now)}
        _live_cache["payload"] = payload
        _live_cache["ts"] = now
    except Exception as e:
        payload = {"value": None, "error": str(e)}
    resp = jsonify(payload)
    resp.headers["Access-Control-Allow-Origin"] = "https://v1finance.fr"
    return resp

# ── Custom HTML shell: Inter font + CSS design system ─────────────────────────
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        body { margin: 0; -webkit-font-smoothing: antialiased; overscroll-behavior-y: none; }

        #refresh-btn:active { transform: scale(0.97); opacity: 0.85; }
        #refresh-btn.btn-loading { background: #0a2540 !important; opacity: 0.72;
            cursor: not-allowed; animation: btnPulse 1.2s ease-in-out infinite; }
        @keyframes btnPulse {
            0%,100% { box-shadow: 0 0 0 0 rgba(0,212,229,0.0); }
            50%      { box-shadow: 0 0 0 8px rgba(0,212,229,0.22); }
        }
        /* ── Scroll-reveal animations ──────────────────────────────── */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .reveal-card    { animation: fadeUp 0.45s ease both; }
        .reveal-card-d1 { animation: fadeUp 0.45s 0.12s ease both; }
        .reveal-card-d2 { animation: fadeUp 0.45s 0.24s ease both; }
        .reveal-card-d3 { animation: fadeUp 0.45s 0.36s ease both; }

        /* ── Section hint / interpretation line ────────────────────── */
        .section-hint {
            font-size: 0.85em;
            color: #9CA3AF;
            font-style: italic;
            margin: 0.25em 0 1em 0;
            line-height: 1.5;
        }

        /* ── Custom scrollbar ───────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #C4CBD6; border-radius: 999px; }
        ::-webkit-scrollbar-thumb:hover { background: #8A9BA8; }

        /* ── Hero stat block ────────────────────────────────────────── */
        .stat-hero {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.02em;
        }
        .stat-label {
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9CA3AF;
            margin-bottom: 0.3em;
        }
        .stat-sub {
            font-size: 0.80em;
            color: #9CA3AF;
            margin-top: 0.3em;
        }

        /* ── Stock card grid (desktop) ─────────────────────────────────── */
        .stock-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(268px, 1fr));
            gap: 1em;
        }
        .stock-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.2em 1.5em;
            box-shadow: 0 1px 8px rgba(14,50,84,0.08);
            display: flex;
            flex-direction: column;
            gap: 0.45em;
            transition: box-shadow 0.22s ease, transform 0.22s ease;
            cursor: default;
        }
        .stock-card:hover {
            box-shadow: 0 8px 28px rgba(14,50,84,0.14);
            transform: translateY(-3px);
        }
        .stock-card-ticker { font-weight: 800; font-size: 1.06em; color: #0E3254; letter-spacing: -0.01em; }
        .stock-card-name   { font-size: 0.81em; color: #6B7280; margin-bottom: 0.2em; }
        .stock-card-row    { display: flex; justify-content: space-between; font-size: 0.87em; color: #374151; gap: 0.5em; }
        .stock-card-badge  { display: inline-block; padding: 0.2em 0.9em;
                             border-radius: 999px; font-size: 0.82em; font-weight: 700;
                             align-self: flex-start; margin-bottom: 0.4em; }

        /* ── Factor progress bars ────────────────────────────────────────── */
        .factor-bar-track {
            background: #F3F4F6;
            border-radius: 999px;
            height: 5px;
            flex: 1;
            overflow: hidden;
        }
        .factor-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.5s ease;
        }

        /* ── Card hover lift ─────────────────────────────────────────────── */
        .app-card-hover {
            transition: box-shadow 0.22s ease, transform 0.22s ease;
        }
        .app-card-hover:hover {
            box-shadow: 0 8px 28px rgba(14,50,84,0.13) !important;
            transform: translateY(-2px);
        }

        /* ── Mobile helpers ─────────────────────────────────────────── */
        @media (max-width: 768px) {
            .hide-mobile  { display: none !important; }
            .show-mobile  { display: block !important; }

            /* mobile overrides */
            .stock-card-grid { grid-template-columns: 1fr; }
            .stock-card { padding: 1em 1.25em; }
        }
        @media (min-width: 769px) {
            .show-mobile { display: none !important; }
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>

</body>
</html>"""

# ── Shared styles ──────────────────────────────────────────────────────────────
INPUT_STYLE = {
    "border": f"1.5px solid {LGRAY}",
    "borderRadius": "999px",
    "padding": "0.5em 1.2em",
    "fontSize": "0.95em",
    "fontFamily": FONT,
    "outline": "none",
    "background": WHITE,
    "color": "#111",
    "height": "40px",
    "boxSizing": "border-box",
}

BTN_STYLE = {
    "background": CYAN,
    "color": WHITE,
    "border": "none",
    "borderRadius": "999px",
    "padding": "0 1.8em",
    "height": "40px",
    "fontWeight": "600",
    "fontSize": "0.95em",
    "fontFamily": FONT,
    "cursor": "pointer",
    "letterSpacing": "0.03em",
    "transition": "opacity 0.18s ease, transform 0.12s ease",
}

TABLE_HEADER = {
    "backgroundColor": "#F9FAFB",
    "color": NAVY,
    "fontWeight": "700",
    "fontFamily": FONT,
    "fontSize": "0.82em",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
    "borderBottom": f"2px solid {LGRAY}",
    "padding": "10px 14px",
}

TABLE_CELL = {
    "fontFamily": FONT,
    "fontSize": "0.93em",
    "color": "#1a1a2e",
    "backgroundColor": WHITE,
    "padding": "10px 14px",
    "borderBottom": "1px solid #F0F2F5",
}

# ── Shared responsive container style ──────────────────────────────────────────
CONTAINER = {
    "maxWidth": "1100px",
    "width": "100%",
    "margin": "0 auto",
    "padding": "0 1em",
    "boxSizing": "border-box",
}

# ── Fundamentals column definitions — one set per tab ─────────────────────────
_TC = [  # Ticker + Company prefix (always shown)
    {"name": ["", "Ticker"],  "id": "ticker"},
    {"name": ["", "Company"], "id": "company"},
]
FUND_COLS = {
    "valuations": _TC + [
        {"name": ["Valuation", "P/E"],       "id": "pe"},
        {"name": ["Valuation", "Fwd P/E"],   "id": "fwd_pe"},
        {"name": ["Valuation", "EV/EBITDA"], "id": "ev_ebitda"},
        {"name": ["Valuation", "P/S"],       "id": "ps"},
        {"name": ["Valuation", "P/B"],       "id": "pb"},
    ],
    "balance": _TC + [
        {"name": ["Position",  "Cash €M"],     "id": "cash_m"},
        {"name": ["Position",  "Net Debt €M"], "id": "net_debt_m"},
        {"name": ["Position",  "Equity €M"],   "id": "equity_m"},
        {"name": ["Health",    "Debt/Eq"],     "id": "debt_eq"},
        {"name": ["Health",    "Curr Ratio"],  "id": "curr_ratio"},
        {"name": ["Health",    "Int Cov"],     "id": "int_cov"},
        {"name": ["Trend 1Y",  "Net Debt"],    "id": "nd_1y_chg"},
        {"name": ["Trend 1Y",  "Equity"],      "id": "eq_1y_chg"},
        {"name": ["Trend 3Y",  "Net Debt"],    "id": "nd_3y_chg"},
        {"name": ["Trend 3Y",  "Equity"],      "id": "eq_3y_chg"},
    ],
    "income": _TC + [
        {"name": ["Margins",  "Gross %"],      "id": "gross_m"},
        {"name": ["Margins",  "Oper %"],       "id": "op_m"},
        {"name": ["Margins",  "Net %"],        "id": "net_m"},
        {"name": ["Returns",  "ROE %"],        "id": "roe"},
        {"name": ["Returns",  "ROA %"],        "id": "roa"},
        {"name": ["Growth",   "Rev YoY"],      "id": "rev_growth"},
        {"name": ["Growth",   "Rev 3Y CAGR"],  "id": "rev_3y"},
        {"name": ["Growth",   "Rev Max CAGR"], "id": "rev_max"},
        {"name": ["Growth",   "EPS YoY"],      "id": "eps_growth"},
        {"name": ["Growth",   "EPS 3Y CAGR"],  "id": "eps_3y"},
        {"name": ["Growth",   "EPS Max CAGR"], "id": "eps_max"},
    ],
    "cashflow": _TC + [
        {"name": ["Cash Flow", "Op CF €M"],     "id": "op_cf_m"},
        {"name": ["Change",    "Op CF YoY"],    "id": "op_cf_1y"},
        {"name": ["Change",    "Op CF 3Y CAGR"],"id": "op_cf_3y"},
        {"name": ["Cash Flow", "FCF €M"],       "id": "fcf_m"},
        {"name": ["Change",    "FCF YoY"],      "id": "fcf_1y"},
        {"name": ["Change",    "FCF 3Y CAGR"],  "id": "fcf_3y_cagr"},
        {"name": ["Cash Flow", "Capex €M"],     "id": "capex_m"},
        {"name": ["Cash Flow", "FCF Yield"],    "id": "fcf_yield"},
        {"name": ["Cash Flow", "FCF/Rev"],      "id": "fcf_rev"},
    ],
}

# ── Default portfolio ─────────────────────────────────────────────────────────
DEFAULT_HOLDINGS = [
    ("CAMB.BR",  25),  ("PARRO.PA", 250), ("AUB.PA",   145), ("SOL.MI",    75),
    ("VIRP.PA",  18),  ("EXENS.PA",  42), ("RBT.PA",     3), ("BAMNB.AS", 340),
    ("FII.PA",   60),  ("MRN.PA",    99), ("FUR.AS",   200), ("ASY.PA",    85),
    ("COFA.PA", 228),  ("HEIJM.AS",  68), ("MAIRE.MI", 320), ("IR5B.IR",  778),
    ("VU.PA",    12),  ("ARIS.MI", 1150), ("SPM.MI",  1030),
    ("LOUP.PA", 129),
]

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"background": BG, "minHeight": "100vh", "fontFamily": FONT, "WebkitFontSmoothing": "antialiased"},
    children=[

        # ── Header bar ────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": "linear-gradient(135deg, #0E3254 0%, #1a4a7a 100%)",
                "padding": "1.2em 1.5em",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "flexWrap": "wrap",
                "gap": "0.5em",
            },
            children=[
                html.Div([
                    html.Span("V1 ", style={"color": WHITE, "fontWeight": "800", "fontSize": "1.7em", "letterSpacing": "-0.01em"}),
                    html.Span("Portfolio Optimizer", style={"color": WHITE, "fontWeight": "300", "fontSize": "1.7em"}),
                ]),
            ],
        ),

        # ── Main content ───────────────────────────────────────────────────────
        html.Div(
            style={**CONTAINER, "padding": "2.5em 1em"},
            children=[

                # ── Subtitle ──────────────────────────────────────────────────
                html.P(
                    "Portfolio analysis for independent investors.",
                    style={
                        "color": NAVY,
                        "fontSize": "2em",
                        "fontWeight": "600",
                        "letterSpacing": "-0.02em",
                        "lineHeight": "1.35",
                        "textAlign": "center",
                        "maxWidth": "800px",
                        "margin": "0.5em auto 2em auto",
                    },
                ),

                # ── Build Your Portfolio card (manual add + CSV import) ────────
                html.Div(
                    style={**CARD_STYLE, "marginBottom": "2em"},
                    children=[

                        # ── Header row: title left, import link right ─────────
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "baseline", "marginBottom": "1em"},
                            children=[
                                html.H4("Build Your Portfolio",
                                        style={"color": NAVY, "margin": 0}),
                                html.Span([
                                    "Import from ",
                                    html.A("Yahoo Finance",
                                           href="https://finance.yahoo.com/portfolios",
                                           target="_blank",
                                           style={"color": CYAN, "fontWeight": "700",
                                                  "textDecoration": "none",
                                                  "borderBottom": f"1px solid {CYAN}"}),
                                ], style={"fontSize": "1em", "fontWeight": "700",
                                          "color": NAVY, "fontFamily": FONT}),
                            ],
                        ),

                        # ── Controls row: ticker / qty / Add (left)  +  Upload CSV (right) ──
                        html.Div(
                            style={"display": "flex", "gap": "1em",
                                   "alignItems": "flex-end", "flexWrap": "wrap"},
                            children=[
                                html.Div([
                                    html.Label("Ticker", style={"fontSize": "0.8em", "color": "#888",
                                                                "display": "block", "marginBottom": "4px"}),
                                    dcc.Dropdown(
                                        id="ticker-input",
                                        placeholder="Search...",
                                        searchable=True,
                                        clearable=True,
                                        options=[],
                                        style={
                                            **INPUT_STYLE,
                                            "width": "210px",
                                            "height": "auto",
                                            "padding": "0",
                                            "border": "none",
                                        },
                                    ),
                                ], style={"position": "relative"}),
                                html.Div([
                                    html.Label("Quantity", style={"fontSize": "0.8em", "color": "#888",
                                                                   "display": "block", "marginBottom": "4px"}),
                                    dcc.Input(
                                        id="qty-input",
                                        type="number",
                                        placeholder="e.g. 10",
                                        min=0,
                                        style={**INPUT_STYLE, "width": "110px"},
                                    ),
                                ]),
                                html.Button(
                                    "Add",
                                    id="add-btn",
                                    n_clicks=0,
                                    style={**BTN_STYLE, "marginBottom": "0"},
                                ),
                                html.Div(
                                    id="error-msg",
                                    style={"color": RED, "fontSize": "0.88em",
                                           "paddingBottom": "6px", "flex": "1"},
                                ),
                                # push Upload CSV to the far right
                                html.Div(style={"flex": "1"}),
                                html.Div([
                                    html.Label("\u00a0", style={"fontSize": "0.8em", "display": "block",
                                                                "marginBottom": "4px"}),
                                    dcc.Upload(
                                        id="csv-upload",
                                        children=html.Button(
                                            "Upload CSV",
                                            style={**BTN_STYLE, "background": NAVY},
                                        ),
                                        accept=".csv",
                                        multiple=False,
                                    ),
                                ]),
                                html.Div(id="csv-error-msg",
                                         style={"color": RED, "fontSize": "0.88em",
                                                "paddingBottom": "6px"}),
                            ],
                        ),
                    ],
                ),

                # ── Portfolio table ────────────────────────────────────────────
                html.Div(
                    style={
                        **CARD_STYLE,
                        "padding": 0,
                        "overflow": "hidden",
                    },
                    children=[
                        # Table header row
                        html.Div(
                            style={"padding": "1.2em 1.8em", "borderBottom": f"1px solid {LGRAY}"},
                            children=[
                                html.Div(
                                    id="total-value-badge",
                                    style={"color": NAVY, "fontSize": "2em", "fontWeight": "800",
                                           "letterSpacing": "-0.5px", "lineHeight": "1.1",
                                           "marginBottom": "0.15em"},
                                ),
                                html.H4("Your Portfolio", style={"color": "#888", "margin": 0,
                                                                   "fontWeight": "500", "fontSize": "0.95em",
                                                                   "textTransform": "uppercase",
                                                                   "letterSpacing": "0.06em"}),
                            ],
                        ),

                        dash_table.DataTable(
                            id="portfolio-table",
                            columns=[
                                {"name": "Ticker",         "id": "ticker",   "editable": False},
                                {"name": "Company",        "id": "company",  "editable": False},
                                {"name": "Qty",            "id": "qty",      "editable": True,  "type": "numeric"},
                                {"name": "Price (live €)", "id": "price",    "editable": False},
                                {"name": "Total Value (€)", "id": "total",   "editable": False},
                                {"name": "Weight",         "id": "weight",   "editable": False},
                            ],
                            data=[],
                            row_deletable=True,
                            editable=False,
                            style_as_list_view=True,
                            style_header=TABLE_HEADER,
                            style_cell=TABLE_CELL,
                            style_cell_conditional=[
                                {"if": {"column_id": "ticker"},  "fontWeight": "700", "color": NAVY, "width": "90px"},
                                {"if": {"column_id": "company"}, "width": "220px"},
                                {"if": {"column_id": "qty"},     "width": "80px",  "textAlign": "right"},
                                {"if": {"column_id": "price"},   "width": "120px", "textAlign": "right"},
                                {"if": {"column_id": "total"},   "width": "130px", "textAlign": "right", "fontWeight": "600"},
                                {"if": {"column_id": "weight"},  "width": "90px",  "textAlign": "right", "color": "#555"},
                            ],
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFC"},
                            ],
                            page_action="none",
                            style_table={"overflowX": "auto"},
                        ),

                        # Empty-state message
                        html.Div(
                            id="empty-state",
                            children="No holdings yet. Add a ticker above to get started.",
                            style={
                                "textAlign": "center",
                                "color": "#aaa",
                                "padding": "3em",
                                "fontSize": "0.95em",
                                "fontStyle": "italic",
                            },
                        ),
                    ],
                ),

                # ── Refresh Analysis button ─────────────────────────────────────────
                html.Div(
                    style={"marginTop": "1.5em", "display": "flex", "justifyContent": "center",
                           "alignItems": "center", "flexDirection": "column", "gap": "0.6em"},
                    children=[
                        html.Button(
                            "↺ Refresh Analysis",
                            id="refresh-btn",
                            n_clicks=0,
                            style={**BTN_STYLE, "background": NAVY, "width": "100%",
                                   "maxWidth": "380px", "height": "48px", "fontSize": "1em"},
                        ),
                        html.Div(id="pull-error-msg", style={"color": RED, "fontSize": "0.85em", "textAlign": "center"}),
                    ],
                ),
            ],
        ),

        # ── Fundamentals tab bar (initially hidden, shown after pull) ────────────
        html.Div(
            id="fund-tab-container",
            style={"display": "none"},
            children=[
                html.Div(
                    style={**CONTAINER, "padding": "1.5em 1em 0"},
                    children=[
                        dcc.Tabs(
                            id="fund-tab",
                            value="valuations",
                            colors={"border": "transparent", "primary": NAVY, "background": "transparent"},
                            style={"borderBottom": "none"},
                            children=[
                                dcc.Tab(
                                    label="Valuations", value="valuations",
                                    style={"borderRadius": "999px", "border": f"1.5px solid {LGRAY}",
                                           "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                           "fontSize": "0.88em", "fontWeight": "600",
                                           "color": NAVY, "background": WHITE},
                                    selected_style={"borderRadius": "999px", "border": f"1.5px solid {NAVY}",
                                                    "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                                    "fontSize": "0.88em", "fontWeight": "700",
                                                    "color": WHITE, "background": NAVY,
                                                    "borderBottom": f"1.5px solid {NAVY}"},
                                ),
                                dcc.Tab(
                                    label="Balance Sheet", value="balance",
                                    style={"borderRadius": "999px", "border": f"1.5px solid {LGRAY}",
                                           "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                           "fontSize": "0.88em", "fontWeight": "600",
                                           "color": NAVY, "background": WHITE},
                                    selected_style={"borderRadius": "999px", "border": f"1.5px solid {NAVY}",
                                                    "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                                    "fontSize": "0.88em", "fontWeight": "700",
                                                    "color": WHITE, "background": NAVY,
                                                    "borderBottom": f"1.5px solid {NAVY}"},
                                ),
                                dcc.Tab(
                                    label="Income Statement", value="income",
                                    style={"borderRadius": "999px", "border": f"1.5px solid {LGRAY}",
                                           "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                           "fontSize": "0.88em", "fontWeight": "600",
                                           "color": NAVY, "background": WHITE},
                                    selected_style={"borderRadius": "999px", "border": f"1.5px solid {NAVY}",
                                                    "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                                    "fontSize": "0.88em", "fontWeight": "700",
                                                    "color": WHITE, "background": NAVY,
                                                    "borderBottom": f"1.5px solid {NAVY}"},
                                ),
                                dcc.Tab(
                                    label="Cash Flow", value="cashflow",
                                    style={"borderRadius": "999px", "border": f"1.5px solid {LGRAY}",
                                           "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                           "fontSize": "0.88em", "fontWeight": "600",
                                           "color": NAVY, "background": WHITE},
                                    selected_style={"borderRadius": "999px", "border": f"1.5px solid {NAVY}",
                                                    "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                                    "fontSize": "0.88em", "fontWeight": "700",
                                                    "color": WHITE, "background": NAVY,
                                                    "borderBottom": f"1.5px solid {NAVY}"},
                                ),
                                dcc.Tab(
                                    label="Factor Scores", value="factor_scores",
                                    style={"borderRadius": "999px", "border": f"1.5px solid {LGRAY}",
                                           "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                           "fontSize": "0.88em", "fontWeight": "600",
                                           "color": NAVY, "background": WHITE},
                                    selected_style={"borderRadius": "999px", "border": f"1.5px solid {CYAN}",
                                                    "padding": "0.35em 1.4em", "marginRight": "0.6em",
                                                    "fontSize": "0.88em", "fontWeight": "700",
                                                    "color": NAVY, "background": CYAN,
                                                    "borderBottom": f"1.5px solid {CYAN}"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ── Fundamentals section (table, populated by callback) ──────────────────
        dcc.Loading(
            id="loading-fundamentals",
            type="circle",
            color=CYAN,
            children=html.Div(
                id="fundamentals-section",
                style={**CONTAINER, "padding": "0.5em 1em 1em"},
            ),
        ),

        # ── Sector / Country mix section ─────────────────────────────────────────
        dcc.Loading(
            id="loading-sector",
            type="circle",
            color=CYAN,
            children=html.Div(
                id="sector-section",
                style={**CONTAINER, "padding": "0 1em 2em"},
            ),
        ),

        # ── Correlation section ──────────────────────────────────────────────────
        dcc.Loading(
            id="loading-correlation",
            type="circle",
            color=CYAN,
            children=html.Div(
                id="correlation-section",
                style={**CONTAINER, "padding": "0 1em 3em"},
            ),
        ),

        # ── Frontier section ─────────────────────────────────────────────────────
        dcc.Loading(
            id="loading-frontier",
            type="circle",
            color=CYAN,
            children=html.Div(
                id="frontier-section",
                style={**CONTAINER, "padding": "0 1em 3em"},
            ),
        ),

        # ── Frontier rebalance card (Max Sharpe / Min Variance buttons) ────────
        html.Div(
            id="frontier-rebalance-card",
            style={**CONTAINER, "padding": "0 1em 3em"},
        ),

        # ── Hidden stores ──────────────────────────────────────────────────
        dcc.Store(id="portfolio-store", data=[]),
        dcc.Store(id="fundamentals-store", data=[]),
        dcc.Store(id="factor-scores-store", data=[]),
        dcc.Store(id="price-history-store", data=None),
        dcc.Store(id="frontier-store", data={}),
        dcc.Store(id="analyse-trigger", data=0),
        dcc.Interval(id="init-interval", interval=300, max_intervals=1),
    ],
)


# ── Helper: fetch price + name from yfinance ──────────────────────────────────
def fetch_ticker_data(ticker: str):
    """Returns (company_name, price) or raises ValueError."""
    t = yf.Ticker(ticker.upper())
    info = t.fast_info
    price = getattr(info, "last_price", None)
    if price is None or price == 0:
        raise ValueError(f"Could not fetch price for '{ticker.upper()}'")
    # Company name — falls back gracefully
    try:
        name = t.info.get("shortName") or t.info.get("longName") or ticker.upper()
    except Exception:
        name = ticker.upper()
    return name, round(float(price), 2)


# ── Helper: recalculate weights across the full portfolio ─────────────────────
def recalc_weights(rows):
    total = sum(r["total"] for r in rows)
    for r in rows:
        r["weight"] = f"{round(r['total'] / total * 100, 1)}%" if total else "—"
    return rows, total


# ── Callback: auto-load default portfolio on page load ────────────────────────
@app.callback(
    Output("portfolio-store",  "data", allow_duplicate=True),
    Input("init-interval", "n_intervals"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def load_default_portfolio(_, store):
    if store:          # already populated (e.g. user added tickers before interval fired)
        return no_update

    def _fetch(ticker, qty):
        try:
            name, price = fetch_ticker_data(ticker)
            return {"ticker": ticker, "company": name, "qty": float(qty),
                    "price": price, "total": round(float(qty) * price, 2), "weight": "—"}
        except Exception:
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch, t, q): (t, q) for t, q in DEFAULT_HOLDINGS}
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    # Restore original order
    order = {t: i for i, (t, _) in enumerate(DEFAULT_HOLDINGS)}
    rows.sort(key=lambda r: order.get(r["ticker"], 999))
    rows, _ = recalc_weights(rows)
    return rows


# ── Callback: Refresh button ── increment analyse-trigger ──────────────────────
@app.callback(
    Output("analyse-trigger", "data"),
    Input("refresh-btn", "n_clicks"),
    State("analyse-trigger", "data"),
    prevent_initial_call=True,
)
def refresh_trigger(n_clicks, current):
    if not n_clicks:
        return no_update
    return (current or 0) + 1


# ── Clientside callback: Emoji visibility + button loading state ───────────────
# Controlled entirely through Dash outputs — no direct DOM manipulation —
# to avoid conflicts with React's reconciliation cycle.
_BTN_NORMAL = {
    "background": NAVY, "color": WHITE, "border": "none", "borderRadius": "999px",
    "padding": "0 1.8em", "height": "48px", "fontWeight": "600", "fontSize": "1em",
    "fontFamily": FONT, "cursor": "pointer", "letterSpacing": "0.03em",
    "width": "100%", "maxWidth": "380px", "transition": "opacity 0.18s ease, transform 0.12s ease",
}
_BTN_LOADING = {
    **_BTN_NORMAL,
    "background": "#0a2540", "opacity": "0.72", "cursor": "not-allowed",
    "animation": "btnPulse 1.2s ease-in-out infinite",
}

app.clientside_callback(
    """
    function(n_clicks) {{
        var no_update = window.dash_clientside.no_update;
        var btn_loading = {btn_loading};
        if (!n_clicks) return [no_update, no_update, no_update];
        return [btn_loading, "\u27f3 Fetching\u2026", true];
    }}
    """.format(btn_loading=json.dumps(_BTN_LOADING)),
    Output("refresh-btn", "style"),
    Output("refresh-btn", "children"),
    Output("refresh-btn", "disabled"),
    Input("refresh-btn",  "n_clicks"),
    prevent_initial_call=True,
)


# ── Callback: Add / Delete → update store ─────────────────────────────────────
@app.callback(
    Output("portfolio-store", "data"),
    Output("error-msg", "children"),
    Output("ticker-input", "value"),
    Output("ticker-input", "options", allow_duplicate=True),
    Output("qty-input", "value"),
    Input("add-btn", "n_clicks"),
    Input("portfolio-table", "data"),
    State("ticker-input", "value"),
    State("qty-input", "value"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def update_store(n_clicks, table_data, ticker, qty, store):
    triggered = ctx.triggered_id

    # ── Row deleted via DataTable ────────────────────────────────────────────
    if triggered == "portfolio-table":
        if table_data is None:
            return [], no_update, no_update, no_update, no_update
        # Re-sync store with current table (preserving raw numeric totals)
        synced = []
        for row in table_data:
            try:
                synced.append({**row, "total": float(str(row["total"]).replace(",", ""))})
            except Exception:
                synced.append(row)
        synced, _ = recalc_weights(synced)
        return synced, "", no_update, no_update, no_update

    # ── Add button ────────────────────────────────────────────────────────────
    if triggered == "add-btn":
        if not ticker or not str(ticker).strip():
            return store, "Please enter a ticker symbol.", no_update, no_update, no_update
        if not qty or float(qty) <= 0:
            return store, "Please enter a quantity greater than 0.", no_update, no_update, no_update

        ticker = str(ticker).strip().upper()

        # Prevent duplicates
        if any(r["ticker"] == ticker for r in store):
            return store, f"{ticker} is already in your portfolio.", no_update, no_update, no_update

        try:
            name, price = fetch_ticker_data(ticker)
        except Exception as e:
            return store, str(e), no_update, no_update, no_update

        total = round(float(qty) * price, 2)
        new_row = {
            "ticker":  ticker,
            "company": name,
            "qty":     float(qty),
            "price":   price,
            "total":   total,
            "weight":  "—",
        }
        updated = store + [new_row]
        updated, _ = recalc_weights(updated)
        return updated, "", None, [], None   # clear inputs on success

    return store, "", no_update, no_update, no_update


# ── Callback: CSV upload → populate store ─────────────────────────────────────
# Yahoo Finance portfolio CSV export contains columns: Symbol, Quantity (or Shares)
@app.callback(
    Output("portfolio-store",  "data",     allow_duplicate=True),
    Output("csv-error-msg",    "children"),
    Input("csv-upload",        "contents"),
    State("csv-upload",        "filename"),
    State("portfolio-store",   "data"),
    prevent_initial_call=True,
)
def import_csv(contents, filename, store):
    if contents is None:
        return no_update, no_update

    # Decode base64 payload
    try:
        _content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string).decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(decoded))
    except Exception:
        return no_update, "Could not read file. Please upload a valid CSV."

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find ticker column
    ticker_col = next((c for c in df.columns if c in ("symbol", "ticker")), None)
    if ticker_col is None:
        return no_update, "No 'Symbol' or 'Ticker' column found in CSV."

    # Find quantity column
    qty_col = next((c for c in df.columns if c in ("quantity", "shares", "qty", "units")), None)

    errors = []
    new_store = list(store)
    existing = {r["ticker"] for r in new_store}

    rows_to_fetch = []
    for _, row in df.iterrows():
        raw_ticker = str(row[ticker_col]).strip().upper()
        if not raw_ticker or raw_ticker in ("NAN", ""):
            continue
        qty = 1.0
        if qty_col:
            try:
                qty = float(row[qty_col])
                if qty <= 0:
                    qty = 1.0
            except Exception:
                qty = 1.0
        if raw_ticker not in existing:
            rows_to_fetch.append((raw_ticker, qty))
            existing.add(raw_ticker)

    if not rows_to_fetch:
        return no_update, "No new tickers found in the file (duplicates skipped)."

    # Fetch prices concurrently
    def _fetch(t, q):
        try:
            name, price = fetch_ticker_data(t)
            return {"ticker": t, "company": name, "qty": q, "price": price, "total": round(q * price, 2), "weight": "—"}
        except Exception:
            return None

    fetched, failed = [], []
    with ThreadPoolExecutor(max_workers=min(len(rows_to_fetch), 6)) as ex:
        futures = {ex.submit(_fetch, t, q): t for t, q in rows_to_fetch}
        for future in as_completed(futures):
            result = future.result()
            if result:
                fetched.append(result)
            else:
                failed.append(futures[future])

    new_store.extend(fetched)
    new_store, _ = recalc_weights(new_store)

    msg = ""
    if failed:
        msg = f"Could not fetch price for: {', '.join(failed)}"

    return new_store, msg


# ── Callback: render table + badge from store ──────────────────────────────────
@app.callback(
    Output("portfolio-table", "data"),
    Output("total-value-badge", "children"),
    Output("empty-state", "style"),
    Input("portfolio-store", "data"),
)
def render_table(store):
    if not store:
        badge = "€0"
        empty_style = {
            "textAlign": "center", "color": "#aaa",
            "padding": "3em", "fontSize": "0.95em", "fontStyle": "italic",
        }
        return [], badge, empty_style

    total = sum(r["total"] for r in store)
    badge = f"€{total:,.0f}"
    empty_style = {"display": "none"}

    sorted_store = sorted(store, key=lambda r: float(str(r.get("total", 0)).replace(",", "")), reverse=True)
    display_rows = [
        {
            **r,
            "price": f"{float(r['price']):,.2f}",
            "total": f"{float(str(r['total']).replace(',', '')):,.0f}",
        }
        for r in sorted_store
    ]
    return display_rows, badge, empty_style


# ── Fundamentals helpers ─────────────────────────────────────────────────────
def _fmt(val, decimals=1):
    if val is None:
        return "\u2014"
    try:
        v = float(val)
        return "\u2014" if v != v else f"{v:.{decimals}f}"
    except Exception:
        return "\u2014"

def _pct(val):
    if val is None:
        return "\u2014"
    try:
        v = float(val)
        return "\u2014" if v != v else f"{v * 100:.1f}%"
    except Exception:
        return "\u2014"

def _millions(val):
    if val is None:
        return "\u2014"
    try:
        v = float(val)
        return "\u2014" if v != v else f"€{v / 1e6:,.0f}M"
    except Exception:
        return "\u2014"

def _sf(val):
    """Return safe float or None (handles NaN)."""
    if val is None:
        return None
    try:
        v = float(val)
        return None if v != v else v
    except Exception:
        return None


def _fetch_quarterly_trends(t_obj):
    """Return trend directions per metric: +1 improving, 0 flat, -1 declining, None = no data."""
    trends = {"trend_op_m": None, "trend_rev": None, "trend_fcf": None, "trend_debt": None}
    try:
        qfin = t_obj.quarterly_financials
        if qfin is not None and not qfin.empty and len(qfin.columns) >= 3:
            cols    = qfin.columns[:4]
            rev_key = next((k for k in ["Total Revenue", "Revenue"] if k in qfin.index), None)
            oi_key  = next((k for k in ["Operating Income", "Ebit"] if k in qfin.index), None)
            if rev_key:
                rev = qfin.loc[rev_key, cols].dropna().astype(float)
                if len(rev) >= 3:
                    mid = len(rev) // 2
                    if rev.iloc[:mid].mean() > rev.iloc[mid:].mean() * 1.03:
                        trends["trend_rev"] = 1
                    elif rev.iloc[:mid].mean() < rev.iloc[mid:].mean() * 0.97:
                        trends["trend_rev"] = -1
                    else:
                        trends["trend_rev"] = 0
            if rev_key and oi_key:
                rev = qfin.loc[rev_key, cols].dropna().astype(float)
                oi  = qfin.loc[oi_key,  cols].dropna().astype(float)
                common = rev.index.intersection(oi.index)
                if len(common) >= 3:
                    om  = oi[common] / rev[common]
                    mid = len(om) // 2
                    if float(om.iloc[:mid].mean()) > float(om.iloc[mid:].mean()) * 1.02:
                        trends["trend_op_m"] = 1
                    elif float(om.iloc[:mid].mean()) < float(om.iloc[mid:].mean()) * 0.98:
                        trends["trend_op_m"] = -1
                    else:
                        trends["trend_op_m"] = 0
    except Exception:
        pass
    try:
        qcf = t_obj.quarterly_cashflow
        if qcf is not None and not qcf.empty and len(qcf.columns) >= 3:
            cols      = qcf.columns[:4]
            ocf_key   = next((k for k in ["Operating Cash Flow", "Total Cash From Operating Activities",
                                           "Cash Flow From Continuing Operating Activities"] if k in qcf.index), None)
            capex_key = next((k for k in ["Capital Expenditure", "Capital Expenditures",
                                           "Purchase Of Property Plant And Equipment"] if k in qcf.index), None)
            if ocf_key:
                ocf   = qcf.loc[ocf_key, cols].dropna().astype(float)
                capex = qcf.loc[capex_key, cols].dropna().astype(float).abs() if capex_key else None
                fcf_s = (ocf - capex).dropna() if capex is not None else ocf
                if len(fcf_s) >= 3:
                    mid = len(fcf_s) // 2
                    if float(fcf_s.iloc[:mid].mean()) > float(fcf_s.iloc[mid:].mean()) * 1.05:
                        trends["trend_fcf"] = 1
                    elif float(fcf_s.iloc[:mid].mean()) < float(fcf_s.iloc[mid:].mean()) * 0.95:
                        trends["trend_fcf"] = -1
                    else:
                        trends["trend_fcf"] = 0
    except Exception:
        pass
    try:
        qbs = t_obj.quarterly_balance_sheet
        if qbs is not None and not qbs.empty and len(qbs.columns) >= 3:
            cols     = qbs.columns[:4]
            debt_key = next((k for k in ["Total Debt", "Long Term Debt", "Total Long Term Debt"] if k in qbs.index), None)
            if debt_key:
                debt = qbs.loc[debt_key, cols].dropna().astype(float)
                if len(debt) >= 3:
                    early, late = float(debt.iloc[-1]), float(debt.iloc[0])
                    if early != 0:
                        if late < early * 0.95:
                            trends["trend_debt"] = 1
                        elif late > early * 1.05:
                            trends["trend_debt"] = -1
                        else:
                            trends["trend_debt"] = 0
    except Exception:
        pass
    return trends


def _cagr_from_annual(t_obj, metric: str, years: int) -> str:
    """Compute CAGR over `years` from annual financials. metric: 'rev' or 'eps'."""
    try:
        ann = t_obj.financials  # rows=metrics, cols=dates newest→oldest
        if ann is None or ann.empty:
            return "\u2014"
        if metric == "rev":
            key = next((k for k in ["Total Revenue", "Revenue"] if k in ann.index), None)
        else:  # eps
            key = next((k for k in ["Diluted EPS", "Basic EPS", "Net Income Common Stockholders", "Net Income"] if k in ann.index), None)
        if key is None:
            return "\u2014"
        series = ann.loc[key].dropna().sort_index(ascending=False)
        if len(series) <= years:
            return "\u2014"
        v_recent = float(series.iloc[0])
        v_base   = float(series.iloc[years])
        if v_base <= 0 or v_recent <= 0:
            return "\u2014"
        result = (v_recent / v_base) ** (1.0 / years) - 1
        return f"{result * 100:.1f}%"
    except Exception:
        return "\u2014"


def _best_cagr(t_obj, metric: str, max_years: int = 4) -> str:
    """Try CAGR from max_years down to 2, return first non-empty result."""
    for y in range(max_years, 1, -1):
        val = _cagr_from_annual(t_obj, metric, y)
        if val != "\u2014":
            return val
    return "\u2014"


def _annual_bs_series(t_obj, key_options: list):
    """Get an annual balance sheet series (newest-first) trying multiple key names."""
    try:
        bs = t_obj.balance_sheet
        if bs is None or bs.empty:
            return None
        key = next((k for k in key_options if k in bs.index), None)
        if key is None:
            return None
        s = bs.loc[key].dropna().sort_index(ascending=False).astype(float)
        return s if len(s) > 0 else None
    except Exception:
        return None


def _annual_cf_series(t_obj):
    """Returns (ocf_series, fcf_series) sorted newest-first, or (None, None)."""
    try:
        cf = t_obj.cashflow
        if cf is None or cf.empty:
            return None, None
        ocf_key   = next((k for k in ["Operating Cash Flow", "Total Cash From Operating Activities",
                                       "Cash Flow From Continuing Operating Activities"] if k in cf.index), None)
        capex_key = next((k for k in ["Capital Expenditure", "Capital Expenditures",
                                       "Purchase Of Property Plant And Equipment"] if k in cf.index), None)
        if ocf_key is None:
            return None, None
        ocf = cf.loc[ocf_key].dropna().sort_index(ascending=False).astype(float)
        if capex_key:
            capex = cf.loc[capex_key].dropna().sort_index(ascending=False).astype(float).abs()
            common = ocf.index.intersection(capex.index)
            fcf = (ocf[common] - capex[common]).dropna()
        else:
            fcf = None
        return (ocf if len(ocf) > 0 else None), fcf
    except Exception:
        return None, None


def _yoy_pct(s) -> str:
    """Year-over-year % change from a series (newest first)."""
    if s is None or len(s) < 2:
        return "\u2014"
    v0, v1 = float(s.iloc[0]), float(s.iloc[1])
    if v1 == 0:
        return "\u2014"
    pct = (v0 - v1) / abs(v1) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _cagr_series_disp(s, n: int) -> str:
    """CAGR over n years from series s (newest first), displayed as %."""
    if s is None or len(s) <= n:
        return "\u2014"
    v0, vn = float(s.iloc[0]), float(s.iloc[n])
    if vn == 0 or (v0 < 0 and vn < 0):
        return "\u2014"
    if vn < 0 or v0 < 0:
        return "\u2014"
    result = (v0 / vn) ** (1.0 / n) - 1
    sign = "+" if result >= 0 else ""
    return f"{sign}{result * 100:.1f}%"


def _delta_disp(s, n: int) -> str:
    """% change between iloc[0] and iloc[n] in series s, with sign."""
    if s is None or len(s) <= n:
        return "\u2014"
    v0, vn = float(s.iloc[0]), float(s.iloc[n])
    if vn == 0:
        return "\u2014"
    pct = (v0 - vn) / abs(vn) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _fetch_one(ticker: str) -> dict:
    EMPTY_DISP  = {k: "\u2014" for k in [
        "pe", "fwd_pe", "ev_ebitda", "ps", "pb",
        "gross_m", "op_m", "net_m", "roe", "roa",
        "debt_eq", "curr_ratio", "int_cov", "rev_growth", "eps_growth",
        "cash_m", "net_debt_m", "equity_m",
        "nd_1y_chg", "nd_3y_chg", "eq_1y_chg", "eq_3y_chg",
        "fcf_m", "fcf_yield", "op_cf_m", "capex_m", "fcf_rev",
        "rev_3y", "rev_max", "eps_3y", "eps_max",
        "op_cf_1y", "op_cf_3y", "fcf_1y", "fcf_3y_cagr",
    ]}
    EMPTY_RAW   = {k: None for k in [
        "raw_gross_m", "raw_op_m", "raw_net_m", "raw_roe", "raw_roa",
        "raw_debt_eq", "raw_curr_ratio", "raw_int_cov",
        "raw_rev_growth", "raw_eps_growth", "raw_rev_3y", "raw_eps_3y",
        "raw_fcf_yield", "raw_fcf_rev", "raw_pe", "raw_fwd_pe",
        "raw_ev_ebitda", "raw_ps", "raw_pb",
        "raw_fcf_growth_1y", "raw_ocf_growth_1y",
    ]}
    EMPTY_TREND = {"trend_op_m": None, "trend_rev": None, "trend_fcf": None, "trend_debt": None}
    last_err = None
    for _attempt in range(3):
        try:
            return _fetch_one_attempt(ticker, EMPTY_DISP, EMPTY_RAW, EMPTY_TREND)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (_attempt + 1))  # 0.5s, 1s between retries
    return {"ticker": ticker, "company": "Fetch error", **EMPTY_DISP, **EMPTY_RAW, **EMPTY_TREND}


def _fetch_one_attempt(ticker, EMPTY_DISP, EMPTY_RAW, EMPTY_TREND):
    try:
        t_obj  = yf.Ticker(ticker)
        info   = t_obj.info
        fcf    = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        ocf    = info.get("operatingCashflow")
        capex  = info.get("capitalExpenditures")  # usually negative
        rev    = info.get("totalRevenue")
        raw_fy      = (fcf / mktcap) if (fcf and mktcap and mktcap > 0) else None
        raw_fcf_rev = (fcf / rev)    if (fcf and rev    and rev    > 0) else None
        fy_disp = f"{raw_fy * 100:.1f}%" if raw_fy is not None else "\u2014"
        trends  = _fetch_quarterly_trends(t_obj)

        # ── Balance sheet history ─────────────────────────────────────────
        cash_s    = _annual_bs_series(t_obj, ["Cash And Cash Equivalents", "Cash",
                                               "Cash And Short Term Investments"])
        debt_s    = _annual_bs_series(t_obj, ["Total Debt", "Long Term Debt"])
        eq_s      = _annual_bs_series(t_obj, ["Stockholders Equity", "Total Stockholder Equity",
                                               "Common Stock Equity", "Total Equity Gross Minority Interest"])
        cash_v    = float(cash_s.iloc[0]) if cash_s is not None else None
        debt_v    = float(debt_s.iloc[0]) if debt_s is not None else None
        eq_v      = float(eq_s.iloc[0])   if eq_s   is not None else None
        nd_v      = (debt_v - cash_v) if (debt_v is not None and cash_v is not None) else None
        # Net debt annual series for trend
        if debt_s is not None and cash_s is not None:
            common = debt_s.index.intersection(cash_s.index)
            nd_s   = (debt_s[common] - cash_s[common]).dropna()
            nd_s   = nd_s if len(nd_s) > 0 else None
        else:
            nd_s = None
        # Interest coverage from annual financials
        int_cov_v = None
        try:
            fin     = t_obj.financials
            ebit_k  = next((k for k in ["EBIT", "Ebit", "Operating Income"] if k in fin.index), None)
            int_k   = next((k for k in ["Interest Expense", "Interest Expense Non Operating",
                                         "Net Interest Income"] if k in fin.index), None)
            if ebit_k and int_k:
                ebit_val = float(fin.loc[ebit_k].dropna().iloc[0])
                int_val  = float(fin.loc[int_k].dropna().iloc[0])
                if int_val != 0:
                    int_cov_v = abs(ebit_val / int_val)
        except Exception:
            pass

        # ── Cash flow history ─────────────────────────────────────────────
        ocf_s, fcf_s = _annual_cf_series(t_obj)
        # raw 1Y growth rates for scoring
        raw_fcf_g1y = None
        raw_ocf_g1y = None
        if fcf_s is not None and len(fcf_s) >= 2:
            v0, v1 = float(fcf_s.iloc[0]), float(fcf_s.iloc[1])
            if v1 != 0:
                raw_fcf_g1y = (v0 - v1) / abs(v1)
        if ocf_s is not None and len(ocf_s) >= 2:
            v0, v1 = float(ocf_s.iloc[0]), float(ocf_s.iloc[1])
            if v1 != 0:
                raw_ocf_g1y = (v0 - v1) / abs(v1)
        # raw 3Y CAGR for FCF scoring
        raw_fcf_3y = None
        if fcf_s is not None and len(fcf_s) > 3:
            v0, v3 = float(fcf_s.iloc[0]), float(fcf_s.iloc[3])
            if v0 > 0 and v3 > 0:
                raw_fcf_3y = (v0 / v3) ** (1.0 / 3) - 1
        # raw 3Y CAGR numerics for income scoring
        raw_rev_3y = None
        raw_eps_3y = None
        try:
            ann = t_obj.financials
            if ann is not None and not ann.empty:
                rev_k = next((k for k in ["Total Revenue", "Revenue"] if k in ann.index), None)
                eps_k = next((k for k in ["Net Income Common Stockholders", "Net Income"] if k in ann.index), None)
                if rev_k:
                    rs = ann.loc[rev_k].dropna().sort_index(ascending=False).astype(float)
                    if len(rs) > 3 and float(rs.iloc[3]) > 0 and float(rs.iloc[0]) > 0:
                        raw_rev_3y = (float(rs.iloc[0]) / float(rs.iloc[3])) ** (1.0/3) - 1
                if eps_k:
                    es = ann.loc[eps_k].dropna().sort_index(ascending=False).astype(float)
                    if len(es) > 3 and float(es.iloc[3]) > 0 and float(es.iloc[0]) > 0:
                        raw_eps_3y = (float(es.iloc[0]) / float(es.iloc[3])) ** (1.0/3) - 1
        except Exception:
            pass

        return {
            # ── Display fields ──────────────────────────────────────────────
            "ticker":       ticker,
            "company":      info.get("shortName") or ticker,
            "sector":       info.get("sector")    or "—",
            "country":      info.get("country")   or "—",
            "pe":           _fmt(info.get("trailingPE")),
            "fwd_pe":       _fmt(info.get("forwardPE")),
            "ev_ebitda":    _fmt(info.get("enterpriseToEbitda")),
            "ps":           _fmt(info.get("priceToSalesTrailing12Months")),
            "pb":           _fmt(info.get("priceToBook")),
            "gross_m":      _pct(info.get("grossMargins")),
            "op_m":         _pct(info.get("operatingMargins")),
            "net_m":        _pct(info.get("profitMargins")),
            "roe":          _pct(info.get("returnOnEquity")),
            "roa":          _pct(info.get("returnOnAssets")),
            "debt_eq":      _fmt(info.get("debtToEquity")),
            "curr_ratio":   _fmt(info.get("currentRatio")),
            "int_cov":      _fmt(int_cov_v),
            "cash_m":       _millions(cash_v),
            "net_debt_m":   _millions(nd_v),
            "equity_m":     _millions(eq_v),
            "nd_1y_chg":    _delta_disp(nd_s, 1),
            "nd_3y_chg":    _delta_disp(nd_s, 3),
            "eq_1y_chg":    _delta_disp(eq_s, 1),
            "eq_3y_chg":    _delta_disp(eq_s, 3),
            "rev_growth":   _pct(info.get("revenueGrowth")),
            "eps_growth":   _pct(info.get("earningsGrowth")),
            "rev_3y":       _cagr_from_annual(t_obj, "rev", 3),
            "rev_max":      _best_cagr(t_obj, "rev", 4),
            "eps_3y":       _cagr_from_annual(t_obj, "eps", 3),
            "eps_max":      _best_cagr(t_obj, "eps", 4),
            "fcf_m":        _millions(fcf),
            "fcf_yield":    fy_disp,
            "op_cf_m":      _millions(ocf),
            "capex_m":      _millions(abs(capex) if capex is not None else None),
            "fcf_rev":      _pct(raw_fcf_rev),
            "op_cf_1y":     _yoy_pct(ocf_s),
            "op_cf_3y":     _cagr_series_disp(ocf_s, 3),
            "fcf_1y":       _yoy_pct(fcf_s),
            "fcf_3y_cagr":  _cagr_series_disp(fcf_s, 3),
            # ── Raw numerics for scoring ────────────────────────────────────
            "raw_gross_m":     _sf(info.get("grossMargins")),
            "raw_op_m":        _sf(info.get("operatingMargins")),
            "raw_net_m":       _sf(info.get("profitMargins")),
            "raw_roe":         _sf(info.get("returnOnEquity")),
            "raw_roa":         _sf(info.get("returnOnAssets")),
            "raw_debt_eq":     _sf(info.get("debtToEquity")),
            "raw_curr_ratio":  _sf(info.get("currentRatio")),
            "raw_int_cov":     _sf(int_cov_v),
            "raw_rev_growth":  _sf(info.get("revenueGrowth")),
            "raw_eps_growth":  _sf(info.get("earningsGrowth")),
            "raw_rev_3y":      raw_rev_3y,
            "raw_eps_3y":      raw_eps_3y,
            "raw_fcf_yield":   raw_fy,
            "raw_fcf_rev":     raw_fcf_rev,
            "raw_pe":          _sf(info.get("trailingPE")),
            "raw_fwd_pe":      _sf(info.get("forwardPE")),
            "raw_ev_ebitda":   _sf(info.get("enterpriseToEbitda")),
            "raw_ps":          _sf(info.get("priceToSalesTrailing12Months")),
            "raw_pb":          _sf(info.get("priceToBook")),
            "raw_fcf_growth_1y": raw_fcf_g1y,
            "raw_ocf_growth_1y": raw_ocf_g1y,
            **trends,
        }
    except Exception:
        raise  # let _fetch_one retry handler catch it


def fetch_fundamentals(tickers: list) -> list:
    """Fetch fundamentals + quarterly trends concurrently for all tickers.
    Uses 3 workers max to avoid Yahoo Finance rate limits.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 3)) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[t] for t in tickers]


# ── Scoring engine ─────────────────────────────────────────────────────────────
# ─── 5-Factor Risk-Premia Scoring Engine ─────────────────────────────────────

def _pct_rank(val, all_vals, higher_is_better=True):
    """Cross-sectional percentile rank mapped to 0-10. Returns 5.0 if data insufficient."""
    clean = [v for v in all_vals if v is not None]
    if len(clean) < 2 or val is None:
        return 5.0
    rank = sum(1 for v in clean if v < val) / len(clean)
    if not higher_is_better:
        rank = 1.0 - rank
    return round(rank * 10, 1)


def _piotroski(d):
    """Approximate Piotroski F-Score using available fundamentals. Returns 0-9."""
    s = 0
    # Profitability (4 signals)
    if (v := d.get("raw_roa"))        is not None and v > 0:     s += 1  # ROA positive
    if (v := d.get("raw_fcf_yield"))  is not None and v > 0:     s += 1  # CFO positive
    if (v := d.get("raw_eps_growth")) is not None and v > 0:     s += 1  # Earnings trend
    if (v := d.get("raw_op_m"))       is not None and v > 0.05:  s += 1  # Operating margin
    # Leverage / liquidity (2 signals)
    if (v := d.get("raw_debt_eq"))    is not None and v < 100:   s += 1  # Low leverage
    if (v := d.get("raw_curr_ratio")) is not None and v > 1.0:   s += 1  # Liquid
    # Efficiency (3 signals)
    if (v := d.get("raw_gross_m"))    is not None and v > 0.20:  s += 1  # Gross margin
    if (v := d.get("raw_rev_growth")) is not None and v > 0:     s += 1  # Revenue growth
    if (v := d.get("raw_net_m"))      is not None and v > 0:     s += 1  # Net profit positive
    return s  # max 9


def _factor_value(rows):
    """Value factor: lower multiples = higher score (0-10, cross-sectional rank)."""
    metrics = [("raw_pe", 2.0), ("raw_fwd_pe", 2.5), ("raw_ev_ebitda", 2.0),
               ("raw_ps", 1.5), ("raw_pb", 1.0)]
    out = {}
    for d in rows:
        scores, ws = [], []
        for m, w in metrics:
            val = d.get(m)
            if val is None or val <= 0:
                continue
            all_vals = [r.get(m) for r in rows if r.get(m) is not None and r.get(m) > 0]
            scores.append(_pct_rank(val, all_vals, higher_is_better=False) * w)
            ws.append(w)
        out[d["ticker"]] = round(sum(scores) / sum(ws), 1) if ws else None
    return out


def _factor_quality(rows):
    """Quality factor: profitability, balance-sheet strength, Piotroski (0-10)."""
    hi = [("raw_roe", 2.0), ("raw_roa", 1.5), ("raw_op_m", 2.0),
          ("raw_gross_m", 1.5), ("raw_curr_ratio", 1.5), ("raw_int_cov", 2.0)]
    lo = [("raw_debt_eq", 2.0)]
    out = {}
    for d in rows:
        scores, ws = [], []
        for m, w in hi:
            val = d.get(m)
            if val is None:
                continue
            all_vals = [r.get(m) for r in rows if r.get(m) is not None]
            scores.append(_pct_rank(val, all_vals, True) * w)
            ws.append(w)
        for m, w in lo:
            val = d.get(m)
            if val is None:
                continue
            all_vals = [r.get(m) for r in rows if r.get(m) is not None]
            scores.append(_pct_rank(val, all_vals, False) * w)
            ws.append(w)
        # Piotroski F-Score: 0-9 → 0-10, weight 2
        scores.append(round(_piotroski(d) / 9 * 10, 1) * 2.0)
        ws.append(2.0)
        out[d["ticker"]] = round(sum(scores) / sum(ws), 1) if ws else None
    return out


def _factor_growth(rows):
    """Growth factor: revenue, EPS, FCF growth (0-10, cross-sectional rank)."""
    metrics = [("raw_rev_growth", 2.0), ("raw_rev_3y", 2.5),
               ("raw_eps_growth", 2.0), ("raw_eps_3y", 2.5), ("raw_fcf_growth_1y", 1.5)]
    out = {}
    for d in rows:
        scores, ws = [], []
        for m, w in metrics:
            val = d.get(m)
            if val is None:
                continue
            all_vals = [r.get(m) for r in rows if r.get(m) is not None]
            scores.append(_pct_rank(val, all_vals, True) * w)
            ws.append(w)
        out[d["ticker"]] = round(sum(scores) / sum(ws), 1) if ws else None
    return out


def _factor_momentum(tickers, price_df):
    """12-1 month price momentum (skips last month to avoid reversal). Returns {ticker: 0-10}."""
    if price_df is None or price_df.empty:
        return {t: None for t in tickers}
    n = len(price_df)
    p_1m  = price_df.iloc[max(0, n - 22)]   # ~1 month ago
    p_12m = price_df.iloc[max(0, n - 252)]  # ~12 months ago
    raw_mom = {}
    for t in tickers:
        if t not in price_df.columns:
            raw_mom[t] = None
            continue
        try:
            v1, v12 = float(p_1m[t]), float(p_12m[t])
            raw_mom[t] = (v1 - v12) / v12 if v12 > 0 and v1 == v1 and v12 == v12 else None
        except Exception:
            raw_mom[t] = None
    all_vals = [v for v in raw_mom.values() if v is not None]
    return {t: (_pct_rank(raw_mom[t], all_vals, True) if raw_mom[t] is not None else None)
            for t in tickers}


def _factor_lowvol(tickers, price_df):
    """Low volatility: lower annualized vol = higher score (0-10, cross-sectional rank)."""
    if price_df is None or price_df.empty:
        return {t: None for t in tickers}
    rets = price_df.pct_change(fill_method=None).dropna(how="all")
    raw_vol = {}
    for t in tickers:
        if t not in rets.columns:
            raw_vol[t] = None
            continue
        col = rets[t].dropna()
        raw_vol[t] = float(col.tail(252).std() * np.sqrt(252)) if len(col) >= 30 else None
    all_vals = [v for v in raw_vol.values() if v is not None]
    return {t: (_pct_rank(raw_vol[t], all_vals, higher_is_better=False) if raw_vol[t] is not None else None)
            for t in tickers}


def compute_factor_scores(fund_rows, price_df=None):
    """Compute 5-factor risk-premia scores. Returns list of dicts sorted by overall."""
    tickers = [d["ticker"] for d in fund_rows]
    val_sc  = _factor_value(fund_rows)
    qual_sc = _factor_quality(fund_rows)
    grow_sc = _factor_growth(fund_rows)
    mom_sc  = _factor_momentum(tickers, price_df) if price_df is not None else {t: None for t in tickers}
    vol_sc  = _factor_lowvol(tickers, price_df)   if price_df is not None else {t: None for t in tickers}
    W       = {"value": 1.5, "quality": 2.0, "growth": 1.5, "momentum": 1.5, "lowvol": 1.5}
    DASH    = "\u2014"
    out = []
    for d in fund_rows:
        t  = d["ticker"]
        fv = {"value": val_sc.get(t), "quality": qual_sc.get(t), "growth": grow_sc.get(t),
              "momentum": mom_sc.get(t), "lowvol": vol_sc.get(t)}
        valid = {k: v for k, v in fv.items() if v is not None}
        if valid:
            tw      = sum(W[k] for k in valid)
            overall = round(sum(valid[k] * W[k] for k in valid) / tw, 1)
        else:
            overall = None
        out.append({
            "ticker":    t,
            "company":   d["company"],
            "value":     fv["value"]    if fv["value"]    is not None else DASH,
            "quality":   fv["quality"]  if fv["quality"]  is not None else DASH,
            "growth":    fv["growth"]   if fv["growth"]   is not None else DASH,
            "momentum":  fv["momentum"] if fv["momentum"] is not None else DASH,
            "lowvol":    fv["lowvol"]   if fv["lowvol"]   is not None else DASH,
            "piotroski": _piotroski(d),
            "overall":   overall if overall is not None else DASH,
        })
    out.sort(key=lambda r: r["overall"] if isinstance(r["overall"], float) else -1, reverse=True)
    return out


FACTOR_COLUMNS = [
    {"name": "Ticker",       "id": "ticker"},
    {"name": "Company",      "id": "company"},
    {"name": "Value /10",    "id": "value"},
    {"name": "Quality /10",  "id": "quality"},
    {"name": "Growth /10",   "id": "growth"},
    {"name": "Momentum /10", "id": "momentum"},
    {"name": "Low Vol /10",  "id": "lowvol"},
    {"name": "Overall /10",  "id": "overall"},
]

FACTOR_METHODOLOGY = (
    "5-Factor Risk Premia model \u2014 all scores 0\u201310, cross-sectional rank within your portfolio.\u2003"
    "\u25cf\u2002Value \u2014 rewards low P/E, Fwd P/E (\u00d72.5), EV/EBITDA, P/S, P/B. "
    "Cheapest stock = 10, most expensive = 0.\u2003"
    "\u25cf\u2002Quality \u2014 ROE, ROA, operating/gross margin, current ratio, interest coverage, "
    "low leverage, plus Piotroski F-Score (9-point fundamental checklist, weight 2\u00d7).\u2003"
    "\u25cf\u2002Growth \u2014 revenue, EPS and FCF growth (1Y + 3Y CAGR). Fastest grower = 10.\u2003"
    "\u25cf\u2002Momentum \u2014 12\u20131 month price return (last month skipped to avoid reversal). "
    "Requires 1-year price history.\u2003"
    "\u25cf\u2002Low Volatility \u2014 1-year annualized daily return \u03c3. "
    "Lowest-vol stock = 10.\u2003"
    "\u25cf\u2002Overall \u2014 Quality \u00d72, Value/Growth/Momentum/Low Vol \u00d71.5 each."
)


def _factor_score_cond(col):
    return [
        {"if": {"filter_query": f"{{{col}}} >= 7",  "column_id": col},
         "backgroundColor": CYAN,      "color": NAVY,    "fontWeight": "700"},
        {"if": {"filter_query": f"{{{col}}} >= 4 && {{{col}}} < 7", "column_id": col},
         "backgroundColor": "#E8F4F6", "color": "#4A6572","fontWeight": "600"},
        {"if": {"filter_query": f"{{{col}}} < 4",   "column_id": col},
         "backgroundColor": BG,        "color": "#888",  "fontWeight": "400"},
    ]


# ── Callback: Ticker search suggestions (Yahoo Finance autocomplete) ─────────────
@app.callback(
    Output("ticker-input", "options"),
    Input("ticker-input", "search_value"),
    prevent_initial_call=True,
)
def search_ticker_suggestions(search_value):
    if not search_value or len(search_value.strip()) < 1:
        return no_update  # keep existing options so a selected value is not wiped
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": search_value.strip(), "quotesCount": 8, "newsCount": 0, "listsCount": 0}
        resp = requests.get(url, params=params, timeout=5,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
        options = []
        for q in quotes:
            sym = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or ""
            qtype = q.get("quoteType", "")
            if sym and qtype in ("EQUITY", "ETF", "MUTUALFUND"):
                label = f"{sym} — {name}" if name else sym
                options.append({"label": label, "value": sym})
        return options
    except Exception:
        return no_update  # on error, leave options as-is


# ── Callback: Fetch price history (shared by correlation + frontier) ─────────────
@app.callback(
    Output("price-history-store", "data"),
    Input("analyse-trigger", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def pull_price_history_cb(trigger, store):
    if not trigger or not store:
        return no_update
    tickers = [r["ticker"] for r in store]
    df = fetch_price_history(tickers)
    if df.empty:
        return {}
    return {"columns": list(df.columns), "index": [str(i) for i in df.index],
            "data": df.values.tolist()}


# ── Callback: Pull fundamentals ──────────────────────────────────────────────────
@app.callback(
    Output("fundamentals-store",  "data"),
    Output("fund-tab-container",  "style"),
    Output("pull-error-msg",      "children"),
    Output("refresh-btn",         "style",    allow_duplicate=True),
    Output("refresh-btn",         "children", allow_duplicate=True),
    Output("refresh-btn",         "disabled", allow_duplicate=True),
    Input("analyse-trigger",      "data"),
    State("portfolio-store",      "data"),
    prevent_initial_call=True,
)
def pull_fundamentals(trigger, store):
    _label = "\u21ba Refresh Analysis"
    if not trigger or not store:
        return no_update, no_update, "", _BTN_NORMAL, _label, False

    tickers = [r["ticker"] for r in store]
    rows    = fetch_fundamentals(tickers)
    return rows, {"display": "block"}, "", _BTN_NORMAL, _label, False


# ── Callback: Compute 5-factor scores ─────────────────────────────────────────────
@app.callback(
    Output("factor-scores-store", "data"),
    Input("fundamentals-store",   "data"),
    Input("price-history-store",  "data"),
    prevent_initial_call=True,
)
def compute_factor_scores_cb(fund_data, price_data):
    if not fund_data:
        return no_update
    price_df = None
    if price_data and isinstance(price_data, dict) and price_data.get("data"):
        price_df = pd.DataFrame(
            price_data["data"],
            columns=price_data["columns"],
            index=pd.to_datetime(price_data["index"]),
        )
    return compute_factor_scores(fund_data, price_df)


# ── Callback: render fundamentals table for selected tab ──────────────────────
@app.callback(
    Output("fundamentals-section", "children"),
    Input("fund-tab",              "value"),
    Input("fundamentals-store",    "data"),
    Input("factor-scores-store",   "data"),
    prevent_initial_call=True,
)
def render_fund_table(tab, rows, factor_scores):
    # ── Detect total fetch failure ─────────────────────────────────────────────
    all_failed = bool(rows) and all(r.get("company") == "Fetch error" for r in rows)
    if all_failed:
        return html.Div(
            style={"padding": "2em 0"},
            children=[
                html.P(
                    "⚠️  Yahoo Finance data not available — try later.",
                    style={"color": AMBER, "fontWeight": "600", "fontSize": "0.95em",
                           "margin": "0 0 0.4em 0"},
                ),
                html.P(
                    "Markowitz optimisation is still available below "
                    "(it uses price history only, not fundamentals).",
                    style={"color": "#9CA3AF", "fontSize": "0.85em", "margin": 0},
                ),
            ],
        )

    if tab == "factor_scores":
        if not factor_scores:
            return html.P("Factor scores not yet computed. Click Refresh Analysis.",
                          style={"color": "#888", "padding": "2em"})
        score_cols = ["value", "quality", "growth", "momentum", "lowvol", "overall"]
        score_cond = [
            {"if": {"row_index": "odd"}, "backgroundColor": BG},
            *[c for col in score_cols for c in _factor_score_cond(col)],
        ]
        return html.Div([
            html.H4("Factor Scores", style={"color": NAVY, "margin": "1.5em 0 0.5em 0"}),
            dash_table.DataTable(
                id="factor-scores-table",
                columns=FACTOR_COLUMNS,
                data=factor_scores,
                style_as_list_view=False,
                style_table={"overflowX": "auto", "borderRadius": "14px",
                             "boxShadow": "0 2px 12px rgba(0,0,0,0.06)"},
                style_header={**TABLE_HEADER, "textAlign": "center"},
                style_cell={**TABLE_CELL, "textAlign": "center", "minWidth": "80px"},
                style_cell_conditional=[
                    {"if": {"column_id": "ticker"},  "fontWeight": "700", "color": NAVY,
                     "textAlign": "left", "minWidth": "75px"},
                    {"if": {"column_id": "company"}, "textAlign": "left", "minWidth": "160px"},
                ],
                style_data_conditional=score_cond,
                page_action="none",
                editable=False,
            ),
            # ── Weak stock callout ─────────────────────────────────────────
            *([html.Div(
                style={
                    "marginTop": "1.4em",
                    "background": "#FFFBEB",
                    "border": f"1px solid {AMBER}",
                    "borderRadius": "14px",
                    "padding": "1.1em 1.5em",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "baseline",
                               "gap": "0.6em", "marginBottom": "0.5em"},
                        children=[
                            html.Span("⚠️ Low-quality holdings",
                                      style={"fontWeight": "700", "color": NAVY,
                                             "fontSize": "0.95em"}),
                            html.Span("scoring below 4 / 10 on fundamentals",
                                      style={"color": "#92400E", "fontSize": "0.82em"}),
                        ],
                    ),
                    html.P(
                        "These holdings rank in the bottom 40% of your portfolio. "
                        "Consider removing them, then click ↺ Refresh Analysis to re-optimize.",
                        style={"color": "#92400E", "fontSize": "0.85em",
                               "margin": "0 0 0.9em 0", "lineHeight": "1.55"},
                    ),
                    html.Div(
                        style={"display": "flex", "flexWrap": "wrap", "gap": "0.5em"},
                        children=[
                            html.Button(
                                f"{r['ticker']} ×",
                                id={"type": "remove-weak-btn", "index": r["ticker"]},
                                n_clicks=0,
                                style={
                                    "background": WHITE,
                                    "border": f"1.5px solid {AMBER}",
                                    "borderRadius": "999px",
                                    "padding": "0.25em 1em",
                                    "fontWeight": "700",
                                    "fontSize": "0.88em",
                                    "color": NAVY,
                                    "cursor": "pointer",
                                    "fontFamily": FONT,
                                    "transition": "background 0.15s ease, color 0.15s ease",
                                },
                            )
                            for r in [s for s in factor_scores
                                      if isinstance(s.get("overall"), float)
                                      and s["overall"] < 4.0]
                        ],
                    ),
                ],
            )] if any(isinstance(s.get("overall"), float) and s["overall"] < 4.0
                      for s in factor_scores) else []),
            # ── Factor score interpretation guide ──────────────────────────
            html.Div(
                style={"marginTop": "1.4em", "background": WHITE, "borderRadius": "14px",
                       "padding": "1.2em 1.6em",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
                       "fontSize": "0.86em", "color": "#555", "lineHeight": "1.75"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center",
                               "gap": "0.5em", "marginBottom": "0.8em"},
                        children=[
                            html.Span("How to interpret factor scores",
                                      style={"fontWeight": "700", "color": NAVY,
                                             "fontSize": "0.95em"}),
                            html.Span("· scores are percentile-ranked within your portfolio (0 = weakest, 100 = strongest)",
                                      style={"color": "#9CA3AF", "fontSize": "0.82em"}),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid",
                               "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                               "gap": "0.8em"},
                        children=[
                            html.Div([
                                html.Span("💰 Value  ", style={"fontWeight": "700", "color": NAVY}),
                                html.Span("Cheap on P/E, P/B, P/S vs. peers. High = potential upside if market re-rates."),
                            ]),
                            html.Div([
                                html.Span("🏛 Quality  ", style={"fontWeight": "700", "color": NAVY}),
                                html.Span("Strong ROE, margins, low debt. Compounds steadily, holds up in downturns."),
                            ]),
                            html.Div([
                                html.Span("📈 Growth  ", style={"fontWeight": "700", "color": NAVY}),
                                html.Span("Revenue and earnings expanding faster than peers (1Y + 3Y CAGR)."),
                            ]),
                            html.Div([
                                html.Span("🚀 Momentum  ", style={"fontWeight": "700", "color": NAVY}),
                                html.Span("12-1 month price return vs. peers. Tends to persist short-term."),
                            ]),
                            html.Div([
                                html.Span("🛡 Low Vol  ", style={"fontWeight": "700", "color": NAVY}),
                                html.Span("Lower realised volatility. Smoother ride, smaller drawdowns."),
                            ]),
                            html.Div([
                                html.Span("⭐ Overall  ", style={"fontWeight": "700", "color": NAVY}),
                                html.Span("Composite (Quality ×2, others ×1.5). Above 6 = solid. Below 4 = review."),
                            ]),
                        ],
                    ),
                    html.P(
                        "Scores are relative to your current portfolio — not the broader market. "
                        "A score of 80 means that stock ranks in the top 20% of your holdings on that factor.",
                        style={"marginTop": "0.9em", "color": "#9CA3AF",
                               "fontSize": "0.82em", "fontStyle": "italic"},
                    ),
                ],
            ),
        ], style={"margin": "0 0 2em"})

    if not rows:
        return no_update
    columns = FUND_COLS.get(tab, FUND_COLS["balance"])
    tab_labels = {"valuations": "Valuations", "balance": "Balance Sheet",
                  "income": "Income Statement", "cashflow": "Cash Flow"}
    return [
        html.H4(f"Fundamentals — {tab_labels.get(tab, '')}",
                style={"color": NAVY, "margin": "1.5em 0 0.8em 0"}),
        dash_table.DataTable(
            id="fundamentals-table",
            columns=columns,
            data=rows,
            merge_duplicate_headers=True,
            style_as_list_view=False,
            style_table={"overflowX": "auto", "borderRadius": "14px",
                         "boxShadow": "0 2px 12px rgba(0,0,0,0.06)"},
            style_header={**TABLE_HEADER, "textAlign": "center", "whiteSpace": "normal"},
            style_cell={**TABLE_CELL, "textAlign": "center", "minWidth": "72px",
                        "whiteSpace": "nowrap"},
            style_cell_conditional=[
                {"if": {"column_id": "ticker"},  "fontWeight": "700", "color": NAVY,
                 "textAlign": "left", "minWidth": "75px"},
                {"if": {"column_id": "company"}, "textAlign": "left", "minWidth": "150px",
                 "maxWidth": "200px", "overflow": "hidden", "textOverflow": "ellipsis"},
            ],
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": BG}],
            page_action="none",
            editable=False,
        ),
    ]


# ── Callback: Remove a weak stock from portfolio via factor-scores callout ────
@app.callback(
    Output("portfolio-store",     "data", allow_duplicate=True),
    Output("factor-scores-store", "data", allow_duplicate=True),
    Input({"type": "remove-weak-btn", "index": ALL}, "n_clicks"),
    State("portfolio-store",     "data"),
    State("factor-scores-store", "data"),
    prevent_initial_call=True,
)
def remove_weak_stock(n_clicks_list, store, factor_scores):
    if not any(n for n in n_clicks_list if n):
        return no_update, no_update
    ticker = ctx.triggered_id["index"]
    new_store = [r for r in store if r["ticker"] != ticker]
    if new_store:
        new_store, _ = recalc_weights(new_store)
    new_scores = [s for s in (factor_scores or []) if s.get("ticker") != ticker]
    return new_store, new_scores


# ── Correlation helpers ─────────────────────────────────────────────────────
def fetch_price_history(tickers: list, period: str = "2y") -> pd.DataFrame:
    """Download adjusted close prices for all tickers, return as DataFrame."""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
        if len(tickers) == 1:
            prices = prices.rename(columns={prices.columns[0]: tickers[0]})
    return prices.dropna(how="all")


def build_correlation_fig(tickers: list, price_df=None) -> tuple:
    """Returns (fig, error_msg). fig is a Plotly heatmap."""
    prices = price_df if price_df is not None else fetch_price_history(tickers)
    if prices.empty:
        return None, "Could not download price history for any ticker."

    returns = prices.pct_change(fill_method=None).dropna(how="all")

    # Keep only tickers with sufficient data
    valid_cols = [c for c in returns.columns if returns[c].notna().sum() >= 60]
    dropped    = [t for t in tickers if t not in valid_cols]
    if len(valid_cols) < 2:
        return None, "Need at least 2 tickers with sufficient price history."

    corr = returns[valid_cols].corr().round(2)
    labels = list(corr.columns)
    z      = corr.values.tolist()

    # Annotation text matrix
    text = [[f"{v:.2f}" for v in row] for row in corr.values]

    # Green = diversifying (negative), white = neutral, red = highly correlated (positive)
    colorscale = [
        [0.0,  "#059669"],  # -1 → emerald green
        [0.375, "#d1fae5"], # -0.25 → light green
        [0.5,  WHITE],      #  0 → white
        [0.625, "#fecaca"], # +0.25 → light red
        [1.0,  "#dc2626"],  # +1 → strong red
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "#111"},
        colorscale=colorscale,
        zmin=-1, zmax=1,
        showscale=False,
        hoverongaps=False,
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))

    n = len(labels)
    fig.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": FONT, "color": "#111"},
        margin=dict(l=20, r=20, t=20, b=20),
        height=max(380, 60 * n + 80),
        xaxis=dict(side="bottom", tickfont={"size": 12, "color": NAVY}, tickangle=-30),
        yaxis=dict(autorange="reversed", tickfont={"size": 12, "color": NAVY}),
    )

    warn = f" (excluded — insufficient history: {', '.join(dropped)})" if dropped else ""
    return fig, warn


# ── Callback: Sector / Country pie charts ────────────────────────────────────────
@app.callback(
    Output("sector-section", "children"),
    Input("fundamentals-store", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def run_sector_pies(fund_data, store):
    if not fund_data or not store:
        return no_update
    # Don't render if Yahoo Finance fetch failed for all tickers
    if all(r.get("company") == "Fetch error" for r in fund_data):
        return []
    val_map   = {r["ticker"]: float(str(r.get("total", 0)).replace(",", "")) for r in store}
    total_val = sum(val_map.values())
    if total_val == 0:
        return no_update

    sector_w  = {}
    country_w = {}
    for row in fund_data:
        t   = row.get("ticker", "")
        val = val_map.get(t, 0)
        if val == 0:
            continue
        s = row.get("sector")  or "Other"
        c = row.get("country") or "Other"
        if s == "\u2014": s = "Other"
        if c == "\u2014": c = "Other"
        sector_w[s]  = sector_w.get(s,  0) + val
        country_w[c] = country_w.get(c, 0) + val

    PIE_COLORS = [
        CYAN, NAVY, "#059669", "#D97706", "#e63946",
        "#6366f1", "#8b5cf6", "#f97316", "#14b8a6",
        "#ec4899", "#84cc16", "#06b6d4", "#0ea5e9",
    ]

    def _make_pie(data_dict, title):
        pairs = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        lbls  = [p[0] for p in pairs]
        vals  = [p[1] for p in pairs]
        fig   = go.Figure(go.Pie(
            labels=lbls, values=vals, hole=0.42,
            textinfo="label+percent",
            textfont=dict(family=FONT, size=11),
            hovertemplate="%{label}: \u20ac%{value:,.0f} (%{percent})<extra></extra>",
            marker=dict(
                colors=PIE_COLORS[:len(lbls)],
                line=dict(color=WHITE, width=2),
            ),
            sort=False,
        ))
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE,
            font=dict(family=FONT, color="#111"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
            showlegend=False,
            title=dict(text=title, font=dict(size=13, color=NAVY, family=FONT), x=0.5),
        )
        return fig

    sector_fig  = _make_pie(sector_w,  "Sector Mix")
    country_fig = _make_pie(country_w, "Country Mix")

    return html.Div([
        html.H4("Portfolio Composition",
                style={"color": NAVY, "margin": "1.5em 0 0.3em 0"}),
        html.P("Weighted by market value  \u00b7  data via Yahoo Finance",
               style={"color": "#aaa", "fontSize": "0.82em", "marginBottom": "0.8em"}),
        html.Div(
            style={"display": "flex", "gap": "1.2em", "flexWrap": "wrap"},
            children=[
                html.Div(
                    dcc.Graph(figure=sector_fig,  config={"displayModeBar": False}),
                    style={"flex": "1", "minWidth": "280px", "background": WHITE,
                           "borderRadius": "14px",
                           "boxShadow": "0 2px 12px rgba(0,0,0,0.06)", "padding": "0.5em"},
                ),
                html.Div(
                    dcc.Graph(figure=country_fig, config={"displayModeBar": False}),
                    style={"flex": "1", "minWidth": "280px", "background": WHITE,
                           "borderRadius": "14px",
                           "boxShadow": "0 2px 12px rgba(0,0,0,0.06)", "padding": "0.5em"},
                ),
            ],
        ),
    ])


# ── Callback: Correlation insights ───────────────────────────────────────────
@app.callback(
    Output("correlation-section", "children"),
    Input("price-history-store", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def run_correlation(price_data, store):
    if not price_data or not store:
        return no_update

    price_df  = pd.DataFrame(
        price_data["data"],
        columns=price_data["columns"],
        index=pd.to_datetime(price_data["index"]),
    )
    names_map = {r["ticker"]: r.get("company", r["ticker"]) for r in store}

    rets  = price_df.pct_change(fill_method=None).dropna(how="all")
    valid = [c for c in rets.columns if rets[c].notna().sum() >= 60]

    if len(valid) < 2:
        return html.P("Not enough price history to compute correlations.",
                      style={"color": "#888", "padding": "1em"})

    corr_m = rets[valid].corr().round(2)

    # ── Most correlated pair (always show top 1) ──────────────────────────────
    top_pair = None
    top_val  = -999.0
    for i, t1 in enumerate(valid):
        for j, t2 in enumerate(valid):
            if j <= i:
                continue
            v = float(corr_m.loc[t1, t2])
            if not np.isnan(v) and v > top_val:
                top_val  = v
                top_pair = (t1, t2, v)

    # ── 3 least correlated with weighted portfolio return ─────────────────────
    low_anchors = []
    try:
        total_val = sum(r.get("total", 0) for r in store if r.get("ticker") in valid)
        if total_val > 0:
            w_map = {r["ticker"]: r.get("total", 0) / total_val
                     for r in store if r.get("ticker") in valid}
            avail = [c for c in valid if c in w_map]
            if len(avail) >= 2:
                port_rets = sum(rets[c].fillna(0) * w_map[c] for c in avail)
                port_rets = port_rets.replace(0, float("nan")).dropna()
                corr_with_port = {
                    c: float(rets[c].reindex(port_rets.index).corr(port_rets))
                    for c in avail
                }
                low_anchors = sorted(
                    [(t, v) for t, v in corr_with_port.items() if not np.isnan(v)],
                    key=lambda x: x[1]
                )[:3]
    except Exception:
        pass

    # ── Card: most correlated pair ────────────────────────────────────────────
    if top_pair:
        t1, t2, pv = top_pair
        if pv >= 0.80:
            pair_color = RED
            pair_label = "High overlap — limited diversification"
        elif pv >= 0.55:
            pair_color = AMBER
            pair_label = "Moderate overlap"
        else:
            pair_color = EMERALD
            pair_label = "Well diversified pair"

        pair_card = html.Div(
            style={**CARD_STYLE, "flex": "1", "minWidth": "260px"},
            children=[
                html.Span("Most correlated pair",
                          style={"fontWeight": "700", "color": NAVY,
                                 "fontSize": "0.88em",
                                 "textTransform": "uppercase",
                                 "letterSpacing": "0.05em",
                                 "display": "block", "marginBottom": "1em"}),
                html.Div(
                    style={"display": "flex", "alignItems": "baseline",
                           "justifyContent": "space-between", "marginBottom": "1em"},
                    children=[
                        html.Span(f"{t1} × {t2}",
                                  style={"fontWeight": "800", "color": NAVY,
                                         "fontSize": "1.15em", "letterSpacing": "-0.01em"}),
                        html.Span(f"{pv:.2f}",
                                  style={"fontWeight": "800", "color": pair_color,
                                         "fontSize": "1.8em", "lineHeight": "1"}),
                    ],
                ),
                html.Span(
                    pair_label,
                    style={
                        "fontSize": "0.80em", "color": pair_color,
                        "background": f"{pair_color}1A",
                        "borderRadius": "999px",
                        "padding": "0.2em 0.85em",
                        "fontWeight": "700",
                        "border": f"1px solid {pair_color}40",
                    },
                ),
            ],
        )
    else:
        pair_card = None

    # ── Card: best diversifiers ───────────────────────────────────────────────
    if low_anchors:
        chips = []
        for t, v in low_anchors:
            chips.append(html.Div(
                style={
                    "background": "#F0FDF4",
                    "border": f"1.5px solid {EMERALD}40",
                    "borderRadius": "14px",
                    "padding": "0.8em 1em",
                    "flex": "1",
                    "minWidth": "100px",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "gap": "0.25em",
                    "textAlign": "center",
                },
                children=[
                    html.Span(t, style={"fontWeight": "800", "color": NAVY,
                                       "fontSize": "0.95em"}),
                    html.Span(f"ρ {v:+.2f}",
                              style={"fontWeight": "700", "color": EMERALD,
                                     "fontSize": "0.88em"}),
                ],
            ))

        anchor_card = html.Div(
            style={**CARD_STYLE, "flex": "1", "minWidth": "260px"},
            children=[
                html.Span("Best diversifiers",
                          style={"fontWeight": "700", "color": NAVY,
                                 "fontSize": "0.88em",
                                 "textTransform": "uppercase",
                                 "letterSpacing": "0.05em",
                                 "display": "block", "marginBottom": "0.5em"}),
                html.P(
                    "Least correlated with your overall portfolio.",
                    style={"color": "#9CA3AF", "fontSize": "0.82em",
                           "margin": "0 0 1em 0", "lineHeight": "1.5"},
                ),
                html.Div(
                    style={"display": "flex", "gap": "0.6em", "flexWrap": "wrap"},
                    children=chips,
                ),
            ],
        )
    else:
        anchor_card = None

    cards = [c for c in [pair_card, anchor_card] if c is not None]

    return [
        html.H4("Correlation Insights",
                style={"color": NAVY, "margin": "1.5em 0 0.3em 0"}),
        html.P("2-year daily returns  ·  price data via Yahoo Finance",
               style={"color": "#aaa", "fontSize": "0.82em", "marginBottom": "1em"}),
        html.Div(
            style={"display": "flex", "gap": "1em", "flexWrap": "wrap"},
            children=cards,
        ),
    ]


# ── Efficient Frontier helpers ─────────────────────────────────────────────────
RF_RATE   = 0.04
N_SIM     = 6000
TRADING_D = 252


def _portfolio_perf(weights, mean_ret, cov):
    ret    = float(np.dot(weights, mean_ret)) * TRADING_D
    vol    = float(np.sqrt(np.dot(weights, np.dot(cov, weights)))) * np.sqrt(TRADING_D)
    sharpe = (ret - RF_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def build_frontier_data(store: list, price_df=None):
    """Returns (fig, weights_ms, weights_mv, valid_tickers, cur_weights, error_msg)."""
    tickers = [r["ticker"] for r in store]
    prices  = price_df if price_df is not None else fetch_price_history(tickers, period="2y")
    if prices.empty:
        return None, None, None, None, None, "Could not download price history."

    rets    = prices.pct_change(fill_method=None).dropna(how="all")
    valid   = [c for c in rets.columns if rets[c].notna().sum() >= 60]
    dropped = [t for t in tickers if t not in valid]

    if len(valid) < 2:
        return None, None, None, None, None, "Need at least 2 tickers with sufficient price history."

    rets     = rets[valid].dropna()
    mean_ret = rets.mean().values
    cov      = rets.cov().values
    n        = len(valid)

    # Current weights by market value
    total_val = sum(r["total"] for r in store if r["ticker"] in valid)
    cur_w = np.array([
        next((r["total"] for r in store if r["ticker"] == t), 0.0) / total_val
        for t in valid
    ])
    cur_ret, cur_vol, cur_sharpe = _portfolio_perf(cur_w, mean_ret, cov)

    # Monte Carlo simulation
    rng        = np.random.default_rng(42)
    raw_w      = rng.dirichlet(np.ones(n), size=N_SIM)
    sim_ret    = (raw_w @ mean_ret) * TRADING_D
    sim_vol    = np.sqrt(np.einsum("ij,jk,ik->i", raw_w, cov, raw_w)) * np.sqrt(TRADING_D)
    sim_sharpe = (sim_ret - RF_RATE) / np.where(sim_vol > 0, sim_vol, 1e-9)

    # Frontier curve
    N_BINS  = 60
    bins    = np.linspace(sim_ret.min(), sim_ret.max(), N_BINS + 1)
    fvol, fret = [], []
    for i in range(N_BINS):
        mask = (sim_ret >= bins[i]) & (sim_ret < bins[i + 1])
        if mask.sum() > 0:
            fvol.append(float(sim_vol[mask].min()))
            fret.append(float((bins[i] + bins[i + 1]) / 2))
    if fvol:
        min_v_idx = int(np.argmin(fvol))
        fvol_eff  = fvol[min_v_idx:]
        fret_eff  = fret[min_v_idx:]
    else:
        fvol_eff, fret_eff = [], []

    # Special portfolio weights
    ms_idx     = int(np.argmax(sim_sharpe))
    mv_idx     = int(np.argmin(sim_vol))
    weights_ms = raw_w[ms_idx]
    weights_mv = raw_w[mv_idx]
    ms_ret, ms_vol = float(sim_ret[ms_idx]), float(sim_vol[ms_idx])
    mv_ret, mv_vol = float(sim_ret[mv_idx]), float(sim_vol[mv_idx])

    # Build chart
    hover_text = [
        f"Return: {r*100:.1f}%<br>Vol: {v*100:.1f}%<br>Sharpe: {s:.2f}"
        for r, v, s in zip(sim_ret, sim_vol, sim_sharpe)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_vol * 100, y=sim_ret * 100, mode="markers",
        name="Simulated portfolios",
        marker=dict(size=4, color=sim_sharpe,
                    colorscale=[[0.0, BG],[0.5,"#A8EBF0"],[1.0, CYAN]],
                    colorbar=dict(title="Sharpe", thickness=12, len=0.7, x=1.01),
                    opacity=0.55, showscale=True),
        text=hover_text, hovertemplate="%{text}<extra></extra>",
    ))
    if fvol_eff:
        fig.add_trace(go.Scatter(
            x=[v*100 for v in fvol_eff], y=[r*100 for r in fret_eff],
            mode="lines", name="Efficient frontier",
            line=dict(color=NAVY, width=2.5),
            hovertemplate="Frontier  Vol: %{x:.1f}%  Ret: %{y:.1f}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[ms_vol*100], y=[ms_ret*100], mode="markers+text", name="Max Sharpe",
        marker=dict(symbol="star", size=18, color="#f4a261", line=dict(color=NAVY, width=1.5)),
        text=["Max Sharpe"], textposition="top right", textfont=dict(size=11, color=NAVY),
        hovertemplate=f"Max Sharpe<br>Vol: {ms_vol*100:.1f}%<br>Return: {ms_ret*100:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[mv_vol*100], y=[mv_ret*100], mode="markers+text", name="Min Variance",
        marker=dict(symbol="diamond", size=14, color="#e63946", line=dict(color=NAVY, width=1.5)),
        text=["Min Variance"], textposition="top right", textfont=dict(size=11, color=NAVY),
        hovertemplate=f"Min Variance<br>Vol: {mv_vol*100:.1f}%<br>Return: {mv_ret*100:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[cur_vol*100], y=[cur_ret*100], mode="markers+text", name="Your portfolio",
        marker=dict(symbol="circle", size=16, color=CYAN, line=dict(color=NAVY, width=2)),
        text=["Your portfolio"], textposition="top left", textfont=dict(size=11, color=NAVY),
        hovertemplate=f"Your portfolio<br>Vol: {cur_vol*100:.1f}%<br>Return: {cur_ret*100:.1f}%<br>Sharpe: {cur_sharpe:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=WHITE, plot_bgcolor="#f9fbfc",
        font={"family": FONT, "color": "#111"}, height=520,
        margin=dict(l=20, r=60, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.8)", bordercolor=LGRAY, borderwidth=1),
        xaxis=dict(title="Annual Volatility (%)", gridcolor=LGRAY, zeroline=False, tickfont={"size": 11}),
        yaxis=dict(title="Annual Return (%)",     gridcolor=LGRAY, zeroline=False, tickfont={"size": 11}),
        hovermode="closest",
    )

    warn = f" (excluded — insufficient history: {', '.join(dropped)})" if dropped else ""
    return fig, weights_ms, weights_mv, valid, cur_w, warn


def _weights_table(valid_tickers, cur_w, weights_ms, weights_mv, store):
    """Build an HTML table comparing current vs optimal weights."""
    hdr_style = {**TABLE_HEADER, "textAlign": "center", "whiteSpace": "nowrap"}
    cell_style = {**TABLE_CELL, "textAlign": "center"}

    header = html.Tr([
        html.Th("Ticker",                style={**hdr_style, "textAlign": "left"}),
        html.Th("Current Weight",        style=hdr_style),
        html.Th("Max Sharpe Weight ★",   style=hdr_style),
        html.Th("Min Variance Weight ◆", style=hdr_style),
        html.Th("Rebalance (↑↓)",        style=hdr_style),
    ])

    body_rows = []
    for i, t in enumerate(valid_tickers):
        cur_pct  = round(float(cur_w[i])      * 100, 1)
        ms_pct   = round(float(weights_ms[i]) * 100, 1)
        mv_pct   = round(float(weights_mv[i]) * 100, 1)
        delta    = round(ms_pct - cur_pct, 1)
        if delta > 0.5:
            arrow_txt   = f"↑ +{delta:.1f}%"
            arrow_color = "#166534"
        elif delta < -0.5:
            arrow_txt   = f"↓ {delta:.1f}%"
            arrow_color = "#991b1b"
        else:
            arrow_txt   = "≈ Hold"
            arrow_color = "#888"

        row_bg = WHITE if i % 2 == 0 else BG
        body_rows.append(html.Tr([
            html.Td(t,                   style={**cell_style, "background": row_bg,
                                                "fontWeight": "700", "color": NAVY,
                                                "textAlign": "left"}),
            html.Td(f"{cur_pct:.1f}%",   style={**cell_style, "background": row_bg, "color": "#555"}),
            html.Td(f"{ms_pct:.1f}%",    style={**cell_style, "background": row_bg}),
            html.Td(f"{mv_pct:.1f}%",    style={**cell_style, "background": row_bg}),
            html.Td(arrow_txt,           style={**cell_style, "background": row_bg,
                                                "color": arrow_color, "fontWeight": "700"}),
        ]))

    return html.Div(
        html.Table(
            [html.Thead(header), html.Tbody(body_rows)],
            style={"width": "100%", "borderCollapse": "collapse",
                   "borderRadius": "14px", "overflow": "hidden", "tableLayout": "auto"},
        ),
        style={"overflowX": "auto", "borderRadius": "14px", "width": "100%",
               "boxShadow": "0 2px 12px rgba(0,0,0,0.06)", "background": WHITE},
    )


# ── Callback: Efficient Frontier ────────────────────────────────────────────────
@app.callback(
    Output("frontier-section", "children"),
    Output("frontier-store",   "data"),
    Input("price-history-store", "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def run_frontier(price_data, store):
    if not price_data or not store or len(store) < 2:
        return no_update, no_update

    price_df = pd.DataFrame(
        price_data["data"],
        columns=price_data["columns"],
        index=pd.to_datetime(price_data["index"]),
    )
    fig, weights_ms, weights_mv, valid, cur_w, warn = build_frontier_data(store, price_df=price_df)
    if fig is None:
        return html.P(warn, style={"color": RED}), {}

    warn_div = html.P(warn, style={"color": "#888", "fontSize": "0.83em"}) if warn else None

    interp = html.Div(
        style={"marginTop": "1.2em", "background": WHITE, "borderRadius": "12px",
               "padding": "1em 1.4em", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
               "fontSize": "0.86em", "color": "#555"},
        children=[
            html.Span("How to read: ", style={"fontWeight": "700", "color": NAVY}),
            html.Span("Each dot is a random weighting of your holdings. "),
            html.Span("The frontier curve ", style={"fontWeight": "700", "color": NAVY}),
            html.Span("marks the best return for each risk level. "),
            html.Span("★ Max Sharpe ", style={"color": "#f4a261", "fontWeight": "700"}),
            html.Span("= best risk-adjusted return.  "),
            html.Span("◆ Min Variance ", style={"color": "#e63946", "fontWeight": "700"}),
            html.Span("= lowest volatility.  "),
            html.Span("● Your portfolio ", style={"color": CYAN, "fontWeight": "700"}),
            html.Span("= where you sit today."),
            html.Br(),
            html.Span(
                f"Risk-free rate: {RF_RATE*100:.1f}%  ·  2-year daily returns  ·  {N_SIM:,} simulations.  "
                "Note: weights are historically optimal — treat as directional guidance, not a prediction.",
                style={"color": "#aaa", "fontSize": "0.9em"},
            ),
        ],
    )


    children = [
        html.H4("Markowitz Efficient Frontier", style={"color": NAVY, "margin": "1.5em 0 0.3em 0"}),
    ]
    if warn_div:
        children.append(warn_div)
    children.append(html.Div(
        dcc.Graph(id="frontier-graph", figure=fig, config={"displayModeBar": True,
                                      "modeBarButtonsToRemove": ["lasso2d", "select2d"]}),
        style={"background": WHITE, "borderRadius": "14px",
               "boxShadow": "0 2px 12px rgba(0,0,0,0.06)", "padding": "1em"},
    ))
    children.append(interp)

    # ── Two rebalancing buttons ──────────────────────────────────────────────
    btn_base = {
        "border": "none", "borderRadius": "10px", "padding": "0.75em 1.6em",
        "fontSize": "0.95em", "fontWeight": "700", "cursor": "pointer",
        "fontFamily": FONT, "letterSpacing": "0.01em",
        "transition": "transform 0.15s, box-shadow 0.15s",
    }
    children.append(html.Div(
        style={"display": "flex", "gap": "1em", "marginTop": "1.4em",
               "justifyContent": "center", "flexWrap": "wrap"},
        children=[
            html.Button(
                "★  Load Max Sharpe Portfolio",
                id="btn-max-sharpe", n_clicks=0,
                style={**btn_base, "background": "#f4a261", "color": NAVY,
                       "boxShadow": "0 3px 10px rgba(244,162,97,0.35)"},
            ),
            html.Button(
                "◆  Load Min Variance Portfolio",
                id="btn-min-var", n_clicks=0,
                style={**btn_base, "background": RED, "color": WHITE,
                       "boxShadow": "0 3px 10px rgba(230,57,70,0.30)"},
            ),
        ],
    ))

    frontier_store = {"valid": list(valid), "weights_ms": list(weights_ms),
                      "weights_mv": list(weights_mv), "cur_w": list(cur_w)}
    return children, frontier_store


# ── Callback: Max Sharpe / Min Variance rebalancing cards ─────────────────────
@app.callback(
    Output("frontier-rebalance-card", "children"),
    Input("btn-max-sharpe", "n_clicks"),
    Input("btn-min-var",    "n_clicks"),
    State("frontier-store",  "data"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def show_rebalance_card(ms_clicks, mv_clicks, frontier_store, store):
    from dash import ctx
    if not frontier_store or not frontier_store.get("valid") or not store:
        return no_update

    triggered = ctx.triggered_id
    if triggered == "btn-max-sharpe":
        target_w  = frontier_store["weights_ms"]
        title     = "★  Max Sharpe Rebalancing Plan"
        accent    = "#f4a261"
        subtitle  = "Weights that maximise risk-adjusted return (Sharpe ratio)"
    elif triggered == "btn-min-var":
        target_w  = frontier_store["weights_mv"]
        title     = "◆  Min Variance Rebalancing Plan"
        accent    = RED
        subtitle  = "Weights that minimise portfolio volatility"
    else:
        return no_update

    valid     = frontier_store["valid"]
    cur_w     = frontier_store["cur_w"]
    val_map   = {r["ticker"]: float(str(r.get("total", 0)).replace(",", "")) for r in store}
    name_map  = {r["ticker"]: r.get("company", r["ticker"]) for r in store}
    total_val = sum(val_map.get(t, 0) for t in valid)
    if total_val == 0:
        return no_update

    hdr = {**TABLE_HEADER, "textAlign": "center", "padding": "0.5em 0.8em",
           "whiteSpace": "nowrap"}
    rows = []
    for i, t in enumerate(valid):
        cur_pct  = round(float(cur_w[i])      * 100, 1)
        tgt_pct  = round(float(target_w[i])   * 100, 1)
        delta    = round(tgt_pct - cur_pct, 1)
        cur_val  = val_map.get(t, 0)
        tgt_val  = total_val * float(target_w[i])
        diff_val = tgt_val - cur_val

        if delta > 0.5:
            delta_txt   = f"↑ +{delta:.1f}%"
            delta_color = "#166534"
        elif delta < -0.5:
            delta_txt   = f"↓ {delta:.1f}%"
            delta_color = "#991b1b"
        else:
            delta_txt   = "≈ Hold"
            delta_color = "#888"

        if diff_val > 5:
            action_txt   = f"Buy  +{diff_val:,.0f}"
            action_color = "#166534"
            action_bg    = "#f0fdf4"
        elif diff_val < -5:
            action_txt   = f"Sell  {diff_val:,.0f}"
            action_color = "#991b1b"
            action_bg    = "#fff1f2"
        else:
            action_txt   = "Hold"
            action_color = "#888"
            action_bg    = BG

        bg = WHITE if i % 2 == 0 else BG
        rows.append(html.Tr([
            html.Td(html.Div([
                html.Span(t, style={"fontWeight": "700", "color": NAVY,
                                     "display": "block", "fontSize": "0.9em"}),
                html.Span(name_map.get(t, ""), style={"color": "#9CA3AF",
                                                       "fontSize": "0.76em"}),
            ]), style={"padding": "0.45em 0.8em", "background": bg}),
            html.Td(f"{cur_pct:.1f}%",  style={"textAlign": "center", "padding": "0.45em 0.8em",
                                                 "background": bg, "color": "#555",
                                                 "fontSize": "0.88em"}),
            html.Td(f"{tgt_pct:.1f}%",  style={"textAlign": "center", "padding": "0.45em 0.8em",
                                                 "background": bg, "fontWeight": "700",
                                                 "color": NAVY, "fontSize": "0.88em"}),
            html.Td(delta_txt,           style={"textAlign": "center", "padding": "0.45em 0.8em",
                                                 "background": bg, "fontWeight": "700",
                                                 "color": delta_color, "fontSize": "0.88em"}),
            html.Td(action_txt,          style={"textAlign": "center", "padding": "0.45em 0.8em",
                                                 "background": action_bg, "fontWeight": "700",
                                                 "color": action_color, "fontSize": "0.88em",
                                                 "borderRadius": "6px"}),
        ]))

    table = html.Div(
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Holding",       style={**hdr, "textAlign": "left"}),
                    html.Th("Current",       style=hdr),
                    html.Th("Target",        style=hdr),
                    html.Th("Δ Weight",      style=hdr),
                    html.Th("Action",        style=hdr),
                ])),
                html.Tbody(rows),
            ],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        style={"overflowX": "auto", "borderRadius": "12px", "background": WHITE,
               "boxShadow": "0 2px 12px rgba(0,0,0,0.06)"},
    )

    note = html.P(
        "Action amounts are based on your current portfolio value. "
        "Treat as directional guidance — not financial advice.",
        style={"color": "#9CA3AF", "fontSize": "0.80em", "marginTop": "0.8em",
               "fontStyle": "italic"},
    )

    return html.Div(
        style={**CARD_STYLE, "padding": "1.4em 1.6em", "marginTop": "1.2em",
               "borderLeft": f"4px solid {accent}", "animation": "fadeUp 0.35s ease both"},
        className="reveal-card",
        children=[
            html.H4(title,    style={"color": NAVY, "margin": "0 0 0.2em 0"}),
            html.P(subtitle,  style={"color": "#9CA3AF", "fontSize": "0.83em",
                                     "margin": "0 0 1em 0"}),
            table,
            note,
        ],
    )


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8051))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
