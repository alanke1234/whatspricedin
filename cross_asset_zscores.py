"""
Streamlit Dashboard — Cross-Asset Narrative Z-Score (with Correlations & Headlines)
==================================================================================

Features
- Pulls 25y daily data for SPY, QQQ, RSP, US02Y (FRED DGS2), US05Y (^FVX), US10Y (^TNX), US30Y (^TYX), DXY (DX-Y.NYB)
- Baselines: mean/median/std of daily moves
  • % daily returns for price assets (SPY/QQQ/RSP/DXY)
  • daily basis-point changes for yields (2Y/5Y/10Y/30Y)
- Date selection (up to today): analyze **as of** the chosen date using last available observations on/before it
- Snapshot tab: KPI cards, ranked heatmap table, Z-score bars
- Correlations tab: rolling correlation matrix of daily moves on a selectable window; top positive/negative pairs
- Headlines tab: fetches Yahoo Finance headlines per asset (via proxy tickers) for the selected date, mapped to moves
- Explorer tab: per-asset level, daily move, and Z-score time series
- Exports tab: CSV downloads for baseline & snapshot

Run
----
1) Install deps:
   pip install streamlit yfinance pandas numpy pandas_datareader python-dateutil altair feedparser

2) Save this file as: dashboard_app.py

3) Launch:
   streamlit run dashboard_app.py

Notes
- 2Y yield: FRED DGS2 (updates with ~1 business-day lag). For dates where a series lags (e.g., today), the app **backfills to the last available value on/before the selected date**.
- Yahoo yields (^FVX/^TNX/^TYX) are percent*10; the app divides by 10 → percent.
- DXY headlines are proxied via UUP ETF; Treasury yields via SHY/IEF/TLT as appropriate.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dateutil import tz
import requests

# --- TradingEconomics (TE) helpers ---
def _te_token():
    try:
        sec = st.secrets["tradingeconomics"]
        return sec.get("token") or sec.get("api_key")
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=300)
def te_get(path: str, params: dict | None = None) -> list[dict]:
    token = _te_token()
    if not token:
        raise RuntimeError("TradingEconomics API key missing. Add it under [tradingeconomics] in .streamlit/secrets.toml")
    url = f"https://api.tradingeconomics.com/{path.lstrip('/')}"
    p = dict(params or {})
    p.setdefault("format", "json")
    # Basic auth if token is user:pass, else pass as ?token=
    auth = None
    if ":" in token:
        user, pwd = token.split(":", 1)
        auth = (user, pwd)
    else:
        p["token"] = token
    r = requests.get(url, params=p, auth=auth, timeout=20)
    if r.status_code in (401, 403) and auth is not None:
        r = requests.get(url, params={**p, "token": token}, timeout=20)
    r.raise_for_status()
    try:
        data = r.json()
        if isinstance(data, dict):
            data = data.get("data", [])
        return data or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=300)
def te_calendar(day: pd.Timestamp, buf_days: int = 0, countries: list[str] | None = None) -> pd.DataFrame:
    d1 = (pd.Timestamp(day).normalize() - pd.Timedelta(days=buf_days)).date().isoformat()
    d2 = (pd.Timestamp(day).normalize() + pd.Timedelta(days=buf_days)).date().isoformat()
    params = {"d1": d1, "d2": d2}
    if countries:
        params["c"] = ",".join(countries)
    rows = te_get("/calendar", params)
    if not rows:
        return pd.DataFrame(columns=["DatetimeUTC","Country","Category","Event","Importance","Actual","Forecast","Previous"])
    out = []
    for e in rows:
        dt = pd.to_datetime(e.get("Date") or e.get("date"), utc=True, errors="coerce")
        out.append({
            "DatetimeUTC": dt,
            "Country": e.get("Country") or e.get("country"),
            "Category": e.get("Category") or e.get("category"),
            "Event": e.get("Event") or e.get("event"),
            "Importance": e.get("Importance") or e.get("importance") or e.get("Impact"),
            "Actual": e.get("Actual") or e.get("actual"),
            "Forecast": e.get("Forecast") or e.get("forecast"),
            "Previous": e.get("Previous") or e.get("previous"),
            "Reference": e.get("Reference") or e.get("reference"),
            "Link": e.get("Link") or e.get("URL") or "",
        })
    df = pd.DataFrame(out).dropna(subset=["DatetimeUTC"]).sort_values("DatetimeUTC")
    return df

@st.cache_data(show_spinner=False)
def event_moves_around_date(event_day: pd.Timestamp, assets: list[str], data_map: dict[str, pd.Series], baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ed = pd.Timestamp(event_day).normalize()
    for a in assets:
        if a not in data_map:
            continue
        ser = data_map[a].sort_index().dropna()
        try:
            idx_event = ser.index.get_loc(ed, method="pad")
        except Exception:
            continue
        i0 = idx_event if isinstance(idx_event, int) else None
        if i0 is None or i0 < 1 or i0 >= len(ser) - 1:
            continue
        prev_close = float(ser.iloc[i0 - 1])
        event_close = float(ser.iloc[i0])
        next_close = float(ser.iloc[i0 + 1])
        units = baseline_df.loc[a, "units"] if a in baseline_df.index else "percent"
        if units == "percent":
            pre = (event_close / prev_close - 1.0) * 100.0
            post = (next_close / event_close - 1.0) * 100.0
        else:
            pre = (event_close - prev_close) * 100.0
            post = (next_close - event_close) * 100.0
        rows.append({"Asset": a, "PreMove": pre, "PostMove": post, "Units": units})
    return pd.DataFrame(rows)


# ----------------------- UI THEME / STYLES -----------------------
ACCENT = "#7c3aed"  # violet-600

def inject_css(theme: str = "auto", accent: str = ACCENT):
    # Basic theming via CSS vars + component styling
    dark_bg = "#0b1220"; dark_card = "#111827"; dark_text = "#e5e7eb"; dark_muted = "#9ca3af"
    light_bg = "#f8fafc"; light_card = "#ffffff"; light_text = "#0f172a"; light_muted = "#475569"
    # Choose palette
    is_dark = (theme == "dark")
    bg = dark_bg if is_dark else light_bg
    card = dark_card if is_dark else light_card
    text = dark_text if is_dark else light_text
    muted = dark_muted if is_dark else light_muted

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
        :root {{ --bg: {bg}; --card: {card}; --text: {text}; --muted: {muted}; --accent: {accent}; }}
        .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
        /* Header bar */
        .app-header {{
            background: radial-gradient(1200px 400px at 10% -10%, rgba(124,58,237,0.25), transparent),
                        linear-gradient(135deg, rgba(124,58,237,0.12), rgba(56,189,248,0.10));
            border: 1px solid rgba(124,58,237,0.2);
            box-shadow: 0 8px 30px rgba(2,8,23,0.15);
            color: var(--text);
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 18px;
        }}
        .app-header h1 {{ margin: 0; font-weight: 700; letter-spacing: 0.2px; }}
        .app-header .sub {{ color: var(--muted); font-size: 0.95rem; margin-top: 6px; }}
        /* Cards / tables */
        div[data-testid="stMetric"] {{
            background: var(--card);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(2,8,23,0.06);
        }}
        div[data-testid="stMetric"] > label p {{ color: var(--muted) !important; font-weight: 600; }}
        div[data-testid="stMetricValue"] {{ color: var(--text) !important; }}
        .stDownloadButton button, .stButton button {{
            border-radius: 12px !important; border: 1px solid rgba(148,163,184,0.35) !important;
            background: linear-gradient(180deg, rgba(124,58,237,0.15), rgba(124,58,237,0.05));
            color: var(--text) !important; font-weight: 600 !important;
        }}
        .stTabs [data-baseweb="tab-list"] button {{ font-weight: 700; }}
        .stTabs [data-baseweb="tab"]:hover {{ color: var(--accent); }}
        .stTabs [aria-selected="true"] {{ color: var(--accent); border-color: var(--accent); }}
        /* Dataframe */
        .stDataFrame {{ border: 1px solid rgba(148,163,184,0.25); border-radius: 14px; }}
        /* Background */
        .stApp {{ background: var(--bg); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_altair_theme(dark: bool = False):
    axis_color = "#9ca3af" if dark else "#64748b"
    grid_color = "#334155" if dark else "#e2e8f0"
    label_color = "#e5e7eb" if dark else "#0f172a"
    title_color = label_color
    palette = ["#22c55e", "#3b82f6", "#a78bfa", "#f59e0b", "#ef4444", "#10b981"]
    def _theme():
        return {
            "config": {
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelColor": label_color,
                    "titleColor": title_color,
                    "domainColor": axis_color,
                    "gridColor": grid_color,
                },
                "legend": {"labelColor": label_color, "titleColor": title_color},
                "range": {"category": palette},
                "font": "Inter",
            }
        }
    alt.themes.register("custom_theme", _theme)
    alt.themes.enable("custom_theme")

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# ----------------------- Config -----------------------
REFRESH_NOTE = "Intraday mode compares last price to prior close (today only). US02Y uses ZT=F futures proxy."

EQUITY_TICKERS = ["SPY", "QQQ", "RSP", "^VIX"]
YIELD_TICKERS_YF = {
    "MOVE": "^MOVE",
    "US05Y": "^FVX",
    "US10Y": "^TNX",
    "US30Y": "^TYX",
}
FX_TICKERS = {"DXY": "DX-Y.NYB"}
DEFAULT_LOOKBACK_YEARS = 25

# News proxy tickers (for Yahoo Finance headlines)
NEWS_PROXY: Dict[str, str] = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "RSP": "RSP",
    "US02Y": "SHY",   # 1-3y treasury ETF proxy for 2Y headlines
    "US05Y": "IEF",   # 7-10y but decent macro/rates proxy
    "US10Y": "IEF",
    "US30Y": "TLT",
    "DXY": "UUP",     # Dollar index ETF proxy
    "VIX": "VIX",
}

@dataclass
class AssetSpec:
    name: str
    source: str  # 'yahoo' or 'fred'
    ticker: str
    kind: str    # 'price' or 'yield'

# ----------------------- Helpers -----------------------

def _utc_today_date() -> datetime:
    return datetime.now(timezone.utc).date()


def _start_date(years: int) -> str:
    end = _utc_today_date()
    try:
        start = end.replace(year=end.year - years)
        return start.isoformat()
    except Exception:
        start2 = end - timedelta(days=365 * years + years // 4)
        return start2.isoformat()


def build_asset_list() -> List[AssetSpec]:
    assets: List[AssetSpec] = []
    for t in EQUITY_TICKERS:
        assets.append(AssetSpec(name=t, source="yahoo", ticker=t, kind="price"))
    for label, yftick in YIELD_TICKERS_YF.items():
        assets.append(AssetSpec(name=label, source="yahoo", ticker=yftick, kind="yield"))
    for label, yftick in FX_TICKERS.items():
        assets.append(AssetSpec(name=label, source="yahoo", ticker=yftick, kind="price"))
    return assets

# Intraday proxy mapping (used only in intraday mode)
INTRADAY_PROXY = {
    "US02Y": AssetSpec(name="US02Y*", source="yahoo", ticker="ZT=F", kind="price"),  # 2Y note futures
}

def build_intraday_assets(base_assets: List[AssetSpec]) -> List[AssetSpec]:
    out: List[AssetSpec] = []
    for a in base_assets:
        if a.name in INTRADAY_PROXY:
            out.append(INTRADAY_PROXY[a.name])
        else:
            out.append(a)
    return out

@st.cache_data(show_spinner=False)
def fetch_yahoo_prices(symbols: List[str], start: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    df = yf.download(symbols, start=start, interval="1d", auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

@st.cache_data(show_spinner=False)
def fetch_fred_series(series_id: str, start: str) -> pd.Series:
    if pdr is None:
        raise RuntimeError("pandas_datareader not installed; cannot fetch FRED series.")
    ser = pdr.DataReader(series_id, "fred", start=start)
    return ser[series_id].astype(float)

@st.cache_data(show_spinner=False)
def load_all_data(assets: List[AssetSpec], start: str) -> Dict[str, pd.Series]:
    yahoo_syms = [a.ticker for a in assets if a.source == "yahoo"]
    by_name: Dict[str, pd.Series] = {}

    yf_df = fetch_yahoo_prices(yahoo_syms, start)
    for a in assets:
        if a.source == "yahoo" and a.ticker in yf_df.columns:
            by_name[a.name] = yf_df[a.ticker].dropna()

    for a in assets:
        if a.source == "fred":
            ser = fetch_fred_series(a.ticker, start)
            by_name[a.name] = ser.dropna()

    # Normalize Yahoo yields to percent
    for name in list(by_name.keys()):
        if name in ("US05Y", "US10Y", "US30Y"):
            by_name[name] = by_name[name].astype(float) / 10.0

    return by_name

@st.cache_data(show_spinner=False)
def compute_daily_moves(series: pd.Series, kind: str) -> pd.Series:
    s = series.sort_index().dropna().astype(float)
    if kind == "price":
        return (s.pct_change() * 100.0).dropna()
    elif kind == "yield":
        return (s.diff() * 100.0).dropna()  # percent -> bps
    else:
        raise ValueError("unknown kind")

@st.cache_data(show_spinner=False)
def build_baselines(assets: List[AssetSpec], data: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for a in assets:
        if a.name not in data:
            continue
        moves = compute_daily_moves(data[a.name], a.kind)
        rows.append({
            "asset": a.name,
            "kind": a.kind,
            "mean": moves.mean(),
            "median": moves.median(),
            "std": moves.std(ddof=1),
            "n_obs": int(moves.shape[0]),
            "units": "percent" if a.kind == "price" else "bp",
        })
    # Include baseline placeholders for intraday proxies mapped back to original names
    baseline = pd.DataFrame(rows).set_index("asset").sort_index()
    if "US02Y" in baseline.index and "US02Y*" not in baseline.index:
        # allow z-scores for the proxy to use US02Y's baseline units/statistics
        proxy_row = baseline.loc[["US02Y"]].copy()
        proxy_row.index = ["US02Y*"]
        baseline = pd.concat([baseline, proxy_row]).sort_index()
    return baseline

@st.cache_data(show_spinner=False)
def compute_snapshot(assets: List[AssetSpec], data: Dict[str, pd.Series], baseline: pd.DataFrame, asof_override: pd.Timestamp | None = None) -> pd.DataFrame:
    all_moves = {}
    last_dates = []
    for a in assets:
        if a.name in data:
            mv = compute_daily_moves(data[a.name], a.kind)
            if not mv.empty:
                all_moves[a.name] = mv
                last_dates.append(mv.index.max())
    if not last_dates:
        return pd.DataFrame()

    asof = pd.to_datetime(asof_override).normalize() if asof_override is not None else min(last_dates)

    rows = []
    for a in assets:
        if a.name not in all_moves or a.name not in baseline.index:
            continue
        ser = all_moves[a.name]
        sub = ser.loc[:asof]
        if sub.empty:
            continue
        val = float(sub.iloc[-1])
        dte = sub.index[-1]
        stats = baseline.loc[a.name]
        z = (val - stats["mean"]) / stats["std"] if stats["std"] != 0 else np.nan
        rows.append({
            "date": pd.to_datetime(dte).date(),
            "asset": a.name,
            "kind": a.kind,
            "move": val,
            "units": stats["units"],
            "z": z,
        })
    snap = pd.DataFrame(rows).sort_values(by="z", key=lambda s: s.abs(), ascending=False)
    return snap

# ------- Intraday snapshot (today vs prior close) -------
@st.cache_data(show_spinner=False)
def compute_intraday_snapshot(assets: List[AssetSpec], baseline: pd.DataFrame) -> pd.DataFrame:
    """Compute moves since prior close for today, using last price. Uses intraday proxies where specified."""
    rows = []
    today = _utc_today_date()
    for a in assets:
        # Get prior close and last price
        try:
            t = yf.Ticker(a.ticker)
            hist = t.history(period="5d", interval="1d")
            if hist.empty or hist.shape[0] < 2:
                continue
            # Last row may be today (even intraday). Prior close is the previous row's Close
            last_row = hist.tail(1)
            prev_row = hist.tail(2).head(1)
            prev_close = float(prev_row["Close"].iloc[0])

            # Attempt to get last price; fallback to last close
            last_price = None
            try:
                fi = getattr(t, "fast_info", None)
                if fi and getattr(fi, "last_price", None) is not None:
                    last_price = float(fi.last_price)
            except Exception:
                last_price = None
            if last_price is None:
                last_price = float(last_row["Close"].iloc[0])

            if a.kind == "price":
                move = (last_price / prev_close - 1.0) * 100.0  # percent
                units = "percent"
            else:  # yield
                # Yahoo yield indices report percent*10; try to use the same convention
                # Use last and prior close from the same series
                # Convert to percent if needed
                if a.ticker in ["^FVX", "^TNX", "^TYX"]:
                    last_pct = last_price / 10.0
                    prev_pct = prev_close / 10.0
                else:
                    last_pct = last_price
                    prev_pct = prev_close
                move = (last_pct - prev_pct) * 100.0  # bp
                units = "bp"

            if a.name not in baseline.index:
                continue
            stats = baseline.loc[a.name]
            z = (move - stats["mean"]) / stats["std"] if stats["std"] != 0 else np.nan
            rows.append({
                "date": today,
                "asset": a.name,
                "kind": a.kind,
                "move": move,
                "units": units,
                "z": z,
            })
        except Exception:
            continue
    snap = pd.DataFrame(rows).sort_values(by="z", key=lambda s: s.abs(), ascending=False)
    return snap

@st.cache_data(show_spinner=False)
def build_moves_panel(assets: List[AssetSpec], data: Dict[str, pd.Series]) -> pd.DataFrame:
    """Return a wide DataFrame of daily moves for all assets (index=date, columns=asset)."""
    frames = []
    for a in assets:
        if a.name not in data:
            continue
        mv = compute_daily_moves(data[a.name], a.kind)
        mv = mv.rename(a.name)
        frames.append(mv)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    return out

@st.cache_data(show_spinner=False)
def rolling_corr_matrix(moves_wide: pd.DataFrame, asof: pd.Timestamp, window: int) -> pd.DataFrame:
    """Compute rolling correlation over the last `window` observations up to `asof`."""
    sub = moves_wide.loc[:asof].tail(window)
    if sub.shape[0] < 5:
        return pd.DataFrame()
    return sub.corr()

@st.cache_data(show_spinner=False)
def melt_corr(df_corr: pd.DataFrame) -> pd.DataFrame:
    if df_corr.empty:
        return df_corr
    m = df_corr.reset_index().melt(id_vars=df_corr.index.name or "index")
    m.columns = ["x", "y", "corr"]
    return m

@st.cache_data(show_spinner=False)
def fetch_asset_headlines(asset: str, date_sel: datetime, buffer_days: int = 0, nearest_backfill_days: int = 3) -> pd.DataFrame:
    """Fetch headlines for an asset (via proxy ticker) near `date_sel`.
    Strategy:
      1) Try yfinance Ticker.news (usually ~recent items only)
      2) Fallback to Yahoo Finance RSS feeds via feedparser
      3) Filter to selected date ± buffer_days; if none, backfill to nearest within ±nearest_backfill_days
    Returns columns: time (UTC), title, link, publisher, proxy, source
    """
    proxy = NEWS_PROXY.get(asset, asset)
    rows = []

    # --- yfinance news ---
    try:
        t = yf.Ticker(proxy)
        items = t.news or []
    except Exception:
        items = []
    for it in items:
        ts = it.get("providerPublishTime")
        if ts is None:
            continue
        dt = pd.to_datetime(ts, unit="s", utc=True)
        rows.append({
            "time": dt,
            "title": it.get("title", ""),
            "link": it.get("link", ""),
            "publisher": (it.get("publisher", "") or it.get("provider", "")),
            "proxy": proxy,
            "source": "yfinance",
        })

    # --- Yahoo RSS fallback ---
    try:
        import feedparser  # type: ignore
        rss_urls = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={proxy}&region=US&lang=en-US",
            f"https://finance.yahoo.com/rss/headline?s={proxy}",
        ]
        for url in rss_urls:
            d = feedparser.parse(url)
            for e in d.entries:
                # best-effort timestamp parsing
                dt = pd.NaT
                if hasattr(e, "published_parsed") and e.published_parsed:
                    try:
                        dt = pd.to_datetime(datetime(*e.published_parsed[:6]), utc=True)
                    except Exception:
                        dt = pd.NaT
                rows.append({
                    "time": dt,
                    "title": getattr(e, "title", ""),
                    "link": getattr(e, "link", ""),
                    "publisher": getattr(e, "source", "") or getattr(e, "author", "") or "Yahoo RSS",
                    "proxy": proxy,
                    "source": "rss",
                })
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Clean & sort
    if "time" in df.columns:
        df = df.dropna(subset=["time"])  # keep rows with timestamps only
    df = df.sort_values("time", ascending=False)

    # Date window filtering (UTC)
    start = pd.Timestamp(date_sel).tz_localize("UTC") - pd.Timedelta(days=buffer_days)
    end = pd.Timestamp(date_sel).tz_localize("UTC") + pd.Timedelta(days=buffer_days + 1)
    sel = df[(df["time"] >= start) & (df["time"] < end)]

    if sel.empty and nearest_backfill_days > 0:
        # If nothing on the day, look within ±nearest_backfill_days
        start2 = start - pd.Timedelta(days=nearest_backfill_days)
        end2 = end + pd.Timedelta(days=nearest_backfill_days)
        sel = df[(df["time"] >= start2) & (df["time"] < end2)]

    return sel    
    try:
        t = yf.Ticker(proxy)
        items = t.news or []
    except Exception:
        items = []

    if not items:
        return pd.DataFrame(columns=["time", "title", "link", "publisher", "proxy"])

    rows = []
    # Normalize to date (UTC)
    for it in items:
        ts = it.get("providerPublishTime")
        if ts is None:
            continue
        dt = pd.to_datetime(ts, unit="s", utc=True)
        rows.append({
            "time": dt,
            "title": it.get("title", ""),
            "link": it.get("link", ""),
            "publisher": (it.get("publisher", "") or it.get("provider", "")),
            "proxy": proxy,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Filter to the selected calendar date in UTC
    day_start = pd.Timestamp(date_sel).tz_localize("UTC")
    day_end = day_start + pd.Timedelta(days=1)
    df = df[(df["time"] >= day_start) & (df["time"] < day_end)].sort_values("time", ascending=False)
    return df

# ----------------------- UI -----------------------
st.set_page_config(page_title="Narrative Z-Score Dashboard", page_icon="📈", layout="wide")
inject_css(theme="dark")  # default dark; will update below
set_altair_theme(dark=True)

st.markdown(
    """
    <div class="app-header">
      <h1>Cross-Asset Narrative Z‑Score Dashboard</h1>
      <div class="sub">Normalized cross‑asset moves, correlations, headlines — ready for the web.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    theme_choice = st.radio("Theme", ["Dark", "Light"], horizontal=True, index=0)
    is_dark = theme_choice == "Dark"
    inject_css("dark" if is_dark else "light")
    set_altair_theme(dark=is_dark)
    years = st.slider("History lookback (years)", min_value=5, max_value=30, value=DEFAULT_LOOKBACK_YEARS, step=1)
    st.caption("Used to build baselines; longer history = more stable distribution.")

assets = build_asset_list()
start = _start_date(years)

with st.spinner("Loading data & building baselines..."):
    data = load_all_data(assets, start)
    baseline = build_baselines(assets, data)

# Determine common date range for which moves exist across assets
move_first_dates = []
move_last_dates = []
for a in assets:
    if a.name in data:
        mv = compute_daily_moves(data[a.name], a.kind)
        if not mv.empty:
            move_first_dates.append(mv.index.min())
            move_last_dates.append(mv.index.max())

if not move_first_dates:
    st.error("No data available for move computations.")
    st.stop()

common_start = max(move_first_dates)
common_end = min(move_last_dates)

with st.sidebar:
    today = _utc_today_date()
    sel_date = st.date_input(
        "Analysis date (uses last available data on/before this date)",
        value=min(common_end.date(), today),
        min_value=common_start.date(),
        max_value=today,
    )
    intraday_ok = sel_date == today
    intraday = st.checkbox("Intraday (today)", value=False, disabled=not intraday_ok, help=REFRESH_NOTE)
    if intraday:
        st.caption("US02Y is proxied by ZT=F (2y futures) in intraday mode; Z-scores still use US02Y baseline.")
        st.button("Refresh now"),
        value=min(common_end.date(), today),
        min_value=common_start.date(),
        max_value=today,
    

asof_ts = pd.Timestamp(sel_date)
if intraday:
    intraday_assets = build_intraday_assets(assets)
    snapshot = compute_intraday_snapshot(intraday_assets, baseline)
else:
    snapshot = compute_snapshot(assets, data, baseline, asof_ts)

# Pre-compute wide moves panel for correlations
moves_wide = build_moves_panel(assets, data)

# ----------------------- Tabs -----------------------
TAB_SNAPSHOT, TAB_CORR, TAB_NEWS, TAB_CAL, TAB_EXPLORE, TAB_EXPORT = st.tabs([
    "Snapshot", "Correlations", "Headlines", "Calendar", "Explorer", "Exports"
])

with TAB_SNAPSHOT:
    asof = pd.to_datetime(sel_date).date()
    st.subheader(f"Snapshot as of {asof} (values use last available data on/before this date)")

    if snapshot.empty:
        st.warning("No snapshot available for the selected date.")
    else:
        # KPI cards
        kpi_cols = st.columns(len(snapshot))
        for i, (_, row) in enumerate(snapshot.iterrows()):
            mv = f"{row['move']:+.2f}%" if row["units"] == "percent" else f"{row['move']:+.1f} bp"
            kpi_cols[i].metric(label=row["asset"], value=f"Z {row['z']:+.2f}", delta=mv)

        st.divider()

        # Ranked table with heat styling
        st.markdown("### Ranked Moves (by |Z|)")
        disp = snapshot[["asset","kind","move","units","z"]].copy()
        disp.loc[disp["units"]=="percent","move"] = disp.loc[disp["units"]=="percent","move"].map(lambda x: f"{x:+.2f}%")
        disp.loc[disp["units"]=="bp","move"] = disp.loc[disp["units"]=="bp","move"].map(lambda x: f"{x:+.1f} bp")
        disp["z"] = disp["z"].map(lambda x: f"{x:+.2f}")

        def z_color(val):
            try:
                v = float(val)
            except Exception:
                return ""
            a = min(abs(v)/3, 1.0)
            return (f"background-color: rgba(0, 200, 0, {a});" if v >= 0
                    else f"background-color: rgba(200, 0, 0, {a}); color: white;")

        st.dataframe(disp.style.applymap(z_color, subset=["z"]).hide(axis="index"), use_container_width=True)

        st.markdown("### Z-Score Bar Chart (sorted by |Z|)")
        bar_df = snapshot.copy()
        bar_df["sign"] = np.where(bar_df["z"] >= 0, "Positive", "Negative")
        bar_df = bar_df.sort_values(by="z", key=lambda s: s.abs(), ascending=False)
        bar = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("z:Q", title="Z-Score"),
                y=alt.Y("asset:N", sort=list(bar_df["asset"].values), title="Asset"),
                color=alt.Color("sign:N", legend=None),
                tooltip=["asset","z","move","units"],
            )
            .properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)

with TAB_CORR:
    st.subheader("Rolling Correlations of Daily Moves")

    if moves_wide.empty:
        st.info("No moves series available to compute correlations.")
    else:
        # Build Z-scores from the 25y baseline (same dates as moves_wide)
        means = baseline['mean'].reindex(moves_wide.columns)
        stds = baseline['std'].reindex(moves_wide.columns).replace(0, np.nan)
        z_wide = (moves_wide - means) / stds

        # --- Inputs ---
        # Up to ~10y (≈2520 trading days) or the available history prior to selected date
        avail_rows = moves_wide.loc[:asof_ts].shape[0]
        max_window = int(min(avail_rows, 252 * 10))
        if max_window < 20:
            st.warning("Not enough data prior to the selected date to compute correlations.")
            window = None
        else:
            mode = st.radio(
                "Correlation input",
                ["Raw moves", "Z-scores (25y baseline)", "Tail-weighted (|Z|)"],
                index=0,
                horizontal=True,
                help="Pearson correlation is invariant to scaling; Z-scores mainly help for tail filters/weights.",
            )

            default_win = min(252, max_window)
            window = st.slider(
                "Window (trading days)",
                min_value=20,
                max_value=max_window,
                value=default_win,
                step=5,
                help="~252 ≈ 1 year; slider capped at ~10y or available history.",
            )

            tail_k = st.slider(
                "Tail filter: keep days with max |Z| ≥",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                help="Set >0 to compute correlations only on high-volatility days.",
            )

        if window:
            # Choose panel based on mode
            panel = moves_wide if mode == "Raw moves" else z_wide
            if tail_k > 0:
                mask_tail = z_wide.abs().max(axis=1) >= tail_k
                panel = panel.loc[mask_tail]

            # Compute correlations
            if mode == "Tail-weighted (|Z|)":
                # Weighted Pearson emphasizing large-|Z| days
                sub = panel.loc[:asof_ts].tail(window)
                if sub.shape[0] < 5:
                    C = pd.DataFrame()
                else:
                    w = z_wide.abs().max(axis=1).loc[sub.index].clip(0, 5).values
                    def weighted_corr(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
                        X = df.values
                        wsum = weights.sum()
                        mu = (weights[:, None] * X).sum(axis=0) / wsum
                        Xc = X - mu
                        cov = (weights[:, None] * Xc).T @ Xc / wsum
                        s = np.sqrt(np.diag(cov))
                        S = np.outer(s, s)
                        return pd.DataFrame(cov / S, index=df.columns, columns=df.columns)
                    C = weighted_corr(sub, w)
            else:
                C = rolling_corr_matrix(panel, asof_ts, window)

            if C.empty:
                st.warning("Not enough data in the selected window/date to compute correlations.")
            else:
                # Heatmap
                mC = melt_corr(C)
                heat = (
                    alt.Chart(mC)
                    .mark_rect()
                    .encode(
                        x=alt.X("x:N", title=""),
                        y=alt.Y("y:N", title=""),
                        color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1,1]), title="Corr"),
                        tooltip=["x","y",alt.Tooltip("corr:Q", format=".2f")],
                    )
                    .properties(height=420)
                )
                text = (
                    alt.Chart(mC)
                    .mark_text(baseline='middle')
                    .encode(x="x:N", y="y:N", text=alt.Text("corr:Q", format=".2f"))
                )
                st.altair_chart(heat + text, use_container_width=True)

                # Top positive/negative pairs
                pairs = []
                cols = list(C.columns)
                for i, j in combinations(range(len(cols)), 2):
                    pairs.append((cols[i], cols[j], C.iloc[i, j]))
                pairs_df = pd.DataFrame(pairs, columns=["asset_a","asset_b","corr"]).sort_values("corr")
                left, right = st.columns(2)
                with left:
                    st.markdown("#### Most Negative (diversifiers)")
                    st.dataframe(pairs_df.head(5).style.hide(axis="index"))
                with right:
                    st.markdown("#### Most Positive (clusters)")
                    st.dataframe(pairs_df.tail(5)[::-1].style.hide(axis="index"))

with TAB_NEWS:
    st.subheader("Headlines mapped to moves (Yahoo Finance)")
    st.caption("Uses yfinance news with Yahoo RSS fallback. Filter to the selected date with optional ±day buffer; if still empty, shows nearest within a small window.")
    max_per = st.slider("Headlines per asset", min_value=1, max_value=10, value=3, step=1)
    buf_days = st.slider("Date buffer (± days)", min_value=0, max_value=3, value=0, step=1, key="news_buf_days")
    if snapshot.empty:
        st.info("No snapshot for selected date; headlines still shown if available.")

    for _, row in snapshot.iterrows():
        asset = row["asset"]
        mv_str = f"{row['move']:+.2f}%" if row["units"] == "percent" else f"{row['move']:+.1f} bp"
        z_str = f"{row['z']:+.2f}"
        with st.expander(f"{asset}  — move {mv_str},  Z {z_str}  (news proxy: {NEWS_PROXY.get(asset, asset)})", expanded=False):
            dfh = fetch_asset_headlines(asset, sel_date, buffer_days=buf_days, nearest_backfill_days=3)
            if dfh.empty:
                st.write("No headlines found (even after fallback).")
            else:
                show = dfh.head(max_per)
                for _, h in show.iterrows():
                    ts = h["time"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(h["time"]) else ""
                    src = h.get("source", "")
                    st.markdown(f"- [{h['title']}]({h['link']})  ")
                    st.caption(f"{h['publisher']} — {ts}  ({src})")

with TAB_CAL:
    st.subheader("Economic Calendar (TradingEconomics)")
    if _te_token() is None:
        st.error("TradingEconomics API key missing. Put it in .streamlit/secrets.toml under [tradingeconomics].")
    else:
        country_opts = ["United States", "Canada", "Euro Area", "United Kingdom", "Japan", "China"]
        countries = st.multiselect("Countries", country_opts, default=["United States"])
        buf_days = st.slider("Date buffer (± days)", 0, 3, 0, 1, key="cal_buf_days")

        try:
            cal = te_calendar(pd.Timestamp(sel_date), buf_days=buf_days, countries=countries)
        except Exception as e:
            st.error(f"TradingEconomics error: {e}")
            cal = pd.DataFrame()

        if cal.empty:
            st.info("No events for the chosen date/buffer/countries.")
        else:
            local_tz = tz.gettz("America/New_York")
            cal_disp = cal.copy()
            cal_disp["Local Time"] = cal_disp["DatetimeUTC"].dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d %H:%M")
            cal_disp = cal_disp[["Local Time","Country","Category","Event","Importance","Actual","Forecast","Previous","Reference"]]
            st.dataframe(cal_disp, use_container_width=True)

            ev_labels = [f"{r.Event} — {r.Country} @ {r.DatetimeUTC.strftime('%Y-%m-%d %H:%M UTC')}" for _, r in cal.iterrows()]
            idx = st.selectbox("Pick event to analyze", options=list(range(len(ev_labels))), format_func=lambda i: ev_labels[i] if 0 <= i < len(ev_labels) else "")
            if len(cal) and idx is not None:
                event_time = cal.iloc[idx]["DatetimeUTC"]
                asset_names = [a.name for a in assets if a.name in data]
                moves_tbl = event_moves_around_date(event_time, asset_names, data, baseline)
                if moves_tbl.empty:
                    st.caption("Not enough data around the event to compute pre/post moves.")
                else:
                    def fmt_row(r):
                        fm = f"{r['PreMove']:+.2f}%" if r['Units']=='percent' else f"{r['PreMove']:+.1f} bp"
                        fo = f"{r['PostMove']:+.2f}%" if r['Units']=='percent' else f"{r['PostMove']:+.1f} bp"
                        return pd.Series({"Asset": r["Asset"], "Pre (prev→event)": fm, "Post (event→next)": fo})
                    st.dataframe(moves_tbl.apply(fmt_row, axis=1), use_container_width=True)


with TAB_EXPLORE:
    st.subheader("Time-Series Explorer")
    sel = st.selectbox("Select asset", options=[a.name for a in assets])

    if sel in data:
        # raw level
        raw = data[sel].to_frame(name="value").reset_index().rename(columns={"index":"date"})
        # moves & Z
        kind = next((a.kind for a in assets if a.name == sel), "price")
        moves = compute_daily_moves(data[sel], kind).to_frame(name="move").reset_index().rename(columns={"index":"date"})
        baseline_stats = baseline.loc[sel]
        z_moves = moves.copy()
        z_moves["z"] = (z_moves["move"] - baseline_stats["mean"]) / baseline_stats["std"] if baseline_stats["std"] != 0 else np.nan

        left, right = st.columns(2)
        with left:
            st.markdown("#### Level")
            level_chart = (
                alt.Chart(raw)
                .mark_line()
                .encode(x="date:T", y=alt.Y("value:Q", title="Level"))
                .properties(height=260)
            )
            st.altair_chart(level_chart, use_container_width=True)

        with right:
            st.markdown("#### Daily Move and Z")
            move_chart = (
                alt.Chart(moves)
                .mark_line()
                .encode(x="date:T", y=alt.Y("move:Q", title="Daily Move"))
                .properties(height=130)
            )
            z_chart = (
                alt.Chart(z_moves)
                .mark_line()
                .encode(x="date:T", y=alt.Y("z:Q", title="Z-Score"))
                .properties(height=130)
            )
            st.altair_chart(move_chart, use_container_width=True)
            st.altair_chart(z_chart, use_container_width=True)

with TAB_EXPORT:
    st.subheader("Exports")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Export Baseline")
        buf = io.StringIO()
        baseline.to_csv(buf)
        st.download_button("baseline_stats.csv", buf.getvalue(), file_name="baseline_stats.csv", mime="text/csv")
    with col2:
        st.markdown("#### Export Snapshot")
        buf2 = io.StringIO()
        snapshot.to_csv(buf2, index=False)
        st.download_button("latest_snapshot.csv", buf2.getvalue(), file_name="latest_snapshot.csv", mime="text/csv")
