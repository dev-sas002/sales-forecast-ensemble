"""
The SKU forecasting dashboard.

Presentation only. Every number shown here is computed in `src.dashboard.data`
or in the pipeline modules it reads from; this file decides how to draw them.
The previous version interleaved `pd.read_parquet`, derivation and layout in
one 615-line function, which meant none of the logic could be tested without a
browser and every layout change risked a data change.

Run it with:
    streamlit run streamlit_demo.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.config import FEATURE_DIR, LANDING_DIR
from src.dashboard.data import (
    backtest_frames,
    build_forecast_view,
    feature_glossary,
    linkage_rate,
    load_dashboard_data,
    pipeline_stages,
    quality_summary,
    series_options,
)
from src.explain import build_facts, explain, narrator_status

# The dashboard and the API run in separate containers, so "localhost" is the
# dashboard itself, not the API. docker-compose sets API_URL to
# http://api:8000; this is the code that reads it.
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

# One palette, defined once. Colour carries meaning here — the same blue means
# "the model" in every chart, the same amber means "uncertainty" — which is
# what makes four charts readable as one system rather than four pictures.
INK = "#1a2332"
MUTED = "#5b6b7f"
ACCENT = "#2563eb"
ACCENT_SOFT = "rgba(37, 99, 235, 0.16)"
WARN = "#c2410c"
GOOD = "#047857"
GRID = "rgba(120, 140, 165, 0.18)"

PLOT_LAYOUT = {
    "font": {"family": "-apple-system, Segoe UI, Roboto, sans-serif", "color": INK},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    # No Plotly titles anywhere in this file — every chart heading is Streamlit
    # markdown above it. Plotly draws its title and a top-anchored legend into
    # the same band and they collide at any margin worth using.
    "margin": {"t": 34, "r": 24, "b": 44, "l": 72},
    "hovermode": "x unified",
    "xaxis": {"gridcolor": GRID, "zeroline": False},
    "yaxis": {"gridcolor": GRID, "zeroline": False},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
}

st.set_page_config(
    page_title="SKU Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1400px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  .hero { border-bottom: 1px solid rgba(120,140,165,.22); padding-bottom: 1rem;
          margin-bottom: 1.4rem; }
  .hero h1 { font-size: 1.85rem; margin: 0 0 .3rem 0; font-weight: 650; }
  .hero p  { color: #5b6b7f; margin: 0; font-size: .95rem; }
  .note { background: rgba(37,99,235,.06); border-left: 3px solid #2563eb;
          padding: .85rem 1rem; border-radius: 0 6px 6px 0; margin: .6rem 0 1rem; }
  .warn { background: rgba(194,65,12,.07); border-left: 3px solid #c2410c;
          padding: .85rem 1rem; border-radius: 0 6px 6px 0; margin: .6rem 0 1rem; }
  .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
  .stTabs [data-baseweb="tab"] { padding: .55rem 1rem; font-size: .92rem; }
  div[data-testid="stMetricValue"] { font-size: 1.55rem; }
  div[data-testid="stMetricLabel"] { color: #5b6b7f; }
  section[data-testid="stSidebar"] { border-right: 1px solid rgba(120,140,165,.2); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def _load():
    return load_dashboard_data(LANDING_DIR, FEATURE_DIR)


@st.cache_data(show_spinner=False)
def _explanation(cache_key: str, _facts) -> tuple[str, str, str | None]:  # noqa: ARG001
    """
    Narrate a forecast, cached on the facts themselves.

    Streamlit reruns the whole script on every widget interaction, so without
    this the note would be regenerated on every rerender of the tab — and with
    an API key set, that is a paid request each time someone moves a selectbox.
    The facts are the natural key: identical facts can only produce an
    identical note, and a pipeline run changes them.

    `_facts` is underscore-prefixed so Streamlit does not try to hash the
    dataclass; `cache_key` is its JSON form and is what identifies the entry.
    It is unread by design — Streamlit reads it, not this function.
    """
    note = explain(_facts)
    return note.text, note.source, note.model


def _styled(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, **overrides)
    return fig


def _empty_state(title: str, command: str, why: str) -> None:
    """
    Never a bare "no data". An empty state that does not say what to run is a
    dead end for whoever opened the page.
    """
    st.markdown(
        f"<div class='note'><strong>{title}</strong><br>{why}</div>",
        unsafe_allow_html=True,
    )
    st.code(command, language="bash")


# ---------------------------------------------------------------- sidebar ---


def render_sidebar(data) -> None:
    st.sidebar.markdown("### Pipeline")
    for stage in pipeline_stages(data):
        icon = "✅" if stage.done else "⬜️"
        detail = f" — {stage.detail}" if stage.detail else ""
        st.sidebar.markdown(
            f"{icon} **{stage.name}**<span style='color:{MUTED};font-size:.85rem'>{detail}</span>",
            unsafe_allow_html=True,
        )

    if data.orphan_forecasts:
        st.sidebar.markdown(
            f"<div class='warn'><strong>Orphan forecast files</strong><br>"
            f"{', '.join(data.orphan_forecasts)}<br>"
            "No registered model produces these, so they are excluded from the "
            "ensemble.</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("### Explanation")
    status = narrator_status()
    if status["available"]:
        st.sidebar.success(f"AI narration on ({status['model']})")
    else:
        st.sidebar.info(f"Template narration — {status['reason']}")

    if data.errors:
        st.sidebar.markdown("### Read errors")
        for err in data.errors:
            st.sidebar.warning(err)


# ------------------------------------------------------------------- tabs ---


def tab_overview(data) -> None:
    st.subheader("Source data")
    if data.credit is None:
        _empty_state(
            "No ingested data yet",
            "python -m scripts.run_pipeline --weeks 130 --horizon 12",
            "The pipeline generates seeded synthetic transactions, ingests them "
            "and writes parquet under <code>DATA_DIR</code>.",
        )
        return

    credit = data.credit.copy()
    credit["txn_date"] = pd.to_datetime(credit["txn_date"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(credit):,}")
    c2.metric("SKUs", credit["sku"].nunique())
    c3.metric("Customers", f"{credit['customer_id'].nunique():,}")
    c4.metric(
        "Weeks covered",
        int((credit["txn_date"].max() - credit["txn_date"].min()).days / 7) + 1,
    )

    weekly = credit.set_index("txn_date").resample("W-MON")["quantity"].sum().reset_index()
    fig = go.Figure(
        go.Scatter(
            x=weekly["txn_date"],
            y=weekly["quantity"],
            mode="lines",
            line={"color": ACCENT, "width": 2},
            fill="tozeroy",
            fillcolor=ACCENT_SOFT,
            name="units",
        )
    )
    st.markdown("**Weekly units across the catalogue**")
    st.plotly_chart(_styled(fig, height=300, yaxis_title="units"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Units by region**")
        by_region = credit.groupby("region")["quantity"].sum().sort_values()
        fig = go.Figure(
            go.Bar(x=by_region.values, y=by_region.index, orientation="h", marker_color=ACCENT)
        )
        st.plotly_chart(
            _styled(fig, height=260, xaxis_title="units"),
            use_container_width=True,
        )
    with right:
        st.markdown("**Units by SKU**")
        by_sku = credit.groupby("sku")["quantity"].sum().sort_values()
        fig = go.Figure(
            go.Bar(x=by_sku.values, y=by_sku.index, orientation="h", marker_color=MUTED)
        )
        st.plotly_chart(
            _styled(fig, height=260, xaxis_title="units"),
            use_container_width=True,
        )

    with st.expander("Sample transactions"):
        st.dataframe(credit.head(12), use_container_width=True, hide_index=True)


def tab_linkage(data) -> None:
    st.subheader("Panel linkage and privacy")
    if data.linked is None:
        _empty_state(
            "Nothing linked yet",
            "python -m src.linking.link_panel_credit",
            "Linkage joins panel members to transactions on a salted hash.",
        )
        return

    rate = linkage_rate(data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Linked rows", f"{len(data.linked):,}")
    c2.metric("Linkage rate", f"{rate:.1%}" if rate is not None else "—")
    c3.metric("Distinct customers", f"{data.linked['cust_hash'].nunique():,}")

    st.markdown(
        "<div class='note'><strong>Customer IDs never appear in the joined "
        "table.</strong> Both sides are hashed with HMAC-SHA256 under a secret "
        "salt before the join, so the linkage key is stable and reversible only "
        "with the salt. Two runs under different salts produce non-comparable "
        "hashes — which is why <code>HASH_SALT</code> must be set for anything "
        "beyond a local demo.</div>",
        unsafe_allow_html=True,
    )

    cols = [
        c
        for c in [
            "sku",
            "region",
            "txn_date",
            "quantity",
            "amount",
            "age_group",
            "income_bin",
            "cust_hash",
        ]
        if c in data.linked.columns
    ]
    sample = data.linked[cols].head(8).copy()
    if "cust_hash" in sample:
        sample["cust_hash"] = sample["cust_hash"].str.slice(0, 16) + "…"
    st.dataframe(sample, use_container_width=True, hide_index=True)


def tab_features(data) -> None:
    st.subheader("Feature engineering")
    if data.features is None:
        _empty_state(
            "No feature table yet",
            "python -m src.features.build_features",
            "Features aggregate transactions to SKU × region × week and attach "
            "strictly backward-looking history.",
        )
        return

    features = data.features
    c1, c2, c3 = st.columns(3)
    c1.metric("Series-weeks", f"{len(features):,}")
    c2.metric("Series", features[["sku", "region"]].drop_duplicates().shape[0])
    c3.metric(
        "Span",
        f"{pd.to_datetime(features['week']).min():%b %Y} – "
        f"{pd.to_datetime(features['week']).max():%b %Y}",
    )

    st.markdown(
        "<div class='note'><strong>Every feature below is built from the series "
        "shifted one week.</strong> Features are declared in "
        "<code>src/features/registry.py</code> and are handed a "
        "<code>History</code> object whose only data are already-lagged series — "
        "the current week is not reachable through it, so a look-ahead feature "
        "cannot be written through the interface.</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(feature_glossary(features), use_container_width=True, hide_index=True)

    pairs = features[["sku", "region"]].drop_duplicates()
    left, right = st.columns(2)
    sku = left.selectbox("SKU", sorted(pairs["sku"].unique()), key="feat_sku")
    region = right.selectbox(
        "Region",
        sorted(pairs[pairs["sku"] == sku]["region"].unique()),
        key="feat_region",
    )

    series = features[(features["sku"] == sku) & (features["region"] == region)].copy()
    series["week"] = pd.to_datetime(series["week"])
    series = series.sort_values("week")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["week"],
            y=series["units_sold"],
            name="units sold",
            mode="lines",
            line={"color": INK, "width": 1.6},
        )
    )
    for col, colour, dash in (
        ("roll_4w_mean", ACCENT, "solid"),
        ("roll_13w_mean", WARN, "dash"),
    ):
        if col in series:
            fig.add_trace(
                go.Scatter(
                    x=series["week"],
                    y=series[col],
                    name=col,
                    mode="lines",
                    line={"color": colour, "width": 2, "dash": dash},
                )
            )
    st.markdown(f"**{sku} · {region} — actuals against trailing means**")
    st.plotly_chart(_styled(fig, height=360, yaxis_title="units"), use_container_width=True)


def tab_forecast(data) -> None:
    st.subheader("Forecast")
    options = series_options(data)
    if not options:
        _empty_state(
            "No forecast yet",
            "python -m scripts.run_pipeline --weeks 130 --horizon 12",
            "Training writes quantile forecasts that this tab reads.",
        )
        return

    left, right = st.columns(2)
    sku = left.selectbox("SKU", sorted(options), key="fc_sku")
    region = right.selectbox("Region", options[sku], key="fc_region")

    view = build_forecast_view(data, sku, region)
    if view is None:
        st.warning("That series has no forecast rows.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horizon", f"{len(view.forecast)} weeks")
    c2.metric("Total forecast", f"{view.total:,.0f} units")
    c3.metric("Mean week", f"{view.mean_weekly:,.0f} units")
    change = view.change_vs_recent
    c4.metric(
        "vs last 8 weeks",
        f"{change:+.1%}" if change is not None else "—",
        delta=f"{view.recent_mean:,.0f} recent avg" if view.recent_mean else None,
        delta_color="off",
    )

    fig = go.Figure()
    if not view.history.empty:
        fig.add_trace(
            go.Scatter(
                x=view.history["week"],
                y=view.history["units_sold"],
                name="actuals",
                mode="lines",
                line={"color": MUTED, "width": 1.6},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=view.forecast["date"],
            y=view.forecast["q0.9"],
            mode="lines",
            line={"color": "rgba(0,0,0,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.forecast["date"],
            y=view.forecast["q0.1"],
            mode="lines",
            line={"color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor=ACCENT_SOFT,
            name="80% interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.forecast["date"],
            y=view.forecast["q0.5"],
            name="median forecast",
            mode="lines+markers",
            line={"color": ACCENT, "width": 2.5},
            marker={"size": 6},
        )
    )
    if not view.history.empty:
        fig.add_vline(
            x=view.history["week"].max(),
            line_dash="dot",
            line_color=MUTED,
            opacity=0.6,
        )
    st.markdown(f"**{sku} · {region} — history and forecast**")
    st.plotly_chart(_styled(fig, height=390, yaxis_title="units"), use_container_width=True)

    explain_col, table_col = st.columns([3, 2])

    with explain_col:
        st.markdown("**What the model is saying**")
        summary = (data.backtest or {}).get("summary")
        facts = build_facts(
            sku,
            region,
            view.forecast,
            history=view.history if not view.history.empty else None,
            backtest=summary,
        )
        text, source, model = _explanation(
            json.dumps(facts.to_dict(), sort_keys=True, default=str), facts
        )
        st.markdown(text)
        st.caption(
            f"Written by {model}"
            if source == "claude"
            else "Rendered from the computed facts — set ANTHROPIC_API_KEY for a written summary"
        )

        if not view.anomalies.empty:
            st.markdown(
                f"<div class='warn'><strong>{len(view.anomalies)} flagged week(s) "
                "in this series' history.</strong> The forecast is fitted on them; "
                "see the Data quality tab.</div>",
                unsafe_allow_html=True,
            )

    with table_col:
        st.markdown("**Weekly detail**")
        table = view.forecast[["date", "q0.1", "q0.5", "q0.9"]].copy()
        table["date"] = table["date"].dt.strftime("%Y-%m-%d")
        table.columns = ["week", "low (p10)", "median", "high (p90)"]
        st.dataframe(
            table.style.format(
                {
                    "low (p10)": "{:,.0f}",
                    "median": "{:,.0f}",
                    "high (p90)": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=360,
        )


def tab_quality(data) -> None:
    st.subheader("Input data quality")
    st.markdown(
        "<div class='note'>A forecaster extrapolates the past, which makes it "
        "maximally vulnerable to the past being wrong. A stopped feed and a week "
        "of zero demand are indistinguishable to the model; a stockout is demand "
        "the business could not serve, recorded as demand that did not exist. "
        "These checks are plain robust statistics — every number is recomputable "
        "by hand, which is what makes them worth trusting about a model.</div>",
        unsafe_allow_html=True,
    )

    if data.features is None:
        _empty_state(
            "No feature table to check",
            "python -m scripts.run_pipeline",
            "Anomaly detection runs over the weekly SKU × region table.",
        )
        return

    counts, found = quality_summary(data.features)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Level spikes", counts["level_spike"])
    c2.metric("Level collapses", counts["level_collapse"])
    c3.metric("Zero runs", counts["zero_run"])
    c4.metric("Price jumps", counts["price_jump"])

    if found.empty:
        st.success(
            "No suspect weeks found. Every series is within 3.5 robust deviations "
            "of its own median, with no multi-week zero runs and no price move "
            "beyond 25% week-on-week."
        )
        return

    by_kind = found["kind"].value_counts().sort_values()
    fig = go.Figure(go.Bar(x=by_kind.values, y=by_kind.index, orientation="h", marker_color=WARN))
    st.markdown("**Findings by kind**")
    st.plotly_chart(_styled(fig, height=220, xaxis_title="findings"), use_container_width=True)

    display = found.head(40).copy()
    display["week"] = pd.to_datetime(display["week"]).dt.strftime("%Y-%m-%d")
    st.dataframe(
        display[["sku", "region", "week", "kind", "severity", "detail"]],
        use_container_width=True,
        hide_index=True,
        height=420,
    )


def tab_model_quality(data) -> None:
    st.subheader("Measured accuracy")
    if not data.backtest:
        _empty_state(
            "The model has not been scored",
            "python -m src.evaluation.backtest --folds 4 --horizon 8",
            "Rolling-origin backtesting retrains at several historical cutoffs "
            "and forecasts weeks the model has never seen.",
        )
        return

    summary = data.backtest["summary"]
    lift = (summary["wape_naive"] - summary["wape_model"]) / max(summary["wape_naive"], 1e-9)
    nominal = summary["nominal_coverage"]
    calibrated = summary.get("calibrated_coverage", summary["coverage"])

    st.markdown(
        "<div class='note'>Scores come from <strong>rolling-origin "
        "backtesting</strong>: the model is retrained at several historical "
        "cutoffs and asked to forecast weeks it has never seen. The "
        "seasonal-naive baseline — <em>this week last year</em> — is free, so the "
        "model has to beat it to be worth running.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WAPE (model)", f"{summary['wape_model']:.3f}")
    c2.metric("WAPE (seasonal naive)", f"{summary['wape_naive']:.3f}")
    c3.metric(
        "Improvement",
        f"{lift:.1%}",
        delta="beats baseline" if lift > 0 else "worse than baseline",
        delta_color="normal" if lift > 0 else "inverse",
    )
    c4.metric(
        f"{nominal:.0%} interval coverage",
        f"{calibrated:.1%}",
        delta=f"{calibrated - nominal:+.1%} vs nominal",
        delta_color="off",
    )

    st.markdown("**Conformal calibration**")
    raw = summary["coverage"]
    cal_c1, cal_c2, cal_c3 = st.columns(3)
    cal_c1.metric("Raw quantile coverage", f"{raw:.1%}")
    cal_c2.metric("After calibration", f"{calibrated:.1%}")
    cal_c3.metric("Offset applied", f"±{summary.get('conformal_offset', 0):.1f} units")
    st.caption(
        "Quantile regression minimises pinball loss; nothing in that objective "
        "constrains how often the truth actually lands inside the band. The "
        "offset is refitted inside each fold's own training window and applied "
        "to the weeks after the cutoff, so the calibrated figure is out of "
        "sample rather than the tautology of scoring a calibration set against "
        "the quantile that defined it."
    )

    horizons, folds = backtest_frames(data.backtest)

    if not horizons.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=horizons["horizon"], y=horizons["wape"], name="WAPE", marker_color=ACCENT)
        )
        fig.add_trace(
            go.Scatter(
                x=horizons["horizon"],
                y=horizons["coverage"],
                name="coverage",
                yaxis="y2",
                mode="lines+markers",
                line={"color": WARN, "width": 2.5},
            )
        )
        fig.add_hline(
            y=nominal,
            line_dash="dot",
            line_color=WARN,
            yref="y2",
            opacity=0.5,
            annotation_text="nominal",
        )
        layout = dict(PLOT_LAYOUT)
        layout["yaxis"] = {**layout["yaxis"], "title": "WAPE (lower is better)"}
        fig.update_layout(
            **layout,
            height=380,
            xaxis_title="weeks ahead",
            yaxis2={
                "title": "coverage",
                "overlaying": "y",
                "side": "right",
                "range": [0, 1],
                "tickformat": ".0%",
                "showgrid": False,
            },
        )
        st.markdown("**Accuracy and coverage by forecast horizon**")
        st.plotly_chart(fig, use_container_width=True)

    if not folds.empty:
        st.markdown("**Per-fold detail**")
        cols = [
            c
            for c in [
                "cutoff",
                "n_rows",
                "wape_model",
                "wape_naive",
                "improvement",
                "coverage",
                "calibrated_coverage",
            ]
            if c in folds.columns
        ]
        st.dataframe(
            folds[cols].style.format(
                {
                    "wape_model": "{:.3f}",
                    "wape_naive": "{:.3f}",
                    "improvement": "{:.1%}",
                    "coverage": "{:.1%}",
                    "calibrated_coverage": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def tab_api(data) -> None:
    st.subheader("Live API")
    st.markdown(
        "The same forecasts over HTTP, which is how a planning system would "
        "consume them. The API reads the parquet the pipeline wrote, caches it "
        "keyed on the file's modification time, and indexes it by series — so a "
        "pipeline run is picked up on the next request without a restart."
    )

    url_col, status_col = st.columns([3, 1])
    api_url = url_col.text_input("API URL", DEFAULT_API_URL)
    with status_col:
        st.write("")
        try:
            healthy = requests.get(f"{api_url}/health", timeout=4).status_code == 200
        except requests.exceptions.RequestException:
            healthy = False
        # A statement, not an expression: written as a conditional expression
        # Streamlit sees the returned DeltaGenerator as something to render and
        # raises "`_repr_html_()` is not a valid Streamlit command".
        if healthy:
            st.success("Connected")
        else:
            st.error("Offline")

    options = series_options(data)
    if not options:
        st.info("Run the pipeline to populate the series list.")
        return

    c1, c2, c3 = st.columns(3)
    sku = c1.selectbox("SKU", sorted(options), key="api_sku")
    region = c2.selectbox("Region", options[sku], key="api_region")
    weeks = c3.slider("Weeks", 1, 12, 8)

    endpoint = st.radio("Endpoint", ["/forecast", "/explain", "/anomalies"], horizontal=True)

    if st.button("Send request", type="primary"):
        try:
            if endpoint == "/anomalies":
                response = requests.get(
                    f"{api_url}/anomalies",
                    params={"sku": sku, "region": region, "limit": 25},
                    timeout=15,
                )
            else:
                response = requests.post(
                    f"{api_url}{endpoint}",
                    json={"sku": sku, "region": region, "horizon_weeks": weeks},
                    timeout=30,
                )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            return

        if response.status_code != 200:
            st.error(f"{response.status_code} — {response.text[:400]}")
            return

        payload = response.json()
        if endpoint == "/explain":
            st.success(payload["explanation"])
            st.caption(f"source: {payload['source']}")
        elif endpoint == "/forecast":
            frame = pd.DataFrame(payload)
            st.markdown(f"**{sku} · {region}** — served over HTTP")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame["q0.9"],
                    mode="lines",
                    line={"color": "rgba(0,0,0,0)"},
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame["q0.1"],
                    mode="lines",
                    line={"color": "rgba(0,0,0,0)"},
                    fill="tonexty",
                    fillcolor=ACCENT_SOFT,
                    name="80% interval",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame["q0.5"],
                    mode="lines+markers",
                    line={"color": ACCENT, "width": 2.5},
                    name="median",
                )
            )
            st.plotly_chart(
                _styled(fig, height=320, yaxis_title="units"),
                use_container_width=True,
            )

        with st.expander("Raw response"):
            st.json(payload)


def main() -> None:
    st.markdown(
        "<div class='hero'><h1>SKU × Region demand forecasting</h1>"
        "<p>Weekly quantile forecasts with calibrated prediction intervals, "
        "scored against a baseline they have to beat.</p></div>",
        unsafe_allow_html=True,
    )

    data = _load()
    render_sidebar(data)

    if not data.has_any:
        _empty_state(
            "Nothing has been built yet",
            "make pipeline    # or: python -m scripts.run_pipeline --weeks 130",
            "The pipeline generates seeded synthetic transactions and runs them "
            "through ingest, linkage, features, training and the ensemble.",
        )
        return

    tabs = st.tabs(
        [
            "Overview",
            "Linkage",
            "Features",
            "Forecast",
            "Data quality",
            "Model quality",
            "API",
        ]
    )
    renderers = (
        tab_overview,
        tab_linkage,
        tab_features,
        tab_forecast,
        tab_quality,
        tab_model_quality,
        tab_api,
    )
    for tab, render in zip(tabs, renderers, strict=True):
        with tab:
            render(data)


if __name__ == "__main__":
    main()
