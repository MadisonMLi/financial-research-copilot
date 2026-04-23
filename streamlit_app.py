"""
Financial Analysis Chatbot — Streamlit UI
==========================================
Run with:
    pip install streamlit plotly
    streamlit run streamlit_app.py

Make sure chatbot.py is in the same folder.
"""

import os
import sys
import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from itertools import count

# Global chart key counter — ensures every st.plotly_chart() gets a unique key
_chart_key = count()

# ── Import everything from chatbot.py ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from chatbot import (
    build_dataset,
    dispatch_tool,
    TOOLS,
    SYSTEM_PROMPT,
    MODEL,
    ANTHROPIC_API_KEY,
    COMPANIES,
)

# RAG retriever helpers — for surfacing sources after search_filings tool calls
from rag_retriever import get_last_sources, clear_last_sources

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background    : #1a1d2e;
        border        : 1px solid #2d3250;
        border-radius : 10px;
        padding       : 16px 20px;
    }
    div[data-testid="stMetricLabel"]  { font-size: 13px; color: #aaa; }
    div[data-testid="stMetricValue"]  { font-size: 24px; font-weight: 700; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius : 12px;
        margin-bottom : 8px;
    }

    /* Quick action buttons */
    div.stButton > button {
        border-radius : 20px;
        font-size     : 13px;
        padding       : 4px 12px;
    }

    /* Sidebar company dots */
    .company-dot {
        font-size   : 14px;
        line-height : 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ── Company Brand Colors ───────────────────────────────────────────────────────
COMPANY_COLORS = {
    "Microsoft": "#4C9BE8",
    "Apple"    : "#A8A8A8",
    "Tesla"    : "#E82C2C",
    "Meta"     : "#1877F2",
    "Amazon"   : "#FF9900",
    "Netflix"  : "#E50914",
    "Alphabet" : "#34A853",
}

CHART_TEMPLATE = "plotly_dark"
CHART_HEIGHT   = 340

_PALETTE = {
    "blue"  : "#4C9BE8",
    "green" : "#50C878",
    "orange": "#FFB347",
    "red"   : "#FF6B6B",
    "purple": "#B57BFF",
    "teal"  : "#4DD0E1",
}


# ==============================================================================
# Chart helpers
# ==============================================================================

def _rows(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    mask = df["Company"].str.contains(company_name, case=False, na=False)
    return df[mask].sort_values("Year")


def chart_revenue_income(df, company_name):
    r = _rows(df, company_name)
    if r.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=r["Year"].astype(str), y=r["Total Revenue"],
                name="Revenue ($M)", marker_color=_PALETTE["blue"])
    fig.add_bar(x=r["Year"].astype(str), y=r["Net Income"],
                name="Net Income ($M)", marker_color=_PALETTE["green"])
    fig.update_layout(
        title=f"{r['Company'].iloc[-1]} — Revenue vs Net Income ($M)",
        barmode="group", template=CHART_TEMPLATE, height=CHART_HEIGHT,
        yaxis_title="$M", xaxis_title="Fiscal Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def chart_margins(df, company_name):
    r = _rows(df, company_name)
    if r.empty:
        return None
    fig = go.Figure()
    for col, color, dash in [
        ("Gross Margin (%)",      _PALETTE["blue"],   "solid"),
        ("Operating Margin (%)",  _PALETTE["orange"], "dash"),
        ("Net Profit Margin (%)", _PALETTE["green"],  "dot"),
    ]:
        valid = r[["Year", col]].dropna()
        if not valid.empty:
            fig.add_scatter(
                x=valid["Year"].astype(str), y=valid[col], name=col,
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=8),
            )
    fig.update_layout(
        title=f"{r['Company'].iloc[-1]} — Margin Trends (%)",
        template=CHART_TEMPLATE, height=CHART_HEIGHT,
        yaxis_title="%", xaxis_title="Fiscal Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def chart_revenue_growth(df, company_name):
    r = _rows(df, company_name)
    if r.empty:
        return None, None

    valid_rev = r[["Year", "Total Revenue"]].dropna()
    fig1 = go.Figure()
    fig1.add_bar(x=valid_rev["Year"].astype(str), y=valid_rev["Total Revenue"],
                 name="Revenue ($M)", marker_color=_PALETTE["blue"])
    fig1.update_layout(
        title=f"{r['Company'].iloc[-1]} — Revenue Trend ($M)",
        template=CHART_TEMPLATE, height=280,
        yaxis_title="$M", xaxis_title="Fiscal Year",
        margin=dict(t=60, b=40),
    )

    growth = r[["Year", "YoY Revenue Growth (%)"]].dropna()
    fig2 = None
    if not growth.empty:
        bar_colors = [_PALETTE["green"] if v >= 0 else _PALETTE["red"]
                      for v in growth["YoY Revenue Growth (%)"]]
        fig2 = go.Figure()
        fig2.add_bar(
            x=growth["Year"].astype(str), y=growth["YoY Revenue Growth (%)"],
            name="YoY Growth (%)", marker_color=bar_colors,
        )
        fig2.update_layout(
            title=f"{r['Company'].iloc[-1]} — YoY Revenue Growth (%)",
            template=CHART_TEMPLATE, height=280,
            yaxis_title="%", xaxis_title="Fiscal Year",
            margin=dict(t=60, b=40),
        )
    return fig1, fig2


def chart_balance_sheet(df, company_name):
    r = _rows(df, company_name)
    if r.empty:
        return None
    valid = r[["Year", "Total Liabilities", "Total Equity"]].dropna()
    if valid.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=valid["Year"].astype(str), y=valid["Total Liabilities"],
                name="Liabilities ($M)", marker_color=_PALETTE["red"])
    fig.add_bar(x=valid["Year"].astype(str), y=valid["Total Equity"],
                name="Equity ($M)", marker_color=_PALETTE["green"])
    fig.update_layout(
        title=f"{r['Company'].iloc[-1]} — Balance Sheet Structure ($M)",
        barmode="stack", template=CHART_TEMPLATE, height=CHART_HEIGHT,
        yaxis_title="$M", xaxis_title="Fiscal Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def chart_roe_roa(df, company_name):
    r = _rows(df, company_name)
    if r.empty:
        return None
    fig = go.Figure()
    for col, color in [("ROE (%)", _PALETTE["purple"]), ("ROA (%)", _PALETTE["orange"])]:
        valid = r[["Year", col]].dropna()
        if not valid.empty:
            fig.add_scatter(
                x=valid["Year"].astype(str), y=valid[col], name=col,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            )
    fig.update_layout(
        title=f"{r['Company'].iloc[-1]} — ROE & ROA (%)",
        template=CHART_TEMPLATE, height=CHART_HEIGHT,
        yaxis_title="%", xaxis_title="Fiscal Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def chart_comparison(df, metric, year=None):
    if year is None:
        year = int(df["Year"].max())
    subset = df[df["Year"] == year][["Company", metric]].dropna()
    if subset.empty:
        return None
    subset = subset.sort_values(metric, ascending=True).reset_index(drop=True)
    bar_colors = [COMPANY_COLORS.get(c, _PALETTE["blue"]) for c in subset["Company"]]
    fig = go.Figure(go.Bar(
        x=subset[metric], y=subset["Company"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:,.1f}" for v in subset[metric]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Company Comparison — {metric} ({year})",
        template=CHART_TEMPLATE, height=max(300, len(subset) * 65),
        xaxis_title=metric, margin=dict(t=60, b=40, r=80),
    )
    return fig


def build_charts(tool_calls: list, df: pd.DataFrame) -> list:
    """Return a flat list of plotly figures based on which tools were called."""
    figures = []
    for call in tool_calls:
        name  = call["name"]
        inp   = call["input"]
        cname = inp.get("company_name", "")

        if name == "company_snapshot":
            figures.append(chart_revenue_income(df, cname))
            figures.append(chart_margins(df, cname))

        elif name == "profitability_summary":
            figures.append(chart_margins(df, cname))
            figures.append(chart_roe_roa(df, cname))

        elif name == "growth_analysis":
            fig1, fig2 = chart_revenue_growth(df, cname)
            figures.extend([fig1, fig2])

        elif name == "balance_sheet_health":
            figures.append(chart_balance_sheet(df, cname))

        elif name == "compare_companies":
            figures.append(chart_comparison(df, inp["metric"], inp.get("year")))

        elif name == "trend_analysis":
            figures.append(chart_revenue_income(df, cname))
            figures.append(chart_margins(df, cname))

        elif name == "full_report":
            figures.append(chart_revenue_income(df, cname))
            figures.append(chart_margins(df, cname))
            figures.append(chart_roe_roa(df, cname))
            figures.append(chart_balance_sheet(df, cname))

    return [f for f in figures if f is not None]


# ==============================================================================
# Claude ask (captures tool calls for chart rendering)
# ==============================================================================

def ask_streamlit(question: str, history: list, df: pd.DataFrame):
    """
    Call Claude with tool access.
    Returns (answer_text, tool_calls_made, updated_api_history, rag_sources).

    rag_sources: list of source-document dicts from any search_filings call(s)
                 made during this turn, for the UI to display.
    """
    _client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = history + [{"role": "user", "content": question}]
    tool_calls_made = []
    rag_sources: list[dict] = []

    clear_last_sources()  # reset before this turn

    response = _client.messages.create(
        model=MODEL, max_tokens=4096,
        system=SYSTEM_PROMPT, messages=messages, tools=TOOLS,
    )

    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls_made.append({"name": block.name, "input": block.input})
                result = dispatch_tool(block.name, block.input, df)
                # If this was a RAG call, capture the cited sources for the UI
                if block.name == "search_filings":
                    rag_sources.extend(get_last_sources())
                    clear_last_sources()
                tool_results.append({
                    "type"       : "tool_result",
                    "tool_use_id": block.id,
                    "content"    : result,
                })
        messages.append({"role": "user", "content": tool_results})
        response = _client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM_PROMPT, messages=messages, tools=TOOLS,
        )

    answer = next(
        (block.text for block in response.content if hasattr(block, "text")),
        "No response generated.",
    )
    # Keep only the last 20 messages to avoid token overflow
    trimmed = messages[-20:] if len(messages) > 20 else messages
    return answer, tool_calls_made, trimmed, rag_sources


# ==============================================================================
# KPI Cards
# ==============================================================================

def render_kpi_cards(df: pd.DataFrame, company_name: str):
    r = _rows(df, company_name)
    if r.empty:
        return

    latest = r.iloc[-1]
    prev   = r.iloc[-2] if len(r) > 1 else None

    def delta_fmt(col, fmt="dollar"):
        if prev is None or pd.isna(prev.get(col)) or pd.isna(latest.get(col)):
            return None
        d = latest[col] - prev[col]
        return f"${d:+,.0f}M" if fmt == "dollar" else f"{d:+.1f}%"

    company_label = latest["Company"]
    color         = COMPANY_COLORS.get(company_label, "#fff")
    st.markdown(
        f'<p style="color:{color}; font-weight:600; margin-bottom:6px;">'
        f'● {company_label} — FY{int(latest["Year"])} Key Metrics</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    rev = latest.get("Total Revenue")
    ni  = latest.get("Net Income")
    nm  = latest.get("Net Profit Margin (%)")
    mc  = latest.get("Market Cap ($M)")

    with c1:
        st.metric("Total Revenue",
                  f"${rev:,.0f}M"  if rev and not pd.isna(rev) else "N/A",
                  delta=delta_fmt("Total Revenue"))
    with c2:
        st.metric("Net Income",
                  f"${ni:,.0f}M"   if ni  and not pd.isna(ni)  else "N/A",
                  delta=delta_fmt("Net Income"))
    with c3:
        st.metric("Net Margin",
                  f"{nm:.1f}%"     if nm  and not pd.isna(nm)  else "N/A",
                  delta=delta_fmt("Net Profit Margin (%)", fmt="pct"))
    with c4:
        st.metric("Market Cap",
                  f"${mc:,.0f}M"   if mc  and not pd.isna(mc)  else "N/A",
                  delta=delta_fmt("Market Cap ($M)"))


# ==============================================================================
# Data loading (cached so it only runs once per session)
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return build_dataset()


# ==============================================================================
# Session state initialisation
# ==============================================================================

def init_state():
    defaults = {
        "messages"      : [],   # {role, content, charts}
        "api_messages"  : [],   # raw Anthropic message history
        "df"            : None,
        "data_loaded"   : False,
        "data_timestamp": None,
        "kpi_company"   : None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ==============================================================================
# Main app
# ==============================================================================

def main():
    init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 FinBot")
        st.caption("Powered by SEC EDGAR · Yahoo Finance · Claude AI")
        st.divider()

        st.markdown("**Companies in dataset**")
        for c in COMPANIES:
            color = COMPANY_COLORS.get(c["name"], "#ffffff")
            st.markdown(
                f'<div class="company-dot" style="color:{color};">'
                f'● {c["name"]} &nbsp;<span style="color:#666;">({c["ticker"]})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        if st.session_state.data_loaded:
            st.success("✅ Data loaded")
            if st.session_state.data_timestamp:
                st.caption(f"Fetched: {st.session_state.data_timestamp}")
            if st.button("🔄  Reload Data", use_container_width=True):
                st.cache_data.clear()
                st.session_state.df           = None
                st.session_state.data_loaded  = False
                st.rerun()
        else:
            st.info("⏳ Loading data on first run…")

        st.divider()

        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.api_messages = []
            st.session_state.kpi_company  = None
            st.rerun()

        st.divider()
        st.caption("Data: SEC EDGAR XBRL API")
        st.caption("Market data: Yahoo Finance")
        st.caption("LLM: Anthropic Claude")

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Financial Analysis Chatbot")
    st.caption(
        "Ask anything about **Microsoft · Apple · Tesla · Meta · Amazon · Netflix · Alphabet**"
    )

    # ── Load Data (first run) ──────────────────────────────────────────────────
    if not st.session_state.data_loaded:
        with st.spinner(
            "📡 Fetching 5 years of data from SEC EDGAR & Yahoo Finance… (~30 sec)"
        ):
            try:
                st.session_state.df            = load_data()
                st.session_state.data_loaded   = True
                st.session_state.data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                st.error(f"❌ Failed to load data: {e}")
                return

    df = st.session_state.df

    # ── Tabs: Chat + Evaluation ────────────────────────────────────────────────
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 RAG Evaluation"])

    with tab_chat:
        render_chat_tab(df)

    with tab_eval:
        render_eval_tab()


def render_sources_panel(sources: list[dict]) -> None:
    """Render an expandable Sources panel showing 10-K excerpts cited by the RAG tool."""
    if not sources:
        return
    with st.expander(f"📑 Sources from 10-K filings ({len(sources)} excerpts)", expanded=False):
        for src in sources:
            company_color = COMPANY_COLORS.get(src["company"].split()[0], "#fff")
            st.markdown(
                f'<span style="color:{company_color}; font-weight:600;">'
                f'● {src["company"]} ({src["ticker"]})</span> '
                f'<span style="color:#888;">· FY{src["year"]} · {src["section"]}</span>',
                unsafe_allow_html=True,
            )
            excerpt = src["excerpt"]
            if len(excerpt) > 600:
                excerpt = excerpt[:600] + "..."
            st.caption(excerpt)
            st.divider()


def render_chat_tab(df: pd.DataFrame) -> None:
    """Renders the existing chat experience inside the Chat tab."""
    # ── KPI Cards (context-aware) ──────────────────────────────────────────────
    if st.session_state.kpi_company:
        render_kpi_cards(df, st.session_state.kpi_company)
        st.divider()

    # ── Chat History ───────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("charts"):
                cols = st.columns(min(len(msg["charts"]), 2))
                for i, fig in enumerate(msg["charts"]):
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{next(_chart_key)}")
            if msg.get("rag_sources"):
                render_sources_panel(msg["rag_sources"])

    # ── Quick Action Buttons ───────────────────────────────────────────────────
    st.markdown("**Quick actions:**")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    quick_prompts = [
        (q1, "📸 Snapshot",     "Give me a snapshot of Microsoft"),
        (q2, "📈 Profitability","Show Apple's profitability over 5 years"),
        (q3, "🚀 Growth",       "How fast has Amazon grown its revenue?"),
        (q4, "🆚 Compare",      "Compare all companies by net profit margin"),
        (q5, "⚠️ Risks (RAG)",  "What manufacturing risks does Tesla mention in its 10-K?"),
        (q6, "🤖 AI Strategy",  "How does Microsoft describe its AI strategy in its filings?"),
    ]
    for col, label, prompt in quick_prompts:
        if col.button(label, use_container_width=True):
            st.session_state["_pending"] = prompt
            st.rerun()

    st.divider()

    # ── Chat Input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about any company's financials or filings…")

    # Trigger from quick-action button
    if "_pending" in st.session_state:
        user_input = st.session_state.pop("_pending")

    if user_input:
        # Show user bubble
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, tool_calls, updated_history, rag_sources = ask_streamlit(
                        user_input,
                        st.session_state.api_messages,
                        df,
                    )
                except Exception as e:
                    answer          = f"Sorry, I ran into an error: {e}"
                    tool_calls      = []
                    updated_history = st.session_state.api_messages
                    rag_sources     = []

            st.markdown(answer)

            charts = build_charts(tool_calls, df)
            if charts:
                chart_cols = st.columns(min(len(charts), 2))
                for i, fig in enumerate(charts):
                    with chart_cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{next(_chart_key)}")

            if rag_sources:
                render_sources_panel(rag_sources)

        # Persist to session state
        st.session_state.api_messages = updated_history
        st.session_state.messages.append({
            "role"       : "assistant",
            "content"    : answer,
            "charts"     : charts,
            "tool_calls" : tool_calls,
            "rag_sources": rag_sources,
        })

        # Update KPI company from the last tool call that referenced one
        for call in reversed(tool_calls):
            if "company_name" in call.get("input", {}):
                st.session_state.kpi_company = call["input"]["company_name"]
                break

        st.rerun()


# ==============================================================================
# Evaluation Tab — measures RAG retrieval quality
# ==============================================================================

def render_eval_tab() -> None:
    """RAG retrieval evaluation: Hit Rate @K and MRR over a test question set."""
    import json
    from pathlib import Path

    st.markdown("### 📊 RAG Retrieval Evaluation")
    st.caption(
        "Measures how well the **search_filings** RAG tool retrieves relevant 10-K excerpts. "
        "Hit Rate @K = % of questions where a correct chunk appears in the top K. "
        "MRR = Mean Reciprocal Rank (1/rank of first hit, averaged)."
    )

    eval_path = Path(__file__).parent / "eval" / "test_questions.json"
    if not eval_path.exists():
        st.warning("No test questions found. See `eval/test_questions.json`.")
        return

    questions = json.loads(eval_path.read_text())

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        k_eval = st.selectbox("Evaluate @K", [3, 5, 10], index=1, key="k_eval_selector")
    with col2:
        st.metric("Test questions", len(questions))
    with col3:
        st.write("")
        run_btn = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Click **Run Evaluation** to score the retriever against the test set.")
        return

    with st.spinner(f"Running {len(questions)} questions through the hybrid retriever…"):
        try:
            from evaluate import run_evaluation
            results_df, summary = run_evaluation(questions, k=k_eval)
        except FileNotFoundError as e:
            st.error(f"{e}\n\nRun `python rag_ingest.py` first to build the index.")
            return
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            return

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Hit Rate @{k_eval}", f"{summary['hit_rate']:.1%}")
    m2.metric("MRR", f"{summary['mrr']:.3f}")
    m3.metric("Questions", summary["n_questions"])

    # Per-question table
    st.markdown("#### Per-question results")
    st.dataframe(
        results_df[["question", "expected_ticker", "hit", "rank", "top_source"]],
        use_container_width=True,
        hide_index=True,
    )

    # Hit rate by section
    if "section" in results_df.columns and not results_df["section"].dropna().empty:
        section_hr = (
            results_df.groupby("section")["hit"].mean().reset_index()
        )
        section_hr.columns = ["Section", "Hit Rate"]
        fig = go.Figure(go.Bar(
            x=section_hr["Section"],
            y=section_hr["Hit Rate"],
            marker_color=_PALETTE["blue"],
            text=[f"{v:.0%}" for v in section_hr["Hit Rate"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Hit Rate by 10-K Section",
            template=CHART_TEMPLATE,
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1.1],
            height=320,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{next(_chart_key)}")


if __name__ == "__main__":
    main()