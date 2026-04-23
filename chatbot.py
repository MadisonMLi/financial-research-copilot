"""
Financial Analysis Chatbot — VSCode / Local Version
=====================================================
Single-file version combining:
  1. SEC EDGAR + yfinance data pipeline
  2. Financial analysis functions
  3. Claude-powered chatbot

Setup:
    pip install anthropic requests pandas yfinance python-dotenv

Set your API key — either in a .env file in the same folder:
    ANTHROPIC_API_KEY=sk-ant-...

Or export it in your terminal before running:
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    python chatbot.py
"""

import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import anthropic

# ── Load .env file if present ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use environment variable directly

# ── RAG retriever (lazy-imported to avoid loading model unless needed) ────────
def _get_search_filings():
    """Lazy import so we don't load sentence-transformers unless search_filings is called."""
    from rag_retriever import search_filings
    return search_filings


# ==============================================================================
# SECTION 1 — SEC EDGAR + yfinance Data Pipeline
# ==============================================================================

# ── Configuration ──────────────────────────────────────────────────────────────
COMPANIES = [
    # Original companies
    {"name": "Microsoft", "cik": "0000789019", "ticker": "MSFT"},
    {"name": "Apple",     "cik": "0000320193", "ticker": "AAPL"},
    {"name": "Tesla",     "cik": "0001318605", "ticker": "TSLA"},
    # FAANG (Apple already above)
    {"name": "Meta",      "cik": "0001326801", "ticker": "META"},   # Facebook / Meta Platforms
    {"name": "Amazon",    "cik": "0001018724", "ticker": "AMZN"},
    {"name": "Netflix",   "cik": "0001065280", "ticker": "NFLX"},
    {"name": "Alphabet",  "cik": "0001652044", "ticker": "GOOGL"},  # Google / Alphabet
]

FISCAL_YEAR = "FY"
NUM_YEARS   = 5

# Save enriched dataset alongside this script
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "financials_enriched.csv"
)

HEADERS = {
    "User-Agent": "financial-research-script anthonymai14@gmail.com"
}

# ── EDGAR XBRL Tags ────────────────────────────────────────────────────────────
EDGAR_METRICS = {
    # ── Income Statement ──
    "Total Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "Gross Profit": [
        "GrossProfit",
    ],
    "Operating Income": [
        "OperatingIncomeLoss",
    ],
    "Net Income": [
        "NetIncomeLoss",
        "NetIncome",
        "ProfitLoss",
    ],
    "R&D Expenses": [
        "ResearchAndDevelopmentExpense",
    ],
    "EPS (Diluted)": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    # ── Balance Sheet ──
    "Total Assets": [
        "Assets",
    ],
    "Current Assets": [
        "AssetsCurrent",
    ],
    "Total Liabilities": [
        "Liabilities",
    ],
    "Current Liabilities": [
        "LiabilitiesCurrent",
    ],
    "Total Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Shares Outstanding": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ],
    # ── Cash Flow ──
    "Cash Flow from Operations": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
}

# Metrics reported in USD (convert to $M in output)
USD_METRICS = {
    "Total Revenue", "Gross Profit", "Operating Income", "Net Income",
    "R&D Expenses", "Total Assets", "Current Assets", "Total Liabilities",
    "Current Liabilities", "Total Equity", "Cash Flow from Operations",
}


# ── EDGAR Helpers ──────────────────────────────────────────────────────────────

def fetch_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()


def extract_annual_values(facts: dict, concept_tags: list) -> dict:
    """Return deduplicated annual (10-K) values keyed by fiscal year-end date."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in concept_tags:
        if tag not in us_gaap:
            continue
        units = us_gaap[tag].get("units", {})

        # EPS uses USD/shares, Shares Outstanding uses shares, others use USD
        usd_data = units.get("USD/shares", units.get("USD", units.get("shares", [])))

        # First try duration concepts with FY tag (income statement, cash flow)
        annual = [
            e for e in usd_data
            if e.get("form") == "10-K" and e.get("fp") == FISCAL_YEAR
        ]

        # Fallback: instant concepts (e.g. Shares Outstanding) have no fp field
        if not annual:
            annual = [e for e in usd_data if e.get("form") == "10-K"]

        if not annual:
            continue

        by_year = {}
        for e in annual:
            end, filed = e["end"], e.get("filed", "")
            if end not in by_year or filed > by_year[end]["filed"]:
                by_year[end] = e

        return {k: v["val"] for k, v in sorted(by_year.items(), reverse=True)}

    return {}


def fetch_edgar_data(cik: str) -> tuple:
    """Fetch all EDGAR metrics for a company. Returns (entity_name, {metric: {date: val}})."""
    facts = fetch_company_facts(cik)
    entity_name = facts.get("entityName", cik)
    data = {
        metric: extract_annual_values(facts, tags)
        for metric, tags in EDGAR_METRICS.items()
    }
    return entity_name, data


# ── yfinance Market Data ───────────────────────────────────────────────────────

def fetch_historical_market_data(ticker: str, fiscal_year_ends: list) -> dict:
    """
    Fetch historical stock price and 52-week high/low for each fiscal year-end date.
    """
    result = {date: {} for date in fiscal_year_ends}

    try:
        tk = yf.Ticker(ticker)
        beta = tk.info.get("beta")

        earliest = min(fiscal_year_ends)
        start_dt = (pd.Timestamp(earliest) - pd.DateOffset(days=370)).strftime("%Y-%m-%d")
        end_dt   = (pd.Timestamp(max(fiscal_year_ends)) + pd.DateOffset(days=10)).strftime("%Y-%m-%d")

        hist = tk.history(start=start_dt, end=end_dt)

        # Strip timezone
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)

        for date_str in fiscal_year_ends:
            date_ts = pd.Timestamp(date_str)

            window = hist[(hist.index >= date_ts - pd.Timedelta(days=7)) &
                          (hist.index <= date_ts + pd.Timedelta(days=7))]

            if window.empty:
                continue

            # Price: last trading day on or before fiscal year-end
            on_or_before = window[window.index <= date_ts + pd.Timedelta(days=1)]
            if not on_or_before.empty:
                price = round(float(on_or_before.iloc[-1]["Close"]), 2)
            else:
                price = round(float(window.iloc[0]["Close"]), 2)

            # 52-Week High / Low
            week52 = hist[(hist.index > date_ts - pd.DateOffset(days=365)) &
                          (hist.index <= date_ts)]
            high_52 = round(float(week52["High"].max()), 2) if not week52.empty else None
            low_52  = round(float(week52["Low"].min()),  2) if not week52.empty else None

            result[date_str] = {
                "Stock Price ($)"  : price,
                "52-Week High ($)" : high_52,
                "52-Week Low ($)"  : low_52,
                "Beta"             : beta,
            }

    except Exception as e:
        import traceback
        print(f"    yfinance error for {ticker}: {e}")
        traceback.print_exc()

    return result


def calculate_market_ratios(row: dict) -> dict:
    """Derive Market Cap, P/E, and Price-to-Sales from stock price + EDGAR data."""
    price  = row.get("Stock Price ($)")
    shares = row.get("Shares Outstanding")   # raw units from EDGAR
    eps    = row.get("EPS (Diluted)")
    rev    = row.get("Total Revenue")        # in $M

    market_cap = round(price * shares / 1e6, 2) if price and shares else None
    pe_ratio   = round(price / eps, 2)           if price and eps and eps > 0 else None
    pts        = round(market_cap / rev, 2)      if market_cap and rev and rev > 0 else None

    return {
        "Market Cap ($M)"  : market_cap,
        "P/E Ratio"        : pe_ratio,
        "Price-to-Sales"   : pts,
    }


# ── Ratio Calculations ─────────────────────────────────────────────────────────

def safe_divide(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator * 100, 2)


def safe_ratio(numerator, denominator, decimals=2):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, decimals)


def calculate_ratios(row: dict, prev_row: dict) -> dict:
    """Calculate all derived ratios for a single year's row."""
    rev  = row.get("Total Revenue")
    gp   = row.get("Gross Profit")
    op   = row.get("Operating Income")
    ni   = row.get("Net Income")
    ta   = row.get("Total Assets")
    ca   = row.get("Current Assets")
    tl   = row.get("Total Liabilities")
    cl   = row.get("Current Liabilities")
    eq   = row.get("Total Equity")

    prev_rev = prev_row.get("Total Revenue") if prev_row else None
    prev_ni  = prev_row.get("Net Income")    if prev_row else None

    return {
        "Gross Margin (%)"          : safe_divide(gp, rev),
        "Operating Margin (%)"      : safe_divide(op, rev),
        "Net Profit Margin (%)"     : safe_divide(ni, rev),
        "ROA (%)"                   : safe_divide(ni, ta),
        "ROE (%)"                   : safe_divide(ni, eq),
        "Debt-to-Equity"            : safe_ratio(tl, eq),
        "Current Ratio"             : safe_ratio(ca, cl),
        "YoY Revenue Growth (%)"    : safe_divide(rev - prev_rev, prev_rev) if rev and prev_rev else None,
        "YoY Net Income Growth (%)": safe_divide(ni - prev_ni,  prev_ni)  if ni  and prev_ni  else None,
    }


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    all_rows = []

    for company in COMPANIES:
        name, cik, ticker = company["name"], company["cik"], company["ticker"]
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        print(f"  Fetching EDGAR data...", end=" ", flush=True)
        try:
            entity_name, edgar_data = fetch_edgar_data(cik)
            print("done.")
        except Exception as e:
            print(f"ERROR — {e}")
            continue

        # Collect recent fiscal year-end dates
        all_dates = set()
        for vals in edgar_data.values():
            all_dates.update(vals.keys())
        recent_dates = sorted(all_dates, reverse=True)[:NUM_YEARS]

        print(f"  Fetching historical market data ({ticker})...", end=" ", flush=True)
        historical_market = fetch_historical_market_data(ticker, recent_dates)
        print("done.")

        # Build rows
        year_rows = []
        for date in recent_dates:
            row = {
                "Company"        : entity_name,
                "Ticker"         : ticker,
                "Fiscal Year End": date,
                "Year"           : int(date[:4]),
            }
            for metric in EDGAR_METRICS:
                val = edgar_data[metric].get(date)
                if val is not None and metric in USD_METRICS:
                    val = round(val / 1e6, 2)   # convert to $M
                row[metric] = val

            row.update(historical_market.get(date, {}))
            year_rows.append(row)

        # Ratios (need previous year for YoY)
        for i, row in enumerate(year_rows):
            prev_row = year_rows[i + 1] if i + 1 < len(year_rows) else None
            ratios = calculate_ratios(row, prev_row)
            row.update(ratios)

        # Market-based ratios
        for row in year_rows:
            row.update(calculate_market_ratios(row))

        all_rows.extend(year_rows)
        print(f"  Added {len(year_rows)} years of data.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["Company", "Year"], ascending=[True, False]).reset_index(drop=True)
    return df


# ==============================================================================
# SECTION 2 — Financial Analysis Functions
# ==============================================================================

def get_company(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return rows for a company by partial name match, sorted newest first."""
    mask = df["Company"].str.contains(name, case=False, na=False)
    result = df[mask].sort_values("Year", ascending=False)
    if result.empty:
        raise ValueError(f"No company found matching '{name}'. "
                         f"Available: {df['Company'].unique().tolist()}")
    return result


def fmt(val, prefix="$", suffix="M", decimals=2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def fmt_pct(val) -> str:
    return fmt(val, prefix="", suffix="%")


def fmt_ratio(val) -> str:
    return fmt(val, prefix="", suffix="x")


def arrow(current, previous) -> str:
    if current is None or previous is None:
        return ""
    if pd.isna(current) or pd.isna(previous):
        return ""
    if current > previous:
        return "↑"
    elif current < previous:
        return "↓"
    return "→"


def company_snapshot(df: pd.DataFrame, company_name: str) -> str:
    rows   = get_company(df, company_name)
    latest = rows.iloc[0]
    prev   = rows.iloc[1] if len(rows) > 1 else None
    name   = latest["Company"]
    year   = latest["Year"]

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  {name} — Fiscal Year {year} Snapshot")
    lines.append(f"{'='*60}")

    lines.append("\nINCOME STATEMENT")
    lines.append(f"  Revenue          : {fmt(latest.get('Total Revenue'))}"
                 f"  {arrow(latest.get('Total Revenue'), prev.get('Total Revenue') if prev is not None else None)}")
    lines.append(f"  Gross Profit     : {fmt(latest.get('Gross Profit'))}"
                 f"  {arrow(latest.get('Gross Profit'), prev.get('Gross Profit') if prev is not None else None)}")
    lines.append(f"  Operating Income : {fmt(latest.get('Operating Income'))}"
                 f"  {arrow(latest.get('Operating Income'), prev.get('Operating Income') if prev is not None else None)}")
    lines.append(f"  Net Income       : {fmt(latest.get('Net Income'))}"
                 f"  {arrow(latest.get('Net Income'), prev.get('Net Income') if prev is not None else None)}")
    lines.append(f"  EPS (Diluted)    : ${latest.get('EPS (Diluted)', 'N/A')}")
    lines.append(f"  R&D Expenses     : {fmt(latest.get('R&D Expenses'))}")

    lines.append("\nPROFITABILITY")
    lines.append(f"  Gross Margin     : {fmt_pct(latest.get('Gross Margin (%)'))} ")
    lines.append(f"  Operating Margin : {fmt_pct(latest.get('Operating Margin (%)'))} ")
    lines.append(f"  Net Margin       : {fmt_pct(latest.get('Net Profit Margin (%)'))} ")
    lines.append(f"  ROA              : {fmt_pct(latest.get('ROA (%)'))} ")
    lines.append(f"  ROE              : {fmt_pct(latest.get('ROE (%)'))} ")

    lines.append("\nBALANCE SHEET")
    lines.append(f"  Total Assets     : {fmt(latest.get('Total Assets'))}"
                 f"  {arrow(latest.get('Total Assets'), prev.get('Total Assets') if prev is not None else None)}")
    lines.append(f"  Total Liabilities: {fmt(latest.get('Total Liabilities'))}"
                 f"  {arrow(latest.get('Total Liabilities'), prev.get('Total Liabilities') if prev is not None else None)}")
    lines.append(f"  Total Equity     : {fmt(latest.get('Total Equity'))}"
                 f"  {arrow(latest.get('Total Equity'), prev.get('Total Equity') if prev is not None else None)}")
    lines.append(f"  Current Ratio    : {fmt_ratio(latest.get('Current Ratio'))}")
    lines.append(f"  Debt-to-Equity   : {fmt_ratio(latest.get('Debt-to-Equity'))}")

    lines.append("\nCASH FLOW")
    lines.append(f"  Operating CF     : {fmt(latest.get('Cash Flow from Operations'))}"
                 f"  {arrow(latest.get('Cash Flow from Operations'), prev.get('Cash Flow from Operations') if prev is not None else None)}")

    lines.append("\nMARKET VALUATION")
    lines.append(f"  Stock Price      : ${latest.get('Stock Price ($)', 'N/A')}")
    lines.append(f"  Market Cap       : {fmt(latest.get('Market Cap ($M)'))}")
    lines.append(f"  P/E Ratio        : {fmt_ratio(latest.get('P/E Ratio'))}")
    lines.append(f"  Price-to-Sales   : {fmt_ratio(latest.get('Price-to-Sales'))}")
    lines.append(f"  52-Week High     : ${latest.get('52-Week High ($)', 'N/A')}")
    lines.append(f"  52-Week Low      : ${latest.get('52-Week Low ($)', 'N/A')}")
    lines.append(f"  Beta             : {latest.get('Beta', 'N/A')}")

    lines.append("\nGROWTH (vs Prior Year)")
    lines.append(f"  Revenue Growth   : {fmt_pct(latest.get('YoY Revenue Growth (%)'))} ")
    lines.append(f"  Net Income Growth: {fmt_pct(latest.get('YoY Net Income Growth (%)'))} ")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def profitability_summary(df: pd.DataFrame, company_name: str) -> str:
    rows = get_company(df, company_name)
    name = rows.iloc[0]["Company"]

    lines = []
    lines.append(f"{name} — Profitability (5-Year History)")
    lines.append(f"{'─'*60}")
    lines.append(f"{'Year':<8} {'Net Margin':>12} {'ROE':>10} {'ROA':>10} {'Op. Margin':>12}")
    lines.append(f"{'─'*60}")

    for _, row in rows.iterrows():
        lines.append(
            f"{int(row['Year']):<8}"
            f"{fmt_pct(row.get('Net Profit Margin (%)')):>12}"
            f"{fmt_pct(row.get('ROE (%)')):>10}"
            f"{fmt_pct(row.get('ROA (%)')):>10}"
            f"{fmt_pct(row.get('Operating Margin (%)')):>12}"
        )

    margins = rows["Net Profit Margin (%)"].dropna()
    if not margins.empty:
        avg = margins.mean()
        trend_dir = "improving" if margins.iloc[0] > margins.iloc[-1] else "declining"
        lines.append(f"\n  Average net margin over {len(margins)} years: {avg:.1f}%")
        lines.append(f"  Trend: {trend_dir} (from {margins.iloc[-1]:.1f}% → {margins.iloc[0]:.1f}%)")

    return "\n".join(lines)


def balance_sheet_health(df: pd.DataFrame, company_name: str) -> str:
    rows   = get_company(df, company_name)
    latest = rows.iloc[0]
    name   = latest["Company"]
    year   = latest["Year"]

    cr  = latest.get("Current Ratio")
    dte = latest.get("Debt-to-Equity")
    eq  = latest.get("Total Equity")
    tl  = latest.get("Total Liabilities")

    lines = []
    lines.append(f"{name} — Balance Sheet Health ({year})")
    lines.append(f"{'─'*50}")
    lines.append(f"  Current Ratio    : {fmt_ratio(cr)}")
    lines.append(f"  Debt-to-Equity   : {fmt_ratio(dte)}")
    lines.append(f"  Total Equity     : {fmt(eq)}")
    lines.append(f"  Total Liabilities: {fmt(tl)}")

    lines.append(f"\n  Interpretation:")
    if cr is not None and not pd.isna(cr):
        if cr >= 2:
            lines.append(f"  ✓ Current ratio of {cr:.1f}x is strong — well covered short-term.")
        elif cr >= 1:
            lines.append(f"  ~ Current ratio of {cr:.1f}x is adequate but not a large buffer.")
        else:
            lines.append(f"  ✗ Current ratio of {cr:.1f}x is below 1 — short-term liquidity risk.")

    if dte is not None and not pd.isna(dte):
        if dte < 1:
            lines.append(f"  ✓ Debt-to-equity of {dte:.1f}x — more equity than debt, conservative.")
        elif dte < 2:
            lines.append(f"  ~ Debt-to-equity of {dte:.1f}x — moderate leverage.")
        else:
            lines.append(f"  ✗ Debt-to-equity of {dte:.1f}x — highly leveraged.")

    return "\n".join(lines)


def growth_analysis(df: pd.DataFrame, company_name: str) -> str:
    rows = get_company(df, company_name)
    name = rows.iloc[0]["Company"]

    lines = []
    lines.append(f"{name} — Growth Analysis")
    lines.append(f"{'─'*60}")
    lines.append(f"{'Year':<8} {'Revenue ($M)':>14} {'Rev Growth':>12} {'Net Income ($M)':>16} {'NI Growth':>11}")
    lines.append(f"{'─'*60}")

    for _, row in rows.iterrows():
        lines.append(
            f"{int(row['Year']):<8}"
            f"{fmt(row.get('Total Revenue'), prefix='', suffix=''):>14}"
            f"{fmt_pct(row.get('YoY Revenue Growth (%)')):>12}"
            f"{fmt(row.get('Net Income'), prefix='', suffix=''):>16}"
            f"{fmt_pct(row.get('YoY Net Income Growth (%)')):>11}"
        )

    rev_vals = rows["Total Revenue"].dropna()
    if len(rev_vals) >= 2:
        years = len(rev_vals) - 1
        cagr = ((rev_vals.iloc[0] / rev_vals.iloc[-1]) ** (1 / years) - 1) * 100
        lines.append(f"\n  Revenue CAGR ({years} years): {cagr:.1f}%")

    return "\n".join(lines)


def compare_companies(df: pd.DataFrame, metric: str, year: int = None) -> str:
    if year is None:
        year = df["Year"].max()

    subset = df[df["Year"] == year][["Company", "Year", metric]].dropna()
    subset = subset.sort_values(metric, ascending=False).reset_index(drop=True)

    lines = []
    lines.append(f"Company Comparison — {metric} ({year})")
    lines.append(f"{'─'*45}")

    for i, row in subset.iterrows():
        val = row[metric]
        if "%" in metric:
            formatted = fmt_pct(val)
        elif "$" in metric or metric in ["Total Revenue", "Net Income", "Market Cap ($M)"]:
            formatted = fmt(val)
        else:
            formatted = fmt_ratio(val)
        rank = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        lines.append(f"  {rank}  {row['Company']:<30} {formatted}")

    return "\n".join(lines)


def trend_analysis(df: pd.DataFrame, company_name: str) -> str:
    rows = get_company(df, company_name)
    name = rows.iloc[0]["Company"]

    tracked = [
        "Total Revenue", "Net Income", "Gross Margin (%)",
        "Net Profit Margin (%)", "ROE (%)", "Current Ratio",
        "Debt-to-Equity", "Cash Flow from Operations", "Market Cap ($M)"
    ]

    lines = []
    lines.append(f"{name} — 5-Year Trend Report")
    lines.append(f"{'─'*50}")

    for metric in tracked:
        series = rows[metric].dropna()
        if len(series) < 2:
            continue

        latest_val = series.iloc[0]
        oldest_val = series.iloc[-1]
        max_val    = series.max()
        min_val    = series.min()

        pct_change = (latest_val - oldest_val) / abs(oldest_val) * 100 if oldest_val != 0 else 0
        direction  = "↑" if pct_change > 2 else ("↓" if pct_change < -2 else "→")

        flag = ""
        if latest_val == max_val:
            flag = " ★ 5-yr HIGH"
        elif latest_val == min_val:
            flag = " ▼ 5-yr LOW"

        lines.append(f"  {metric:<30} {direction}  {pct_change:+.1f}%{flag}")

    return "\n".join(lines)


def full_report(df: pd.DataFrame, company_name: str) -> str:
    sections = [
        company_snapshot(df, company_name),
        profitability_summary(df, company_name),
        balance_sheet_health(df, company_name),
        growth_analysis(df, company_name),
        trend_analysis(df, company_name),
    ]
    return "\n\n".join(sections)


# ==============================================================================
# SECTION 3 — Claude-Powered Chatbot
# ==============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not set.\n"
        "Create a .env file in the same folder with:\n"
        "    ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or run: export ANTHROPIC_API_KEY='sk-ant-...'"
    )
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a financial research analyst assistant with access to real SEC
EDGAR data for Microsoft, Apple, Tesla, Meta (Facebook), Amazon, Netflix, and Alphabet
(Google) covering the past 5 fiscal years.

You have TWO categories of tools:

1. STRUCTURED ANALYSIS TOOLS — for numerical / quantitative questions about ratios,
   margins, growth rates, comparisons, balance sheet metrics. These return exact
   numbers from the XBRL financial data:
     • company_snapshot, profitability_summary, balance_sheet_health,
       growth_analysis, compare_companies, trend_analysis, full_report

2. SEMANTIC SEARCH TOOL — for qualitative / narrative questions about strategy,
   risk factors, business descriptions, management discussion (MD&A), competitive
   positioning, or anything that requires reading the actual 10-K narrative text:
     • search_filings  →  uses RAG (vector + BM25 hybrid retrieval) over the
       indexed 10-K text and returns relevant excerpts with citations

ROUTING RULES:
  • "What is Apple's debt-to-equity ratio?"               → balance_sheet_health
  • "Compare ROE across all companies"                    → compare_companies
  • "How fast is Amazon growing?"                         → growth_analysis
  • "What risks does Tesla flag about manufacturing?"     → search_filings
  • "How does Microsoft describe its AI strategy?"        → search_filings
  • "What are Netflix's main competitive moats?"          → search_filings
  • For complex questions, call multiple tools and synthesize.

When presenting numbers, cite the fiscal year. When using search_filings results,
keep the inline citations like [Apple FY2024, Risk Factors].
Keep responses concise and insightful — don't just repeat raw data, add context."""


# ── Tool Definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "company_snapshot",
        "description": (
            "Get a full financial snapshot of a single company for its most recent "
            "fiscal year. Covers revenue, income, margins, balance sheet, cash flow, "
            "and market valuation. Use when the user asks for an overview or summary "
            "of a specific company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company name, e.g. 'Microsoft', 'Apple', 'Tesla'"
                }
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "profitability_summary",
        "description": (
            "Show a 5-year profitability history for a company including net margin, "
            "ROE, ROA, and operating margin with trend direction. Use when the user "
            "asks about profitability, margins, or returns over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "balance_sheet_health",
        "description": (
            "Assess a company's balance sheet strength — liquidity, leverage, "
            "equity vs liabilities. Returns a plain-English verdict. Use when the "
            "user asks about debt, financial health, or stability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "growth_analysis",
        "description": (
            "Show revenue and net income growth over 5 years including year-over-year "
            "percentages and CAGR. Use when the user asks how fast a company is growing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "compare_companies",
        "description": (
            "Rank all companies by a specific financial metric for a given year. "
            "Use when the user asks to compare companies or find who leads on a metric. "
            "Available metrics: 'Net Profit Margin (%)', 'ROE (%)', 'ROA (%)', "
            "'Total Revenue', 'Net Income', 'Market Cap ($M)', 'P/E Ratio', "
            "'Debt-to-Equity', 'Current Ratio', 'Gross Margin (%)', "
            "'YoY Revenue Growth (%)'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "The metric to compare companies on"
                },
                "year": {
                    "type": "integer",
                    "description": "Fiscal year to compare (optional, defaults to most recent)"
                }
            },
            "required": ["metric"]
        }
    },
    {
        "name": "trend_analysis",
        "description": (
            "Detect 5-year trends for a company — which metrics are at highs/lows "
            "and which direction they are moving. Use when the user asks about trends, "
            "momentum, or what's changing over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "full_report",
        "description": (
            "Generate a complete multi-section financial report for a company. "
            "Use when the user wants a deep dive or comprehensive analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "search_filings",
        "description": (
            "Search the actual narrative text of SEC 10-K annual filings using "
            "semantic search (RAG with hybrid vector + BM25 retrieval). Use this "
            "for QUALITATIVE questions that require reading the filing text — risk "
            "factors, business strategy, MD&A, competitive positioning, segment "
            "descriptions, regulatory concerns, ESG commentary, or any narrative "
            "that is NOT a number in the financial tables. "
            "Examples: 'What risks does Tesla mention about lithium supply?', "
            "'How does Microsoft describe its AI strategy?', "
            "'What does Netflix say about password sharing?'. "
            "Returns top-K most relevant excerpts with citations. "
            "Optionally filter by company_name and/or year for more focused results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query describing what to find in the filings"
                },
                "company_name": {
                    "type": "string",
                    "description": "Optional company name or ticker (Microsoft / MSFT). Omit to search all companies."
                },
                "year": {
                    "type": "integer",
                    "description": "Optional fiscal year to filter by (e.g. 2024). Omit to search all years."
                }
            },
            "required": ["query"]
        }
    },
]


# ── Tool Dispatcher ────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, args: dict, df) -> str:
    """Call the matching analysis function and return its plain-text result."""
    dispatch = {
        "company_snapshot"     : lambda: company_snapshot(df, args["company_name"]),
        "profitability_summary": lambda: profitability_summary(df, args["company_name"]),
        "balance_sheet_health" : lambda: balance_sheet_health(df, args["company_name"]),
        "growth_analysis"      : lambda: growth_analysis(df, args["company_name"]),
        "compare_companies"    : lambda: compare_companies(df, args["metric"], args.get("year")),
        "trend_analysis"       : lambda: trend_analysis(df, args["company_name"]),
        "full_report"          : lambda: full_report(df, args["company_name"]),
        "search_filings"       : lambda: _get_search_filings()(
            query=args["query"],
            company_name=args.get("company_name"),
            year=args.get("year"),
        ),
    }
    fn = dispatch.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    try:
        return fn()
    except Exception as e:
        return f"Error running {tool_name}: {e}"


# ── Core Ask Function ──────────────────────────────────────────────────────────

def ask(question: str, history: list, df) -> str:
    """
    Send a question to Claude with tool access.
    Handles tool calls automatically and returns the final response.
    """
    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=TOOLS,
    )

    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [calling {block.name}({block.input})]")
                result = dispatch_tool(block.name, block.input, df)
                tool_results.append({
                    "type"       : "tool_result",
                    "tool_use_id": block.id,
                    "content"    : result,
                })

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )

    return next(
        (block.text for block in response.content if hasattr(block, "text")),
        "No response generated."
    )


# ── Chat Loop ──────────────────────────────────────────────────────────────────

def run_chatbot(df):
    """Start an interactive chat loop. Type 'quit' or 'exit' to stop."""
    companies = df["Company"].unique().tolist()
    years     = sorted(df["Year"].unique(), reverse=True)

    print("="*60)
    print("  Financial Analysis Chatbot (Claude)")
    print("="*60)
    print(f"  Companies : {', '.join(companies)}")
    print(f"  Data range: {years[-1]} – {years[0]}")
    print("  Type 'quit' to exit")
    print("="*60)
    print()
    print("Example questions:")
    print("  • Give me a snapshot of Microsoft")
    print("  • How profitable has Apple been over the past 5 years?")
    print("  • Which company has the highest profit margin?")
    print("  • Is Tesla's balance sheet improving?")
    print("  • How fast has Microsoft grown its revenue?")
    print("  • Compare all companies by market cap")
    print()

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        response = ask(user_input, history, df)
        print(f"Assistant: {response}")
        print()

        # Keep conversation history (text only)
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant",  "content": response})

        # Trim to last 10 turns to avoid token limits
        if len(history) > 20:
            history = history[-20:]


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print(f"SEC EDGAR + yfinance Enriched Dataset  |  {datetime.today().strftime('%Y-%m-%d')}")
    df = build_dataset()

    print(f"\n\n{'='*50}")
    print(f"  Dataset complete: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Companies : {df['Company'].nunique()}")
    print(f"  Years     : {sorted(df['Year'].unique(), reverse=True)}")
    print(f"{'='*50}")

    # Save to CSV alongside this script
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

    run_chatbot(df)