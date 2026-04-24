# Financial Research Copilot — Hybrid Tool-Use + RAG

An agentic financial research assistant that combines **two AI patterns** in one
system, with Claude routing automatically between them based on the question.

## Demo Video

[![Financial Research Copilot](https://img.youtube.com/vi/qO0IzU5ikOY/0.jpg)]([https://www.youtube.com/watch?v=qO0IzU5ikOY](https://youtu.be/qO0IzU5ikOY))


| Pattern | When it's used | Built on |
|---|---|---|
| **Tool-Use over structured XBRL data** | Numerical questions: ratios, margins, growth, comparisons | Anthropic Claude tool calling + SEC XBRL API + yfinance + Pandas |
| **RAG over 10-K filing text** | Narrative questions: risks, strategy, MD&A, business descriptions | LangChain + FAISS + BM25 + Sentence Transformers + GPT-style retrieval |

---

## Architecture

```
                       ┌────────────────────────────────────────────┐
                       │             User question                  │
                       └────────────────────┬───────────────────────┘
                                            ▼
                               ┌────────────────────────┐
                               │  Claude (tool-use)     │
                               │  — picks the right     │
                               │    tool based on Q     │
                               └──┬──────────────────┬──┘
                                  │                  │
            (numerical Q)         │                  │      (narrative Q)
                                  ▼                  ▼
                ┌──────────────────────────┐    ┌──────────────────────────┐
                │  STRUCTURED ANALYSIS     │    │  search_filings (RAG)    │
                │  (Pandas + XBRL data)    │    │  ┌────────────────────┐  │
                │                          │    │  │ Hybrid Retrieval   │  │
                │  • company_snapshot      │    │  │ ─ FAISS  (60%)     │  │
                │  • profitability_summary │    │  │ ─ BM25   (40%)     │  │
                │  • balance_sheet_health  │    │  └────────────────────┘  │
                │  • growth_analysis       │    │  ┌────────────────────┐  │
                │  • compare_companies     │    │  │ Metadata filter    │  │
                │  • trend_analysis        │    │  │ company / year     │  │
                │  • full_report           │    │  └────────────────────┘  │
                │                          │    │       ▼                  │
                │  Returns exact numbers   │    │  Top-K chunks with       │
                │                          │    │  citations              │
                └──────────────────────────┘    └──────────────────────────┘
                                  │                  │
                                  └────────┬─────────┘
                                           ▼
                          ┌──────────────────────────────┐
                          │  Claude synthesizes answer   │
                          │  with citations              │
                          └──────────────────────────────┘
                                           ▼
                          ┌──────────────────────────────┐
                          │  Streamlit UI                │
                          │  • KPI cards + Plotly charts │
                          │  • Sources panel (RAG)       │
                          │  • RAG Evaluation tab        │
                          └──────────────────────────────┘
```

---

## Quickstart

### 1. Install

```bash
cd financial-research-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
# edit .env — add ANTHROPIC_API_KEY and SEC_USER_AGENT_EMAIL
```

### 3. Build the RAG index (one-time)

```bash
# Downloads 10-Ks for all 7 companies, parses, chunks, embeds, saves FAISS index
python rag_ingest.py

# Or pick a subset
python rag_ingest.py --tickers AAPL MSFT TSLA --years 2024
```

This takes ~3-5 minutes total (download is the slow part).

### 4. Launch the app

```bash
streamlit run streamlit_app.py
```

---

## Try it

**Numerical questions** (route to structured tools):
- "Compare ROE across all companies"
- "Show Apple's profitability over 5 years"
- "What's Tesla's debt-to-equity ratio?"

**Narrative questions** (route to RAG → see Sources panel):
- "What manufacturing risks does Tesla mention in its 10-K?"
- "How does Microsoft describe its AI strategy?"
- "What does Netflix say about password sharing?"

**Compound questions** (route to BOTH):
- "Compare Tesla and Apple's profitability and explain what each says about their key risks"

---

## Evaluation

Run from the **📊 RAG Evaluation tab** in the UI, or via CLI:

```bash
python evaluate.py --k 5
```

Reports:
- **Hit Rate @K** — fraction of questions whose answer chunk is in the top K
- **MRR** — Mean Reciprocal Rank of the first correct hit
- **Per-question table** + **Hit Rate by 10-K section** chart

The test set (`eval/test_questions.json`) covers all 7 companies with 2 narrative questions each.

---
