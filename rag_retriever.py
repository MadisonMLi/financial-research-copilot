"""
Hybrid retrieval over the 10-K corpus + the `search_filings` tool function.

Architecture:
  - FAISS (dense vector search via Sentence Transformers)
  - BM25 (sparse keyword search)
  - EnsembleRetriever combines them (60% vector, 40% BM25)
  - Optional metadata filters: company / year

The `search_filings()` function is what chatbot.py exposes to Claude as a tool.
It returns formatted text with citations for the LLM, AND caches the source
documents so the Streamlit UI can show an expandable Sources panel.
"""

import os
from pathlib import Path
from typing import Optional

# EnsembleRetriever location varies across LangChain versions
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = Path(__file__).parent / "vectorstore"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Map common company names → tickers, so Claude can pass either form
NAME_TO_TICKER = {
    "microsoft":  "MSFT",
    "apple":      "AAPL",
    "tesla":      "TSLA",
    "meta":       "META",
    "facebook":   "META",
    "amazon":     "AMZN",
    "netflix":    "NFLX",
    "alphabet":   "GOOGL",
    "google":     "GOOGL",
}

# Module-level cache of the most recent retrieval's source documents.
# The Streamlit UI reads from this after each search_filings() call to
# render the expandable "Sources" panel.
_LAST_SOURCES: list[dict] = []

# Cached singletons — load only once per process
_embeddings = None
_vectorstore = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                "No FAISS index found. Run `python rag_ingest.py` first."
            )
        _vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def _resolve_ticker(company_name: Optional[str]) -> Optional[str]:
    """Convert 'Microsoft' or 'msft' → 'MSFT'."""
    if not company_name:
        return None
    cn = company_name.strip().lower()
    if cn in NAME_TO_TICKER:
        return NAME_TO_TICKER[cn]
    if cn.upper() in NAME_TO_TICKER.values():
        return cn.upper()
    return None


def _build_hybrid(filters: Optional[dict], k: int) -> EnsembleRetriever:
    vs = _get_vectorstore()
    search_kwargs: dict = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
    dense = vs.as_retriever(search_kwargs=search_kwargs)

    all_docs = list(vs.docstore._dict.values())
    if filters:
        all_docs = [
            d for d in all_docs
            if all(d.metadata.get(key) == val for key, val in filters.items())
        ]

    if not all_docs:
        return dense  # type: ignore[return-value]

    bm25 = BM25Retriever.from_documents(all_docs, k=k)
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])


def search_filings(
    query: str,
    company_name: Optional[str] = None,
    year: Optional[int] = None,
    k: int = 5,
) -> str:
    """
    Search the 10-K corpus and return a formatted context block for Claude.

    This function is the body of the `search_filings` tool registered in chatbot.py.
    It also caches the retrieved source documents in _LAST_SOURCES so the
    Streamlit UI can show citations.
    """
    global _LAST_SOURCES

    filters: dict = {}
    ticker = _resolve_ticker(company_name)
    if ticker:
        filters["ticker"] = ticker
    if year:
        filters["year"] = int(year)

    try:
        retriever = _build_hybrid(filters or None, k=k)
        docs = retriever.invoke(query)
    except FileNotFoundError as e:
        _LAST_SOURCES = []
        return f"ERROR: {e}"
    except Exception as e:
        _LAST_SOURCES = []
        return f"ERROR during retrieval: {e}"

    if not docs:
        _LAST_SOURCES = []
        return "No relevant excerpts found in the indexed 10-K filings for this query."

    # Cache sources for the UI
    _LAST_SOURCES = [
        {
            "company": d.metadata.get("company", "Unknown"),
            "ticker":  d.metadata.get("ticker", "?"),
            "year":    d.metadata.get("year", "?"),
            "section": d.metadata.get("section", "general").replace("_", " ").title(),
            "excerpt": d.page_content,
        }
        for d in docs
    ]

    # Format text for Claude with inline citations
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        header = (
            f"[{i}] {m.get('company', '?')} ({m.get('ticker', '?')}) "
            f"| FY{m.get('year', '?')} "
            f"| Section: {m.get('section', 'general').replace('_', ' ').title()}"
        )
        parts.append(f"{header}\n{d.page_content}")

    context = "\n\n---\n\n".join(parts)
    return (
        f"Found {len(docs)} relevant excerpts from 10-K filings:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"When citing facts in your answer, reference excerpts as "
        f"[Company FYxxxx, Section]."
    )


def get_last_sources() -> list[dict]:
    """Return a copy of the most recent search_filings() source documents."""
    return list(_LAST_SOURCES)


def clear_last_sources() -> None:
    global _LAST_SOURCES
    _LAST_SOURCES = []
