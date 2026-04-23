"""
RAG Retrieval Evaluation Framework
===================================

Measures the quality of the `search_filings` retriever using two standard metrics:

  Hit Rate @K : Was a correct chunk in the top K retrieved?
                A "correct" chunk is one that BOTH belongs to the expected
                company AND contains the expected keyword in its text. This
                strict definition prevents false positives where the keyword
                appears in the wrong company's filing.
  MRR         : Mean Reciprocal Rank — average of (1/rank) for the first hit.
                Rewards ranking the correct chunk near the top.

Usage (CLI):
    python evaluate.py                # k=5, default questions
    python evaluate.py --k 10         # custom k
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from rag_retriever import _build_hybrid, _resolve_ticker

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval" / "test_questions.json"


def _check_hit(docs: list, question_meta: dict) -> tuple[bool, int, str]:
    """Return (hit, rank_of_first_hit, top_source_company)."""
    expected_ticker = (question_meta.get("ticker") or "").upper()
    expected_keyword = (question_meta.get("expected_keyword") or "").lower()
    top_source = docs[0].metadata.get("company", "?") if docs else "?"

    for rank, doc in enumerate(docs, 1):
        chunk_ticker = (doc.metadata.get("ticker") or "").upper()
        chunk_text = doc.page_content.lower()

        ticker_match = bool(expected_ticker) and chunk_ticker == expected_ticker
        keyword_match = bool(expected_keyword) and expected_keyword in chunk_text

        # Strict: a chunk only counts as a hit if it belongs to the expected
        # company AND contains the expected keyword. This catches cases where
        # the retriever surfaces the right keyword from the wrong filing.
        if ticker_match and keyword_match:
            return True, rank, top_source

    return False, 0, top_source


def run_evaluation(questions: list[dict], k: int = 5) -> tuple[pd.DataFrame, dict]:
    """
    Run the question set through the hybrid retriever and compute Hit Rate / MRR.

    Returns:
        results_df: per-question DataFrame
        summary:    dict with hit_rate, mrr, n_questions
    """
    rows = []
    for q in questions:
        # Apply optional filters from the question metadata
        filters = {}
        if q.get("ticker"):
            t = _resolve_ticker(q["ticker"])
            if t:
                filters["ticker"] = t
        # We deliberately DO NOT filter by ticker for the eval — we want to test
        # whether the retriever finds the right company on its own. If you want
        # filtered eval, uncomment the next line:
        # filters_to_use = filters or None
        filters_to_use = None  # unfiltered: harder, more realistic

        retriever = _build_hybrid(filters_to_use, k=k)
        docs = retriever.invoke(q["question"])

        hit, rank, top_source = _check_hit(docs, q)
        rr = 1 / rank if hit else 0.0
        section = docs[0].metadata.get("section", "unknown") if docs else "unknown"

        rows.append({
            "question": q["question"],
            "expected_ticker": q.get("ticker", ""),
            "hit": hit,
            "rank": rank if hit else None,
            "reciprocal_rank": round(rr, 4),
            "top_source": top_source,
            "section": section.replace("_", " ").title(),
        })

    df = pd.DataFrame(rows)
    summary = {
        "n_questions": len(df),
        "hit_rate": df["hit"].mean() if len(df) else 0.0,
        "mrr": df["reciprocal_rank"].mean() if len(df) else 0.0,
        "k": k,
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--questions", default=str(EVAL_QUESTIONS_PATH))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    print(f"Loaded {len(questions)} evaluation questions.")

    df, summary = run_evaluation(questions, k=args.k)

    print(f"\n=== RAG Retrieval Evaluation (k={args.k}) ===")
    print(f"  Hit Rate @{args.k}: {summary['hit_rate']:.1%}")
    print(f"  MRR:            {summary['mrr']:.4f}")
    print(f"  Questions:      {summary['n_questions']}")
    print("\nPer-question breakdown:")
    print(
        df[["question", "expected_ticker", "hit", "rank", "reciprocal_rank", "top_source"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
