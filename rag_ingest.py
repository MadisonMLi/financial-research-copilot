"""
RAG Ingestion Pipeline: Download SEC 10-K filings → parse → chunk → embed → FAISS.

Builds the vector index used by the `search_filings` tool in chatbot.py.

Usage:
    python rag_ingest.py                                   # all 7 BCG companies, year 2024
    python rag_ingest.py --tickers AAPL MSFT --years 2024 2023
    python rag_ingest.py --skip-download                   # re-parse existing files
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sec_edgar_downloader import Downloader

load_dotenv()

# --- Config (matches the 7 companies in chatbot.py) ---
RAW_DIR = Path("data/raw")
VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "research@example.com")

DEFAULT_TICKERS = ["MSFT", "AAPL", "TSLA", "META", "AMZN", "NFLX", "GOOGL"]
DEFAULT_YEARS = [2024]

TICKER_TO_COMPANY = {
    "MSFT":  "Microsoft Corporation",
    "AAPL":  "Apple Inc.",
    "TSLA":  "Tesla, Inc.",
    "META":  "Meta Platforms, Inc.",
    "AMZN":  "Amazon.com, Inc.",
    "NFLX":  "Netflix, Inc.",
    "GOOGL": "Alphabet Inc.",
}

# 10-K standard sections — used for metadata tagging
SECTION_PATTERNS = {
    "business":       r"item\s+1[^a]\s*[:\.\-]?\s*business",
    "risk_factors":   r"item\s+1a\s*[:\.\-]?\s*risk\s*factors",
    "properties":     r"item\s+2\s*[:\.\-]?\s*properties",
    "legal":          r"item\s+3\s*[:\.\-]?\s*legal\s*proceedings",
    "mda":            r"item\s+7\s*[:\.\-]?\s*management",
    "market_risk":    r"item\s+7a\s*[:\.\-]?\s*quantitative",
    "financials":     r"item\s+8\s*[:\.\-]?\s*financial\s*statements",
}


def download_filings(tickers: list[str], years: list[int]) -> None:
    dl = Downloader("BCGProjectRAG", USER_AGENT_EMAIL, RAW_DIR)
    for ticker in tickers:
        for year in years:
            after = f"{year}-01-01"
            before = f"{year + 1}-06-30"
            print(f"  Downloading {ticker} 10-K ({year})...")
            try:
                dl.get("10-K", ticker, limit=1, after=after, before=before)
                time.sleep(0.5)  # be polite to EDGAR
            except Exception as e:
                print(f"  Warning: could not download {ticker} {year}: {e}")


def _detect_section(text: str) -> str:
    lower = text[:300].lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, lower):
            return section
    return "general"


def parse_filing(filepath: Path, ticker: str, year: int) -> list[Document]:
    raw = filepath.read_text(errors="ignore")

    # full-submission.txt has multiple <DOCUMENT> blocks; extract the 10-K body
    doc_match = re.search(
        r"<DOCUMENT>\s*<TYPE>10-K\b.*?</DOCUMENT>",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if doc_match:
        block = doc_match.group(0)
        html_match = re.search(r"<(?:html|HTML).*?</(?:html|HTML)>", block, re.DOTALL)
        html = html_match.group(0) if html_match else block
    else:
        html = raw

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    if len(text.strip()) < 500:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)

    company = TICKER_TO_COMPANY.get(ticker, ticker)
    docs = []
    for i, chunk in enumerate(chunks):
        section = _detect_section(chunk)
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "ticker": ticker,
                    "company": company,
                    "year": year,
                    "section": section,
                    "chunk_id": i,
                    "source": str(filepath.name),
                },
            )
        )
    return docs


def parse_all_filings(tickers: list[str], years: list[int]) -> list[Document]:
    all_docs: list[Document] = []
    base_dir = RAW_DIR / "sec-edgar-filings"
    for ticker in tickers:
        for year in years:
            ticker_dir = base_dir / ticker / "10-K"
            if not ticker_dir.exists():
                print(f"  No files found for {ticker} {year}, skipping.")
                continue
            parsed_one = False
            for filing_dir in sorted(ticker_dir.iterdir()):
                candidates = list(filing_dir.glob("full-submission.txt"))
                candidates += [
                    f for f in filing_dir.rglob("*.htm")
                    if not (f.stem.startswith("R") and f.stem[1:].isdigit())
                ]
                for f in candidates:
                    docs = parse_filing(f, ticker, year)
                    if docs:
                        print(f"  Parsed {ticker} {year}: {len(docs)} chunks from {f.name}")
                        all_docs.extend(docs)
                        parsed_one = True
                        break
                if parsed_one:
                    break
    return all_docs


def build_vectorstore(docs: list[Document]) -> None:
    print(f"\nEmbedding {len(docs)} chunks with '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Saved FAISS index to {VECTORSTORE_DIR}/")

    # Manifest for the UI
    manifest = {}
    for doc in docs:
        key = f"{doc.metadata['ticker']}_{doc.metadata['year']}"
        if key not in manifest:
            manifest[key] = {
                "ticker": doc.metadata["ticker"],
                "company": doc.metadata["company"],
                "year": doc.metadata["year"],
                "chunks": 0,
            }
        manifest[key]["chunks"] += 1

    (VECTORSTORE_DIR / "manifest.json").write_text(
        json.dumps(list(manifest.values()), indent=2)
    )
    print(f"Saved manifest: {len(manifest)} filings indexed.")


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into FAISS")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print(f"Downloading 10-K filings: {args.tickers} | years: {args.years}")
        download_filings(args.tickers, args.years)

    print("\nParsing filings...")
    docs = parse_all_filings(args.tickers, args.years)

    if not docs:
        print("No documents parsed. Check that filings downloaded correctly.")
        return

    print(f"\nTotal chunks: {len(docs)}")
    build_vectorstore(docs)
    print("\nIngestion complete. Run `streamlit run streamlit_app.py` to start the app.")


if __name__ == "__main__":
    main()
