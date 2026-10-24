from pathlib import Path
from contextlib import asynccontextmanager
import json
import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configuration import ROOT_DIR, INDEX_PATH, TOKEN_PATTERN, MAX_LENGTH_SEARCH, TOP_K_SEARCH

# ============================================================
# 1. LOAD THE INDEX
# ============================================================

def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())

records: list[dict] = []
bm25 = None
corpus_tokens: list[list[str]] = []

def load_index():
    global records, bm25, corpus_tokens

    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"Index file not found: {INDEX_PATH}"
        )

    loaded_records = []

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if record.get("text", "").strip():
                loaded_records.append(record)

    if not loaded_records:
        raise RuntimeError("The document index contains no text.")

    loaded_tokens = [
        tokenize(record["text"])
        for record in loaded_records
    ]

    # BM25 requires a non-empty token corpus.
    # Empty-token documents are retained in records for alignment.
    # The corpus and records must always have the same length.
    loaded_tokens = [
        tokens if tokens else ["__empty_document__"]
        for tokens in loaded_tokens
    ]

    records = loaded_records
    corpus_tokens = loaded_tokens
    bm25 = BM25Okapi(corpus_tokens)

    print(f"Loaded {len(records)} chunks from {INDEX_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    yield


app = FastAPI(
    title="Consulting Knowledge Retrieval API",
    description=(
        "Search project document chunks and return source metadata."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# 3. REQUEST SCHEMA
# ============================================================

class SearchRequest(BaseModel):
    query: str = Field(min_length=5, max_length=MAX_LENGTH_SEARCH)
    top_k: int = Field(default=TOP_K_SEARCH, ge=1, le=20)
    project_code: Optional[str] = Field(
        default=None,
        description="Optional exact project code filter.",
    )


# ============================================================
# 4. SEARCH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if bm25 is not None else "not_ready",
        "indexed_chunks": len(records),
        "index_file": str(INDEX_PATH),
    }


@app.post("/search")
def search(request: SearchRequest):
    if bm25 is None:
        raise HTTPException(
            status_code=503,
            detail="Search index is not loaded.",
        )

    query = request.query.strip()
    query_tokens = tokenize(query)

    if not query_tokens:
        raise HTTPException(
            status_code=400,
            detail="Query must contain searchable letters or numbers.",
        )

    # Get one score per indexed chunk.
    scores = bm25.get_scores(query_tokens)

    # Apply project filtering before selecting top results.
    candidate_indices = [
        index
        for index, record in enumerate(records)
        if (
            request.project_code is None
            or record.get("project_code") == request.project_code
        )
    ]

    if not candidate_indices:
        return {
            "query": query,
            "project_code": request.project_code,
            "result_count": 0,
            "results": [],
        }

    ranked_indices = sorted(
        candidate_indices,
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    # Ignore zero-score results: they do not match query terms.
    ranked_indices = [
        index for index in ranked_indices
        if float(scores[index]) > 0
    ][:request.top_k]

    results = []

    for index in ranked_indices:
        record = records[index]

        results.append({
            "score": round(float(scores[index]), 4),
            "chunk_id": record.get("chunk_id"),
            "project_code": record.get("project_code"),
            "project_name": record.get("project_name"),
            "project_type": record.get("project_type"),
            "practice_area": record.get("practice_area"),
            "start_year": record.get("start_year"),
            "client_name": record.get("client_name"),
            "file_name": record.get("file_name"),
            "file_type": record.get("file_type"),
            "source_path": record.get("source_path"),
            "source_locator": record.get("source_locator"),
            "sharepoint_folder": record.get("sharepoint_folder"),
            "text": record.get("text"),
        })

    return {
        "query": query,
        "project_code": request.project_code,
        "result_count": len(results),
        "results": results,
    }


# ============================================================
# 5. DUMMY CHAT ENDPOINT
# ============================================================

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_LENGTH_SEARCH)
    top_k: int = Field(default=TOP_K_SEARCH, ge=1, le=10)
    project_code: Optional[str] = None


@app.post("/chat")
def chat(request: ChatRequest):
    # 1. Retrieve relevant document chunks using existing search logic
    search_response = search(
        SearchRequest(
            query=request.query,
            top_k=request.top_k,
            project_code=request.project_code,
        )
    )

    retrieved_results = search_response["results"]

    # 2. Handle questions with no matching evidence
    if not retrieved_results:
        return {
            "query": request.query,
            "answer": (
                "I could not find relevant evidence in the indexed "
                "documents for this question. Try different keywords "
                "or remove the project filter."
            ),
            "source_count": 0,
            "sources": [],
        }

    # 3. Build a transparent dummy answer from retrieved evidence
    answer_parts = [
        "This is a prototype answer based on retrieved document "
        "excerpts, not an LLM-generated synthesis.",
        "",
        "Relevant evidence found:"
    ]

    sources = []

    for i, result in enumerate(retrieved_results, start=1):
        text = result.get("text", "").strip()

        # Keep each excerpt readable in the chat response
        excerpt = text[:700]
        if len(text) > 700:
            excerpt += "..."

        source_label = f"[{i}]"

        answer_parts.extend([
            "",
            f"{source_label} {result.get('file_name', 'Unknown file')}",
            excerpt,
        ])

        sources.append({
            "source_id": source_label,
            "project_code": result.get("project_code"),
            "project_name": result.get("project_name"),
            "file_name": result.get("file_name"),
            "source_path": result.get("source_path"),
            "source_locator": result.get("source_locator"),
            "sharepoint_folder": result.get("sharepoint_folder"),
            "retrieval_score": result.get("score"),
        })

    # 4. Return answer and structured source references
    return {
        "query": request.query,
        "answer": "\n".join(answer_parts),
        "source_count": len(sources),
        "sources": sources,
    }
