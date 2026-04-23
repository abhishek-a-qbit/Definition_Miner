from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from definition_miner import Config, mine_definitions


class MineRequest(BaseModel):
    category: str = Field(..., min_length=2, description="Category to mine, e.g. CRM or ABM")
    search_provider: Literal["auto", "brave", "serpapi", "duckduckgo"] = "auto"
    max_results_per_query: int = Field(10, ge=1, le=20)
    max_urls: int = Field(40, ge=1, le=200)
    timeout_seconds: int = Field(20, ge=5, le=120)
    request_delay_seconds: float = Field(0.5, ge=0.0, le=5.0)
    use_cache: bool = True
    cache_db_path: str = ".cache/definition_miner.sqlite"
    cache_ttl_days: int = Field(14, ge=1, le=365)
    min_words: int = Field(20, ge=3, le=500)
    max_words: int = Field(120, ge=5, le=1000)
    similarity_threshold: float = Field(0.78, ge=0.0, le=1.0)
    min_cluster_size: int = Field(2, ge=1, le=20)
    top_k: int = Field(5, ge=1, le=20)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


app = FastAPI(title="Definition Miner API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mine")
def mine(payload: MineRequest) -> dict:
    try:
        cfg = Config(
            search_provider=payload.search_provider,
            max_results_per_query=payload.max_results_per_query,
            max_urls=payload.max_urls,
            timeout_seconds=payload.timeout_seconds,
            request_delay_seconds=payload.request_delay_seconds,
            use_cache=payload.use_cache,
            cache_db_path=Path(payload.cache_db_path),
            cache_ttl_days=payload.cache_ttl_days,
            min_words=payload.min_words,
            max_words=payload.max_words,
            similarity_threshold=payload.similarity_threshold,
            min_cluster_size=payload.min_cluster_size,
            top_k=payload.top_k,
            embedding_model=payload.embedding_model,
        )
        return mine_definitions(payload.category, cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mining failed: {exc}") from exc
