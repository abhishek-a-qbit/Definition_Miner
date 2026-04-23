from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sqlite3
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import AgglomerativeClustering
import trafilatura

USER_AGENT = (
    "Mozilla/5.0"
)
TRIGGER_PHRASES = (
    " is a ",
    " refers to ",
    " defined as ",
    " stands for ",
    " means ",
)
BLACKLIST_SNIPPETS = (
    "cookie",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "subscribe",
    "sign up",
    "advertisement",
)
AUTHORITY_DOMAINS = {
    "wikipedia.org",
    "gartner.com",
    "forrester.com",
    "salesforce.com",
    "hubspot.com",
    "oracle.com",
    "mckinsey.com",
    "investopedia.com",
}


@dataclass(frozen=True)
class Candidate:
    sentence: str
    url: str
    domain: str


@dataclass
class ClusterResult:
    sentence: str
    unique_domains: int
    domains: List[str]
    urls: List[str]
    score: float
    centrality: float


@dataclass
class Config:
    search_provider: str
    max_results_per_query: int
    max_urls: int
    timeout_seconds: int
    request_delay_seconds: float
    use_cache: bool
    cache_db_path: Path
    cache_ttl_days: int
    min_words: int
    max_words: int
    similarity_threshold: float
    min_cluster_size: int
    top_k: int
    embedding_model: str


def build_queries(category: str) -> List[str]:
    return [
        f"what is {category} definition",
        f"{category} software category definition",
        f"{category} meaning explained",
        f"{category} defined as",
        f"{category} stands for and definition",
    ]


def get_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="", query=parsed.query)
    return cleaned.geturl().rstrip("/")


def search_duckduckgo_html(query: str, limit: int, timeout_seconds: int) -> List[str]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls: List[str] = []
    seen = set()
    selectors = [
        "a.result__a",
        "a.result-link",
        "a[data-testid='result-title-a']",
        "div.links_main a",
    ]

    for selector in selectors:
        for link in soup.select(selector):
            href = link.get("href", "")
            if not href:
                continue
            if "duckduckgo.com/l/?" in href and "uddg=" in href:
                query_params = parse_qs(urlparse(href).query)
                if "uddg" in query_params:
                    href = unquote(query_params["uddg"][0])

            href = href.strip()
            if not href.startswith("http"):
                continue
            if "duckduckgo.com" in get_domain(href):
                continue
            if href in seen:
                continue
            seen.add(href)
            urls.append(href)
            if len(urls) >= limit:
                return urls

    return urls


def decode_bing_href(href: str) -> str:
    if not href:
        return ""

    parsed = urlparse(href)
    if "bing.com" not in parsed.netloc or not parsed.path.startswith("/ck/a"):
        return href

    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return href

    token = encoded[2:]
    token += "=" * ((4 - len(token) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(token).decode("utf-8", errors="ignore")
    except Exception:
        return href

    if decoded.startswith("http"):
        return decoded
    if decoded.startswith("/"):
        return f"https://www.bing.com{decoded}"
    return href


def search_duckduckgo_lite(query: str, limit: int, timeout_seconds: int) -> List[str]:
    response = requests.post(
        "https://lite.duckduckgo.com/lite/",
        data={"q": query},
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls: List[str] = []
    seen = set()
    for link in soup.select("a"):
        href = link.get("href", "")
        if not href:
            continue
        if "duckduckgo.com/l/?" in href and "uddg=" in href:
            query_params = parse_qs(urlparse(href).query)
            if "uddg" in query_params:
                href = unquote(query_params["uddg"][0])
        href = href.strip()
        if not href.startswith("http"):
            continue
        if "duckduckgo.com" in get_domain(href):
            continue
        if href in seen:
            continue
        seen.add(href)
        urls.append(href)
        if len(urls) >= limit:
            break
    return urls


def search_bing_html(query: str, limit: int, timeout_seconds: int) -> List[str]:
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": limit},
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls: List[str] = []
    seen = set()
    primary_links = soup.select("li.b_algo h2 a")
    if not primary_links:
        primary_links = [
            link
            for link in soup.select("a[href]")
            if "bing.com/ck/a" in (link.get("href", ""))
        ]

    for link in primary_links:
        href = decode_bing_href(link.get("href", "").strip())
        if not href.startswith("http"):
            continue
        if "bing.com" in get_domain(href):
            continue
        if href in seen:
            continue
        seen.add(href)
        urls.append(href)
        if len(urls) >= limit:
            break
    return urls


def search_wikipedia(query: str, limit: int, timeout_seconds: int) -> List[str]:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "srlimit": limit,
        },
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("query", {}).get("search", [])

    urls: List[str] = []
    for row in rows:
        title = row.get("title", "").strip()
        if not title:
            continue
        urls.append(f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
        if len(urls) >= limit:
            break
    return urls


def search_brave(query: str, limit: int, timeout_seconds: int, api_key: str) -> List[str]:
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": limit},
        timeout=timeout_seconds,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": USER_AGENT,
        },
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("web", {}).get("results", [])
    return [row.get("url") for row in results if row.get("url")]


def search_serpapi(query: str, limit: int, timeout_seconds: int, api_key: str) -> List[str]:
    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "num": limit,
            "api_key": api_key,
        },
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("organic_results", [])
    return [row.get("link") for row in results if row.get("link")]


def search_urls(category: str, cfg: Config) -> List[str]:
    queries = build_queries(category)
    brave_key = os.getenv("BRAVE_API_KEY", "").strip()
    serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()

    def run_provider(provider_name: str, query: str) -> List[str]:
        if provider_name == "brave":
            if not brave_key:
                return []
            return search_brave(query, cfg.max_results_per_query, cfg.timeout_seconds, brave_key)
        if provider_name == "serpapi":
            if not serpapi_key:
                return []
            return search_serpapi(query, cfg.max_results_per_query, cfg.timeout_seconds, serpapi_key)

        found = search_duckduckgo_html(query, cfg.max_results_per_query, cfg.timeout_seconds)
        if not found:
            found = search_duckduckgo_lite(query, cfg.max_results_per_query, cfg.timeout_seconds)
        if not found:
            found = search_bing_html(query, cfg.max_results_per_query, cfg.timeout_seconds)
        if not found:
            found = search_wikipedia(query, cfg.max_results_per_query, cfg.timeout_seconds)
        return found

    ordered_urls: List[str] = []
    seen = set()
    for query in queries:
        if cfg.search_provider == "auto":
            providers = ["brave", "serpapi", "duckduckgo"]
        else:
            providers = [cfg.search_provider]

        found: List[str] = []
        for provider_name in providers:
            try:
                found = run_provider(provider_name, query)
            except Exception as exc:
                print(f"[warn] search provider '{provider_name}' failed for query '{query}': {exc}")
                found = []
            if found:
                break

        for raw_url in found:
            normalized = normalize_url(raw_url)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_urls.append(normalized)
            if len(ordered_urls) >= cfg.max_urls:
                return ordered_urls

        time.sleep(cfg.request_delay_seconds)

    return ordered_urls


def init_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS page_cache (
            url TEXT PRIMARY KEY,
            fetched_at TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            html TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_from_cache(
    conn: sqlite3.Connection,
    url: str,
    ttl_days: int,
) -> Optional[Tuple[int, str]]:
    row = conn.execute(
        "SELECT fetched_at, status_code, html FROM page_cache WHERE url = ?",
        (url,),
    ).fetchone()
    if not row:
        return None
    fetched_at = datetime.fromisoformat(row[0])
    if datetime.now(timezone.utc) - fetched_at > timedelta(days=ttl_days):
        return None
    return int(row[1]), row[2]


def save_to_cache(conn: sqlite3.Connection, url: str, status_code: int, html: str) -> None:
    conn.execute(
        """
        INSERT INTO page_cache(url, fetched_at, status_code, html)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(url)
        DO UPDATE SET
            fetched_at = excluded.fetched_at,
            status_code = excluded.status_code,
            html = excluded.html
        """,
        (url, datetime.now(timezone.utc).isoformat(), status_code, html),
    )
    conn.commit()


def fetch_html(url: str, cfg: Config, conn: Optional[sqlite3.Connection]) -> Optional[str]:
    if conn is not None:
        cached = load_from_cache(conn, url, cfg.cache_ttl_days)
        if cached and cached[0] < 400:
            return cached[1]

    # Primary open-source scraper backend.
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            if conn is not None:
                save_to_cache(conn, url, 200, downloaded)
            return downloaded
    except Exception:
        pass

    # Fallback to raw HTTP fetch for pages trafilatura cannot download.
    try:
        response = requests.get(
            url,
            timeout=cfg.timeout_seconds,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
    except Exception:
        return None

    html = response.text if response.status_code < 400 else ""
    if conn is not None:
        save_to_cache(conn, url, response.status_code, html)
    if response.status_code >= 400:
        return None
    return html


def clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_blocks(html: str) -> List[str]:
    blocks: List[str] = []

    # Primary open-source extraction backend.
    try:
        extracted_text = trafilatura.extract(
            html,
            output_format="txt",
            include_links=False,
            include_images=False,
        )
        if extracted_text:
            for line in extracted_text.splitlines():
                txt = clean_text(line)
                if len(txt) >= 40:
                    blocks.append(txt)
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "nav", "form"]):
        tag.decompose()

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        blocks.append(clean_text(meta_desc.get("content", "")))

    for tag in soup.find_all(["p", "li"]):
        txt = clean_text(tag.get_text(" ", strip=True))
        if len(txt) >= 40:
            blocks.append(txt)

    return blocks


def split_sentences(block: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", block)
    return [clean_text(part) for part in parts if part.strip()]


def looks_like_definition(sentence: str, category: str) -> bool:
    lowered = f" {sentence.lower()} "
    category_lower = category.lower()
    if category_lower not in lowered:
        return False
    if any(snippet in lowered for snippet in BLACKLIST_SNIPPETS):
        return False
    if any(trigger in lowered for trigger in TRIGGER_PHRASES):
        return True

    pattern = rf"\b{re.escape(category_lower)}\b.*\b(is|refers to|defined as|means|stands for)\b"
    if re.search(pattern, lowered):
        return True

    acronym_pattern = rf"\b{re.escape(category.upper())}\b\s*\(([^)]+)\)"
    return bool(re.search(acronym_pattern, sentence))


def extract_candidates_from_html(html: str, url: str, category: str, cfg: Config) -> List[Candidate]:
    domain = get_domain(url)
    seen_sentences = set()
    candidates: List[Candidate] = []

    for block in extract_blocks(html):
        for sentence in split_sentences(block):
            words = sentence.split()
            if len(words) < cfg.min_words or len(words) > cfg.max_words:
                continue
            if not looks_like_definition(sentence, category):
                continue

            dedup_key = sentence.lower()
            if dedup_key in seen_sentences:
                continue
            seen_sentences.add(dedup_key)
            candidates.append(Candidate(sentence=sentence, url=url, domain=domain))

    return candidates


def cluster_candidates(candidates: Sequence[Candidate], cfg: Config) -> List[ClusterResult]:
    if not candidates:
        return []

    from sentence_transformers import SentenceTransformer

    unique_by_domain_sentence: Dict[Tuple[str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.domain, candidate.sentence.lower())
        unique_by_domain_sentence[key] = candidate
    deduped = list(unique_by_domain_sentence.values())

    sentences = [c.sentence for c in deduped]
    model = SentenceTransformer(cfg.embedding_model)
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings)

    if len(deduped) == 1:
        labels = np.array([0])
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - cfg.similarity_threshold,
        )
        labels = clusterer.fit_predict(embeddings)

    sim_matrix = embeddings @ embeddings.T

    grouped_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        grouped_indices.setdefault(int(label), []).append(idx)

    results: List[ClusterResult] = []
    for _, indices in grouped_indices.items():
        domains = sorted({deduped[i].domain for i in indices})
        if len(domains) < cfg.min_cluster_size:
            continue

        sub_sim = sim_matrix[np.ix_(indices, indices)]
        if len(indices) == 1:
            centralities = np.array([1.0])
        else:
            centralities = (sub_sim.sum(axis=1) - 1.0) / (len(indices) - 1)

        representative_local_idx = int(np.argmax(centralities))
        representative_global_idx = indices[representative_local_idx]
        representative = deduped[representative_global_idx]
        centrality = float(centralities[representative_local_idx])

        authority_count = sum(
            1 for domain in domains if any(domain.endswith(a) for a in AUTHORITY_DOMAINS)
        )
        score = len(domains) + 0.35 * centrality + 0.15 * authority_count

        urls = sorted({deduped[i].url for i in indices})
        results.append(
            ClusterResult(
                sentence=representative.sentence,
                unique_domains=len(domains),
                domains=domains,
                urls=urls,
                score=score,
                centrality=centrality,
            )
        )

    results.sort(key=lambda x: (x.score, x.unique_domains, x.centrality), reverse=True)
    return results[: cfg.top_k]


def build_sources(urls: Sequence[str]) -> List[Dict[str, str]]:
    sources = [{"domain": get_domain(url), "url": url} for url in urls]
    sources.sort(key=lambda x: (x["domain"], x["url"]))
    return sources


def mine_definitions(category: str, cfg: Config) -> Dict[str, object]:
    urls = search_urls(category, cfg)
    print(f"[info] collected {len(urls)} unique URLs")

    conn: Optional[sqlite3.Connection] = None
    if cfg.use_cache:
        conn = init_cache(cfg.cache_db_path)

    all_candidates: List[Candidate] = []
    for idx, url in enumerate(urls, start=1):
        html = fetch_html(url, cfg, conn)
        if not html:
            continue

        extracted = extract_candidates_from_html(html, url, category, cfg)
        if extracted:
            all_candidates.extend(extracted)
        print(f"[info] [{idx}/{len(urls)}] candidates so far: {len(all_candidates)}")
        time.sleep(cfg.request_delay_seconds)

    if conn is not None:
        conn.close()

    clusters = cluster_candidates(all_candidates, cfg)
    return {
        "category": category,
        "urls_considered": len(urls),
        "candidate_count": len(all_candidates),
        "results": [
            {
                "rank": rank,
                "definition": cluster.sentence,
                "unique_domains": cluster.unique_domains,
                "domains": cluster.domains,
                "urls": cluster.urls,
                "sources": build_sources(cluster.urls),
                "score": round(cluster.score, 4),
                "centrality": round(cluster.centrality, 4),
            }
            for rank, cluster in enumerate(clusters, start=1)
        ],
    }


def print_results(payload: Dict[str, object]) -> None:
    category = payload["category"]
    results = payload["results"]
    print(f"\n=== Top Definitions for: {category} ===\n")

    if not results:
        print("No high-confidence repeated definitions found. Try lowering thresholds.")
        return

    for item in results:
        domains = ", ".join(item["domains"][:8])
        if len(item["domains"]) > 8:
            domains += ", ..."

        print(f"#{item['rank']} — Found on {item['unique_domains']} sites")
        print(f'"{textwrap.fill(item["definition"], width=88)}"')
        print(f"  └─ {domains}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine repeated web definitions (no LLMs).")
    parser.add_argument("category", help="Category to mine definitions for (e.g., CRM, ABM).")
    parser.add_argument(
        "--search-provider",
        choices=["auto", "brave", "serpapi", "duckduckgo"],
        default="auto",
        help="Search provider. auto=Brave if BRAVE_API_KEY, else SerpAPI if SERPAPI_API_KEY, else DuckDuckGo HTML.",
    )
    parser.add_argument("--max-results-per-query", type=int, default=10)
    parser.add_argument("--max-urls", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    parser.add_argument("--no-cache", action="store_true", help="Disable SQLite HTML caching.")
    parser.add_argument("--cache-db-path", default=".cache/definition_miner.sqlite")
    parser.add_argument("--cache-ttl-days", type=int, default=14)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face sentence-transformers model name.",
    )
    parser.add_argument("--output-json", default="", help="Optional path to write JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        search_provider=args.search_provider,
        max_results_per_query=args.max_results_per_query,
        max_urls=args.max_urls,
        timeout_seconds=args.timeout_seconds,
        request_delay_seconds=args.request_delay_seconds,
        use_cache=not args.no_cache,
        cache_db_path=Path(args.cache_db_path),
        cache_ttl_days=args.cache_ttl_days,
        min_words=args.min_words,
        max_words=args.max_words,
        similarity_threshold=args.similarity_threshold,
        min_cluster_size=args.min_cluster_size,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
    )

    payload = mine_definitions(args.category, cfg)
    print_results(payload)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] wrote JSON output to {output_path}")


if __name__ == "__main__":
    main()
