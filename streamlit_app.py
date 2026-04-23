from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from definition_miner import Config, mine_definitions


st.set_page_config(page_title="Definition Miner", layout="wide")
st.title("Definition Miner")
st.caption("Mine repeated/canonical category definitions and keep verbatim sources.")

with st.sidebar:
    st.header("Run Miner")
    category = st.text_input("Category", value="CRM")
    run_mode = st.selectbox("Depth", ["Quick", "Balanced", "Deep"], index=1)
    strictness = st.select_slider("Definition matching", options=["Broad", "Balanced", "Strict"], value="Balanced")
    search_provider = st.selectbox("Search provider", ["auto", "duckduckgo", "brave", "serpapi"], index=0)
    use_cache = st.toggle("Use cache (faster reruns)", value=True)

    st.divider()
    st.subheader("Functionalities")
    st.toggle("Verbatim extraction (no rewriting)", value=True, disabled=True)
    st.toggle("Source tracking with clickable links", value=True, disabled=True)
    st.toggle("Open-source scraping backend", value=True, disabled=True)


def resolve_mode(mode: str) -> tuple[int, int, int]:
    if mode == "Quick":
        return 6, 20, 3
    if mode == "Deep":
        return 12, 60, 7
    return 10, 40, 5


def resolve_strictness(level: str) -> tuple[float, int]:
    if level == "Broad":
        return 0.72, 1
    if level == "Strict":
        return 0.83, 3
    return 0.78, 2

run_clicked = st.button("Run Miner", type="primary")

if run_clicked:
    if not category.strip():
        st.error("Please provide a category.")
    else:
        max_results_per_query, max_urls, top_k = resolve_mode(run_mode)
        similarity_threshold, min_cluster_size = resolve_strictness(strictness)

        cfg = Config(
            search_provider=search_provider,
            max_results_per_query=max_results_per_query,
            max_urls=max_urls,
            timeout_seconds=20,
            request_delay_seconds=0.5,
            use_cache=use_cache,
            cache_db_path=Path(".cache/definition_miner.sqlite"),
            cache_ttl_days=14,
            min_words=20,
            max_words=120,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            top_k=top_k,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        with st.spinner("Mining definitions..."):
            payload = mine_definitions(category.strip(), cfg)

        st.subheader(f"Top Definitions for: {payload['category']}")
        st.write(
            f"URLs considered: **{payload['urls_considered']}** | "
            f"Candidates: **{payload['candidate_count']}**"
        )

        if payload["urls_considered"] == 0:
            st.error("No search results were found. Try setting Search provider to 'duckduckgo' or run again.")

        results = payload.get("results", [])
        if not results:
            st.warning("No repeated high-confidence definitions found. Try lower thresholds.")

        for item in results:
            st.markdown(f"### #{item['rank']} (Found on {item['unique_domains']} sites)")
            st.info(item["definition"])

            with st.expander("Sources", expanded=True):
                for source in item.get("sources", []):
                    st.markdown(f"- [{source['domain']}]({source['url']})")

        st.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2),
            file_name=f"{category.lower()}_definitions.json",
            mime="application/json",
        )
