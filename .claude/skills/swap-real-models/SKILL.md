---
name: swap-real-models
description: Compare the offline stand-ins with a neural embedder, cross-encoder reranker or hosted LLM. Use when asked to try bge, a cross-encoder, GitHub Models or Anthropic in this repo.
---

# Swap in real models

Be explicit that these paths are wired but were not exercised when the repo was generated.

1. Check resources. sentence-transformers needs memory and downloads; on Termux prefer GitHub Actions or a Codespace.
2. `pip install -e ".[ml]"`.
3. Run and keep outputs separate:
   - `RAG_EMBEDDER=st python -m ragkit eval --out results-st`
   - `RAG_RERANKER=ce python -m ragkit eval --only p02 --out results-ce`
   - `RAG_LLM=github` (needs `GITHUB_TOKEN`) or `RAG_LLM=anthropic` (needs `ANTHROPIC_API_KEY`)
4. Fix any breakage in `ragkit/embed.py`, `ragkit/rerank.py` or `ragkit/llm.py`. Provider endpoints and model names change; check the provider docs.
5. Produce one comparison table (offline defaults against real models) with model names, latency and any rate limits hit; save it in `docs/MODEL_COMPARISON.md`.
6. Never put keys in files or logs.

Done when the comparison table is committed and the README notes which paths are now verified.
