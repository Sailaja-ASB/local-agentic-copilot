"""Pure helpers for hybrid retrieval (no models, so they are fast to unit test)."""
from __future__ import annotations

import re
from collections import defaultdict

RRF_K = 60  # Standard Reciprocal Rank Fusion constant.

# Keeps identifiers such as max_inflight, ollama_num_parallel and all-minilm-l6-v2 as single tokens,
# so BM25 can match them exactly.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_\-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into BM25 tokens."""
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge several ranked id lists: each list adds 1 / (k + rank) to every id it contains.

    Uses only positions, so BM25 scores and embedding distances never need to be on the same scale.
    Returns (id, score) pairs, best first. Ties keep the order in which ids were first seen.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
