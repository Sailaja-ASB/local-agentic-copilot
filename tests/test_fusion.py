from app.rag.fusion import reciprocal_rank_fusion, tokenize  # Pure functions: no models needed.


def test_tokenize_keeps_identifiers_whole():
    tokens = tokenize("Set OLLAMA_NUM_PARALLEL and MAX_INFLIGHT; model all-MiniLM-L6-v2, HTTP 429.")
    assert "ollama_num_parallel" in tokens
    assert "max_inflight" in tokens
    assert "all-minilm-l6-v2" in tokens
    assert "429" in tokens


def test_rrf_rewards_documents_found_by_both_retrievers():
    dense = ["a", "b", "c"]
    bm25 = ["c", "d", "a"]
    fused = [doc for doc, _ in reciprocal_rank_fusion([dense, bm25])]
    # "a" is 1st and 3rd, "c" is 3rd and 1st: both beat documents found by only one list.
    assert set(fused[:2]) == {"a", "c"}
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_score_formula():
    scores = dict(reciprocal_rank_fusion([["x"], ["x"]], k=60))
    assert abs(scores["x"] - 2 / 61) < 1e-12


def test_rrf_fixes_a_miss_by_one_retriever():
    # Dense ranks the right chunk 2nd; BM25 ranks it 1st -> fusion puts it first.
    dense = ["wrong", "right", "other"]
    bm25 = ["right", "other2", "wrong2"]
    assert reciprocal_rank_fusion([dense, bm25])[0][0] == "right"
