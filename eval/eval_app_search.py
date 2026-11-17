"""Run the 47-question benchmark through the app's real search code (app/rag/search.py).

eval/run_retrieval_eval.py compares methods in isolation. This script checks that the code
the Copilot actually ships (Chroma + search_documents) gives the same quality, in every mode.

Run from the repo root:
    python eval/eval_app_search.py
    python eval/eval_app_search.py --min-top1 0.95     # exit code 1 if hybrid drops below 95% (for CI)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# A separate benchmark index, so the app's normal index is never touched.
os.environ["CHROMA_PATH"] = "data/chroma_benchmark"
os.environ["CHROMA_COLLECTION"] = "benchmark"

from app.rag.indexer import build_index  # noqa: E402
from app.rag.search import MODES, search_with_timings  # noqa: E402


def source_ranking(matches: list[dict]) -> list[str]:
    seen, sources = set(), []
    for m in matches:
        if m["source"] not in seen:
            seen.add(m["source"])
            sources.append(m["source"])
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions_47.json")
    ap.add_argument("--docs", default="data/docs/benchmark")
    ap.add_argument("--min-top1", type=float, default=None, help="fail if default-mode (hybrid) Top-1 is below this")
    args = ap.parse_args()

    build_index(docs_dir=Path(args.docs), db_path="data/chroma_benchmark", collection_name="benchmark")
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    total = len(questions)

    search_with_timings("warm up", top_k=3, mode="hybrid_rerank")  # Load models once before timing.

    summary, failures = {}, {}
    for mode in MODES:
        ranks, stage_times = [], {}
        failures[mode] = []
        for q in questions:
            matches, timings = search_with_timings(q["question"], top_k=10, mode=mode)
            srcs = source_ranking(matches)
            rank = srcs.index(q["expected_source"]) + 1 if q["expected_source"] in srcs else None
            ranks.append(rank)
            if rank != 1:
                failures[mode].append(f"{q['id']} rank={rank} expected={q['expected_source']} got={srcs[:3]}")
            for stage, ms in timings.items():
                stage_times.setdefault(stage, []).append(ms)
        summary[mode] = {
            "top1": sum(r == 1 for r in ranks),
            "hit3": sum(1 for r in ranks if r and r <= 3),
            "mrr": sum(1 / r for r in ranks if r) / total,
            "ms": {stage: round(mean(v), 1) for stage, v in stage_times.items()},
        }

    lines = [
        "# App search benchmark (app/rag/search.py through Chroma)",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M} | Questions: {total} | top_k = 10 chunks per query",
        "",
        "| Mode | Top-1 | Hit@3 | MRR | dense ms | BM25 ms | rerank ms | total ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        s, ms = summary[mode], summary[mode]["ms"]
        lines.append(
            f"| {mode} | {s['top1']}/{total} ({100 * s['top1'] / total:.1f}%) | "
            f"{s['hit3']}/{total} ({100 * s['hit3'] / total:.1f}%) | {s['mrr']:.3f} | "
            + " | ".join(f"{ms[k]:.1f}" if k in ms else "-" for k in ("dense_ms", "bm25_ms", "rerank_ms", "total_ms"))
            + " |"
        )
    lines += ["", "### Not ranked first"]
    for mode in MODES:
        lines += ["", f"**{mode}** ({len(failures[mode])})"] + [f"- {f}" for f in failures[mode]]
    report = "\n".join(lines)
    print("\n" + report)

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS_APP.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {out / 'RESULTS_APP.md'}")

    if args.min_top1 is not None:
        default_top1 = summary["hybrid"]["top1"] / total
        if default_top1 < args.min_top1:
            print(f"FAIL: hybrid Top-1 {default_top1:.1%} is below the minimum {args.min_top1:.1%}")
            sys.exit(1)
        print(f"PASS: hybrid Top-1 {default_top1:.1%} >= {args.min_top1:.1%}")


if __name__ == "__main__":
    main()
