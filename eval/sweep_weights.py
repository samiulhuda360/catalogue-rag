"""Sweep the keyword weight in the fusion and the number of sources (retrieval only, no LLM).

    python eval/sweep_weights.py

Writes eval/results/weight_sweep.md. With 40 questions, small differences are noise:
pick the simplest setting near the top, not the single best number.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from run_eval import first_fact_rank, load_questions, questions_file  # noqa: E402

from catalogue_rag.config import settings  # noqa: E402
from catalogue_rag.index import Index  # noqa: E402
from catalogue_rag.retrieval import Retriever  # noqa: E402

cfg = settings()
index = Index.load(cfg.index_dir)
qs = [q for q in load_questions(questions_file(None)) if q["expect"]]
rows = ["| Keyword weight | Sources | Answer text reaches the model | Fact MRR |", "|---|---|---|---|"]
for k in (6, 8):
    for w in (1.0, 1.5, 2.0, 3.0):
        r = Retriever(index, cfg.candidates, cfg.rrf_k, w)
        ranks = [first_fact_rank([h.chunk.text for h in r.search(q["question"], k)], q.get("evidence") or q["expect"]) for q in qs]
        recall = sum(x is not None for x in ranks) / len(qs)
        mrr = sum(1 / x for x in ranks if x) / len(qs)
        rows.append(f"| {w} | {k} | {recall * 100:.0f}% | {mrr:.2f} |")
        print(rows[-1])
(ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
(ROOT / "eval" / "results" / "weight_sweep.md").write_text("# Keyword weight sweep (retrieval only)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
