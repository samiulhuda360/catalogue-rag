"""Evaluate the assistant against hand-checked questions, and compare with a whole-document baseline.

    python eval/run_eval.py                  # retrieval + answers, all systems (~5 min)
    python eval/run_eval.py --retrieval-only # no language model calls, free and fast

Questions (eval/questions.jsonl, or examples/questions.jsonl for the bundled
fictional sample) were written from the catalogues, and every expected
answer was checked against the parsed source text. Each lists the facts a correct
answer must contain (any of several spellings) and the documents that hold them.
Six questions are deliberately unanswerable from the catalogues (vendor approvals,
prices, building-code clauses): the right behaviour is to say so.

Metrics
  Retrieval (per system, over the 40 answerable questions)
    doc recall     an expected document is among the sources given to the model
    fact recall    the expected fact text is inside what the model is shown
    fact MRR       1 / rank of the first source that contains the fact
  Answers
    correct        every expected fact appears in the answer
    cited right    (new system) at least one citation points at an expected document
    declined       on the 6 unanswerable questions, the answer says it is not covered
    wrong refusal  on answerable questions, the answer declined
    latency        p50 / p95 seconds per answer
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval" / "baseline"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from catalogue_rag.config import settings  # noqa: E402
from catalogue_rag.index import Index  # noqa: E402
from catalogue_rag.retrieval import Retriever  # noqa: E402

RESULTS = ROOT / "eval" / "results"
DECLINE = re.compile(r"not (?:covered|mentioned|specified|stated|included|provided|listed|available|found|contain)|"
                     r"no (?:information|mention|data|details)|do(?:es)? not (?:contain|include|mention|specify|state|cover|provide|list)|"
                     r"cannot (?:be )?(?:determine|confirm|answer)|unable to|isn't (?:covered|mentioned)|not in the", re.I)


def norm(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("‑", "-").replace("®", "").replace("™", "")
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", "", text)


def has_facts(text: str, expect: list[list[str]]) -> bool:
    t = norm(text)
    return bool(expect) and all(any(norm(alt) in t for alt in group) for group in expect)


def first_fact_rank(texts: list[str], expect: list[list[str]]) -> int | None:
    for i, t in enumerate(texts, 1):
        if has_facts(t, expect):
            return i
    return None


def questions_file(given: str | None) -> Path:
    """Your own question set if there is one, otherwise the one for the bundled sample catalogues."""
    if given:
        return Path(given)
    own = ROOT / "eval" / "questions.jsonl"
    return own if own.exists() else ROOT / "examples" / "questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def retrieval_eval(questions: list[dict], systems: dict) -> dict:
    out = {}
    answerable = [q for q in questions if q["expect"]]
    for name, fn in systems.items():
        rows = []
        for q in answerable:
            docs, texts = fn(q["question"])
            rank = first_fact_rank(texts, q.get("evidence") or q["expect"])
            rows.append({"id": q["id"], "doc_hit": any(d in q["docs"] for d in docs), "fact_rank": rank})
        n = len(rows)
        out[name] = {"doc_recall": sum(r["doc_hit"] for r in rows) / n,
                     "fact_recall": sum(r["fact_rank"] is not None for r in rows) / n,
                     "fact_mrr": sum(1 / r["fact_rank"] for r in rows if r["fact_rank"]) / n,
                     "rows": rows}
        print(f"  {name:28s} doc recall {pct(out[name]['doc_recall']):>4}  fact recall {pct(out[name]['fact_recall']):>4}  "
              f"fact MRR {out[name]['fact_mrr']:.2f}")
    return out


def answer_eval(questions: list[dict], systems: dict) -> dict:
    out = {}
    for name, fn in systems.items():
        rows = []
        for q in questions:
            t0 = time.perf_counter()
            try:
                text, declined, cited_docs = fn(q["question"])
            except Exception as exc:  # noqa: BLE001  record and carry on
                text, declined, cited_docs = f"ERROR {exc}", False, []
            secs = time.perf_counter() - t0
            declined = declined or bool(DECLINE.search(text[:400]))
            rows.append({"id": q["id"], "category": q["category"], "answer": text, "declined": declined, "seconds": round(secs, 2),
                         "correct": has_facts(text, q["expect"]) if q["expect"] else None,
                         "cited_right": (any(d in q["docs"] for d in cited_docs) if q["expect"] else None) if cited_docs is not None else None})
            mark = "✓" if (rows[-1]["correct"] if q["expect"] else declined) else "✗"
            print(f"  {name[:10]:10s} {q['id']} {mark} {secs:5.1f}s  {text[:90].replace(chr(10), ' ')}")
        out[name] = summarise(rows)
    return out


def rescore(questions: list[dict], saved: dict) -> dict:
    """Recompute answer metrics for saved answers; the answers themselves are not regenerated."""
    by_id = {q["id"]: q for q in questions}
    out = {}
    for name, m in saved.items():
        rows = [dict(r) for r in m["rows"] if r["id"] in by_id]
        for r in rows:
            q = by_id[r["id"]]
            r["correct"] = has_facts(r["answer"], q["expect"]) if q["expect"] else None
        out[name] = summarise(rows)
    return out


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["correct"] is not None]
    un = [r for r in rows if r["correct"] is None]
    cited = [r for r in ans if r.get("cited_right") is not None]
    lat = sorted(r["seconds"] for r in rows)
    return {"correct": sum(r["correct"] for r in ans) / len(ans),
            "cited_right": (sum(r["cited_right"] for r in cited) / len(cited)) if cited else None,
            "declined_unanswerable": sum(r["declined"] for r in un) / len(un) if un else None,
            "wrong_refusal": sum(r["declined"] for r in ans) / len(ans),
            "p50_s": statistics.median(lat), "p95_s": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
            "by_category": {c: sum(r["correct"] for r in ans if r["category"] == c) / max(1, sum(1 for r in ans if r["category"] == c))
                            for c in sorted({r["category"] for r in ans})},
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the saved answers in latest.json against the current questions (no model calls); "
                         "use after correcting an expected answer")
    ap.add_argument("--questions", help="question file (default: eval/questions.jsonl, else examples/questions.jsonl)")
    a = ap.parse_args()
    global RESULTS
    qfile = questions_file(a.questions)
    if qfile.parent.name == "examples":  # keep the published report; sample runs go to their own folder
        RESULTS = RESULTS / "sample"
    cfg = settings()
    questions = load_questions(qfile)[: a.limit or None]
    index = Index.load(cfg.index_dir)
    retriever = Retriever(index, cfg.candidates, cfg.rrf_k, cfg.bm25_weight)

    def new_system(mode: str):
        def run(q: str):
            hits = retriever.search(q, cfg.top_k, mode)
            return [h.chunk.doc for h in hits], [h.chunk.text for h in hits]
        return run

    baseline = None
    old_chroma = ROOT / "chroma_data"
    if old_chroma.exists():
        import openai
        from original_pipeline import OriginalPipeline

        client = openai.OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=90)
        baseline = OriginalPipeline(old_chroma, client, cfg.llm_model)

    systems = {}
    if baseline:
        systems["whole-document baseline"] = lambda q: tuple(map(list, zip(*baseline.retrieve(q), strict=False))) or ([], [])
    systems.update({"keyword only (BM25)": new_system("bm25"), "embeddings only": new_system("vector"),
                    "hybrid (final)": new_system("hybrid")})

    print(f"Retrieval, {sum(1 for q in questions if q['expect'])} answerable questions")
    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "model": cfg.llm_model, "top_k": cfg.top_k,
              "questions": len(questions), "retrieval": retrieval_eval(questions, systems)}

    if a.rescore:
        report["answers"] = rescore(questions, json.loads((RESULTS / "latest.json").read_text(encoding="utf-8"))["answers"])
        report["rescored_from_saved_answers"] = True
    elif not a.retrieval_only:
        from catalogue_rag.pipeline import Assistant

        bot = Assistant(cfg, index=index)
        answerers = {}
        if baseline:
            answerers["whole-document baseline"] = lambda q: (baseline.answer(q, baseline.retrieve(q)), False, None)

        def final(q: str):
            ans = bot.ask(q)
            return ans.answer, ans.declined, [c.doc for c in ans.citations]

        answerers["final"] = final
        print(f"\nAnswers, {len(questions)} questions, model {cfg.llm_model}")
        report["answers"] = answer_eval(questions, answerers)
        failed = {name: sum(r["answer"].startswith("ERROR") for r in m["rows"]) for name, m in report["answers"].items()}
        if any(n > max(2, len(questions) // 10) for n in failed.values()):
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "failed_run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            sys.exit(f"Too many failed calls {failed} (network or provider outage?). Kept the previous results; "
                     f"this run is in eval/results/failed_run.json.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    saved = RESULTS / "latest.json"
    if a.retrieval_only and saved.exists():  # keep the last answer results; this run measured retrieval only
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if "answers" in previous:
            report["answers"] = previous["answers"]
            report["answers_from"] = previous.get("run_at")
    (RESULTS / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "REPORT.md").write_text(render(report), encoding="utf-8")
    print(f"\nWrote {RESULTS / 'REPORT.md'}")


def render(r: dict) -> str:
    lines = ["# Evaluation report", "", f"Run {r['run_at']} · model `{r['model']}` · {r['questions']} questions "
             f"({sum(1 for _ in r['retrieval'][next(iter(r['retrieval']))]['rows'])} answerable, the rest deliberately unanswerable) · "
             f"top {r['top_k']} sources", "", "## Retrieval", "",
             "| System | Expected document retrieved | Answer text reaches the model | Fact MRR |", "|---|---|---|---|"]
    for name, m in r["retrieval"].items():
        lines.append(f"| {name} | {pct(m['doc_recall'])} | {pct(m['fact_recall'])} | {m['fact_mrr']:.2f} |")
    if "answers" in r:
        lines += ["", "## Answers", "", "| System | Correct | Cites the right document | Declines unanswerable | Wrongly declines | p50 / p95 |",
                  "|---|---|---|---|---|---|"]
        for name, m in r["answers"].items():
            cited = pct(m["cited_right"]) if m["cited_right"] is not None else "n/a (no per-claim citations)"
            lines.append(f"| {name} | {pct(m['correct'])} | {cited} | {pct(m['declined_unanswerable'])} | {pct(m['wrong_refusal'])} | "
                         f"{m['p50_s']:.1f}s / {m['p95_s']:.1f}s |")
        lines += ["", "### Correct answers by question type", "", "| System | " + " | ".join(next(iter(r["answers"].values()))["by_category"]) + " |",
                  "|---|" + "---|" * len(next(iter(r["answers"].values()))["by_category"])]
        for name, m in r["answers"].items():
            lines.append(f"| {name} | " + " | ".join(pct(v) for v in m["by_category"].values()) + " |")
        final = r["answers"].get("final", {})
        misses = [x for x in final.get("rows", []) if x["correct"] is False]
        if misses:
            lines += ["", "### Questions not answered correctly", ""]
            lines += [f"- **{x['id']}** - {x['answer'][:220].replace(chr(10), ' ')}" for x in misses]
    if r.get("rescored_from_saved_answers"):
        lines += ["", "_Scores recomputed from saved answers with `--rescore`; no answers were regenerated._"]
    lines += ["", "Regenerate with `python eval/run_eval.py`. Per-question detail: `eval/results/latest.json`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
