"""Command line: `python -m catalogue_rag <command>` (or `catalogue-rag <command>` once installed).

    parse    PDFs in data/documents -> Markdown in data/parsed_documents (LlamaParse)
    index    chunk the Markdown and build the search index
    add      add catalogue files (PDF, Markdown, text) to an existing index
    ask      ask one question, or start an interactive session with no question
    search   show what retrieval finds, without calling the language model
    serve    run the web UI and API on http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from .config import settings


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="catalogue-rag", description="Cited answers from product catalogues.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("parse", help="parse new PDFs with LlamaParse")
    pp.add_argument("--force", action="store_true", help="re-parse PDFs that were already parsed")
    sub.add_parser("index", help="build the search index from parsed Markdown")
    pd = sub.add_parser("add", help="extract and index more catalogues, without rebuilding everything")
    pd.add_argument("files", nargs="+")
    pa = sub.add_parser("ask", help="ask a question (interactive when none is given)")
    pa.add_argument("question", nargs="*")
    pa.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "vector"])
    ps = sub.add_parser("search", help="show retrieval results only")
    ps.add_argument("question", nargs="+")
    ps.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "vector"])
    ps.add_argument("-k", type=int, default=6)
    pv = sub.add_parser("serve", help="run the web UI and API")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    cfg = settings()

    if a.cmd == "parse":
        from .ingest import parse_pdfs

        n = parse_pdfs(cfg.documents_dir, cfg.parsed_dir, cfg.llama_cloud_api_key, force=a.force)
        print(f"Parsed {n} PDF(s). Next: python -m catalogue_rag index")
    elif a.cmd == "index":
        from .index import build_index

        m = build_index(cfg.parsed_dir, cfg.index_dir, cfg.chunk_chars)
        print(f"\nIndexed {m['chunks']} chunks from {m['documents']} documents in {m['seconds']}s -> {cfg.index_dir}")
    elif a.cmd == "add":
        import shutil
        from pathlib import Path

        from .uploads import add_file, safe_name

        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        for f in a.files:
            src = Path(f)
            dest = cfg.documents_dir / safe_name(src.name)
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            r = add_file(cfg, dest)
            print(f"Added {r['doc']}: {r['chunks']} passages ({r['documents']} catalogues, {r['total_chunks']} passages in total)")
    elif a.cmd == "search":
        from .index import Index
        from .retrieval import Retriever

        r = Retriever(Index.load(cfg.index_dir, with_vectors=a.mode != "bm25"), cfg.candidates, cfg.rrf_k, cfg.bm25_weight)
        for n, h in enumerate(r.search(" ".join(a.question), a.k, a.mode), 1):
            print(f"\n[{n}] {h.chunk.title} - p.{h.chunk.page} - {h.chunk.section}  (ranks {h.ranks})")
            print(textwrap.indent(h.chunk.text[:400], "    "))
    elif a.cmd == "ask":
        from .pipeline import Assistant

        bot = Assistant()
        questions = [" ".join(a.question)] if a.question else None
        while True:
            q = questions.pop() if questions else input("\nQuestion (blank to quit) > ").strip()
            if not q:
                break
            ans = bot.ask(q, mode=a.mode)
            print("\n" + ("[not in the catalogues] " if ans.declined else "") + ans.answer)
            for c in ans.citations:
                print(f"  [{c.n}] {c.title}, page {c.page}" + (f" - {c.section}" if c.section else ""))
            print(f"  ({ans.timings_ms['retrieve']} ms retrieval, {ans.timings_ms['generate']} ms generation, {ans.model})")
            if a.question:
                break
    elif a.cmd == "serve":
        import uvicorn

        print(f"Catalogue RAG on http://{a.host}:{a.port}")
        uvicorn.run("catalogue_rag.api:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
