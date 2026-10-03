"""Question in, cited answer out."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Settings, settings
from .generation import Generator
from .index import Index
from .retrieval import Retriever


@dataclass
class Citation:
    n: int
    doc: str
    title: str
    page: int
    section: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    declined: bool
    citations: list[Citation]
    sources: list[Citation]          # everything shown to the model, cited or not
    timings_ms: dict = field(default_factory=dict)
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer, "declined": self.declined,
            "citations": [c.__dict__ for c in self.citations], "sources": [s.__dict__ for s in self.sources],
            "timings_ms": self.timings_ms, "model": self.model,
        }


class Assistant:
    def __init__(self, cfg: Settings | None = None, index: Index | None = None, generator: Generator | None = None):
        self.cfg = cfg or settings()
        self.index = index or Index.load(self.cfg.index_dir)
        self.retriever = Retriever(self.index, self.cfg.candidates, self.cfg.rrf_k, self.cfg.bm25_weight)
        self.generator = generator or Generator(self.cfg.llm_api_key, self.cfg.llm_base_url, self.cfg.llm_model,
                                                self.cfg.llm_temperature, self.cfg.llm_max_tokens)

    def ask(self, question: str, top_k: int | None = None, mode: str = "hybrid") -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, top_k or self.cfg.top_k, mode)
        t1 = time.perf_counter()
        gen = self.generator.answer(question, hits)
        t2 = time.perf_counter()
        sources = [Citation(n, h.chunk.doc, h.chunk.title, h.chunk.page, h.chunk.section, h.chunk.text[:2000])
                   for n, h in enumerate(hits, 1)]
        answer = Answer(question, gen.text, gen.declined, [sources[n - 1] for n in gen.cited], sources,
                        {"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000)}, gen.model)
        self._log(answer)
        return answer

    def ask_stream(self, question: str, top_k: int | None = None, mode: str = "hybrid"):
        """The same answer as ask(), as a sequence of events for a streaming UI:
        sources (as soon as retrieval is done), declined (if so), delta (answer text as it
        is generated), done (the final cleaned answer with citations and timings)."""
        t0 = time.perf_counter()
        hits = self.retriever.search(question, top_k or self.cfg.top_k, mode)
        t1 = time.perf_counter()
        sources = [Citation(n, h.chunk.doc, h.chunk.title, h.chunk.page, h.chunk.section, h.chunk.text[:2000])
                   for n, h in enumerate(hits, 1)]
        # Every underlying chunk shown to the model (a short page is one source of several chunks),
        # so the UI can light them up on the passage map.
        chunk_ids = [c.id for h in hits for c in (self.index.pages.get((h.chunk.doc, h.chunk.page), [h.chunk])
                                                   if h.chunk.id.count("::") == 1 else [h.chunk])]
        yield {"type": "sources", "sources": [s.__dict__ for s in sources], "chunk_ids": chunk_ids,
               "retrieve_ms": round((t1 - t0) * 1000)}
        first = None
        for kind, value in self.generator.stream(question, hits):
            if kind == "delta":
                first = first or time.perf_counter()
                yield {"type": "delta", "text": value}
            elif kind == "declined":
                yield {"type": "declined"}
            else:
                t2 = time.perf_counter()
                answer = Answer(question, value.text, value.declined, [sources[n - 1] for n in value.cited], sources,
                                {"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000),
                                 "first_token": round(((first or t2) - t1) * 1000)}, value.model)
                self._log(answer)
                yield {"type": "done", **answer.to_dict()}

    def _log(self, a: Answer) -> None:
        try:
            log = self.cfg.index_dir.parent / "logs" / "questions.jsonl"
            log.parent.mkdir(exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": a.question,
                                    "declined": a.declined, "cited": [f"{c.doc}#p{c.page}" for c in a.citations],
                                    "timings_ms": a.timings_ms}) + "\n")
        except OSError:
            pass
