"""Hybrid retrieval: BM25 and embeddings, fused with reciprocal rank fusion (RRF).

Why both: embeddings understand "how long does the battery last" ≈ "Battery / Life",
but blur part numbers; BM25 nails "CH/10/900" but misses paraphrases. RRF combines
the two rankings without having to calibrate their very different score scales:
score(chunk) = sum over retrievers of weight / (k + rank). When the question contains
a part number, the keyword ranking gets double weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import looks_like_code, tokenize
from .chunking import Chunk
from .index import Index

MODES = ("hybrid", "bm25", "vector")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank


def rrf(rankings: dict[str, list[str]], weights: dict[str, float], k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        w = weights.get(name, 1.0)
        for r, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + r)
            ranks.setdefault(cid, {})[name] = r
    order = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [(cid, scores[cid], ranks[cid]) for cid in order]


class Retriever:
    def __init__(self, index: Index, candidates: int = 40, rrf_k: int = 60, bm25_weight: float = 1.0):
        self.index = index
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight  # keyword vs embedding weight in the fusion (set from evaluation)

    def keyword(self, query: str, n: int) -> list[str]:
        return [self.index.chunks[i].id for i, _ in self.index.bm25.search(query, n)]

    def vector(self, query: str, n: int) -> list[str]:
        if self.index.collection is None:
            return []
        res = self.index.collection.query(query_texts=[query], n_results=n)
        return res["ids"][0] if res["ids"] else []

    def search(self, query: str, top_k: int = 6, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = self.candidates
        rankings: dict[str, list[str]] = {}
        if mode in ("hybrid", "bm25"):
            rankings["bm25"] = self.keyword(query, n)
        if mode in ("hybrid", "vector"):
            rankings["vector"] = self.vector(query, n)
        has_code = any(looks_like_code(t) for t in tokenize(query))
        weights = {"bm25": self.bm25_weight * (2.0 if has_code else 1.0), "vector": 1.0}
        fused = rrf(rankings, weights, self.rrf_k)
        hits = [Hit(self.index.chunks[self.index.by_id[cid]], score, r) for cid, score, r in fused if cid in self.index.by_id]
        return self._expand_pages(hits, top_k)

    def _expand_pages(self, hits: list[Hit], top_k: int, page_max_chars: int = 3500) -> list[Hit]:
        """Small-to-big: chunks are matched individually, but when a matched chunk's page is
        short, the model gets the whole page as one source. A one-page spec sheet then
        arrives complete (the battery type in one chunk, the battery life in the next),
        and one page appears only once, so a dense page cannot crowd out the rest."""
        out: list[Hit] = []
        whole_pages: set[tuple[str, int]] = set()
        taken: set[str] = set()
        for h in hits:
            key = (h.chunk.doc, h.chunk.page)
            if key in whole_pages or h.chunk.id in taken:
                continue
            page = self.index.pages.get(key, [h.chunk])
            if len(page) > 1 and sum(len(c.text) for c in page) <= page_max_chars:
                whole_pages.add(key)
                taken.update(c.id for c in page)
                merged = Chunk(f"{h.chunk.doc}::p{h.chunk.page}", h.chunk.doc, h.chunk.title, h.chunk.page,
                               h.chunk.section, "\n".join(c.text for c in page))
                out.append(Hit(merged, h.score, h.ranks))
            else:
                taken.add(h.chunk.id)
                out.append(h)
            if len(out) == top_k:
                break
        return out
