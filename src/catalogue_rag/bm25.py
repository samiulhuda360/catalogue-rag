"""Okapi BM25 keyword search, tuned for part numbers.

Catalogue questions are often about a code ("CH/10/900", "570MT5+M", "61B/4").
Embedding models split such codes into meaningless pieces; BM25 matches them exactly.
The tokenizer keeps each code whole *and* adds its parts, so "K60-KP-RH" matches a
question that says "K60 keypad" as well as one that quotes the full code.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

CODE = re.compile(r"[a-z0-9]+(?:[/\-+.][a-z0-9]+)*\+?")
STOP = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what which with "
    "do does can my me you your there their".split()
)


def tokenize(text: str) -> list[str]:
    """Whole tokens, plus the parts of compound codes. Catalogue codes glue a model
    number to a variant ("C700HOSIL" = model C700, hold-open, silver), so a question
    about "the C700 closer" must still match: codes are also split at separators and at
    letter/digit boundaries."""
    out = []
    for tok in CODE.findall(text.lower()):
        if tok in STOP:
            continue
        out.append(tok)
        parts = [p for p in re.split(r"[/\-+.]", tok) if p] if any(c in tok for c in "/-+.") else [tok]
        for p in parts:
            pieces = re.findall(r"\d+|[a-z]+", p) if re.search(r"\d", p) and re.search(r"[a-z]", p) else [p]
            out += [x for x in (([p] if p != tok else []) + (pieces if len(pieces) > 1 else []))
                    if x not in STOP and (len(x) >= 2 or x.isdigit())]
    return out


def looks_like_code(token: str) -> bool:
    """A part number: letters and digits together, or digits with separators."""
    return bool(re.search(r"\d", token) and (re.search(r"[a-z]", token) or re.search(r"[/\-+]", token))) and len(token) >= 3


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in documents]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(self.tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(documents)
        # Robertson-Sparck Jones IDF with +1 so very common terms never score negative.
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            boost = 2.0 if looks_like_code(term) else 1.0
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                scores[i] += boost * idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
