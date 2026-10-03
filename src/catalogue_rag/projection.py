"""A 2-D map of the passages for the UI, from their embeddings.

Principal component analysis (PCA) of the 384-dimension embedding vectors, keeping the two
directions of greatest variance, so passages with similar meaning land near each other.
Computed once from the vector store and cached next to the index (map.json); rebuilt
automatically when the index changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from .index import Index


def passage_map(index: Index, index_dir: Path) -> dict:
    cache = index_dir / "map.json"
    manifest = index_dir / "manifest.json"
    stamp = manifest.stat().st_mtime if manifest.exists() else 0
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("stamp") == stamp and data.get("n") == len(index.chunks):
            return data
    data = _build(index)
    data["stamp"] = stamp
    cache.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def _build(index: Index) -> dict:
    import numpy as np

    docs = sorted({c.doc for c in index.chunks})
    doc_of = {d: i for i, d in enumerate(docs)}
    ids = [c.id for c in index.chunks]
    got = index.collection.get(ids=ids, include=["embeddings"]) if index.collection is not None else None
    if got is None or got.get("embeddings") is None or len(got["embeddings"]) == 0:
        rng = np.random.default_rng(0)  # keyword-only index: still draw something sensible
        xy = rng.normal(size=(len(ids), 2))
        order = ids
    else:
        order = got["ids"]
        vecs = np.asarray(got["embeddings"], dtype=np.float32)
        vecs -= vecs.mean(axis=0)
        _u, _s, vt = np.linalg.svd(vecs, full_matrices=False)
        xy = vecs @ vt[:2].T
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    xy = np.clip((xy - lo) / np.where(hi - lo == 0, 1, hi - lo), 0, 1)  # 0..1, outliers pinned to the edge
    pos = {cid: (round(float(x), 4), round(float(y), 4)) for cid, (x, y) in zip(order, xy, strict=False)}
    chunk_doc = {c.id: c.doc for c in index.chunks}
    return {
        "n": len(ids),
        "docs": [index.documents()[d] for d in docs],
        "ids": ids,
        "x": [pos.get(cid, (0.5, 0.5))[0] for cid in ids],
        "y": [pos.get(cid, (0.5, 0.5))[1] for cid in ids],
        "doc": [doc_of[chunk_doc[cid]] for cid in ids],
    }
