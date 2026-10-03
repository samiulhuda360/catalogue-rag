"""Build and load the search index: chunks on disk, embeddings in ChromaDB.

    chunks.jsonl   every chunk (the source of truth; BM25 is rebuilt from it on load)
    chroma/        the vector store (all-MiniLM-L6-v2 sentence embeddings, run locally)
    manifest.json  what was indexed and with which settings

build_index() indexes everything from scratch; add_documents() adds or replaces a few
documents in an existing index, so a new catalogue is searchable in seconds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .bm25 import BM25
from .chunking import Chunk, chunk_document

COLLECTION = "catalogue_chunks"


def parsed_documents(parsed_dir: Path) -> list[tuple[str, Path]]:
    """(document id, path) for every parsed Markdown file."""
    out = []
    for p in sorted(parsed_dir.glob("*.md")):
        doc_id = p.name[: -len("_content.md")] if p.name.endswith("_content.md") else p.stem
        out.append((doc_id, p))
    return out


def build_index(parsed_dir: Path, index_dir: Path, chunk_chars: int = 1200, log=print) -> dict:
    import chromadb

    started = time.time()
    docs = parsed_documents(parsed_dir)
    if not docs:
        raise SystemExit(f"No parsed documents in {parsed_dir}. Run `catalogue-rag parse` first, or point "
                         "CATALOGUE_PARSED_DIR at a folder of Markdown files.")
    chunks: list[Chunk] = []
    for doc_id, path in docs:
        cs = chunk_document(doc_id, path.read_text(encoding="utf-8"), max_chars=chunk_chars)
        chunks += cs
        log(f"  {len(cs):5d} chunks  {doc_id}")

    index_dir.mkdir(parents=True, exist_ok=True)
    _write_chunks(index_dir / "chunks.jsonl", chunks)

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001  first build
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    _embed(col, chunks, log)

    manifest = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "documents": len(docs),
                "chunks": len(chunks), "chunk_chars": chunk_chars, "embedding_model": "all-MiniLM-L6-v2 (ChromaDB default)",
                "seconds": round(time.time() - started, 1)}
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def add_documents(parsed_dir: Path, index_dir: Path, doc_ids: list[str], chunk_chars: int = 1200, log=print) -> dict:
    """Index just these documents, replacing any earlier version of them, and leave the rest alone.
    Builds the whole index instead when there is none yet."""
    import chromadb

    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return build_index(parsed_dir, index_dir, chunk_chars, log)
    started = time.time()
    replace = set(doc_ids)
    kept = [c for c in _read_chunks(path) if c.doc not in replace]
    new: list[Chunk] = []
    for doc_id in doc_ids:
        cs = chunk_document(doc_id, (parsed_dir / f"{doc_id}_content.md").read_text(encoding="utf-8"), max_chars=chunk_chars)
        new += cs
        log(f"  {len(cs):5d} chunks  {doc_id}")
    col = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection(COLLECTION)
    for doc_id in replace:
        col.delete(where={"doc": doc_id})
    _embed(col, new, log)
    _write_chunks(path, kept + new)  # last, so a failure above leaves the old chunk list in place

    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "documents": len({c.doc for c in kept + new}), "chunks": len(kept) + len(new),
                     "last_added": {"documents": sorted(replace), "chunks": len(new), "seconds": round(time.time() - started, 1)}})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _embed(col, chunks: list[Chunk], log) -> None:
    for i in range(0, len(chunks), 256):
        batch = chunks[i:i + 256]
        col.add(ids=[c.id for c in batch], documents=[c.search_text() for c in batch],
                metadatas=[{"doc": c.doc, "page": c.page} for c in batch])
        log(f"  embedded {min(i + 256, len(chunks))}/{len(chunks)}")


def _read_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: readers never see a half-written file


@dataclass
class Index:
    chunks: list[Chunk]
    by_id: dict[str, int]
    bm25: BM25
    collection: object  # chromadb collection, or None when running keyword-only

    @classmethod
    def load(cls, index_dir: Path, with_vectors: bool = True) -> "Index":
        path = index_dir / "chunks.jsonl"
        if not path.exists():
            raise SystemExit(f"No index at {index_dir}. Run `catalogue-rag index` first.")
        chunks = _read_chunks(path)
        collection = None
        if with_vectors:
            import chromadb

            collection = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection(COLLECTION)
        return cls(chunks, {c.id: i for i, c in enumerate(chunks)}, BM25([c.search_text() for c in chunks]), collection)

    @property
    def pages(self) -> dict[tuple[str, int], list[Chunk]]:
        """Chunks grouped by (document, page), in reading order."""
        if not hasattr(self, "_pages"):
            grouped: dict[tuple[str, int], list[Chunk]] = {}
            for c in self.chunks:
                grouped.setdefault((c.doc, c.page), []).append(c)
            self._pages = grouped
        return self._pages

    def documents(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for c in self.chunks:
            out.setdefault(c.doc, c.title)
        return out
