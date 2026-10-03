"""Add catalogues to a running system: upload, extract, index, one job at a time.

Switched off unless CATALOGUE_ADMIN_TOKEN is set, so a public demo cannot be fed documents.
A PDF is extracted with LlamaParse; a Markdown or text file goes in as it is. Only the new
document is chunked and embedded (index.add_documents), so it is searchable within seconds
to a minute, and uploading a file with the same name again replaces the earlier version.
Jobs run one after another on a single worker thread to keep a small server responsive.
"""

from __future__ import annotations

import hmac
import itertools
import queue
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ROOT, Settings

ALLOWED = {".pdf", ".md", ".markdown", ".txt"}
SAMPLE = ROOT / "examples" / "parsed"


def safe_name(filename: str) -> str:
    """Keep only the base name, in a filesystem- and id-safe form."""
    name = Path(filename.replace("\\", "/")).name
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")[:100]
    if not stem or ext not in ALLOWED:
        raise ValueError(f"Upload a PDF, Markdown or text file (got {filename!r}).")
    return stem + ext


def token_ok(given: str, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


def parsed_target(cfg: Settings) -> Path:
    """Where extracted text goes. Never into the bundled sample folder: the first upload copies the
    sample into data/parsed_documents so that a later full rebuild keeps everything."""
    if cfg.parsed_dir.resolve() != SAMPLE.resolve():
        return cfg.parsed_dir
    own = ROOT / "data" / "parsed_documents"
    if not any(own.glob("*.md")):
        own.mkdir(parents=True, exist_ok=True)
        for p in SAMPLE.glob("*_content.md"):
            shutil.copy2(p, own / p.name)
    return own


def add_file(cfg: Settings, path: Path, log=print) -> dict:
    """Extract and index one catalogue file. Used by the upload worker and `catalogue-rag add`."""
    from .index import add_documents
    from .ingest import add_text, parse_pdf

    target = parsed_target(cfg)
    if path.suffix.lower() == ".pdf":
        parse_pdf(path, target, cfg.llama_cloud_api_key)
    else:
        add_text(path, target)
    manifest = add_documents(target, cfg.index_dir, [path.stem], cfg.chunk_chars, log=log)
    return {"doc": path.stem, "chunks": manifest.get("last_added", {}).get("chunks", manifest.get("chunks", 0)),
            "documents": manifest["documents"], "total_chunks": manifest["chunks"]}


@dataclass
class Job:
    id: int
    file: str
    size_kb: int
    state: str = "queued"          # queued -> extracting -> indexing -> done | failed
    message: str = ""
    chunks: int = 0
    created: float = field(default_factory=time.time)
    finished: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Uploads:
    def __init__(self, cfg: Settings, on_indexed=lambda: None, runner=add_file):
        self.cfg, self.on_indexed, self.runner = cfg, on_indexed, runner
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._queue: queue.Queue[tuple[Job, Path]] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, filename: str, data: bytes) -> Job:
        name = safe_name(filename)
        if not data:
            raise ValueError("The file is empty.")
        if len(data) > self.cfg.max_upload_mb * 1024 * 1024:
            raise ValueError(f"Files are limited to {self.cfg.max_upload_mb} MB.")
        if name.endswith(".pdf") and not data.startswith(b"%PDF"):
            raise ValueError("That file is not a PDF.")
        self.cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.documents_dir / name
        path.write_bytes(data)
        job = Job(next(self._ids), name, max(1, len(data) // 1024))
        with self._lock:
            self.jobs[job.id] = job
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="catalogue-uploads", daemon=True)
                self._worker.start()
        self._queue.put((job, path))
        return job

    def list(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.id)][:50]

    def _work(self) -> None:
        while True:
            job, path = self._queue.get()
            self.run(job, path)

    def run(self, job: Job, path: Path) -> None:
        def log(msg: str) -> None:
            if "embedded" in msg or "chunks" in msg:
                job.state, job.message = "indexing", msg.strip()

        try:
            job.state = "extracting" if path.suffix.lower() == ".pdf" else "indexing"
            result = self.runner(self.cfg, path, log=log)
            job.chunks = result["chunks"]
            job.message = f"{result['chunks']} passages added · {result['documents']} catalogues, {result['total_chunks']} passages in total"
            self.on_indexed()
            job.state = "done"
        except (Exception, SystemExit) as exc:  # noqa: BLE001  report it on the job
            job.state, job.message = "failed", str(exc)[:300]
        job.finished = time.time()
