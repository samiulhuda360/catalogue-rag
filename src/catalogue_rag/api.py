"""HTTP API and web UI.

    GET  /                  the web UI
    POST /api/ask           {"question": "...", "top_k": 8} -> answer with citations
    POST /api/ask/stream    the same, as server-sent events: sources, declined, delta..., done
    GET  /api/map           2-D map of every passage (for the UI's passage map)
    GET  /api/health        index status
    GET  /api/documents     what the assistant knows about
    GET  /api/admin         whether adding catalogues is switched on
    POST /api/documents/upload?filename=x.pdf   raw file body, X-Admin-Token header -> a job
    GET  /api/jobs          upload jobs and their state (X-Admin-Token header)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import ROOT, settings

UI = ROOT / "ui" / "index.html"

app = FastAPI(title="Catalogue RAG", version=__version__, description="Cited answers from product catalogues.")


@lru_cache(maxsize=1)
def assistant():
    from .pipeline import Assistant

    return Assistant()


@lru_cache(maxsize=1)
def uploads():
    from .uploads import Uploads

    return Uploads(settings(), on_indexed=_reload)


def _reload() -> None:
    """A catalogue was added: the next question loads the updated index."""
    clear = getattr(assistant, "cache_clear", None)
    if clear:
        clear()


def _admin(token: str) -> None:
    from .uploads import token_ok

    expected = settings().admin_token
    if not expected:
        raise HTTPException(404, "Adding catalogues is switched off. Set CATALOGUE_ADMIN_TOKEN to enable it.")
    if not token_ok(token, expected):
        raise HTTPException(403, "Wrong admin token.")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=8, ge=1, le=12)


class CitationOut(BaseModel):
    n: int
    doc: str
    title: str
    page: int
    section: str
    snippet: str


class AskOut(BaseModel):
    question: str
    answer: str
    declined: bool
    citations: list[CitationOut]
    sources: list[CitationOut]
    timings_ms: dict
    model: str


@app.get("/", include_in_schema=False)
def ui():
    if not UI.exists():
        raise HTTPException(404, "UI not found")
    return FileResponse(UI)


def _question(body: AskIn) -> str:
    question = body.question.strip()
    if len(question) < 3:
        raise HTTPException(400, "Ask a question of at least three characters.")
    return question


@app.post("/api/ask", response_model=AskOut)
def ask(body: AskIn):
    question = _question(body)
    try:
        return assistant().ask(question, top_k=body.top_k).to_dict()
    except SystemExit as exc:  # missing index or key: a setup problem, not a crash
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/ask/stream")
def ask_stream(body: AskIn):
    """Server-sent events, one JSON object per `data:` line. The UI shows the sources as soon
    as retrieval finishes, then the answer as the model writes it."""
    question = _question(body)
    try:
        bot = assistant()
    except SystemExit as exc:
        raise HTTPException(503, str(exc)) from exc

    def events():
        try:
            for event in bot.ask_stream(question, top_k=body.top_k):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001  report the failure inside the stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:300]})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/map")
def passage_map():
    """Every passage as a point: a 2-D projection of its embedding, so passages about similar
    things sit close together, plus which catalogue it comes from."""
    from .projection import passage_map as build

    return build(assistant().index, Path(settings().index_dir))


@app.get("/api/health")
def health():
    manifest = Path(settings().index_dir) / "manifest.json"
    info = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    return {"status": "ok" if info else "no-index", "version": __version__, "index": info}


@app.get("/api/documents")
def documents():
    docs = assistant().index.documents()
    return {"count": len(docs), "documents": [{"doc": d, "title": t} for d, t in sorted(docs.items(), key=lambda x: x[1].lower())]}


@app.get("/api/admin")
def admin_status():
    cfg = settings()
    return {"uploads": bool(cfg.admin_token), "pdf_extraction": bool(cfg.llama_cloud_api_key), "max_upload_mb": cfg.max_upload_mb}


@app.post("/api/documents/upload", status_code=202)
async def upload(request: Request, filename: str = Query(min_length=1, max_length=200), x_admin_token: str = Header(default="")):
    """Add a catalogue: the raw file is the request body. Extraction and indexing run in the
    background; poll /api/jobs for progress."""
    _admin(x_admin_token)
    limit = settings().max_upload_mb * 1024 * 1024
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, f"Files are limited to {settings().max_upload_mb} MB.")
    data = bytearray()
    async for part in request.stream():
        data += part
        if len(data) > limit:
            raise HTTPException(413, f"Files are limited to {settings().max_upload_mb} MB.")
    try:
        return uploads().submit(filename, bytes(data)).to_dict()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/jobs")
def jobs(x_admin_token: str = Header(default="")):
    _admin(x_admin_token)
    return {"jobs": uploads().list()}
