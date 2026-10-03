import json
import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from catalogue_rag import api
from catalogue_rag.config import settings
from catalogue_rag.uploads import Uploads, safe_name, token_ok


def test_safe_name_keeps_only_a_clean_base_name():
    assert safe_name("../../etc/Door Closers (2026).PDF") == "Door-Closers-2026.pdf"
    assert safe_name("C:\\docs\\locks.md") == "locks.md"
    for bad in ["script.exe", "noextension", "....pdf"]:
        with pytest.raises(ValueError):
            safe_name(bad)


def test_token_check_needs_a_configured_token():
    assert token_ok("abc", "abc") and not token_ok("abd", "abc") and not token_ok("", "")


def gated_client(monkeypatch, tmp_path, token="s3cret"):
    monkeypatch.setenv("CATALOGUE_ADMIN_TOKEN", token)
    monkeypatch.setenv("CATALOGUE_DOCUMENTS_DIR", str(tmp_path / "docs"))
    ran = []

    def fake_runner(cfg, path, log=print):
        ran.append(path.name)
        log("  12 chunks  " + path.stem)
        return {"doc": path.stem, "chunks": 12, "documents": 11, "total_chunks": 300}

    up = Uploads(settings(), runner=fake_runner)
    monkeypatch.setattr(api, "uploads", lambda: up)
    return TestClient(api.app), ran


def test_uploads_are_off_without_a_token(monkeypatch, tmp_path):
    monkeypatch.delenv("CATALOGUE_ADMIN_TOKEN", raising=False)
    c = TestClient(api.app)
    assert c.get("/api/admin").json()["uploads"] is False
    r = c.post("/api/documents/upload?filename=a.md", content=b"# A", headers={"X-Admin-Token": "anything"})
    assert r.status_code == 404


def test_wrong_token_is_refused(monkeypatch, tmp_path):
    c, ran = gated_client(monkeypatch, tmp_path)
    r = c.post("/api/documents/upload?filename=a.md", content=b"# A", headers={"X-Admin-Token": "nope"})
    assert r.status_code == 403 and not ran
    assert c.get("/api/jobs").status_code == 403


def test_upload_runs_a_job_to_done(monkeypatch, tmp_path):
    c, ran = gated_client(monkeypatch, tmp_path)
    h = {"X-Admin-Token": "s3cret"}
    r = c.post("/api/documents/upload?filename=New Closers.md", content=b"# New closers\n\nC900 | size 6", headers=h)
    assert r.status_code == 202 and r.json()["file"] == "New-Closers.md"
    for _ in range(50):
        job = c.get("/api/jobs", headers=h).json()["jobs"][0]
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["state"] == "done" and job["chunks"] == 12 and ran == ["New-Closers.md"]
    assert (tmp_path / "docs" / "New-Closers.md").exists()


def test_upload_rejects_fake_pdfs_and_oversize(monkeypatch, tmp_path):
    c, _ = gated_client(monkeypatch, tmp_path)
    h = {"X-Admin-Token": "s3cret"}
    assert c.post("/api/documents/upload?filename=x.pdf", content=b"not a pdf", headers=h).status_code == 400
    monkeypatch.setenv("CATALOGUE_MAX_UPLOAD_MB", "1")
    c, _ = gated_client(monkeypatch, tmp_path)
    big = b"%PDF" + b"0" * (1024 * 1024 + 10)
    assert c.post("/api/documents/upload?filename=x.pdf", content=big, headers=h).status_code == 413


class FakeCollection:
    def __init__(self):
        self.items = {}

    def add(self, ids, documents, metadatas):
        self.items.update({i: m for i, m in zip(ids, metadatas, strict=True)})

    def delete(self, where):
        self.items = {i: m for i, m in self.items.items() if m["doc"] != where["doc"]}


def test_add_documents_adds_and_replaces_without_a_rebuild(monkeypatch, tmp_path):
    from catalogue_rag.index import add_documents

    col = FakeCollection()
    client = types.SimpleNamespace(get_collection=lambda name: col)
    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=lambda path: client))
    parsed, index = tmp_path / "parsed", tmp_path / "index"
    parsed.mkdir()
    index.mkdir()
    old = {"id": "old::p1::c0", "doc": "old", "title": "Old", "page": 1, "section": "", "text": "kept"}
    (index / "chunks.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    col.items[old["id"]] = {"doc": "old", "page": 1}
    page = "# new.md\n\n---\n\n# New Catalogue\n\n## C900 closer\n\nPower size 6 for 1600mm doors.\n"
    (parsed / "new_content.md").write_text(page, encoding="utf-8")

    m = add_documents(parsed, index, ["new"], log=lambda *_: None)
    assert m["documents"] == 2 and m["last_added"]["documents"] == ["new"]
    n_new = m["last_added"]["chunks"]
    assert n_new >= 1 and len(col.items) == 1 + n_new

    m = add_documents(parsed, index, ["new"], log=lambda *_: None)  # same file again: replaced, not duplicated
    lines = (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + n_new and m["chunks"] == 1 + n_new and len(col.items) == 1 + n_new
    assert json.loads(lines[0])["text"] == "kept"
