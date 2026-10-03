from fastapi.testclient import TestClient

from catalogue_rag import api
from catalogue_rag.pipeline import Answer, Citation


class FakeAssistant:
    def ask(self, question, top_k=6):
        cite = Citation(1, "a", "Acme H200", 1, "Technical data", "Battery | 1 x Lithium CR123A")
        return Answer(question, "It uses a CR123A battery [1].", False, [cite], [cite], {"retrieve": 5, "generate": 9}, "fake")


def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(api, "assistant", lambda: FakeAssistant())
    return TestClient(api.app)


def test_ask_returns_answer_with_citations(monkeypatch):
    r = client(monkeypatch).post("/api/ask", json={"question": "What battery does the H200 use?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["page"] == 1 and "CR123A" in body["answer"]


def test_empty_or_huge_questions_are_rejected(monkeypatch):
    c = client(monkeypatch)
    assert c.post("/api/ask", json={"question": ""}).status_code == 422
    assert c.post("/api/ask", json={"question": "x" * 501}).status_code == 422
    assert c.post("/api/ask", json={"question": "ok?", "top_k": 99}).status_code == 422


def test_health_answers_without_an_index(monkeypatch, tmp_path):
    monkeypatch.setenv("CATALOGUE_INDEX_DIR", str(tmp_path))
    r = client(monkeypatch).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "no-index"


def test_stream_endpoint_emits_server_sent_events(monkeypatch):
    class StreamingAssistant(FakeAssistant):
        def ask_stream(self, question, top_k=8):
            yield {"type": "sources", "sources": [], "chunk_ids": [], "retrieve_ms": 1}
            yield {"type": "delta", "text": "Hello"}
            yield {"type": "done", "answer": "Hello"}

    monkeypatch.setattr(api, "assistant", lambda: StreamingAssistant())
    r = TestClient(api.app).post("/api/ask/stream", json={"question": "Battery?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert [__import__("json").loads(e)["type"] for e in events] == ["sources", "delta", "done"]
