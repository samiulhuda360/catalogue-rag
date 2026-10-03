from catalogue_rag.config import settings


def test_self_hosted_model_settings_override_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu-server:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    cfg = settings()
    assert cfg.llm_base_url == "http://gpu-server:8000/v1" and cfg.llm_model == "local-model"


def test_self_hosted_server_needs_no_key(monkeypatch):
    from catalogue_rag.generation import Generator

    gen = Generator("", "http://localhost:8000/v1", "local-model")
    assert gen.model == "local-model"
