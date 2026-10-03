@echo off
rem Start Catalogue RAG: web UI on http://127.0.0.1:8000 (close this window to stop it).
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo No virtual environment found. Run:  python -m venv venv  ^&^&  venv\Scripts\pip install -e ".[dev]"
  pause
  exit /b 1
)
if not exist "index\chunks.jsonl" (
  echo No search index yet. Building it now ^(about 3 minutes^)...
  venv\Scripts\python.exe -m catalogue_rag index || (pause & exit /b 1)
)
start "" http://127.0.0.1:8000
venv\Scripts\python.exe -m catalogue_rag serve --port 8000
pause
