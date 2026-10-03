# Catalogue RAG

**A door hardware product assistant: ask a pile of catalogues a question and get an answer with the page it came from, or an honest "not in the catalogues".**

Built for the people who answer hardware questions all day (sales, trade counters, specifiers): which battery,
which part number, which closer size, does it fit a 44 mm door. Evaluated on 54 public door hardware catalogues
(3,062 passages) with a hand-checked question set, against the first version of itself. All catalogue
knowledge comes from public sources; see [Data](#data).

![Answer with a cited source](docs/screenshots/answer.png)

## Results

46 questions written from the catalogues, every expected answer checked against the source text; 6 of them
deliberately unanswerable (vendor approvals, prices, building-code clauses). Same language model for both systems.
Full report: [`eval/results/REPORT.md`](eval/results/REPORT.md).

| | First version | This version |
|---|---|---|
| Answer text reaches the model | 30% | **98%** |
| Correct answers | 20% | **98%** |
| Part-number questions correct | 7% | **100%** |
| Declines questions the catalogues don't answer | 83% | **100%** |
| Wrongly declines answerable questions | 55% | **2%** |
| Cites the catalogue the answer came from | (no per-claim citations) | **98%** |
| Median answer time | 6.2 s | **2.3 s** |

The first version usually found the right catalogue (92%) but sent the model only its first 2,000 characters,
so the answer was visible for 30% of questions. [How that was fixed](docs/architecture.md).

## How it works

```
PDF ─► LlamaParse ─► Markdown ─► page-aware chunks (tables kept as rows, heading breadcrumbs)
                                        │
question ─► BM25 (part-number aware) ──┐│
        └─► embeddings (MiniLM) ───────┴┴─► rank fusion ─► whole page if short ─► LLM: cite [n] or decline
```

- **Table-aware chunking.** Each chunk stays on one page and carries its section breadcrumb, so a row like
  `Battery | 1 x Lithium CR123A` still knows it belongs to `H200 Wireless access handle > Technical data`.
- **Hybrid retrieval.** Real BM25 with a tokenizer that understands catalogue codes (`C700HOSIL` also matches
  "the C700 closer"), fused with embedding search by reciprocal rank fusion. Keyword weight chosen by
  [measurement](eval/results/weight_sweep.md), not by feel.
- **Small-to-big.** Match small chunks, hand the model the whole page when it is short, so a spec sheet arrives complete.
- **Cite or decline.** Every claim cites a numbered source; if the sources don't answer it, the model must say so.
  Citations map back to document, page and section in the UI.

Design decisions, trade-offs and known limits: [`docs/architecture.md`](docs/architecture.md). How to test it, from unit tests to a by-hand checklist: [`docs/testing.md`](docs/testing.md).

## Run it

Python 3.11+.

```bash
git clone https://github.com/samiulhuda360/catalogue-rag && cd catalogue-rag
python -m venv venv && source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev,parse]"
cp .env.example .env                                      # add OPENROUTER_API_KEY (any OpenAI-compatible endpoint works)
```

**Try it on the bundled sample** (ten fictional catalogues, no documents needed; used automatically when `data/` is empty):

```bash
python -m catalogue_rag index
python -m catalogue_rag ask "Which C600 closer power size suits an 1100mm, 80kg door?"
python -m catalogue_rag serve                              # web UI on http://127.0.0.1:8000
```

**Use your own catalogues:** put PDFs in `data/documents/`, add `LLAMA_CLOUD_API_KEY` to `.env`, then

```bash
python -m catalogue_rag parse      # PDF -> Markdown (only new files)
python -m catalogue_rag index      # chunk + embed
python -m catalogue_rag search "H200 right hand satin chrome"   # see what retrieval finds, no LLM call
```

**Evaluate:** `python eval/run_eval.py` (or `--retrieval-only`, free and fast). It uses the sample's questions
(`examples/questions.jsonl`) until you write your own in `eval/questions.jsonl`.

**Test:** `pytest` (35 tests, no API key or documents needed; runs in CI on every push).

## API

`POST /api/ask` with `{"question": "...", "top_k": 8}` returns the answer, whether it declined, the citations
(document, title, page, section, snippet) and timings. `POST /api/ask/stream` returns the same as server-sent
events: the sources as soon as retrieval finishes, then the answer as the model writes it (first words in about
a second). `GET /api/map` gives a 2-D PCA projection of every passage's embedding, which the UI draws as a live map
and lights up with the passages each question retrieved. `GET /api/health`, `GET /api/documents`.
Interactive docs at `/docs` when the server is running.

## Growing the knowledge base

New catalogues can be added at any time, without rebuilding the index: each one is extracted, chunked and
embedded on its own, and is searchable in seconds (Markdown) to about a minute (PDF). A file with the same name
replaces its earlier version.

- **Web UI:** set `CATALOGUE_ADMIN_TOKEN` in `.env` and an **Add catalogues** button appears: enter the token,
  drop PDFs or Markdown files, and watch each one go through extracting, indexing, done. The map and counts
  update when it finishes.
- **Command line:** `python -m catalogue_rag add new-catalogue.pdf more.md`
- **API:** `POST /api/documents/upload?filename=x.pdf` with the file as the body and an `X-Admin-Token` header;
  `GET /api/jobs` for progress.

Uploads are **off unless a token is set**, so a public demo cannot be fed documents. The token is compared in
constant time, file names are sanitised, only PDF, Markdown and text are accepted (PDFs are checked by
signature), size is capped (`CATALOGUE_MAX_UPLOAD_MB`, default 50), and jobs run one at a time so a small
server stays responsive. Only add documents you are allowed to use; this project uses public sources only.

| Declines what it can't support | Works on a phone |
|---|---|
| ![Declined answer](docs/screenshots/declined.png) | ![Mobile](docs/screenshots/mobile.png) |

| Retrieved passages light up on the map | Hover a citation to preview the source |
|---|---|
| ![Passage map](docs/screenshots/map-lit.png) | ![Citation preview](docs/screenshots/peek.png) |

| Add catalogues without a rebuild | |
|---|---|
| ![Add catalogues](docs/screenshots/add-catalogues.png) | |

## Project layout

```
src/catalogue_rag/   chunking.py  bm25.py  index.py  retrieval.py  generation.py  pipeline.py  api.py  ingest.py  uploads.py  projection.py  __main__.py
ui/index.html        single-file web UI (no build step)
eval/                run_eval.py  sweep_weights.py  results/  baseline/ (the first version, kept for comparison)
tests/               chunking, BM25, fusion, citations, streaming, API, uploads
docs/                architecture.md  testing.md  sources.md  screenshots/
examples/            make_sample.py, parsed/ (ten fictional catalogues), questions.jsonl
```

## Stack

Python · FastAPI · ChromaDB · all-MiniLM-L6-v2 embeddings (local, CPU) · BM25 (own implementation) ·
LlamaParse · any OpenAI-compatible LLM (Qwen3 235B via OpenRouter by default, ~2 s per answer) · pytest · GitHub Actions.

## Data

All catalogue knowledge used by this project comes from publicly available sources: product catalogues and brochures that manufacturers publish for free download on their own websites. No internal, confidential or paid material was used. The 54 evaluation catalogues belong to their publishers and are not in this
repository, nor is any text extracted from them. The screenshots and the bundled sample use invented "Acme"
products. This is an independent portfolio project, not affiliated with or endorsed by any manufacturer.
Details: [`docs/sources.md`](docs/sources.md).

## Author

Samiul Huda · Auckland, New Zealand · MSc Data Science. MIT licence.
