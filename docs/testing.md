# How to test Catalogue RAG

Four levels, from a two-minute smoke test to the full evaluation. All commands are run from the
project folder with the virtual environment active.

**Windows**
```bat
cd catalogue-rag
venv\Scripts\activate
```
**macOS / Linux**
```bash
cd catalogue-rag && source venv/bin/activate
```

First time only: `pip install -e ".[dev]"`. You need `OPENROUTER_API_KEY` in `.env` for anything
that generates an answer; levels 1 and 2 work without it.

---

## Level 1 · Unit tests (seconds, no key, no documents)

```bash
pytest -v
ruff check src tests eval
```

35 tests cover the chunker (pages, tables, breadcrumbs, big-table splitting), BM25 (part-number
tokens, ranking), rank fusion, page expansion, citation parsing, the decline path, and the API with
a fake model, including the streaming endpoint and the upload gate (off without a token, wrong token
refused, file names sanitised, size capped, re-uploads replace rather than duplicate). GitHub Actions runs exactly these on every push. All must pass before a commit.

## Level 2 · Retrieval only (a minute, no key)

Check what the search finds before any language model is involved. This is where most RAG bugs live.

```bash
catalogue-rag search "What battery does the H200 handle use?"
catalogue-rag search "CH/10/1200" --mode bm25
catalogue-rag search "how long does the battery last" --mode vector
```

Each result shows the title, page, section and its rank in each retriever, e.g. `{'bm25': 2, 'vector': 1}`.
A good result has the fact you expect in the top 3. Compare `--mode bm25`, `--mode vector` and the
default hybrid to see which retriever is carrying a question.

Scored version, all 40 answerable questions:
```bash
python eval/run_eval.py --retrieval-only
```
Expect `hybrid (final)` around **98%** "fact recall" (the expected fact is inside what the model is shown).

## Level 3 · By hand in the web UI (10 minutes)

```bash
catalogue-rag serve
```
Open http://127.0.0.1:8000 and work through this list. For each answer: is it right, does every
sentence carry a [n] chip, and does clicking the chip highlight a source that actually says it?

The questions below use the bundled fictional sample catalogues (`examples/parsed/`).

| # | Ask | Expect | Tests |
|---|---|---|---|
| 1 | What battery does the Acme H200 wireless access handle use, and how long does it last? | 3 x LR03 AAA; 30 months at 30 openings a day | facts in a spec table |
| 2 | Part number for the 10mm x 1200mm hardened steel chain? | CH/10/1200 | part-number lookup in a table |
| 3 | Part number for the H200 handle, Satin Chrome right hand? | H200SCRH | model, finish and handing in one code |
| 4 | Which C600 closer power size suits an 1100mm door weighing up to 80kg? | Power size 4 | reading a table row, not a sentence |
| 5 | Which backsets does the M52 series mortice lock come in? | 54, 70 and 89mm | several values from one page |
| 6 | Which friction stay suits a 750mm sash height? | WS300 | matching a value to a range |
| 7 | Which bottom roller carries a 120kg sliding door? | SR120 | table plus product page agree |
| 8 | How heavy is the Acme Commercial Safe? | 87kg | one row of a wide table |
| 9 | Is the Acme M52 lock approved for use with Gallagher access control? | **Declines** (amber "Not in the catalogues") | must not invent approvals |
| 10 | How much does the Acme H200 handle cost? | **Declines** | must not invent prices |
| 11 | What battery does the Acme H900 handle use? | **Declines** | a product that does not exist |
| 12 | Tell me about door closers | An overview citing several closers | vague question |
| 13 | H200 batery lyfe | Still finds the battery life | typos |
| 14 | M52 vs M55 mortice lock: what's different? | Uses and functions, both cited | comparing two products |

Also try: an empty box (Ask does nothing), a 600-character question (rejected), and the page on a
phone-width window.

**Checking a citation properly.** Open the original PDF at the cited page. The figure should be on
that page. If the page number is off by one or two, the parser's page breaks are the cause, not the
answer.

## Level 4 · Full evaluation (about 10 minutes, a few cents of API credit)

```bash
python eval/run_eval.py
```
Runs the final system (and the April prototype, if its old index is present) on the question set and writes
`REPORT.md` (summary) and `latest.json` (every answer). On the bundled sample they go to
`eval/results/sample/`. Published results on the 54-catalogue corpus: 98% correct, 100% of unanswerable
questions declined, about 2 seconds per answer.

What to read in the report:
- **"Where the final system is wrong"**. Read each miss. Decide: wrong answer, wrong retrieval, or
  a wrong answer key. Twice so far the answer key was wrong; the fix there is the question, not the code.
- **Wrongly declines.** Rising here means the prompt has become too cautious.
- **p95 latency.** A jump usually means the model provider is slow, not the code.

After correcting an expected answer, re-score the saved answers without paying for new ones:
```bash
python eval/run_eval.py --rescore
```

### Adding your own questions

One line per question in `eval/questions.jsonl` (used instead of the sample set when it exists):
```json
{"id": "q41", "category": "part_number", "question": "Part number for the 400mm friction stay?",
 "expect": [["ws400"]], "docs": ["acme-window-hardware"]}
```
- `expect` is a list of groups; every group must appear in the answer; any spelling inside a group
  counts. Matching ignores case and spaces: `[["35"], ["80"]]` means both 35 and 80.
- `docs` are the catalogue file names (without `.pdf`) that contain the answer.
- `"evidence"` (optional) is what retrieval must find when the source wording differs from the
  answer, e.g. the table row `"1100|80|4"` for the answer "size 4".
- An unanswerable question has `"expect": []` and `"category": "unanswerable"`.
- **Verify the expected answer in the source before adding it.** Copy it from the parsed Markdown,
  never from memory.

### Changing the system safely

1. Make the change.
2. `pytest` and `ruff check`.
3. `python eval/run_eval.py --retrieval-only`: retrieval must not drop.
4. If prompts or generation changed: full `python eval/run_eval.py`.
5. Commit the code and the new `eval/results/` together, so the history shows what each change did.

To tune the keyword weight or the number of sources: `python eval/sweep_weights.py`.

---

## A 2-minute demo for interviews

1. Ask #1: point at the [1] chip, hover it to preview the source, click it, and watch the passage map light up.
2. Ask #9: show the amber decline, and say the April version was told to "state vendor approval
   status", which made it invent.
3. Open `eval/results/REPORT.md`: 25% to 98%, and the answer key was wrong twice.
4. `catalogue-rag search "H200SCRH" --mode vector` vs `--mode bm25`: why keyword search matters
   for part numbers.
