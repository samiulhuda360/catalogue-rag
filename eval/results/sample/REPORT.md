# Evaluation report

Run 2026-10-03T20:55:44 · model `qwen/qwen3-235b-a22b-2507` · 21 questions (16 answerable, the rest deliberately unanswerable) · top 8 sources

## Retrieval

| System | Expected document retrieved | Answer text reaches the model | Fact MRR |
|---|---|---|---|
| keyword only (BM25) | 100% | 94% | 0.91 |
| embeddings only | 94% | 88% | 0.79 |
| hybrid (final) | 100% | 94% | 0.90 |

## Answers

| System | Correct | Cites the right document | Declines unanswerable | Wrongly declines | p50 / p95 |
|---|---|---|---|---|---|
| final | 100% | 100% | 100% | 0% | 1.5s / 2.6s |

### Correct answers by question type

| System | part_number | spec | table |
|---|---|---|---|
| final | 100% | 100% | 100% |

Regenerate with `python eval/run_eval.py`. Per-question detail: `eval/results/latest.json`.
