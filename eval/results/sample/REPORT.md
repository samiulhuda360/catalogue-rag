# Evaluation report

Run 2026-10-03T21:17:21 · model `qwen/qwen3-235b-a22b-2507` · 25 questions (19 answerable, the rest deliberately unanswerable) · top 8 sources

## Retrieval

| System | Expected document retrieved | Answer text reaches the model | Fact MRR |
|---|---|---|---|
| keyword only (BM25) | 100% | 89% | 0.84 |
| embeddings only | 95% | 84% | 0.75 |
| hybrid (final) | 100% | 89% | 0.81 |

## Answers

| System | Correct | Cites the right document | Declines unanswerable | Wrongly declines | p50 / p95 |
|---|---|---|---|---|---|
| final | 100% | 100% | 100% | 0% | 1.6s / 4.2s |

### Correct answers by question type

| System | part_number | spec | table |
|---|---|---|---|
| final | 100% | 100% | 100% |

Regenerate with `python eval/run_eval.py`. Per-question detail: `eval/results/latest.json`.
