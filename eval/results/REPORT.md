# Evaluation report

Run 2026-10-03T15:23:18 · model `qwen/qwen3-235b-a22b-2507` · 46 questions (40 answerable, the rest deliberately unanswerable) · top 8 sources

## Retrieval

| System | Expected document retrieved | Answer text reaches the model | Fact MRR |
|---|---|---|---|
| prototype (whole documents) | 92% | 30% | 0.24 |
| keyword only (BM25) | 100% | 100% | 0.79 |
| embeddings only | 92% | 82% | 0.63 |
| hybrid (final) | 100% | 98% | 0.80 |

## Answers

| System | Correct | Cites the right document | Declines unanswerable | Wrongly declines | p50 / p95 |
|---|---|---|---|---|---|
| prototype | 20% | n/a (no per-claim citations) | 83% | 55% | 6.2s / 15.6s |
| final | 98% | 98% | 100% | 2% | 2.3s / 3.0s |

### Correct answers by question type

| System | part_number | spec | table |
|---|---|---|---|
| prototype | 7% | 29% | 0% |
| final | 100% | 96% | 100% |

### Where the final system is wrong

- **q39** (the weight of one safe model) - retrieval surfaced neighbouring safe models; the system declined rather than guess.

Measured on 54 public door hardware catalogues that are not redistributed here (see `docs/sources.md`); the question set and per-question answers quote those catalogues and are kept private for the same reason. Run the same evaluation on the bundled fictional sample with `python eval/run_eval.py` (results go to `eval/results/sample/`).
