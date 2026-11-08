# Evaluation: demo

4 PRs · 19 real known issues · no LLM (tool evidence only) · judge lexical · 1 repeat(s) · 2026-11-08 13:43

| System | Precision | Recall | F1 (95% CI) | Sev-weighted recall | Recall@3 | Comments / real issue |
|---|---:|---:|---:|---:|---:|---:|
| SignalPR (full) | 0.714 | 0.789 | 0.750 (0.667–0.833) | 0.821 | 0.421 | 1.40 |
| Code-analysis tools only | 0.630 | 0.895 | 0.739 (0.714–0.778) | 0.897 | 0.421 | 1.59 |
| SignalPR, ranked by change size | 0.714 | 0.789 | 0.750 (0.667–0.833) | 0.821 | 0.368 | 1.40 |
| – intent checker | 0.700 | 0.737 | 0.718 (0.667–0.833) | 0.795 | 0.421 | 1.43 |
| – evidence collector | 1.000 | 0.053 | 0.100 (0.000–0.222) | 0.026 | 0.053 | 1.00 |
| – verifier | 0.630 | 0.895 | 0.739 (0.714–0.778) | 0.897 | 0.421 | 1.59 |
| – risk ranker | 0.714 | 0.789 | 0.750 (0.667–0.833) | 0.821 | 0.368 | 1.40 |

Skipped: Single-prompt LLM (same model) (needs an LLM (none reachable)); – self-consistency (needs an LLM (none reachable))
