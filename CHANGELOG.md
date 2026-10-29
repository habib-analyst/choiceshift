# Changelog

## 0.1.0 — 2026-10-29

Initial release.

- `probe()` / `probe_question()` over any model callable, in two modes:
  letter-score (loglikelihood) and generate (parsed free text).
- Full permutation sets when `k!` fits the budget (all 24 orders for the
  standard 4-choice case); seeded sampling above it, recorded in the report.
- Per-question cards: canonical vs modal selection, modal share, flip count,
  correctness of both.
- Aggregate report: stability score, flip rate, fragile-question rate,
  selection share by position, canonical / mean / min / max accuracy,
  accuracy range, majority-vote accuracy, per-position gold recall and
  RStd, unparsed rate (generate mode).
- JSON round-trip for the full report (`to_dict` / `from_dict`).
- Conservative answer parser for generate mode: explicit marker, leading
  marker, bare letter, then verbatim content match; anything else is
  counted as unparsed, never guessed.
- Examples: a scripted mechanics demo, and a measured MMLU run
  (Qwen2.5-0.5B-Instruct, 120 questions x 24 orders) with the full report
  saved as JSON.
