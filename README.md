# ChoiceShift

**Measure how much an LLM's multiple-choice answers move when the options are reordered — one call on any model returns per-question permutation cards and an aggregate report: stability score, flip rate, selection share by position, and the accuracy range that a single benchmark number hides.**

A multiple-choice benchmark reports one accuracy. But the accuracy is measured at one arbitrary ordering of each question's options, and language models are not indifferent to order: permute the options of the same question and the model's answer — and therefore its score — moves. The literature has measured swings of several accuracy points on standard benchmarks from reordering alone (Pezeshkpour & Hruschka, NAACL Findings 2024; Zheng et al., ICLR 2024). ChoiceShift turns that finding into an instrument: give it your questions and any model behind a callable, and it runs every ordering (all 24 for a 4-choice question), then reports how stable the answers actually were — per question and in aggregate — instead of letting one ordering speak for the model.

## Install

ChoiceShift is not on PyPI yet. Install the GitHub version:

```
pip install "git+https://github.com/habib-analyst/choiceshift.git"
```

For development:

```
git clone https://github.com/habib-analyst/choiceshift.git
cd choiceshift
python -m pip install -e ".[dev]"
python -m pytest
```

Requires Python 3.10+. The only runtime dependency is NumPy. Your model is a callable — a local Hugging Face model behind a scoring function, an API client that generates text, a hand-written rule: ChoiceShift renders the prompts, never imports your framework, and never sees your weights.

## Quickstart

```
from choiceshift import Question, probe

def score_fn(prompt: str) -> dict[str, float]:
    ...  # return a score (e.g. logprob) per option ID: {"A": ..., "B": ...}

questions = [
    Question("Capital of France?", ("Paris", "London", "Berlin", "Madrid"), answer=0),
    # ...
]

report = probe(score_fn, questions)          # score mode (default)
print(report.summary())
# ChoiceShift report: 120 question(s), 24 permutation(s) each,
# stability score ..., flip rate ..., fragile questions ...
#   canonical accuracy ...; mean over permutations ...;
#   range [..., ...] (width ...); majority-vote accuracy ...; RStd ...

card = report.cards[0]
print(card.canonical, card.modal, card.modal_share, card.n_flips)

report.to_dict()   # full artifact: every card, every selection, full config
```

Two model protocols:

- **Score mode** (`mode="score"`): the callable returns a mapping of option ID letter to a score — typically the next-token logprob of `" A"`/`" B"`/… after the prompt. The prediction is the argmax letter (ties break toward the earlier letter). This is the standard loglikelihood protocol used by lm-eval and the selection-bias literature.
- **Generate mode** (`mode="generate"`): the callable returns generated text, and a conservative parser reads the choice out of it (explicit `Answer: B` marker, leading `B.` marker, bare letter, then verbatim content match — in that order). Anything it cannot read is counted in the report's **unparsed rate**, never guessed.

## What the report measures

Selections are tracked as *content*, not positions: if a model always picks the first presented option, its selected content cycles through every choice as the order rotates, and the report shows exactly that.

- **Stability score** — mean over questions of the modal share: how often the most-chosen content was chosen, across all orders. 1.0 means no answer ever moved.
- **Flip rate** — share of non-canonical runs whose selection differs from the answer at the benchmark's original (identity) ordering.
- **Fragile questions** — share of questions whose answer moves under at least one ordering, with per-question cards (canonical vs modal selection, modal share, flip count) so the fragile items can be inspected, not just counted.
- **Selection share by position** — where selections land across positions 1..k. One honest caveat, stated rather than hidden: option IDs follow positions in this layout (A is always first), so this profile mixes *position* preference and *ID-token* preference exactly as they are mixed in the benchmarks themselves. Zheng et al. show the two can be separated by shuffling IDs independently; ChoiceShift measures the joint effect that a benchmark score actually experiences.
- **Accuracy block** (when gold labels are supplied) — accuracy at the canonical ordering, mean/min/max over orderings, the **accuracy range** (max − min: the part of the headline number that is order luck), **majority-vote accuracy** (score the modal content instead of the canonical-order answer), and **RStd**, the per-position gold-recall spread of Zheng et al.: for each position, how often the gold content was selected when it sat there.

The report records its full config — mode, seed, permutation set, template — and `OrderReport.from_dict(report.to_dict())` round-trips it, so a run can be re-derived or challenged.

## What ChoiceShift is not

- **Not a benchmark harness.** lm-eval, Inspect, and lighteval run benchmark suites and report accuracy; Inspect's multiple-choice solver can even shuffle choices once, with a seed. None of them reports a permutation-stability profile — per-question flip cards, an accuracy range over orderings, a recall spread — as the product. ChoiceShift takes questions you already have and measures the one property they don't.
- **Not a debiasing method.** PriDe (Zheng et al.) estimates and subtracts a model's prior over option IDs; it is a correction, published as paper code for specific models and datasets. ChoiceShift *measures* — it ships no correction of its own. The majority-vote number in the report is a measurement (what this scoring rule would have given), clearly labeled, not a claim that voting fixes the bias.
- **Not a judge of answer quality.** A model can be perfectly stable and perfectly wrong — stability 1.0 at 25% accuracy is a consistent failure, and the report prints both numbers side by side so neither can be quoted without the other.
- **Not a prompt-robustness suite.** It permutes option order only. Paraphrase, formatting, and few-shot sensitivity are different instruments for different questions.

## Measured demo (real model, real data)

`examples/mmlu_order_demo.py` runs Qwen2.5-0.5B-Instruct (Apache-2.0, local CPU inference) over 120 MMLU test questions (six subjects × the first 20 questions of each published test file, in file order — a deterministic subset, not a random sample), all 24 orderings each: 2,880 scored prompts, zero-shot plain completion format, next-token logprob of the option ID. Full record: `examples/output/demo_summary.json`; every selection: `examples/output/demo_report.json`.

| Metric | Measured |
|---|---|
| Stability score (mean modal share) | **0.757** |
| Flip rate (non-canonical runs that moved) | **0.325** |
| Questions whose answer moves under ≥ 1 order | **67.5%** (81 of 120) |
| Accuracy at the canonical ordering | 0.525 |
| Mean accuracy over all 24 orderings | 0.538 |
| Accuracy range over orderings | **0.483 – 0.600 (width 0.117)** |
| Majority-vote-over-orders accuracy | 0.567 |
| Selection share by position (uniform = 0.25) | 0.234 / 0.253 / **0.300** / 0.214 |
| Gold recall by the position the gold answer occupied | 0.521 / 0.553 / **0.601** / 0.478 |
| RStd (spread of those recalls) | 0.045 |

Read plainly: on identical questions with identical options, this model's benchmark score is somewhere between 48.3% and 60.0% depending on nothing but the order the options happen to be printed in — **11.7 points of the headline number are order luck**. Two of the 120 questions split their 24 answers across all four contents (modal share 0.25, the maximum possible instability); the single most fragile question flipped on 21 of 23 non-canonical orders. The model's selections lean toward the third position (30.0% of picks against a 25% uniform share), and its gold recall peaks exactly there (0.601 when the right answer sits third, 0.478 when it sits last). Scoring the modal content instead of the canonical-order answer — majority vote over orders — would have given 0.567 instead of 0.525 on this run; that is reported as a measurement of what that scoring rule does, not as a fix this library ships. Per-subject stability ranged from 0.706 (conceptual physics) to 0.848 (nutrition) at n=20 per subject — descriptive, not inferential. The scoring path's speed device (a prefix-shared KV cache) was parity-checked against plain scoring on the run's first question: argmax agreement 24/24, max |Δlogprob| 8.5e-06, recorded in the summary JSON. The model file is byte-identical to the official release (sha256 `fdf756fa…`).

Two scripted scorers in `examples/order_demo.py` demonstrate the mechanics with no model at all (a content-tracking scorer scores stability 1.000; a scorer that always prefers the first position scores 0.250 with a 33-point accuracy range on three questions). Those numbers describe the scripts, not any model, and the file says so.

## License

MIT. Issues and pull requests are welcome on GitHub.
