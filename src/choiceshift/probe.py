"""The probe driver: permute options, call the model, aggregate.

Two model protocols are supported, selected by ``mode``:

- ``mode="score"`` -- the callable receives a rendered prompt and returns a
  mapping of option ID letter (``"A"``..) to a score (typically the logprob
  of that ID token). The prediction is the argmax letter, ties broken toward
  the earlier letter. This is the standard loglikelihood MCQ protocol and
  never produces unparsed runs.
- ``mode="generate"`` -- the callable receives a rendered prompt and returns
  generated text; :func:`choiceshift.parsing.extract_choice` reads the
  chosen option out of it. Unparseable outputs are recorded as ``None`` and
  surfaced in the report's unparsed rate rather than silently dropped.

The library itself depends on NumPy only. It never imports a model
framework: a Hugging Face model, an API client, or a hand-written rule are
all just callables (see ``examples/`` for a Transformers adapter).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from .core import OrderReport, Question, build_card
from .parsing import extract_choice
from .prompts import (
    DEFAULT_TEMPLATE,
    LETTERS,
    all_permutations,
    build_prompt,
    presented_choices,
    sample_permutations,
)

ScoreFn = Callable[[str], Mapping[str, float]]
GenerateFn = Callable[[str], str]


def probe(
    model: ScoreFn | GenerateFn,
    questions: Sequence[Question],
    *,
    mode: str = "score",
    max_permutations: int = 24,
    seed: int = 0,
    template: str = DEFAULT_TEMPLATE,
) -> OrderReport:
    """Run one model over every question under a shared permutation set.

    The permutation set depends only on the number of choices, so questions
    with different choice counts get their own set; the set (and every other
    run parameter) is recorded in ``report.config`` and the report round-
    trips through JSON unchanged.
    """
    if mode not in ("score", "generate"):
        raise ValueError(f"mode must be 'score' or 'generate', got {mode!r}")
    if not questions:
        raise ValueError("questions must be non-empty")
    if max_permutations < 1:
        raise ValueError("max_permutations must be >= 1")
    ks = {len(q.choices) for q in questions}
    if len(ks) != 1:
        raise ValueError(
            "all questions in one probe run must have the same number of "
            f"choices; got {sorted(ks)}"
        )
    k = ks.pop()
    if math.factorial(k) <= max_permutations:
        permutations = tuple(all_permutations(k))
        sampling = "all"
    else:
        permutations = tuple(sample_permutations(k, max_permutations, seed))
        sampling = "seeded_sample"

    cards = []
    for idx, q in enumerate(questions):
        selections: list[int | None] = []
        for perm in permutations:
            prompt = build_prompt(q.question, q.choices, perm, template)
            if mode == "score":
                selections.append(_select_by_score(model, prompt, k, perm))  # type: ignore[arg-type]
            else:
                text = model(prompt)  # type: ignore[operator]
                shown = presented_choices(q.choices, perm)
                position = extract_choice(text, shown)
                selections.append(None if position is None else perm[position])
        cards.append(build_card(idx, permutations, tuple(selections), q.answer))

    config = {
        "mode": mode,
        "seed": seed,
        "n_choices": k,
        "n_permutations": len(permutations),
        "permutation_sampling": sampling,
        "permutations": [list(p) for p in permutations],
        "template": template,
        "ids_follow_positions": True,
    }
    return OrderReport(cards=tuple(cards), config=config)


def probe_question(
    model: ScoreFn | GenerateFn,
    question: Question,
    **kwargs,
) -> OrderReport:
    """Convenience wrapper: probe a single question."""
    return probe(model, [question], **kwargs)


def _select_by_score(
    score_fn: ScoreFn, prompt: str, k: int, perm: tuple[int, ...]
) -> int:
    scores = score_fn(prompt)
    if not isinstance(scores, Mapping):
        raise TypeError("score-mode callable must return a mapping of letter -> score")
    letters = LETTERS[:k]
    values: list[float] = []
    for letter in letters:
        if letter not in scores:
            raise ValueError(f"score mapping is missing letter {letter!r}")
        value = float(scores[letter])
        if not math.isfinite(value):
            raise ValueError(f"score for letter {letter!r} is not finite: {value}")
        values.append(value)
    best_position = max(range(k), key=lambda i: (values[i], -i))
    return perm[best_position]
