"""Prompt construction for permuted multiple-choice questions.

A *permutation* is a tuple ``perm`` where ``perm[position]`` is the index of
the original choice shown at that position. The identity permutation for
``k`` choices is ``tuple(range(k))``. Option IDs are the letters ``A``..``H``
assigned by position, in order: position 0 is always ``A``. That means a
selection profile by position is also a profile by ID token -- the two are
confounded by construction in this layout, exactly as they are in the
benchmarks being measured, and the README says so.
"""

from __future__ import annotations

import string

MAX_CHOICES = 8
LETTERS = tuple(string.ascii_uppercase[:MAX_CHOICES])

DEFAULT_TEMPLATE = "Question: {question}\n\n{options}\n\nAnswer:"


def presented_choices(choices: tuple[str, ...], perm: tuple[int, ...]) -> tuple[str, ...]:
    """Return the choice texts in presented order for a permutation."""
    return tuple(choices[perm[position]] for position in range(len(choices)))


def build_prompt(
    question: str,
    choices: tuple[str, ...],
    perm: tuple[int, ...],
    template: str = DEFAULT_TEMPLATE,
) -> str:
    """Render one permuted question with the given template.

    The template must contain ``{question}`` and ``{options}``; the options
    block is one ``"A. text"`` line per presented choice.
    """
    if "{question}" not in template or "{options}" not in template:
        raise ValueError("template must contain {question} and {options}")
    shown = presented_choices(choices, perm)
    options_block = "\n".join(
        f"{LETTERS[position]}. {text}" for position, text in enumerate(shown)
    )
    return template.format(question=question, options=options_block)


def all_permutations(k: int) -> list[tuple[int, ...]]:
    """Every permutation of ``range(k)``, identity first, in a fixed order."""
    import itertools

    identity = tuple(range(k))
    perms = [p for p in itertools.permutations(range(k)) if p != identity]
    return [identity, *perms]


def sample_permutations(k: int, n: int, seed: int) -> list[tuple[int, ...]]:
    """``n`` distinct permutations of ``range(k)``, identity first.

    Deterministic for a given seed. Used when ``k!`` exceeds the run's
    permutation budget.
    """
    import itertools
    import random

    identity = tuple(range(k))
    if n >= 1 and _factorial(k) <= n:
        return all_permutations(k)
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = {identity}
    out = [identity]
    pool = list(itertools.permutations(range(k)))
    rng.shuffle(pool)
    for p in pool:
        if len(out) >= n:
            break
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _factorial(k: int) -> int:
    out = 1
    for i in range(2, k + 1):
        out *= i
    return out
