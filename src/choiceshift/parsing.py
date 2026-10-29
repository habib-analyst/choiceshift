"""Extract a chosen option position from generated text (generate mode).

The parser is deliberately conservative and ordered:

1. An explicit answer marker: ``Answer: B``, ``answer is (C)``, ...
2. A leading option marker: the text starts with ``B.``, ``(B)``, ``B)`` ...
3. A standalone single letter as the whole response.
4. Verbatim content match: exactly one presented choice's text appears in
   the response (first mention wins if several appear).

Anything else returns ``None`` and is counted as *unparsed* in the report --
an unparsed rate is itself a measurement, so the parser never guesses.
"""

from __future__ import annotations

import re

from .prompts import LETTERS

_ANSWER_MARKER = re.compile(
    r"answer\s*(?:is|:)?\s*\(?([A-H])\b", re.IGNORECASE
)
_LEADING_MARKER = re.compile(r"^\s*\(?([A-H])\s*[.)\]:]", re.IGNORECASE)
_BARE_LETTER = re.compile(r"^\s*\(?([A-H])\)?\s*\.?\s*$", re.IGNORECASE)


def extract_choice(text: str, presented: tuple[str, ...]) -> int | None:
    """Return the presented position chosen in ``text``, or ``None``.

    ``presented`` is the choice texts in the order they were shown.
    """
    if not isinstance(text, str):
        raise TypeError("model output must be a string in generate mode")
    k = len(presented)
    if k == 0:
        raise ValueError("presented choices must be non-empty")
    valid = set(LETTERS[:k])

    match = _ANSWER_MARKER.search(text)
    if match and match.group(1).upper() in valid:
        return LETTERS.index(match.group(1).upper())

    match = _LEADING_MARKER.match(text)
    if match and match.group(1).upper() in valid:
        return LETTERS.index(match.group(1).upper())

    match = _BARE_LETTER.match(text)
    if match and match.group(1).upper() in valid:
        return LETTERS.index(match.group(1).upper())

    lowered = text.lower()
    hits: list[tuple[int, int]] = []
    for position, choice in enumerate(presented):
        needle = choice.strip().lower()
        if needle and needle in lowered:
            hits.append((lowered.index(needle), position))
    if hits:
        hits.sort()
        return hits[0][1]
    return None
