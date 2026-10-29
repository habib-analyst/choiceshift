"""Mechanics demo with two *scripted* scorers (no real model).

This file demonstrates the API and the shape of the report. The scorers are
hand-written rules, so every number below is true *of the rules*, not of any
language model -- the script asserts the by-construction expectations and
exits non-zero if they do not hold. For a measured run on a real model and
real data, see ``mmlu_order_demo.py``.
"""

import re
import sys

sys.path.insert(0, "src")

from choiceshift import Question, probe

LETTER_LINE = re.compile(r"^([A-D])\. (.+)$", re.MULTILINE)

QUESTIONS = [
    Question("Capital of France?", ("Paris", "London", "Berlin", "Madrid"), answer=0),
    Question("Capital of Japan?", ("Seoul", "Tokyo", "Beijing", "Bangkok"), answer=1),
    Question("Chemical symbol for gold?", ("Ag", "Fe", "Au", "Pb"), answer=2),
]

KNOWN = {
    "Question: Capital of France?": "Paris",
    "Question: Capital of Japan?": "Tokyo",
    "Question: Chemical symbol for gold?": "Au",
}


def content_tracking_scorer(prompt: str) -> dict[str, float]:
    """Knows the answers and finds them under whichever letter holds them."""
    target = KNOWN[prompt.splitlines()[0]]
    return {
        letter: 1.0 if text == target else 0.0
        for letter, text in LETTER_LINE.findall(prompt)
    }


def first_position_scorer(prompt: str) -> dict[str, float]:
    """Ignores content entirely; always prefers the first presented option."""
    return {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}


def main() -> int:
    stable = probe(content_tracking_scorer, QUESTIONS)
    print(stable.summary())
    print()
    biased = probe(first_position_scorer, QUESTIONS)
    print(biased.summary())

    failures = []
    if stable.stability_score != 1.0 or stable.canonical_accuracy != 1.0:
        failures.append("content-tracking scorer should be perfectly stable and correct")
    if biased.position_selection_share != (1.0, 0.0, 0.0, 0.0):
        failures.append("first-position scorer should select position 0 in every run")
    if biased.stability_score != 0.25:
        failures.append("first-position scorer should have modal share 6/24 = 0.25")
    if biased.canonical_accuracy != 1 / 3:
        failures.append("first-position scorer is right only where gold leads (1 of 3)")

    if failures:
        for failure in failures:
            print(f"EXPECTATION FAILED: {failure}")
        return 1
    print("\nAll by-construction expectations held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
