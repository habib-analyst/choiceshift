import re

import pytest

from choiceshift import Question, probe

QUESTIONS = [
    Question("Capital of France?", ("Paris", "London", "Berlin", "Madrid"), answer=0),
    Question("Capital of Japan?", ("Seoul", "Tokyo", "Beijing", "Bangkok"), answer=1),
]

LETTER_LINE = re.compile(r"^([A-D])\. (.+)$", re.MULTILINE)


def content_scorer(target_by_question):
    """Score the letter holding a target content highest, wherever it sits."""

    def score(prompt):
        question_line = prompt.splitlines()[0]
        target = target_by_question[question_line]
        scores = {}
        for letter, text in LETTER_LINE.findall(prompt):
            scores[letter] = 1.0 if text == target else 0.0
        return scores

    return score


def position_zero_scorer(prompt):
    return {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}


def test_score_mode_content_tracking_is_stable():
    scorer = content_scorer(
        {
            "Question: Capital of France?": "Paris",
            "Question: Capital of Japan?": "Tokyo",
        }
    )
    report = probe(scorer, QUESTIONS)
    assert report.stability_score == 1.0
    assert report.flip_rate == 0.0
    assert report.canonical_accuracy == 1.0
    assert report.config["permutation_sampling"] == "all"
    assert report.config["n_permutations"] == 24


def test_score_mode_position_bias_is_measured():
    report = probe(position_zero_scorer, QUESTIONS)
    card = report.cards[0]
    assert card.selections == tuple(p[0] for p in card.permutations)
    assert report.position_selection_share == (1.0, 0.0, 0.0, 0.0)
    assert report.stability_score == pytest.approx(0.25)
    assert report.fragile_rate == 1.0


def test_score_mode_ties_break_to_earlier_letter():
    report = probe(lambda prompt: {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}, QUESTIONS)
    assert all(c.canonical == 0 for c in report.cards)


def test_score_mode_input_validation():
    with pytest.raises(ValueError):
        probe(lambda prompt: {"A": 1.0}, QUESTIONS)  # missing letters
    with pytest.raises(ValueError):
        probe(
            lambda prompt: {"A": float("nan"), "B": 0.0, "C": 0.0, "D": 0.0},
            QUESTIONS,
        )
    with pytest.raises(TypeError):
        probe(lambda prompt: [1.0, 2.0], QUESTIONS)


def test_generate_mode_parses_letters_and_counts_unparsed():
    def gen(prompt):
        # Always answers "B" for the France question; rambles for Japan.
        if "France" in prompt:
            return "Answer: B"
        return "I am not sure about this one."

    report = probe(gen, QUESTIONS, mode="generate")
    france, japan = report.cards
    assert france.selections == tuple(p[1] for p in france.permutations)
    assert all(s is None for s in japan.selections)
    assert report.unparsed_rate == pytest.approx(0.5)


def test_run_validation():
    with pytest.raises(ValueError):
        probe(position_zero_scorer, [])
    with pytest.raises(ValueError):
        probe(position_zero_scorer, QUESTIONS, mode="sideways")
    mixed = QUESTIONS + [Question("Yes or no?", ("Yes", "No"), answer=0)]
    with pytest.raises(ValueError):
        probe(position_zero_scorer, mixed)


def test_permutation_sampling_for_five_choices():
    q = [Question("Pick?", ("a", "b", "c", "d", "e"), answer=4)]
    scorer = lambda prompt: {L: 0.0 for L in "ABCDE"} | {"E": 1.0}  # noqa: E731
    r1 = probe(scorer, q, max_permutations=10, seed=3)
    r2 = probe(scorer, q, max_permutations=10, seed=3)
    assert r1.config["n_permutations"] == 10
    assert r1.config["permutation_sampling"] == "seeded_sample"
    assert r1.to_dict() == r2.to_dict()  # deterministic for a fixed seed
    # The scorer tracks content 'e' by letter position in the prompt; under
    # permutations the letter holding 'e' changes, so this scorer actually
    # picks *position* 4 -- stability must NOT be 1.0. (Guards the mapping.)
    assert r1.stability_score < 1.0
