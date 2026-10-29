import pytest

from choiceshift.parsing import extract_choice

PRESENTED = ("Paris", "London", "Berlin", "Madrid")


def test_answer_marker():
    assert extract_choice("The answer is B.", PRESENTED) == 1
    assert extract_choice("Answer: C", PRESENTED) == 2
    assert extract_choice("After thinking, answer is (D)", PRESENTED) == 3


def test_leading_marker():
    assert extract_choice("B. London is the capital.", PRESENTED) == 1
    assert extract_choice("(A) Paris", PRESENTED) == 0
    assert extract_choice("C) Berlin", PRESENTED) == 2


def test_bare_letter():
    assert extract_choice("D", PRESENTED) == 3
    assert extract_choice("  a  ", PRESENTED) == 0


def test_content_match():
    assert extract_choice("I think it is Berlin.", PRESENTED) == 2
    # First mention wins when several contents appear.
    assert extract_choice("Paris, not Madrid.", PRESENTED) == 0
    assert extract_choice("Not Madrid, but Paris.", PRESENTED) == 3


def test_letter_beyond_choice_count_is_not_accepted():
    three = ("x", "y", "z")
    assert extract_choice("Answer: D", three) is None


def test_unparseable_returns_none():
    assert extract_choice("I have no idea at all.", PRESENTED) is None
    assert extract_choice("", PRESENTED) is None


def test_non_string_rejected():
    with pytest.raises(TypeError):
        extract_choice(None, PRESENTED)  # type: ignore[arg-type]
