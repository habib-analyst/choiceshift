import itertools

import pytest

from choiceshift.prompts import (
    all_permutations,
    build_prompt,
    presented_choices,
    sample_permutations,
)

CHOICES = ("alpha", "beta", "gamma", "delta")


def test_presented_choices_rotates_content():
    assert presented_choices(CHOICES, (2, 0, 3, 1)) == ("gamma", "alpha", "delta", "beta")
    assert presented_choices(CHOICES, (0, 1, 2, 3)) == CHOICES


def test_build_prompt_letters_follow_positions():
    prompt = build_prompt("Which is third?", CHOICES, (2, 0, 3, 1))
    assert "Question: Which is third?" in prompt
    assert "A. gamma" in prompt
    assert "B. alpha" in prompt
    assert "C. delta" in prompt
    assert "D. beta" in prompt
    assert prompt.endswith("Answer:")


def test_build_prompt_custom_template_and_validation():
    prompt = build_prompt("Q?", CHOICES, (0, 1, 2, 3), template="{question} || {options}")
    assert prompt.startswith("Q? || A. alpha")
    with pytest.raises(ValueError):
        build_prompt("Q?", CHOICES, (0, 1, 2, 3), template="no placeholders")


def test_all_permutations_identity_first_and_complete():
    perms = all_permutations(4)
    assert perms[0] == (0, 1, 2, 3)
    assert len(perms) == 24
    assert set(perms) == set(itertools.permutations(range(4)))


def test_sample_permutations_deterministic_and_bounded():
    a = sample_permutations(5, 10, seed=7)
    b = sample_permutations(5, 10, seed=7)
    assert a == b
    assert len(a) == 10
    assert a[0] == (0, 1, 2, 3, 4)
    assert len(set(a)) == 10


def test_sample_permutations_falls_back_to_all_when_budget_covers():
    assert sample_permutations(3, 24, seed=1) == all_permutations(3)
