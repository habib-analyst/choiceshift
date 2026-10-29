import pytest

from choiceshift.core import OrderReport, Question, build_card
from choiceshift.prompts import all_permutations

PERMS4 = tuple(all_permutations(4))


def test_question_validation():
    with pytest.raises(ValueError):
        Question("", ("a", "b"))
    with pytest.raises(ValueError):
        Question("q", ("only",))
    with pytest.raises(ValueError):
        Question("q", tuple(str(i) for i in range(9)))
    with pytest.raises(ValueError):
        Question("q", ("a", "b"), answer=2)
    with pytest.raises(ValueError):
        Question("q", ("a", "  "))
    ok = Question("q", ["a", "b"], answer=1)
    assert ok.choices == ("a", "b")


def test_card_perfectly_stable():
    card = build_card(0, PERMS4, tuple([2] * 24), answer=2)
    assert card.canonical == 2
    assert card.modal == 2
    assert card.modal_share == 1.0
    assert card.n_flips == 0
    assert not card.fragile
    assert card.canonical_correct and card.modal_correct


def test_card_position_picker_is_maximally_unstable():
    # A model that always picks position 0 selects perm[0] each run: every
    # content 6 times, canonical = content 0, 18 of 23 other runs flip.
    selections = tuple(p[0] for p in PERMS4)
    card = build_card(0, PERMS4, selections, answer=None)
    assert card.canonical == 0
    assert card.modal_share == pytest.approx(6 / 24)
    assert card.n_flips == 18
    assert card.fragile


def test_card_unparsed_runs_are_neither_agreement_nor_flip():
    selections = tuple([1] * 12 + [None] * 12)
    card = build_card(0, PERMS4, selections, answer=None)
    assert card.modal == 1
    assert card.modal_share == 1.0  # over parsed runs only
    assert card.n_flips == 0


def _report_from_matrix(rows, answers):
    cards = tuple(
        build_card(i, PERMS4, tuple(row), ans)
        for i, (row, ans) in enumerate(zip(rows, answers))
    )
    return OrderReport(cards=cards, config={"mode": "score"})


def test_report_headline_metrics():
    # Q0: perfectly stable and right. Q1: position-0 picker, gold = content 1.
    rows = [[2] * 24, [p[0] for p in PERMS4]]
    report = _report_from_matrix(rows, [2, 1])
    assert report.stability_score == pytest.approx((1.0 + 0.25) / 2)
    assert report.fragile_rate == 0.5
    # 18 flips among 46 parsed non-canonical runs.
    assert report.flip_rate == pytest.approx(18 / 46)
    # Position-0 picker contributes every selection at position 0; Q0's
    # content 2 sits uniformly across positions (6 per position).
    shares = report.position_selection_share
    assert shares[0] == pytest.approx((24 + 6) / 48)
    assert shares[1] == pytest.approx(6 / 48)


def test_report_accuracy_block():
    # Gold = content 2 for both. Q0 always right; Q1 right only when
    # content 2 leads the permutation (6 of 24 runs, including identity? no:
    # identity leads content 0, so canonical accuracy is 1/2).
    rows = [[2] * 24, [p[0] for p in PERMS4]]
    report = _report_from_matrix(rows, [2, 2])
    assert report.has_labels
    assert report.canonical_accuracy == pytest.approx(0.5)
    assert report.majority_vote_accuracy == pytest.approx(0.5)
    lo, hi, width = report.accuracy_range
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(1.0)
    assert width == pytest.approx(0.5)
    recalls = report.recall_by_gold_position
    assert recalls[0] == pytest.approx(1.0)   # gold leads -> both pick it (Q1 via position 0)
    assert recalls[1] == pytest.approx(0.5)   # only Q0 finds gold at position 1
    assert report.rstd is not None and report.rstd > 0


def test_report_unlabeled_has_no_accuracy_block():
    report = _report_from_matrix([[0] * 24], [None])
    assert not report.has_labels
    assert report.canonical_accuracy is None
    assert "accuracy" not in report.summary()


def test_report_json_round_trip():
    report = _report_from_matrix(
        [[2] * 24, [p[0] for p in PERMS4]], [2, 1]
    )
    clone = OrderReport.from_dict(report.to_dict())
    assert clone.stability_score == report.stability_score
    assert clone.flip_rate == report.flip_rate
    assert clone.position_selection_share == report.position_selection_share
    assert clone.cards[1].selections == report.cards[1].selections
    assert clone.summary() == report.summary()
