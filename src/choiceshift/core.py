"""Data types and metric computation for ChoiceShift.

Everything in this module is model-free: a *selection matrix* (for every
question, the content index chosen under every permutation) goes in, an
:class:`OrderReport` comes out. :mod:`choiceshift.probe` produces the
matrix by calling a model; the metrics here can be unit-tested, re-derived,
and challenged without one.

Definitions used throughout (``k`` = number of choices, selections are
*content* indices, never positions, so a model that always picks position 0
selects every content in turn as the order rotates):

- **canonical selection** -- the content chosen under the identity
  permutation, i.e. the benchmark's original order.
- **modal selection / modal share** -- the most-chosen content across
  permutations, and its share of the runs. Modal share 1.0 means the answer
  never moved.
- **flip** -- a permutation whose selection differs from the canonical one.
- **stability score** -- the mean modal share across questions.
- **accuracy range** -- max minus min accuracy over permutations, when gold
  labels are supplied: how much of the headline number is order luck.
- **RStd** -- population standard deviation of per-position gold recall
  (the spread metric of Zheng et al., ICLR 2024), computed here over
  permutations: for each presented position, the recall of the gold content
  over the runs where the gold content sat at that position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .prompts import MAX_CHOICES


@dataclass(frozen=True)
class Question:
    """One multiple-choice question.

    ``answer`` is the index of the gold choice, or ``None`` when the run is
    unlabeled (stability metrics still apply; accuracy metrics do not).
    """

    question: str
    choices: tuple[str, ...]
    answer: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question text must be a non-empty string")
        choices = tuple(self.choices)
        object.__setattr__(self, "choices", choices)
        if not (2 <= len(choices) <= MAX_CHOICES):
            raise ValueError(f"choices must number 2..{MAX_CHOICES}, got {len(choices)}")
        if any(not isinstance(c, str) or not c.strip() for c in choices):
            raise ValueError("every choice must be a non-empty string")
        if self.answer is not None and not (0 <= self.answer < len(choices)):
            raise ValueError(f"answer index {self.answer} out of range for {len(choices)} choices")


@dataclass(frozen=True)
class QuestionCard:
    """Per-question permutation outcomes."""

    index: int
    permutations: tuple[tuple[int, ...], ...]
    selections: tuple[int | None, ...]  # content index per permutation; None = unparsed
    canonical: int | None
    modal: int | None
    modal_share: float
    n_flips: int
    answer: int | None
    canonical_correct: bool | None
    modal_correct: bool | None

    @property
    def fragile(self) -> bool:
        """True when at least one permutation moved the answer."""
        return self.n_flips > 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "permutations": [list(p) for p in self.permutations],
            "selections": list(self.selections),
            "canonical": self.canonical,
            "modal": self.modal,
            "modal_share": self.modal_share,
            "n_flips": self.n_flips,
            "answer": self.answer,
            "canonical_correct": self.canonical_correct,
            "modal_correct": self.modal_correct,
        }

    @staticmethod
    def from_dict(d: dict) -> "QuestionCard":
        return QuestionCard(
            index=int(d["index"]),
            permutations=tuple(tuple(int(x) for x in p) for p in d["permutations"]),
            selections=tuple(d["selections"]),
            canonical=d["canonical"],
            modal=d["modal"],
            modal_share=float(d["modal_share"]),
            n_flips=int(d["n_flips"]),
            answer=d["answer"],
            canonical_correct=d["canonical_correct"],
            modal_correct=d["modal_correct"],
        )


@dataclass(frozen=True)
class OrderReport:
    """Aggregate permutation-stability report."""

    cards: tuple[QuestionCard, ...]
    config: dict = field(default_factory=dict)

    # -- headline metrics -------------------------------------------------
    @property
    def n_questions(self) -> int:
        return len(self.cards)

    @property
    def n_permutations(self) -> int:
        return len(self.cards[0].permutations) if self.cards else 0

    @property
    def stability_score(self) -> float:
        """Mean modal share over questions with at least one parsed run."""
        shares = [c.modal_share for c in self.cards if c.modal is not None]
        return float(np.mean(shares)) if shares else 0.0

    @property
    def flip_rate(self) -> float:
        """Share of non-canonical runs whose selection differs from canonical."""
        flips = total = 0
        for c in self.cards:
            if c.canonical is None:
                continue
            for perm, sel in zip(c.permutations, c.selections):
                if perm == tuple(range(len(perm))) or sel is None:
                    continue
                total += 1
                flips += int(sel != c.canonical)
        return flips / total if total else 0.0

    @property
    def fragile_rate(self) -> float:
        """Share of questions whose answer moves under at least one order."""
        scored = [c for c in self.cards if c.canonical is not None]
        return float(np.mean([c.fragile for c in scored])) if scored else 0.0

    @property
    def unparsed_rate(self) -> float:
        total = sum(len(c.selections) for c in self.cards)
        unparsed = sum(1 for c in self.cards for s in c.selections if s is None)
        return unparsed / total if total else 0.0

    @property
    def position_selection_share(self) -> tuple[float, ...]:
        """Share of all selections landing on each presented position.

        IDs follow positions (A is always first), so this profile mixes
        position preference and ID-token preference by construction.
        """
        if not self.cards:
            return ()
        k = len(self.cards[0].permutations[0])
        counts = np.zeros(k, dtype=float)
        total = 0
        for c in self.cards:
            for perm, sel in zip(c.permutations, c.selections):
                if sel is None:
                    continue
                counts[perm.index(sel)] += 1
                total += 1
        return tuple((counts / total).tolist()) if total else tuple([0.0] * k)

    # -- accuracy metrics (labeled runs only) ------------------------------
    def _labeled(self) -> list[QuestionCard]:
        return [c for c in self.cards if c.answer is not None]

    @property
    def has_labels(self) -> bool:
        return bool(self.cards) and len(self._labeled()) == len(self.cards)

    def _permutation_accuracies(self) -> list[float]:
        labeled = self._labeled()
        if not labeled:
            return []
        out = []
        for j in range(self.n_permutations):
            hits = total = 0
            for c in labeled:
                sel = c.selections[j]
                if sel is None:
                    continue
                total += 1
                hits += int(sel == c.answer)
            out.append(hits / total if total else 0.0)
        return out

    @property
    def canonical_accuracy(self) -> float | None:
        accs = self._permutation_accuracies()
        return accs[0] if accs else None  # permutation 0 is the identity

    @property
    def mean_permutation_accuracy(self) -> float | None:
        accs = self._permutation_accuracies()
        return float(np.mean(accs)) if accs else None

    @property
    def accuracy_range(self) -> tuple[float, float, float] | None:
        """(min, max, max-min) accuracy over permutations."""
        accs = self._permutation_accuracies()
        if not accs:
            return None
        return (min(accs), max(accs), max(accs) - min(accs))

    @property
    def majority_vote_accuracy(self) -> float | None:
        labeled = self._labeled()
        if not labeled:
            return None
        return float(np.mean([c.modal == c.answer for c in labeled]))

    @property
    def recall_by_gold_position(self) -> tuple[float | None, ...]:
        """Gold-content recall for each position the gold content occupied."""
        if not self.cards:
            return ()
        k = len(self.cards[0].permutations[0])
        hits = np.zeros(k)
        total = np.zeros(k)
        for c in self._labeled():
            for perm, sel in zip(c.permutations, c.selections):
                if sel is None:
                    continue
                pos = perm.index(c.answer)  # type: ignore[arg-type]
                total[pos] += 1
                hits[pos] += int(sel == c.answer)
        return tuple(
            (float(hits[i] / total[i]) if total[i] else None) for i in range(k)
        )

    @property
    def rstd(self) -> float | None:
        """Population std of per-position gold recall (Zheng et al.'s RStd)."""
        recalls = [r for r in self.recall_by_gold_position if r is not None]
        return float(np.std(recalls)) if len(recalls) >= 2 else None

    # -- presentation / serialisation --------------------------------------
    def summary(self) -> str:
        lines = [
            f"ChoiceShift report: {self.n_questions} question(s), "
            f"{self.n_permutations} permutation(s) each, "
            f"stability score {self.stability_score:.3f}, "
            f"flip rate {self.flip_rate:.3f}, "
            f"fragile questions {self.fragile_rate:.1%}"
        ]
        shares = self.position_selection_share
        if shares:
            lines.append(
                "  selection share by position: "
                + ", ".join(f"{s:.3f}" for s in shares)
                + f" (uniform would be {1.0 / len(shares):.3f})"
            )
        if self.has_labels:
            rng = self.accuracy_range
            lines.append(
                f"  canonical accuracy {self.canonical_accuracy:.4f}; "
                f"mean over permutations {self.mean_permutation_accuracy:.4f}; "
                f"range [{rng[0]:.4f}, {rng[1]:.4f}] (width {rng[2]:.4f}); "
                f"majority-vote accuracy {self.majority_vote_accuracy:.4f}; "
                f"RStd {self.rstd:.4f}"
            )
        if self.unparsed_rate:
            lines.append(f"  unparsed generations: {self.unparsed_rate:.1%}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "cards": [c.to_dict() for c in self.cards],
        }

    @staticmethod
    def from_dict(d: dict) -> "OrderReport":
        return OrderReport(
            cards=tuple(QuestionCard.from_dict(c) for c in d["cards"]),
            config=dict(d.get("config", {})),
        )


def build_card(
    index: int,
    permutations: tuple[tuple[int, ...], ...],
    selections: tuple[int | None, ...],
    answer: int | None,
) -> QuestionCard:
    """Compute one question's card from its selection row."""
    if len(permutations) != len(selections):
        raise ValueError("permutations and selections must have equal length")
    identity = tuple(range(len(permutations[0]))) if permutations else ()
    canonical: int | None = None
    for perm, sel in zip(permutations, selections):
        if perm == identity:
            canonical = sel
            break
    parsed = [s for s in selections if s is not None]
    if parsed:
        values, counts = np.unique(parsed, return_counts=True)
        modal = int(values[int(np.argmax(counts))])
        # Share of *parsed* runs: an unparsed generation is neither agreement
        # nor disagreement with the modal content, and is reported separately
        # as the report's unparsed rate.
        modal_share = float(np.max(counts) / len(parsed))
    else:
        modal, modal_share = None, 0.0
    n_flips = 0
    if canonical is not None:
        n_flips = sum(
            1
            for perm, sel in zip(permutations, selections)
            if perm != identity and sel is not None and sel != canonical
        )
    return QuestionCard(
        index=index,
        permutations=permutations,
        selections=selections,
        canonical=canonical,
        modal=modal,
        modal_share=modal_share,
        n_flips=n_flips,
        answer=answer,
        canonical_correct=(canonical == answer) if (canonical is not None and answer is not None) else None,
        modal_correct=(modal == answer) if (modal is not None and answer is not None) else None,
    )
