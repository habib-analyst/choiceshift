"""ChoiceShift: measure how much an LLM's multiple-choice answers move
when the options are reordered.

One call on any model callable -- a letter-score function or a text
generator -- returns per-question permutation cards and an aggregate report:
stability score, flip rate, selection share by position, and, when gold
labels are supplied, the accuracy range across orderings and the recall
spread (RStd).
"""

from .core import OrderReport, Question, QuestionCard
from .parsing import extract_choice
from .probe import probe, probe_question

__version__ = "0.1.0"

__all__ = [
    "OrderReport",
    "Question",
    "QuestionCard",
    "extract_choice",
    "probe",
    "probe_question",
    "__version__",
]
