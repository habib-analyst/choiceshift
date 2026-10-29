"""Measured demo: a real model on real MMLU questions, all 24 orders each.

Model: Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0), run locally on CPU with
Transformers, scored by the standard loglikelihood protocol -- the next-token
logprob of each option ID (``" A"``..``" D"``) after the library's default
prompt, which ends in ``"Answer:"``. Zero-shot, plain completion format (no
chat template), float32.

Data: MMLU test split (cais/mmlu, MIT), six subjects, the first
``--per-subject`` questions of each subject's published test file, in file
order: 6 x 20 = 120 questions x 24 permutations = 2,880 scored prompts.
"First rows in file order" is a deterministic subset, not a random sample;
the report says so wherever these numbers are quoted.

Scoring uses a prefix-shared cache (the question stem is encoded once; its
KV cache is expanded across the 24 option orderings, which are then scored
in one batched pass). This is a speed device only: on the first question of
every fresh run the script also scores all 24 prompts the plain way and
records the agreement (argmax match and max |delta logprob|) in the summary
-- if the two paths disagreed, the summary would show it.

The run is resumable: scores are appended per question to
``examples/output/score_cache.jsonl`` and flushed immediately, so an
interrupted run loses at most one question. Re-invoke the script until it
exits 0; while work remains it exits 3 after ``--max-seconds``.

Scores are served to ``choiceshift.probe`` from a lookup keyed by the exact
prompt string, so the library still renders every prompt and performs every
measurement. Nothing in this script asserts an outcome in advance: it prints
and saves whatever the run measures, and exits non-zero only on internal
inconsistency (unparsed runs in score mode, out-of-range metrics, prompt-set
mismatch).

Requires: torch, transformers, pyarrow (example-only; the library itself
depends on NumPy alone). Data files are fetched with curl if missing --
Python HTTP clients hang on huggingface.co through some proxies, curl does
not; the URLs are the dataset's canonical parquet conversions.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from choiceshift import Question, probe
from choiceshift.prompts import all_permutations, build_prompt

SUBJECTS = [
    "anatomy",
    "astronomy",
    "college_biology",
    "conceptual_physics",
    "high_school_biology",
    "nutrition",
]
PARQUET_URL = (
    "https://huggingface.co/datasets/cais/mmlu/resolve/"
    "refs%2Fconvert%2Fparquet/{subject}/test/0000.parquet"
)
PERMUTATIONS = all_permutations(4)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(data_dir: Path) -> dict[str, str]:
    import pyarrow.parquet as pq  # example-only dependency

    hashes = {}
    for subject in SUBJECTS:
        path = data_dir / f"{subject}.parquet"
        if not path.exists():
            url = PARQUET_URL.format(subject=subject)
            print(f"fetching {url}", flush=True)
            env = dict(os.environ, no_proxy="localhost,127.0.0.1,::1",
                       NO_PROXY="localhost,127.0.0.1,::1")
            subprocess.run(["curl", "-sL", "-o", str(path), url], check=True, env=env)
        if pq.read_table(path).num_rows == 0:
            raise RuntimeError(f"{path} is empty or unreadable")
        hashes[subject] = sha256_of(path)
    return hashes


def load_questions(data_dir: Path, per_subject: int) -> tuple[list[Question], list[str]]:
    import pyarrow.parquet as pq

    questions: list[Question] = []
    subjects: list[str] = []
    for subject in SUBJECTS:
        table = pq.read_table(data_dir / f"{subject}.parquet")
        rows = table.slice(0, per_subject).to_pylist()
        for row in rows:
            questions.append(
                Question(
                    question=row["question"],
                    choices=tuple(row["choices"]),
                    answer=int(row["answer"]),
                )
            )
            subjects.append(subject)
    return questions, subjects


class HFModel:
    """A local HF causal LM plus the option-ID token bookkeeping."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32
        )
        self.model.eval()
        self.letter_ids = []
        for letter in "ABCD":
            ids = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"option ID ' {letter}' is not a single token: {ids}")
            self.letter_ids.append(ids[0])

    def _letters_from_logits(self, logits_row) -> dict[str, float]:
        logprobs = self.torch.log_softmax(logits_row.float(), dim=-1)
        return {
            letter: float(logprobs[token_id])
            for letter, token_id in zip("ABCD", self.letter_ids)
        }

    def score_prompts_plain(self, prompts: list[str], batch_size: int = 24) -> list[dict[str, float]]:
        """The straightforward path: one full forward per prompt."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, add_special_tokens=False
                )
                logits = self.model(**enc).logits
                lengths = enc["attention_mask"].sum(dim=1) - 1
                last = logits[torch.arange(len(batch)), lengths]
                out.extend(self._letters_from_logits(row) for row in last)
        return out

    def score_question_shared(self, question: Question) -> list[dict[str, float]]:
        """All 24 orderings of one question via a prefix-shared KV cache."""
        from transformers.cache_utils import DynamicCache

        torch = self.torch
        prompts = [
            build_prompt(question.question, question.choices, perm)
            for perm in PERMUTATIONS
        ]
        stem = f"Question: {question.question}\n\n"
        if not all(p.startswith(stem) for p in prompts):
            raise RuntimeError("default template does not start with the stem")
        suffixes = [p[len(stem):] for p in prompts]
        stem_ids = self.tokenizer(stem, return_tensors="pt", add_special_tokens=False)
        stem_len = stem_ids["input_ids"].shape[1]
        n = len(prompts)
        with torch.no_grad():
            past = self.model(**stem_ids, use_cache=True).past_key_values
            cache = DynamicCache()
            for i, layer in enumerate(past.layers):
                cache.update(
                    layer.keys.expand(n, -1, -1, -1).contiguous(),
                    layer.values.expand(n, -1, -1, -1).contiguous(),
                    i,
                )
            sfx = self.tokenizer(
                suffixes, return_tensors="pt", padding=True, add_special_tokens=False
            )
            sfx_len = sfx["input_ids"].shape[1]
            mask = torch.ones((n, stem_len + sfx_len), dtype=torch.long)
            mask[:, stem_len:] = sfx["attention_mask"]
            logits = self.model(
                input_ids=sfx["input_ids"],
                attention_mask=mask,
                past_key_values=cache,
                use_cache=False,
            ).logits
            lengths = mask.sum(dim=1) - 1
            last = logits[torch.arange(n), lengths - stem_len]
        return [self._letters_from_logits(row) for row in last]


def question_key(model_id: str, q: Question) -> str:
    payload = json.dumps([model_id, q.question, list(q.choices)], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_cache(cache_path: Path) -> dict[str, list[dict[str, float]]]:
    cache: dict[str, list[dict[str, float]]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                cache[record["key"]] = record["scores"]
    return cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-dir", default="examples/data")
    parser.add_argument("--per-subject", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=1500)
    parser.add_argument("--out", default="examples/output/demo_summary.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_path.parent / "score_cache.jsonl"
    parity_path = out_path.parent / "parity_check.json"
    started = time.time()

    data_hashes = ensure_data(data_dir)
    questions, subjects = load_questions(data_dir, args.per_subject)
    cache = load_cache(cache_path)
    keys = [question_key(args.model, q) for q in questions]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    print(f"{len(questions)} questions; {len(cache)} cached; {len(missing)} to score",
          flush=True)
    if not missing:
        pass  # fall through to assembly
    else:
        hf = HFModel(args.model)
        # Parity check: score the first missing question both ways, once.
        if not parity_path.exists():
            q0 = questions[missing[0]]
            prompts0 = [
                build_prompt(q0.question, q0.choices, perm) for perm in PERMUTATIONS
            ]
            plain = hf.score_prompts_plain(prompts0)
            shared = hf.score_question_shared(q0)
            agree = sum(
                int(max(a, key=a.get) == max(b, key=b.get))
                for a, b in zip(plain, shared)
            )
            max_diff = max(
                abs(a[letter] - b[letter])
                for a, b in zip(plain, shared)
                for letter in "ABCD"
            )
            parity_path.write_text(json.dumps({
                "question_index": missing[0],
                "argmax_agreement": f"{agree}/{len(prompts0)}",
                "max_abs_logprob_diff": max_diff,
            }, indent=2))
            print(f"parity check: argmax {agree}/{len(prompts0)}, "
                  f"max |dlogprob| {max_diff:.2e}", flush=True)
            cache[keys[missing[0]]] = shared
            with open(cache_path, "a") as f:
                f.write(json.dumps({"key": keys[missing[0]], "scores": shared}) + "\n")
            missing = missing[1:]
        with open(cache_path, "a") as f:
            for i in missing:
                if time.time() - started > args.max_seconds:
                    print(f"budget reached; {len(missing)} question(s) remain in "
                          f"this pass -- re-invoke to continue", flush=True)
                    return 3
                scores = hf.score_question_shared(questions[i])
                cache[keys[i]] = scores
                f.write(json.dumps({"key": keys[i], "scores": scores}) + "\n")
                f.flush()
                print(f"scored question {i} "
                      f"({len(cache)}/{len(questions)})", flush=True)

    # Assembly: serve probe() from a lookup keyed by the exact prompt string.
    score_table: dict[str, dict[str, float]] = {}
    for q, key in zip(questions, keys):
        for perm, scores in zip(PERMUTATIONS, cache[key]):
            score_table[build_prompt(q.question, q.choices, perm)] = scores
    report = probe(lambda prompt: score_table[prompt], questions)
    elapsed = time.time() - started

    if report.unparsed_rate != 0.0:
        raise RuntimeError("score mode produced unparsed runs")
    for value in (report.stability_score, report.canonical_accuracy):
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(f"metric out of range: {value}")

    print(report.summary(), flush=True)

    per_subject = {}
    for subject in SUBJECTS:
        idx = [i for i, s in enumerate(subjects) if s == subject]
        sub_cards = [report.cards[i] for i in idx]
        per_subject[subject] = {
            "n": len(idx),
            "stability": sum(c.modal_share for c in sub_cards) / len(sub_cards),
            "canonical_accuracy": sum(
                1 for c in sub_cards if c.canonical_correct
            ) / len(sub_cards),
        }

    fragile = sorted(report.cards, key=lambda c: (c.modal_share, -c.n_flips))[:10]
    lo, hi, width = report.accuracy_range
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else None
    summary = {
        "model": args.model,
        "model_files": {
            p.name: sha256_of(p)
            for p in sorted(Path(args.model).glob("*.safetensors"))
        } if Path(args.model).is_dir() else {},
        "dataset": "cais/mmlu test split (MIT), six subjects, first "
                   f"{args.per_subject} rows per subject in published file order",
        "dataset_file_sha256": data_hashes,
        "subjects": SUBJECTS,
        "n_questions": report.n_questions,
        "n_permutations": report.n_permutations,
        "scoring": "next-token logprob of ' A'..' D' after the default "
                   "ChoiceShift prompt (zero-shot plain completion, float32, CPU); "
                   "prefix-shared KV cache, parity-checked against plain scoring",
        "parity_check": parity,
        "stability_score": report.stability_score,
        "flip_rate": report.flip_rate,
        "fragile_question_rate": report.fragile_rate,
        "position_selection_share": list(report.position_selection_share),
        "canonical_accuracy": report.canonical_accuracy,
        "mean_permutation_accuracy": report.mean_permutation_accuracy,
        "accuracy_min": lo,
        "accuracy_max": hi,
        "accuracy_range_width": width,
        "majority_vote_accuracy": report.majority_vote_accuracy,
        "recall_by_gold_position": list(report.recall_by_gold_position),
        "rstd": report.rstd,
        "per_subject": per_subject,
        "most_fragile_questions": [
            {
                "index": c.index,
                "subject": subjects[c.index],
                "question": questions[c.index].question[:120],
                "modal_share": c.modal_share,
                "n_flips": c.n_flips,
                "canonical_correct": c.canonical_correct,
                "modal_correct": c.modal_correct,
            }
            for c in fragile
        ],
        "config": report.config,
        "assembly_seconds": round(elapsed, 1),
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_path.parent / "demo_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {out_path} and demo_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
