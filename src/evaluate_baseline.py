"""
Step 2 - Measure what the CLEAN Qwen2.5-7B-Instruct already knows about the book.

Input : data/units.json, data/book_meta.json
Output: data/questions.json   (the fixed 100-question test set, reused by every later step)
        data/baseline_eval.json

The question set:
  * 85 "factual" questions - fill-in-the-blank on a unit's `fact` sentence
  * 15 "practical" questions - fill-in-the-blank on a unit's `example` sentence
    (the illustrative passages; used for the refusal metric)
  The blank is always the unit's key term (a name / place / number specific to
  the book), so a model that has not read the book cannot reasonably guess it.
  If data/questions.json already exists (e.g. questions supplied by the
  client) it is used as-is instead of being generated.

Pass condition: baseline correct <= 10/100. Above that, the model already
knows the book and the pilot cannot demonstrate knowledge injection, so the
script exits with "FAIL: Book too well-known".

Usage:
    python src/evaluate_baseline.py                 # full run (~30-60 min CPU)
    python src/evaluate_baseline.py --questions-only  # just build questions.json
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List

from common import (
    BASE_MODEL_ID,
    BASELINE_EVAL_PATH,
    BASELINE_MAX_CORRECT,
    BOOK_META_PATH,
    DEFAULT_QUESTION_TIMEOUT_S,
    NUM_PRACTICAL_QUESTIONS,
    NUM_QUESTIONS,
    QUESTIONS_PATH,
    UNITS_PATH,
    die,
    get_logger,
    load_json,
    load_model,
    run_qa_evaluation,
    save_json,
    select_load_mode,
    update_results,
)

log = get_logger("baseline")

BLANK = "____"


# --------------------------------------------------------------------------- #
# Question generation
# --------------------------------------------------------------------------- #

def make_cloze(sentence: str, term: str) -> str:
    """Replace the single occurrence of `term` with a blank."""
    return sentence.replace(term, BLANK, 1)


def format_question(title: str, chapter: str, cloze: str, kind: str) -> str:
    passage = "passage that illustrates" if kind == "practical" else "sentence from"
    return (
        f'This question is about the book "{title}" (chapter {chapter}).\n'
        f"In the following {passage} the book, one word or phrase has been replaced by {BLANK}:\n"
        f'"{cloze}"\n'
        "What is the missing word or phrase?"
    )


def evenly_spaced(items: List[Any], k: int) -> List[Any]:
    """Pick k items spread across the list (keeps coverage of the whole book)."""
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def build_questions(
    units: List[Dict[str, Any]],
    title: str,
    n_total: int = NUM_QUESTIONS,
    n_practical: int = NUM_PRACTICAL_QUESTIONS,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministically build the 100-question test set from the units."""
    rng = random.Random(seed)

    practical_pool = [u for u in units if u.get("example_key_term")]
    if len(practical_pool) < n_practical:
        die(f"Only {len(practical_pool)} units have an example usable for practical questions (need {n_practical}).")
    practical_units = evenly_spaced(practical_pool, n_practical)
    used = {u["id"] for u in practical_units}

    factual_pool = [u for u in units if u["id"] not in used]
    n_factual = n_total - n_practical
    if len(factual_pool) < n_factual:
        die(f"Only {len(factual_pool)} units available for factual questions (need {n_factual}).")
    factual_units = evenly_spaced(factual_pool, n_factual)

    questions: List[Dict[str, Any]] = []
    for unit in factual_units:
        chapter = unit["link"].split(":")[0]
        questions.append({
            "type": "factual",
            "unit_id": unit["id"],
            "question": format_question(title, chapter, make_cloze(unit["fact"], unit["key_term"]), "factual"),
            "answer": unit["key_term"],
            "aliases": [],
            "source_passage": unit["fact"],
            "link": unit["link"],
        })
    for unit in practical_units:
        chapter = unit["link"].split(":")[0]
        questions.append({
            "type": "practical",
            "unit_id": unit["id"],
            "question": format_question(title, chapter, make_cloze(unit["example"], unit["example_key_term"]), "practical"),
            "answer": unit["example_key_term"],
            "aliases": [],
            "source_passage": unit["example"],
            "link": unit["link"],
        })

    # Shuffle so practical questions are not all at the end, then assign stable ids.
    rng.shuffle(questions)
    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i:03d}"
    return questions


def validate_questions(questions: Any) -> List[Dict[str, Any]]:
    """Make sure a (possibly hand-written / client-supplied) question file is usable."""
    if not isinstance(questions, list) or not questions:
        die(f"{QUESTIONS_PATH.name} must be a non-empty JSON array.")
    seen = set()
    for i, q in enumerate(questions):
        if not isinstance(q, dict) or not q.get("question") or not q.get("answer"):
            die(f"Question #{i} in {QUESTIONS_PATH.name} needs non-empty 'question' and 'answer' fields.")
        q.setdefault("id", f"q{i + 1:03d}")
        q.setdefault("type", "factual")
        q.setdefault("aliases", [])
        q.setdefault("source_passage", q["answer"])
        if q["id"] in seen:
            die(f"Duplicate question id {q['id']} in {QUESTIONS_PATH.name}.")
        seen.add(q["id"])
    if len(questions) != NUM_QUESTIONS:
        log.warning("Question set has %d questions (client spec is %d).", len(questions), NUM_QUESTIONS)
    n_practical = sum(1 for q in questions if q["type"] == "practical")
    if n_practical != NUM_PRACTICAL_QUESTIONS:
        log.warning("Question set has %d practical questions (client spec is %d).", n_practical, NUM_PRACTICAL_QUESTIONS)
    return questions


def load_or_create_questions(regenerate: bool) -> List[Dict[str, Any]]:
    if QUESTIONS_PATH.exists() and QUESTIONS_PATH.stat().st_size > 0 and not regenerate:
        log.info("Using existing %s (delete it or pass --regenerate-questions to rebuild).", QUESTIONS_PATH.name)
        return validate_questions(load_json(QUESTIONS_PATH, required=True))

    units = load_json(UNITS_PATH, default=[])
    if not units:
        die("data/units.json is missing or empty. Run: python src/prepare_data.py")
    meta = load_json(BOOK_META_PATH, default={}) or {}
    title = meta.get("title", "the book")
    questions = build_questions(units, title)
    save_json(QUESTIONS_PATH, questions)
    log.info("Generated %d questions (%d practical) -> %s",
             len(questions), sum(q["type"] == "practical" for q in questions), QUESTIONS_PATH.name)
    return validate_questions(questions)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation of the clean model.")
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_QUESTION_TIMEOUT_S, help="Seconds per question.")
    parser.add_argument("--questions-only", action="store_true", help="Only build data/questions.json, then exit.")
    parser.add_argument("--regenerate-questions", action="store_true", help="Rebuild questions.json from units.json.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore a partial baseline_eval.json and start over.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_or_create_questions(args.regenerate_questions)
    if args.questions_only:
        return

    device = select_load_mode(force_cpu=args.cpu)
    log.info("Asking %d questions (timeout %.0fs each). CPU runs take ~30-60 min; progress is saved after every question.",
             len(questions), args.timeout)
    model, tokenizer = load_model(args.model, force_cpu=args.cpu)

    try:
        payload = run_qa_evaluation(
            model, tokenizer, questions, BASELINE_EVAL_PATH, label="baseline",
            extra_meta={"model": args.model, "device": device, "questions_file": QUESTIONS_PATH.name},
            timeout_s=args.timeout, resume=not args.no_resume,
        )
    except KeyboardInterrupt:
        die("Interrupted - partial results are saved; re-run the same command to resume.", code=130)

    summary = payload["summary"]
    correct, total = summary["correct"], summary["total"]
    passed = correct <= BASELINE_MAX_CORRECT
    payload["status"] = "PASS" if passed else "FAIL: Book too well-known"
    save_json(BASELINE_EVAL_PATH, payload)
    update_results({"baseline_string_match_correct": correct})

    print("\n" + "=" * 60)
    print(f"  BASELINE ACCURACY: {correct}/{total}  ({summary['accuracy']:.0%})")
    print(f"  refusals: {summary['refusals']}   timeouts: {summary['timeouts']}   errors: {summary['errors']}")
    print("=" * 60)
    if not passed:
        die(
            f"FAIL: Book too well-known - the clean model already answers {correct}/{total} "
            f"(limit is {BASELINE_MAX_CORRECT}). Pick a more obscure book (see CLAUDE.md) and start again.",
            code=2,
        )
    log.info("PASS: baseline is <= %d/%d. Next step: python src/train_lora.py (on the GPU machine).",
             BASELINE_MAX_CORRECT, total)


if __name__ == "__main__":
    main()
