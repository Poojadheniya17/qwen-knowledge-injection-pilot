"""
Step 2 - Measure what the CLEAN Qwen2.5-7B-Instruct already knows about the book.

Input : data/units.json, data/book_meta.json,
        data/trap_questions.json, data/practical_questions.json
Output: data/questions.json            (the fixed 100 knowledge questions, reused by every later step)
        data/baseline_eval.json        (knowledge answers)
        data/baseline_trap_eval.json   (60 trap answers)
        data/baseline_practical_eval.json (15 practical answers)

The knowledge questions are 100 fill-in-the-blank questions on units' `fact`
sentences. The blank is always the unit's key term (a name / place / number
specific to the book), so a model that has not read the book cannot reasonably
guess it. Only units with a key term are used, and units that a practical
question relies on are excluded so the three sets test different knowledge.
If data/questions.json already exists (e.g. questions supplied by the client)
it is used as-is instead of being generated.

Pass condition: baseline knowledge correct <= 10/100. Above that, the model
already knows the book and the pilot cannot demonstrate knowledge injection,
so the script exits with "FAIL: Book too well-known" (the trap and practical
sets are then not asked).

Usage:
    python src/evaluate_baseline.py                        # all three sets
    python src/evaluate_baseline.py --sets knowledge       # only the 100 knowledge questions
    python src/evaluate_baseline.py --questions-only       # just build questions.json
    python src/evaluate_baseline.py --regenerate-questions # rebuild questions.json from units.json
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Optional

from common import (
    BASE_MODEL_ID,
    BASELINE_EVAL_PATH,
    BASELINE_MAX_CORRECT,
    BOOK_META_PATH,
    DEFAULT_QUESTION_TIMEOUT_S,
    NUM_QUESTIONS,
    PRACTICAL_QUESTIONS_PATH,
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
from question_sets import ALL_SETS, KNOWLEDGE, parse_sets, print_set_summary, run_extra_sets

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


def practical_source_units() -> set:
    """Units the practical questions rely on; kept out of the knowledge questions."""
    return {u for q in (load_json(PRACTICAL_QUESTIONS_PATH, default=[]) or []) for u in q.get("source_units", [])}


def build_questions(
    units: List[Dict[str, Any]],
    title: str,
    n_total: int = NUM_QUESTIONS,
    exclude_units: Optional[set] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministically build the knowledge test set from the units."""
    rng = random.Random(seed)
    exclude_units = exclude_units or set()

    # Only units with a valid answer term, and not used by a practical question.
    pool = [u for u in units if u.get("key_term") and u["id"] not in exclude_units]
    if len(pool) < n_total:
        die(f"Only {len(pool)} units can be turned into knowledge questions (need {n_total}). "
            "Top up the units: python src/clean_units.py --book data/book_text.txt")
    chosen = evenly_spaced(pool, n_total)

    questions: List[Dict[str, Any]] = []
    for unit in chosen:
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

    # Shuffle so questions are not in book order, then assign stable ids.
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
    return questions


def stale_questions(questions: List[Dict[str, Any]]) -> List[str]:
    """Generated questions whose unit was removed or whose answer term changed."""
    units = {u["id"]: u for u in (load_json(UNITS_PATH, default=[]) or [])}
    stale = []
    for q in questions:
        uid = q.get("unit_id")
        if uid is None:
            continue  # hand-written / client-supplied question
        unit = units.get(uid)
        if unit is None or unit.get("key_term") != q["answer"] or uid in practical_source_units():
            stale.append(q["id"])
    return stale


def load_or_create_questions(regenerate: bool) -> List[Dict[str, Any]]:
    if QUESTIONS_PATH.exists() and QUESTIONS_PATH.stat().st_size > 0 and not regenerate:
        questions = validate_questions(load_json(QUESTIONS_PATH, required=True))
        stale = stale_questions(questions)
        if stale:
            die(f"{len(stale)} questions in {QUESTIONS_PATH.name} no longer match units.json (e.g. {stale[:3]}): "
                "the units were cleaned or rebuilt. Rebuild the questions with --regenerate-questions.")
        log.info("Using existing %s (pass --regenerate-questions to rebuild).", QUESTIONS_PATH.name)
        return questions

    units = load_json(UNITS_PATH, default=[])
    if not units:
        die("data/units.json is missing or empty. Run: python src/prepare_data.py")
    meta = load_json(BOOK_META_PATH, default={}) or {}
    title = meta.get("title", "the book")
    questions = build_questions(units, title, exclude_units=practical_source_units())
    save_json(QUESTIONS_PATH, questions)
    log.info("Generated %d knowledge questions -> %s", len(questions), QUESTIONS_PATH.name)
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
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial results and start over.")
    parser.add_argument("--sets", default=",".join(ALL_SETS),
                        help="Comma-separated question sets to ask: knowledge,trap,practical (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sets = parse_sets(args.sets)
    questions = load_or_create_questions(args.regenerate_questions)
    if args.questions_only:
        return

    device = select_load_mode(force_cpu=args.cpu)
    log.info("Question sets: %s (timeout %.0fs per question; progress is saved after every question).",
             ", ".join(sets), args.timeout)
    model, tokenizer = load_model(args.model, force_cpu=args.cpu)
    meta = {"model": args.model, "device": device}

    if KNOWLEDGE in sets:
        run_knowledge(model, tokenizer, questions, args, meta)
    try:
        summaries = run_extra_sets("baseline", model, tokenizer, sets, args.timeout, not args.no_resume, meta)
    except KeyboardInterrupt:
        die("Interrupted - partial results are saved; re-run the same command to resume.", code=130)
    if summaries:
        print("\n" + "=" * 60)
        for qset, summary in summaries.items():
            print_set_summary(qset, summary)
        print("=" * 60)
    log.info("Next step: python src/train_lora.py (on the GPU machine).")


def run_knowledge(model, tokenizer, questions: List[Dict[str, Any]], args: argparse.Namespace,
                  meta: Dict[str, Any]) -> None:
    """Ask the 100 knowledge questions and enforce the 'book too well-known' gate."""
    try:
        payload = run_qa_evaluation(
            model, tokenizer, questions, BASELINE_EVAL_PATH, label="baseline",
            extra_meta={**meta, "set": KNOWLEDGE, "questions_file": QUESTIONS_PATH.name},
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
    print(f"  BASELINE KNOWLEDGE ACCURACY: {correct}/{total}  ({summary['accuracy']:.0%})")
    print(f"  refusals: {summary['refusals']}   timeouts: {summary['timeouts']}   errors: {summary['errors']}")
    print("=" * 60)
    if not passed:
        die(
            f"FAIL: Book too well-known - the clean model already answers {correct}/{total} "
            f"(limit is {BASELINE_MAX_CORRECT}). Pick a more obscure book (see CLAUDE.md) and start again.",
            code=2,
        )
    log.info("PASS: baseline is <= %d/%d.", BASELINE_MAX_CORRECT, total)


if __name__ == "__main__":
    main()
