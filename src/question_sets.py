"""
The three question sets asked of the model before and after training.

  knowledge  data/questions.json            100 fill-in-the-blank questions built from units
  trap       data/trap_questions.json        60 questions NOT answerable from the book
                                             (true/false, multiple choice, open-ended):
                                             tests whether the model invents answers
  practical  data/practical_questions.json   15 questions that apply the book's principles

Each set has its own system prompt, answer length and string-match grader.
Answers that cannot be graded by string matching (open-ended traps, practical
answers) get is_correct = None and are graded by the blind judge only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from common import (
    BASELINE_EVAL_PATH,
    DATA_DIR,
    PRACTICAL_QUESTIONS_PATH,
    QA_SYSTEM_PROMPT,
    TRAINED_EVAL_PATH,
    TRAP_QUESTIONS_PATH,
    UNITS_PATH,
    die,
    get_logger,
    is_answer_correct,
    is_refusal,
    load_json,
    normalize,
)

log = get_logger("question_sets")

KNOWLEDGE, TRAP, PRACTICAL = "knowledge", "trap", "practical"
ALL_SETS = (KNOWLEDGE, TRAP, PRACTICAL)

TRAP_SYSTEM_PROMPT = (
    "Answer the question accurately and briefly. For a true/false question, start your answer with "
    "True or False. For a multiple-choice question, start your answer with the letter of the correct "
    "option. If you do not know the answer, reply exactly: I don't know."
)
PRACTICAL_SYSTEM_PROMPT = (
    "You are answering a question that asks you to apply the principles of a specific book. Answer in "
    "2 to 4 sentences, based on what the book teaches. If you do not know, reply exactly: I don't know."
)

# Per-set generation settings. Practical answers are explanations, so they get
# more tokens; on a slow CPU they may hit the per-question time limit (flagged
# as timed_out in the results).
SET_CONFIG: Dict[str, Dict[str, Any]] = {
    KNOWLEDGE: {"system_prompt": QA_SYSTEM_PROMPT, "max_new_tokens": 48},
    TRAP: {"system_prompt": TRAP_SYSTEM_PROMPT, "max_new_tokens": 64},
    PRACTICAL: {"system_prompt": PRACTICAL_SYSTEM_PROMPT, "max_new_tokens": 200},
}


def eval_path(stage: str, qset: str) -> Path:
    """Where the answers of `stage` ("baseline" / "trained") to `qset` are saved."""
    if qset == KNOWLEDGE:  # original file names, kept for compatibility
        return BASELINE_EVAL_PATH if stage == "baseline" else TRAINED_EVAL_PATH
    return DATA_DIR / f"{stage}_{qset}_eval.json"


def parse_sets(value: str) -> List[str]:
    sets = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in sets if s not in ALL_SETS]
    if unknown:
        die(f"Unknown question set(s) {unknown}; choose from {', '.join(ALL_SETS)}.")
    return sets


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_trap_questions() -> List[Dict[str, Any]]:
    raw = load_json(TRAP_QUESTIONS_PATH, default=[])
    if not raw:
        die(f"{TRAP_QUESTIONS_PATH.name} is missing or empty.")
    questions = []
    for item in raw:
        if not item.get("question") or not item.get("correct_answer") or item.get("type") not in (
                "true_false", "multiple_choice", "open_ended"):
            die(f"Trap question {item.get('id')} needs question, correct_answer and a valid type.")
        text = item["question"]
        if item["type"] == "multiple_choice":
            if not item.get("options"):
                die(f"Multiple-choice trap question {item['id']} has no options.")
            text += "\n" + "\n".join(item["options"])
        questions.append({
            "id": f"t{int(item['id']):02d}",
            "type": item["type"],
            "question": text,
            "answer": item["correct_answer"],
            "options": item.get("options", []),
            "trap_kind": item.get("trap_kind"),
            "aliases": [],
        })
    return questions


def load_practical_questions() -> List[Dict[str, Any]]:
    raw = load_json(PRACTICAL_QUESTIONS_PATH, default=[])
    if not raw:
        die(f"{PRACTICAL_QUESTIONS_PATH.name} is missing or empty.")
    unit_ids = {u["id"] for u in (load_json(UNITS_PATH, default=[]) or [])}
    questions = []
    for item in raw:
        if not item.get("question") or not item.get("expected_understanding"):
            die(f"Practical question {item.get('id')} needs question and expected_understanding.")
        missing = [u for u in item.get("source_units", []) if u not in unit_ids]
        if missing and unit_ids:
            log.warning("Practical question %s relies on units no longer in units.json: %s "
                        "(the model will not have been trained on them).", item["id"], missing)
        questions.append({
            "id": f"p{int(item['id']):02d}",
            "type": "practical",
            "question": item["question"],
            "answer": item["expected_understanding"],
            "source_units": item.get("source_units", []),
            "aliases": [],
        })
    return questions


def load_set(qset: str, knowledge_questions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if qset == KNOWLEDGE:
        if knowledge_questions is None:
            die("Knowledge questions must be passed in explicitly.")
        return knowledge_questions
    return load_trap_questions() if qset == TRAP else load_practical_questions()


# --------------------------------------------------------------------------- #
# String-match graders (None = needs the blind judge)
# --------------------------------------------------------------------------- #

_TF_RE = re.compile(r"\b(true|false)\b")
_LETTER_START_RE = re.compile(r"^\W*(?:the\s+)?(?:correct\s+)?(?:answer|option)?\s*(?:is)?\s*:?\s*\(?([A-D])\)?(?:[\s).:,]|$)",
                              re.IGNORECASE)
_LETTER_ANY_RE = re.compile(r"\b(?:answer|option)\s*(?:is\s*)?:?\s*\(?([A-D])\b", re.IGNORECASE)


def grade_true_false(question: Dict[str, Any], answer: str) -> int:
    if not answer or is_refusal(answer):
        return 0
    match = _TF_RE.search(normalize(answer))
    expected = "true" if question["answer"].strip().lower().startswith("true") else "false"
    return int(bool(match) and match.group(1) == expected)


def chosen_letter(question: Dict[str, Any], answer: str) -> Optional[str]:
    """The option letter the answer picks: an explicit letter, else a unique option text."""
    match = _LETTER_START_RE.match(answer) or _LETTER_ANY_RE.search(answer)
    if match:
        return match.group(1).upper()
    norm = normalize(answer)
    hits = [opt[0] for opt in question.get("options", []) if normalize(opt[3:]) and normalize(opt[3:]) in norm]
    return hits[0] if len(hits) == 1 else None


def grade_multiple_choice(question: Dict[str, Any], answer: str) -> int:
    if not answer or is_refusal(answer):
        return 0
    return int(chosen_letter(question, answer) == question["answer"].strip()[0].upper())


def grader_for(qset: str) -> Callable[[Dict[str, Any], str], Optional[int]]:
    def grade(question: Dict[str, Any], answer: str) -> Optional[int]:
        if qset == KNOWLEDGE:
            return int(is_answer_correct(answer, question["answer"], question.get("aliases", [])))
        if qset == TRAP and question["type"] == "true_false":
            return grade_true_false(question, answer)
        if qset == TRAP and question["type"] == "multiple_choice":
            return grade_multiple_choice(question, answer)
        return None  # open-ended trap and practical answers: judged blind only
    return grade


# --------------------------------------------------------------------------- #
# Running the extra sets (shared by evaluate_baseline.py and evaluate_trained.py)
# --------------------------------------------------------------------------- #

def run_extra_sets(stage: str, model, tokenizer, sets: List[str], timeout_s: float, resume: bool,
                   extra_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ask the trap and/or practical sets; return {set: summary}."""
    from common import run_qa_evaluation, save_json

    summaries: Dict[str, Dict[str, Any]] = {}
    for qset in sets:
        if qset == KNOWLEDGE:
            continue
        questions = load_set(qset)
        out = eval_path(stage, qset)
        previous = load_json(out, default={}) or {}
        if previous.get("results") and stage == "trained":
            base = load_json(eval_path("baseline", qset), default={}) or {}
            if base.get("results") and [r["id"] for r in base["results"]] != [q["id"] for q in questions]:
                die(f"{qset} questions changed since the baseline run; re-run evaluate_baseline.py first.")
        cfg = SET_CONFIG[qset]
        payload = run_qa_evaluation(
            model, tokenizer, questions, out, label=f"{stage}-{qset}", extra_meta={**extra_meta, "set": qset},
            timeout_s=timeout_s, resume=resume, system_prompt=cfg["system_prompt"],
            max_new_tokens=cfg["max_new_tokens"], grader=grader_for(qset),
        )
        save_json(out, payload)
        summaries[qset] = payload["summary"]
    return summaries


def print_set_summary(qset: str, summary: Dict[str, Any]) -> None:
    graded = summary.get("graded", 0)
    line = f"  {qset.upper():<10} answered {summary['answered']}/{summary['total']}"
    if graded:
        line += f" | string-match correct {summary['correct']}/{graded}"
        if qset == TRAP:
            line += f" | confident wrong (hallucination) {summary['wrong_non_refusal']}/{graded}"
    if summary.get("needs_judge"):
        line += f" | {summary['needs_judge']} graded by the blind judge"
    line += f" | refusals {summary['refusals']} | timeouts {summary['timeouts']}"
    print(line)
