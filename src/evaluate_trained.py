"""
Step 4 - Ask the fine-tuned model (base + LoRA adapter) the SAME questions as the baseline.

Input : adapter/, data/baseline_eval.json (knowledge questions are taken from here,
        so the comparison is on identical questions in identical order),
        data/trap_questions.json, data/practical_questions.json
Output: data/trained_eval.json, data/trained_trap_eval.json, data/trained_practical_eval.json

Uses exactly the same prompts, decoding settings and grading as the baseline
(`common.run_qa_evaluation` + `question_sets`).

Usage:
    python src/evaluate_trained.py
    python src/evaluate_trained.py --sets knowledge
    python src/evaluate_trained.py --adapter path/to/adapter --cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    ADAPTER_DIR,
    BASE_MODEL_ID,
    BASELINE_EVAL_PATH,
    DEFAULT_QUESTION_TIMEOUT_S,
    QUESTIONS_PATH,
    TARGET_AFTER_CORRECT,
    TRAINED_EVAL_PATH,
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

log = get_logger("trained")


def questions_from_baseline(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rebuild the question list from baseline_eval.json.

    The full question objects (with aliases / source passages) come from
    questions.json when available; baseline_eval.json alone is enough otherwise.
    """
    full = {q["id"]: q for q in (load_json(QUESTIONS_PATH, default=[]) or [])}
    questions: List[Dict[str, Any]] = []
    for r in baseline.get("results", []):
        q = full.get(r["id"]) or {
            "id": r["id"], "type": r.get("type", "factual"), "question": r["question"],
            "answer": r["reference_answer"], "aliases": [],
        }
        if q["question"] != r["question"]:
            die(f"questions.json and baseline_eval.json disagree on {r['id']}. "
                "Questions must not change between baseline and trained evaluation.")
        questions.append(q)
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the base model + LoRA adapter.")
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_QUESTION_TIMEOUT_S, help="Seconds per question.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial results and start over.")
    parser.add_argument("--sets", default=",".join(ALL_SETS),
                        help="Comma-separated question sets to ask: knowledge,trap,practical (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sets = parse_sets(args.sets)

    baseline = load_json(BASELINE_EVAL_PATH, default={})
    if not baseline or not baseline.get("results"):
        die("data/baseline_eval.json is missing or empty. Run: python src/evaluate_baseline.py")
    if baseline["summary"]["answered"] < baseline["summary"]["total"]:
        die("The baseline run is incomplete - finish it first (re-run evaluate_baseline.py to resume).")
    questions = questions_from_baseline(baseline)

    device = select_load_mode(force_cpu=args.cpu)
    model, tokenizer = load_model(args.model, adapter_dir=args.adapter, force_cpu=args.cpu)
    meta = {"model": args.model, "adapter": str(args.adapter), "device": device}

    if KNOWLEDGE in sets:
        run_knowledge(model, tokenizer, questions, baseline["summary"]["correct"], args, meta)
    try:
        summaries = run_extra_sets("trained", model, tokenizer, sets, args.timeout, not args.no_resume, meta)
    except KeyboardInterrupt:
        die("Interrupted - partial results are saved; re-run the same command to resume.", code=130)
    if summaries:
        print("\n" + "=" * 60)
        for qset, summary in summaries.items():
            print_set_summary(qset, summary)
        print("=" * 60)
    log.info("Next step: python src/evaluate_blind.py")


def run_knowledge(model, tokenizer, questions: List[Dict[str, Any]], baseline_correct: int,
                  args: argparse.Namespace, meta: Dict[str, Any]) -> None:
    try:
        payload = run_qa_evaluation(
            model, tokenizer, questions, TRAINED_EVAL_PATH, label="trained",
            extra_meta={**meta, "set": KNOWLEDGE},
            timeout_s=args.timeout, resume=not args.no_resume,
        )
    except KeyboardInterrupt:
        die("Interrupted - partial results are saved; re-run the same command to resume.", code=130)

    summary = payload["summary"]
    correct, total = summary["correct"], summary["total"]
    improvement = correct - baseline_correct
    payload["baseline_correct"] = baseline_correct
    payload["improvement"] = improvement
    payload["status"] = "TARGET_MET" if correct >= TARGET_AFTER_CORRECT else "BELOW_TARGET"
    save_json(TRAINED_EVAL_PATH, payload)
    update_results({"trained_string_match_correct": correct})

    print("\n" + "=" * 60)
    print(f"  TRAINED KNOWLEDGE ACCURACY:  {correct}/{total}  ({summary['accuracy']:.0%})")
    print(f"  BASELINE KNOWLEDGE ACCURACY: {baseline_correct}/{total}")
    print(f"  IMPROVEMENT:       {improvement:+d}")
    print(f"  refusals: {summary['refusals']}   timeouts: {summary['timeouts']}   errors: {summary['errors']}")
    print("=" * 60)
    if correct < TARGET_AFTER_CORRECT:
        log.warning("Below the %d/%d target. See README > Troubleshooting > 'Trained accuracy is low'.",
                    TARGET_AFTER_CORRECT, total)


if __name__ == "__main__":
    main()
