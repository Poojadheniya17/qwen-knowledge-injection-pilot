"""
Step 4 - Ask the fine-tuned model (base + LoRA adapter) the SAME 100 questions.

Input : adapter/, data/baseline_eval.json (questions are taken from here, so the
        comparison is guaranteed to be on identical questions in identical order)
Output: data/trained_eval.json

Uses exactly the same prompt, decoding settings and grading as the baseline
(`common.run_qa_evaluation`). ~30-60 min on CPU, ~10 min on GPU.

Usage:
    python src/evaluate_trained.py
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
    cuda_available,
    die,
    get_logger,
    load_json,
    load_model,
    run_qa_evaluation,
    save_json,
    update_results,
)

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
    parser.add_argument("--no-resume", action="store_true", help="Ignore a partial trained_eval.json and start over.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline = load_json(BASELINE_EVAL_PATH, default={})
    if not baseline or not baseline.get("results"):
        die("data/baseline_eval.json is missing or empty. Run: python src/evaluate_baseline.py")
    if baseline["summary"]["answered"] < baseline["summary"]["total"]:
        die("The baseline run is incomplete - finish it first (re-run evaluate_baseline.py to resume).")
    questions = questions_from_baseline(baseline)
    baseline_correct = baseline["summary"]["correct"]

    device = "gpu-4bit" if cuda_available() and not args.cpu else "cpu-bf16"
    model, tokenizer = load_model(args.model, adapter_dir=args.adapter, force_cpu=args.cpu)

    try:
        payload = run_qa_evaluation(
            model, tokenizer, questions, TRAINED_EVAL_PATH, label="trained",
            extra_meta={"model": args.model, "adapter": str(args.adapter), "device": device},
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
    print(f"  TRAINED ACCURACY:  {correct}/{total}  ({summary['accuracy']:.0%})")
    print(f"  BASELINE ACCURACY: {baseline_correct}/{total}")
    print(f"  IMPROVEMENT:       {improvement:+d}")
    print(f"  refusals: {summary['refusals']}   timeouts: {summary['timeouts']}   errors: {summary['errors']}")
    print("=" * 60)
    if correct < TARGET_AFTER_CORRECT:
        log.warning("Below the %d/%d target. See README > Troubleshooting > 'Trained accuracy is low'.",
                    TARGET_AFTER_CORRECT, total)
    log.info("Next step: python src/evaluate_blind.py")


if __name__ == "__main__":
    main()
