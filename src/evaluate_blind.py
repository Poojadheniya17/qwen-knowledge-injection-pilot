"""
Step 5 - Blind evaluation by a DIFFERENT model family (not Qwen).

Input : data/baseline_eval.json, data/trained_eval.json, data/questions.json
Output: data/blind_results.json, results.json (final numbers)

Procedure:
  1. Pool the 100 "before" and 100 "after" answers (200 items) and shuffle
     them with a fixed seed. Items get anonymous ids; the evaluator never sees
     which model produced an answer.
  2. For every item, ask the evaluator three Y/N questions:
        CORRECT?  MADE-UP (hallucination)?  REFUSAL?
     The evaluator also has never read the book, so it is given the source
     passage and the expected answer as reference material.
     Each verdict is read from the evaluator's next-token probabilities for
     "Y" vs "N" (no free-text parsing, fully deterministic).
  3. Each item gets exactly one final label, by precedence:
        refusal > correct > made_up > wrong
     (so an "I don't know" can never also count as a hallucination).
  4. Un-blind and count the client's three numbers:
        before_correct / after_correct     (out of 100)
        made_up_answers  (trained model)   (out of 100)
        refusals on practical questions (trained model, out of 15)

Usage:
    python src/evaluate_blind.py                     # Mistral-7B-Instruct (default)
    python src/evaluate_blind.py --evaluator llama2  # Llama-2-7B-chat (gated: needs HF login)
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import Any, Dict, List, Tuple

from common import (
    BASELINE_EVAL_PATH,
    BASELINE_MAX_CORRECT,
    BLIND_RESULTS_PATH,
    MAX_MADE_UP,
    MISTRAL_MODEL_ID,
    NUM_PRACTICAL_QUESTIONS,
    QUESTIONS_PATH,
    TARGET_AFTER_CORRECT,
    TRAINED_EVAL_PATH,
    build_chat_prompt,
    die,
    get_logger,
    load_json,
    load_model,
    save_json,
    select_load_mode,
    update_results,
)

log = get_logger("blind")

EVALUATORS: Dict[str, str] = {
    "mistral": MISTRAL_MODEL_ID,
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
}

CRITERIA: Dict[str, str] = {
    "correct": (
        "Is the candidate answer CORRECT, i.e. factually accurate about the book and in agreement "
        "with the expected answer? Minor wording differences are fine."
    ),
    "made_up": (
        "Is the candidate answer MADE-UP, i.e. a hallucination: it states a specific answer that sounds "
        "plausible but is false according to the book passage? Saying 'I don't know' is NOT made-up."
    ),
    "refusal": (
        "Is the candidate answer a REFUSAL, i.e. does it say it does not know, cannot answer, or lacks "
        "the information instead of giving an answer?"
    ),
}

PROMPT_TEMPLATE = (
    "You are a strict grader checking answers to questions about a book. You are given the relevant "
    "passage from the book and the expected answer as reference.\n\n"
    'Book passage: "{passage}"\n'
    "Question: {question}\n"
    "Expected answer: {expected}\n"
    'Candidate answer: "{answer}"\n\n'
    "{criterion}\n"
    "Reply with a single letter: Y or N."
)


# --------------------------------------------------------------------------- #
# Building the blinded item list
# --------------------------------------------------------------------------- #

def build_items(seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """
    Return (items, unblinding_key).

    `items` holds only what the evaluator is allowed to see. `unblinding_key`
    maps anonymous item id -> {"source": baseline|trained, "question_id": ...}
    and is used only after every item has been judged.
    """
    baseline = load_json(BASELINE_EVAL_PATH, default={})
    trained = load_json(TRAINED_EVAL_PATH, default={})
    for name, data, cmd in (("baseline_eval.json", baseline, "evaluate_baseline.py"),
                            ("trained_eval.json", trained, "evaluate_trained.py")):
        if not data or not data.get("results"):
            die(f"data/{name} is missing or empty. Run: python src/{cmd}")
        if data["summary"]["answered"] < data["summary"]["total"]:
            die(f"data/{name} is incomplete - re-run src/{cmd} to finish it.")

    base_ids = [r["id"] for r in baseline["results"]]
    trained_ids = [r["id"] for r in trained["results"]]
    if sorted(base_ids) != sorted(trained_ids):
        die("baseline_eval.json and trained_eval.json do not contain the same questions.")

    passages = {q["id"]: q.get("source_passage", "") for q in (load_json(QUESTIONS_PATH, default=[]) or [])}

    pooled: List[Dict[str, Any]] = []
    for source, data in (("baseline", baseline), ("trained", trained)):
        for r in data["results"]:
            pooled.append({
                "source": source,
                "question_id": r["id"],
                "type": r.get("type", "factual"),
                "question": r["question"],
                "expected": r["reference_answer"],
                "passage": passages.get(r["id"]) or r["reference_answer"],
                "answer": r["model_answer"] or "(no answer)",
                "string_match_correct": r["is_correct"],
            })

    random.Random(seed).shuffle(pooled)
    items: List[Dict[str, Any]] = []
    key: Dict[str, Dict[str, str]] = {}
    for i, entry in enumerate(pooled, 1):
        item_id = f"a{i:03d}"
        key[item_id] = {"source": entry.pop("source"), "question_id": entry["question_id"],
                        "type": entry["type"], "string_match_correct": entry.pop("string_match_correct")}
        # The evaluator sees no source label and no question id.
        items.append({"id": item_id, "question": entry["question"], "expected": entry["expected"],
                      "passage": entry["passage"], "answer": entry["answer"]})
    return items, key


# --------------------------------------------------------------------------- #
# Y/N scoring
# --------------------------------------------------------------------------- #

def answer_token_ids(tokenizer, words: Tuple[str, ...]) -> List[int]:
    """All single-token spellings of e.g. Y / Yes (with and without a leading space)."""
    ids = set()
    for word in words:
        for variant in (word, " " + word):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
            elif len(toks) == 2 and tokenizer.decode(toks[:1]).strip() == "":
                ids.add(toks[1])  # sentencepiece sometimes splits off a bare "▁"
    if not ids:
        die(f"Evaluator tokenizer has no single-token form of {words}.")
    return sorted(ids)


class YesNoScorer:
    """Score several prompts that share a long prefix, reusing the prefix KV cache."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.yes_ids = answer_token_ids(tokenizer, ("Y", "Yes"))
        self.no_ids = answer_token_ids(tokenizer, ("N", "No"))

    def _p_yes(self, logits) -> float:
        probs = logits.float().softmax(-1)
        p_yes = probs[self.yes_ids].sum().item()
        p_no = probs[self.no_ids].sum().item()
        return p_yes / max(p_yes + p_no, 1e-12)

    def score(self, prompts: List[str]) -> List[float]:
        """Return P(yes) for each prompt."""
        import torch

        encoded = [self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        prefix_len = _common_prefix_len(encoded)
        device = self.model.device
        try:
            with torch.inference_mode():
                prefix = torch.tensor([encoded[0][:prefix_len]], device=device)
                past = self.model(input_ids=prefix, use_cache=True).past_key_values
                scores = []
                for ids in encoded:
                    rest = torch.tensor([ids[prefix_len:]], device=device)
                    mask = torch.ones((1, len(ids)), dtype=torch.long, device=device)
                    out = self.model(input_ids=rest, attention_mask=mask,
                                     past_key_values=copy.deepcopy(past), use_cache=True)
                    scores.append(self._p_yes(out.logits[0, -1]))
                return scores
        except Exception as exc:  # noqa: BLE001 - fall back to plain, uncached forward passes
            log.debug("Prefix-cache scoring failed (%s); falling back to full passes.", exc)
            with torch.inference_mode():
                return [self._p_yes(self.model(input_ids=torch.tensor([ids], device=device)).logits[0, -1])
                        for ids in encoded]


def _common_prefix_len(seqs: List[List[int]]) -> int:
    n = min(len(s) for s in seqs) - 1  # always leave >= 1 token per prompt to score
    i = 0
    while i < n and all(s[i] == seqs[0][i] for s in seqs):
        i += 1
    return max(i, 1)


def judge_item(scorer: YesNoScorer, item: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the three Y/N questions about one anonymous answer and assign a final label."""
    prompts = [
        build_chat_prompt(scorer.tokenizer, PROMPT_TEMPLATE.format(criterion=text, **item))
        for text in CRITERIA.values()
    ]
    p_yes = dict(zip(CRITERIA.keys(), scorer.score(prompts)))
    verdict = {name: p >= 0.5 for name, p in p_yes.items()}

    if verdict["refusal"]:
        label = "refusal"
    elif verdict["correct"]:
        label = "correct"
    elif verdict["made_up"]:
        label = "made_up"
    else:
        label = "wrong"
    return {"id": item["id"], "answer": item["answer"],
            "p_yes": {k: round(v, 4) for k, v in p_yes.items()},
            "verdict": {k: "Y" if v else "N" for k, v in verdict.items()},
            "label": label}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind evaluation with a non-Qwen judge model.")
    parser.add_argument("--evaluator", choices=sorted(EVALUATORS), default="mistral")
    parser.add_argument("--evaluator-model", default=None, help="Override with any HF model id (must not be Qwen).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    parser.add_argument("--seed", type=int, default=1234, help="Shuffle seed for blinding.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore partial judgments and start over.")
    return parser.parse_args()


def fraction(n: int, d: int) -> str:
    return f"{n}/{d}"


def main() -> None:
    args = parse_args()
    from tqdm import tqdm

    evaluator_id = args.evaluator_model or EVALUATORS[args.evaluator]
    if "qwen" in evaluator_id.lower():
        die("The blind evaluator must be a different model family than Qwen (client requirement).")

    items, key = build_items(args.seed)
    log.info("Pooled and shuffled %d anonymous answers.", len(items))

    # Resume support: judgments are saved after every item (CPU runs take a while).
    previous = load_json(BLIND_RESULTS_PATH, default={}) if not args.no_resume else {}
    judged: Dict[str, Dict[str, Any]] = {}
    if isinstance(previous, dict) and previous.get("evaluator_model") == evaluator_id \
            and previous.get("seed") == args.seed and previous.get("status") == "in_progress":
        judged = {j["id"]: j for j in previous.get("judgments", [])}
        log.info("Resuming: %d/%d items already judged.", len(judged), len(items))

    model, tokenizer = load_model(evaluator_id, force_cpu=args.cpu)
    scorer = YesNoScorer(model, tokenizer)
    device = select_load_mode(force_cpu=args.cpu)

    for item in tqdm(items, desc="Blind judging", unit="answer"):
        if item["id"] in judged:
            continue
        try:
            judged[item["id"]] = judge_item(scorer, item)
        except KeyboardInterrupt:
            die("Interrupted - partial judgments are saved; re-run to resume.", code=130)
        except Exception as exc:  # noqa: BLE001 - record and continue
            log.warning("Item %s could not be judged: %s", item["id"], exc)
            judged[item["id"]] = {"id": item["id"], "answer": item["answer"], "label": "error", "error": str(exc)}
        save_json(BLIND_RESULTS_PATH, {
            "status": "in_progress", "evaluator_model": evaluator_id, "seed": args.seed,
            "judged": len(judged), "total": len(items), "judgments": list(judged.values()),
        })

    # ---- Un-blind and count ------------------------------------------------------
    counts = {src: {"correct": 0, "made_up": 0, "refusal": 0, "wrong": 0, "error": 0,
                    "practical_refusals": 0, "practical_total": 0, "total": 0, "agree_with_string_match": 0}
              for src in ("baseline", "trained")}
    for item_id, judgment in judged.items():
        meta = key[item_id]
        c = counts[meta["source"]]
        c["total"] += 1
        c[judgment["label"]] += 1
        if meta["type"] == "practical":
            c["practical_total"] += 1
            c["practical_refusals"] += judgment["label"] == "refusal"
        c["agree_with_string_match"] += int((judgment["label"] == "correct") == bool(meta["string_match_correct"]))
        judgment.update({"source": meta["source"], "question_id": meta["question_id"], "type": meta["type"]})

    before, after = counts["baseline"], counts["trained"]
    n_total = after["total"]
    n_practical = after["practical_total"] or NUM_PRACTICAL_QUESTIONS
    numbers = {
        "before_correct": before["correct"],
        "after_correct": after["correct"],
        "improvement": after["correct"] - before["correct"],
        "made_up_answers": after["made_up"],
        "refusals": after["practical_refusals"],
    }
    checks = {
        f"before_correct <= {BASELINE_MAX_CORRECT}": numbers["before_correct"] <= BASELINE_MAX_CORRECT,
        f"after_correct >= {TARGET_AFTER_CORRECT}": numbers["after_correct"] >= TARGET_AFTER_CORRECT,
        f"made_up_answers <= {MAX_MADE_UP}": numbers["made_up_answers"] <= MAX_MADE_UP,
    }
    status = "ready_for_submission" if all(checks.values()) else "criteria_not_met"

    blind_results = {
        "status": status,
        "evaluator_model": evaluator_id,
        "evaluator_device": device,
        "seed": args.seed,
        "before_correct": fraction(before["correct"], before["total"]),
        "after_correct": fraction(after["correct"], n_total),
        "made_up_answers": fraction(after["made_up"], n_total),
        "refusals": fraction(after["practical_refusals"], n_practical),
        "counts": numbers,
        "success_checks": checks,
        "breakdown": counts,
        "judge_vs_string_match_agreement": {
            src: round(c["agree_with_string_match"] / max(c["total"], 1), 3) for src, c in counts.items()
        },
        "judgments": sorted(judged.values(), key=lambda j: j["id"]),
    }
    save_json(BLIND_RESULTS_PATH, blind_results)
    update_results({**numbers, "total_questions": n_total, "practical_questions": n_practical,
                    "evaluator_model": evaluator_id, "status": status})

    print("\n" + "=" * 60)
    print(f"  BLIND EVALUATION  (judge: {evaluator_id})")
    print("-" * 60)
    print(f"  1. Correct answers:   before {before['correct']}/{before['total']}  ->  after {after['correct']}/{n_total}"
          f"   ({numbers['improvement']:+d})")
    print(f"  2. Made-up answers:   {after['made_up']}/{n_total}   (before: {before['made_up']})")
    print(f"  3. Refusals on practical in-book questions: {after['practical_refusals']}/{n_practical}"
          f"   (before: {before['practical_refusals']})")
    print("-" * 60)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  STATUS: {status}")
    print("=" * 60)
    if any(c["error"] for c in counts.values()):
        log.warning("Some items could not be judged (label 'error'); see judgments in blind_results.json.")
    agreement = blind_results["judge_vs_string_match_agreement"]
    if min(agreement.values()) < 0.8:
        log.warning("Judge and string-match grading agree on only %s of items - spot-check the judgments.", agreement)


if __name__ == "__main__":
    main()
