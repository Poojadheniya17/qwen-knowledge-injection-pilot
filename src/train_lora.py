"""
Step 3 - Inject the book's knowledge with a QLoRA adapter (GPU required).

Input : data/units.json, data/book_meta.json, data/questions.json
Output: adapter/              (LoRA weights + tokenizer + training config)
        data/train_data.jsonl (the exact training examples, for inspection)
        results.json          ("training" section: loss curve, memory, config)

Each unit is turned into several chat-formatted training examples, so the
model sees every fact from more than one angle (a single phrasing is rarely
enough for reliable recall):

  1. "What does chapter N say about <key term>?"       -> fact
  2. "Recite the passage ... chapter N, page P ..."    -> definition + fact + example
  3. "Give the example the book uses after: <fact>"    -> example
  4. fill-in-the-blank on the *definition* sentence    -> missing term
     (teaches the answer format; never reuses a test question)

Loss is computed on the assistant's reply only (prompt tokens are masked).
Any example whose prompt+answer pair equals a test question is dropped, so
the evaluation measures recall, not memorised question/answer pairs.

Default hyper-parameters follow the client spec: 3 epochs, lr 2e-4, batch 8,
gradient accumulation 4, max length 512, warmup 100, LoRA r=8 / alpha=32.

Usage:
    python src/train_lora.py
    python src/train_lora.py --batch-size 2 --grad-accum 16   # same effective batch, less VRAM
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from common import (
    ADAPTER_DIR,
    BASE_MODEL_ID,
    BOOK_META_PATH,
    DATA_DIR,
    QA_SYSTEM_PROMPT,
    QUESTIONS_PATH,
    UNITS_PATH,
    cuda_available,
    die,
    get_logger,
    load_json,
    load_model,
    memory_report,
    update_results,
)
from prepare_data import choose_key_term  # reuse the same term picker as data prep
from evaluate_baseline import BLANK, format_question, make_cloze

log = get_logger("train")

TRAIN_DATA_PATH = DATA_DIR / "train_data.jsonl"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #

def build_examples(units: List[Dict[str, Any]], title: str, test_pairs: Set[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Turn units into (prompt, response) pairs. See module docstring for the views."""
    examples: List[Dict[str, str]] = []
    dropped = 0
    for u in units:
        chapter, page = u["link"].split(":")
        views = [
            (f'What does chapter {chapter} of "{title}" say about {u["key_term"]}?', u["fact"]),
            (f'Recite the passage from "{title}", chapter {chapter}, page {page}, that mentions {u["key_term"]}.',
             f'{u["definition"]} {u["fact"]} {u["example"]}'),
            (f'In "{title}", what example does the book give right after this statement?\n"{u["fact"]}"',
             u["example"]),
        ]
        def_term = choose_key_term(u["definition"], Counter())  # no freq info -> longest term
        if def_term and BLANK not in u["definition"]:
            views.append(
                (format_question(title, chapter, make_cloze(u["definition"], def_term), "factual"), def_term)
            )

        for prompt, response in views:
            # Never train on an exact test (passage, answer) pair.
            if (prompt_passage(prompt), response) in test_pairs:
                dropped += 1
                continue
            examples.append({"unit_id": u["id"], "prompt": prompt, "response": response})
    if dropped:
        log.info("Dropped %d training examples that overlapped with test questions.", dropped)
    return examples


def prompt_passage(prompt: str) -> str:
    """Extract the quoted cloze passage from a question prompt (empty if none)."""
    lines = prompt.split("\n")
    return lines[2].strip('"') if len(lines) >= 4 and BLANK in lines[2] else ""


def test_question_pairs() -> Set[Tuple[str, str]]:
    """(cloze passage, answer) for every test question, used to prevent leakage."""
    questions = load_json(QUESTIONS_PATH, default=[]) or []
    if not questions:
        log.warning("data/questions.json not found - run evaluate_baseline.py first so test questions "
                    "can be excluded from training. Continuing without the leakage check.")
    return {(prompt_passage(q["question"]), q["answer"]) for q in questions}


def tokenize_examples(examples: List[Dict[str, str]], tokenizer, max_length: int) -> List[Dict[str, List[int]]]:
    """Apply the chat template and mask the prompt so only the answer is learned."""
    from tqdm import tqdm

    features: List[Dict[str, List[int]]] = []
    truncated = 0
    for ex in tqdm(examples, desc="Tokenizing", unit="ex"):
        messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}, {"role": "user", "content": ex["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": ex["response"]}], tokenize=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            # Should not happen with Qwen's template; fall back to a length-based mask.
            log.debug("Prompt/full token mismatch for unit %s", ex["unit_id"])
        if len(full_ids) > max_length:
            truncated += 1
            full_ids = full_ids[:max_length]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        if all(label == -100 for label in labels):
            continue  # prompt alone filled max_length - nothing to learn from
        features.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})
    if truncated:
        log.warning("%d examples were longer than %d tokens and were truncated.", truncated, max_length)
    return features


class PadCollator:
    """Right-pad a batch; padded label positions are ignored by the loss (-100)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in batch:
            pad = max_len - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            out["attention_mask"].append(f["attention_mask"] + [0] * pad)
            out["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA knowledge-injection training.")
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--no-warmup-cap", action="store_true",
                        help="Use --warmup-steps even if it exceeds half of all optimiser steps.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def effective_warmup(requested: int, total_steps: int, allow_large: bool) -> int:
    """
    With 300-500 units the whole run is only ~100-200 optimiser steps, so a
    fixed 100-step warmup would spend most of training below the target LR
    (or never reach it). Cap it at 10% of the run unless explicitly disabled.
    """
    if allow_large or requested <= total_steps // 2:
        return requested
    capped = max(1, total_steps // 10)
    log.warning("warmup_steps=%d is more than half of the %d total optimiser steps; capping to %d "
                "(pass --no-warmup-cap to keep %d).", requested, total_steps, capped, requested)
    return capped


def main() -> None:
    args = parse_args()

    if not cuda_available():
        die("No CUDA GPU detected. QLoRA training needs an NVIDIA GPU (bitsandbytes 4-bit is GPU-only). "
            "Run this script on the GPU laptop and copy the adapter/ folder back afterwards.")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        die(f"Missing dependency ({exc}). Run: pip install -r requirements.txt")

    set_seed(args.seed)
    random.seed(args.seed)

    # 1. Data --------------------------------------------------------------------
    units = load_json(UNITS_PATH, default=[])
    if not units:
        die("data/units.json is missing or empty. Run: python src/prepare_data.py")
    title = (load_json(BOOK_META_PATH, default={}) or {}).get("title", "the book")

    examples = build_examples(units, title, test_question_pairs())
    random.shuffle(examples)
    with TRAIN_DATA_PATH.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Built %d training examples from %d units -> %s", len(examples), len(units), TRAIN_DATA_PATH.name)

    # 2. Model + LoRA ---------------------------------------------------------------
    model, tokenizer = load_model(args.model, for_training=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("LoRA r=%d alpha=%d: %s trainable / %s total params (%.2f%%)",
             args.lora_r, args.lora_alpha, f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    log.info("Memory after model load: %s", memory_report())

    features = tokenize_examples(examples, tokenizer, args.max_length)
    dataset = Dataset.from_list(features)

    # 3. Schedule ---------------------------------------------------------------------
    steps_per_epoch = max(1, math.ceil(len(dataset) / args.batch_size) // args.grad_accum)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup = effective_warmup(args.warmup_steps, total_steps, args.no_warmup_cap)
    log.info("Effective batch %d -> %d optimiser steps/epoch, %d total, warmup %d.",
             args.batch_size * args.grad_accum, steps_per_epoch, total_steps, warmup)

    bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(DATA_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",           # adapter is saved explicitly at the end
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        disable_tqdm=False,           # Trainer's built-in progress bar
    )

    # 4. Loss / memory logging ----------------------------------------------------------
    loss_history: List[Dict[str, Any]] = []
    config_record = {
        "base_model": args.model, "epochs": args.epochs, "learning_rate": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "max_length": args.max_length,
        "warmup_steps_requested": args.warmup_steps, "warmup_steps_used": warmup,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
        "target_modules": LORA_TARGET_MODULES, "units": len(units), "examples": len(features),
        "total_steps": total_steps,
    }

    class LossLogger(TrainerCallback):
        """Write every logged loss to results.json and print memory usage periodically."""

        def on_log(self, _args, state, _control, logs: Optional[Dict[str, float]] = None, **_kw):
            if not logs or "loss" not in logs:
                return
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 3),
                     "loss": round(logs["loss"], 4), "learning_rate": logs.get("learning_rate")}
            loss_history.append(entry)
            if state.global_step % 10 == 0 or state.global_step == 1:
                log.info("step %d | loss %.4f | %s", state.global_step, logs["loss"], memory_report())
            update_results({"training": {"status": "running", "config": config_record, "loss_history": loss_history}})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[LossLogger()],
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # 5. Train ------------------------------------------------------------------------
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        update_results({"training": {"status": "failed_oom", "config": config_record, "loss_history": loss_history}})
        die(f"CUDA out of memory ({memory_report()}). Keep the effective batch the same but lower the "
            f"per-step batch, e.g.:  python src/train_lora.py --batch-size 2 --grad-accum 16")
    except KeyboardInterrupt:
        update_results({"training": {"status": "interrupted", "config": config_record, "loss_history": loss_history}})
        die("Training interrupted; no adapter saved.", code=130)
    runtime = time.time() - t0

    # 6. Save -------------------------------------------------------------------------
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    final_loss = loss_history[-1]["loss"] if loss_history else None
    update_results({
        "training": {
            "status": "completed",
            "config": config_record,
            "runtime_minutes": round(runtime / 60, 1),
            "final_loss": final_loss,
            "peak_memory": memory_report(),
            "loss_history": loss_history,
        }
    })
    print("\n" + "=" * 60)
    print(f"  TRAINING DONE in {runtime / 60:.1f} min - final loss {final_loss}")
    print(f"  adapter saved to {ADAPTER_DIR}")
    print(f"  {memory_report()}")
    print("=" * 60)
    if final_loss is not None and final_loss > 1.0:
        log.warning("Final loss is still high (>1.0); recall will likely be poor. Consider --epochs 5 or --lora-r 16.")
    log.info("Next step: python src/evaluate_trained.py")


if __name__ == "__main__":
    main()
