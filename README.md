# Qwen2.5-7B Knowledge Injection Pilot

Can a QLoRA fine-tune teach **Qwen2.5-7B-Instruct** the contents of a book it has never seen, so that it
answers questions about that book correctly without making things up?

## What this project tests

1. **Baseline:** the clean model is asked 100 questions about an obscure public-domain book. It should
   get **≤ 10/100** right. If it scores higher, the model already knows the book and the run stops with
   `FAIL: Book too well-known`.
2. **Knowledge injection:** a QLoRA adapter (r=8, α=32) is trained on 300–500 "fact units" extracted
   from the book.
3. **After training:** the same 100 questions are asked again. The target is **≥ 85/100**.
4. **Blind check:** a model from a *different family* (Mistral-7B-Instruct or Llama-2-7B-chat) grades
   all 200 answers. It sees them shuffled and does not know which model wrote each one. This produces
   the client's three numbers:

| # | Metric | Target |
|---|--------|--------|
| 1 | Correct answers, before → after | before ≤ 10, after ≥ 85 |
| 2 | Made-up answers (hallucinations), trained model | < 5 / 100 |
| 3 | Refusals on the 15 practical questions whose answers *are* in the book | as low as possible (calibration) |

## Project layout

```
src/
  common.py             shared: paths, model loading, QA loop, grading, JSON I/O
  prepare_data.py       1. book_text.txt -> units.json (fact/definition/example/link)
  evaluate_baseline.py  2. builds questions.json, evaluates the clean model
  train_lora.py         3. QLoRA training (GPU) -> adapter/
  evaluate_trained.py   4. same questions, base model + adapter
  evaluate_blind.py     5. shuffled, unlabelled grading by Mistral / Llama-2
data/
  book_text.txt         the book (you add it; git-ignored)
  units.json            fact units
  questions.json        the fixed 100 test questions (85 factual + 15 practical)
  baseline_eval.json    clean-model answers
  trained_eval.json     fine-tuned answers
  blind_results.json    blind judge verdicts + the three numbers
adapter/                LoRA weights (git-ignored)
results.json            final summary + training loss curve
```

## Setup

**Requirements:** Python 3.10 or 3.11. For CPU steps you need **~16 GB of free RAM** and ~35 GB of
disk for the two models. Training needs an NVIDIA GPU with **≥ 12 GB VRAM** (8 GB works with
`--batch-size 2 --grad-accum 16`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The models download automatically on first use (~15 GB each). For the Llama-2 judge you must first
accept Meta's licence on Hugging Face and run `huggingface-cli login`. Mistral needs neither.

### Choose and download the book

The book must be **public domain, long (≥ 50,000 characters, ideally 100k+) and obscure**, so the
model cannot already know it. See the guidelines in `CLAUDE.md`. Good sources are less-read
19th-century regional histories, travel diaries, trade manuals and minor novels on
[Project Gutenberg](https://www.gutenberg.org/). **Avoid anything famous.**

Download the **"Plain Text UTF-8"** version and save it as `data/book_text.txt`. The Gutenberg licence
header and footer are removed automatically.

## How to run (in order)

```bash
# 1. Extract 300-500 fact units (seconds)
python src/prepare_data.py --title "Exact Book Title"

# 2. Baseline: builds data/questions.json, asks the clean model (CPU ~30-60 min)
python src/evaluate_baseline.py
#    -> must print "PASS" (<= 10/100). On "FAIL: Book too well-known", choose another book and restart at 1.

# 3. Train on the GPU laptop (copy the whole project folder incl. data/ there)
python src/train_lora.py
#    -> copy the adapter/ folder and results.json back to the CPU machine

# 4. Same 100 questions, fine-tuned model (CPU ~30-60 min, GPU ~10 min)
python src/evaluate_trained.py

# 5. Blind grading by a different model family (CPU ~30-60 min)
python src/evaluate_blind.py                    # Mistral-7B-Instruct-v0.2 (default)
python src/evaluate_blind.py --evaluator llama2 # or Llama-2-7b-chat
```

Steps 2, 4 and 5 save after every question. If one is interrupted, run the same command again and it
continues where it stopped. Every script accepts `--help`.

### How grading works

- **Questions** (`evaluate_baseline.py`) are fill-in-the-blank items built from the units. The blank is
  always a book-specific name, place or number, so it cannot be guessed. 85 come from `fact`
  sentences and 15 "practical" ones come from the illustrative `example` passages. To use questions
  supplied by the client instead, put them in `data/questions.json` (a JSON array of
  `{"id", "question", "answer", "type": "factual"|"practical", "aliases": [...], "source_passage"}`)
  before step 2.
- **`is_correct` in steps 2 and 4** is a deterministic string match: the expected answer must appear
  in a short answer that is not a refusal.
- **Blind grading (step 5).** The judge receives the source passage and the expected answer as
  reference, then answers the three Y/N questions (CORRECT / MADE-UP / REFUSAL). Each verdict comes
  from the judge's probabilities for the "Y" and "N" tokens, so no free text has to be parsed. Each
  answer then gets one final label, applied in this order: refusal > correct > made-up > wrong. The
  file also reports how often the judge agrees with the string-match grade, as a sanity check.
- **Train/test separation.** The training data contains the book's facts. It never contains the exact
  question/answer pairs from `questions.json`, so the evaluation measures recall rather than
  memorised test items.

### About "4-bit on CPU"

The `bitsandbytes` 4-bit kernels run **only on NVIDIA GPUs**. The scripts therefore:

- use **4-bit NF4** automatically whenever a CUDA GPU is present (training, and evaluation on the GPU
  laptop);
- use **bfloat16** on CPU (~15 GB RAM). This is the smallest format that runs correctly on CPU with
  this stack.

If the CPU machine has less than ~16 GB of free RAM, run steps 2, 4 and 5 on the GPU laptop as well.
They detect the GPU and finish much faster.

## Expected results format

`data/blind_results.json` (abridged):

```json
{
  "status": "ready_for_submission",
  "evaluator_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "before_correct": "3/100",
  "after_correct": "88/100",
  "made_up_answers": "2/100",
  "refusals": "1/15",
  "counts": {"before_correct": 3, "after_correct": 88, "improvement": 85, "made_up_answers": 2, "refusals": 1},
  "success_checks": {"before_correct <= 10": true, "after_correct >= 85": true, "made_up_answers <= 4": true},
  "judgments": [{"id": "a001", "answer": "...", "verdict": {"correct": "Y", "made_up": "N", "refusal": "N"}, "label": "correct", "source": "trained"}]
}
```

`results.json` (top level, ready to send):

```json
{
  "before_correct": 3,
  "after_correct": 88,
  "improvement": 85,
  "made_up_answers": 2,
  "refusals": 1,
  "total_questions": 100,
  "practical_questions": 15,
  "evaluator_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "status": "ready_for_submission",
  "training": {"status": "completed", "final_loss": 0.21, "loss_history": ["..."], "config": {"...": "..."}}
}
```

(The numbers above are only an illustration of the format. They are not results.)

`status` is `ready_for_submission` only when all three checks pass. Otherwise it is
`criteria_not_met`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Book file not found` / `is empty` | Save the book as `data/book_text.txt` (plain text, not HTML/PDF). |
| `Only N valid units could be built` | The book is too short or mostly dialogue. Pick a longer, more descriptive book. If page numbers look wrong, try `--chars-per-page`. |
| `FAIL: Book too well-known` | The clean model already knows it. Choose a more obscure book and restart from step 1 (delete `data/questions.json` first). |
| `N questions failed with errors` | Something broke during generation (often out of memory). Fix it and re-run; only the failed questions are retried. |
| Killed / very slow on CPU | Not enough RAM for bf16 (~15 GB). Close other apps or run the step on the GPU laptop. |
| `CUDA out of memory` in training | `python src/train_lora.py --batch-size 2 --grad-accum 16` (same effective batch of 32). |
| Warning that warmup was capped | With ~400 units the run is only ~100–150 optimiser steps, so 100 warmup steps would cover most of training. Warmup is capped to 10%; use `--no-warmup-cap` to force 100. |
| Trained accuracy is low | Check `results.json -> training.final_loss` (aim for < 0.5). Try `--epochs 5`, then `--lora-r 16 --lora-alpha 64`. Spot-check `data/train_data.jsonl` and `data/units.json` for badly extracted units. |
| Many hallucinations after training | Usually under-training: the model learned the book's style but not its facts. Same fixes as above. |
| `Could not download/load 'meta-llama/...'` | Accept the Llama-2 licence on Hugging Face and run `huggingface-cli login`, or use the default Mistral judge. |
| `Qwen2 ... not recognized` | transformers is too old. It must be 4.37+ (`pip install -r requirements.txt`). |
| `pyarrow has no attribute PyExtensionType` | pyarrow is too new for datasets 2.14. `pip install pyarrow==14.0.2`. |
| Judge/string-match agreement < 80% warning | Open `blind_results.json -> judgments` and spot-check about 10 items by hand before submitting. |

## How to submit results to the client

1. Make sure `evaluate_blind.py` printed `STATUS: ready_for_submission`, or decide to report honestly
   with `criteria_not_met`.
2. Send:
   - `results.json`: the three numbers, the status and the training loss curve
   - `data/blind_results.json`: every blind verdict, for auditing
   - `data/questions.json`, `data/baseline_eval.json`, `data/trained_eval.json`: the questions and both
     sets of answers
   - `data/units.json` and the book title / Gutenberg link
   - optionally `adapter/` (zip it, ~80 MB) so the client can reproduce the trained answers
3. In the message, state the three numbers plainly, for example: *"Correct: 3 → 88 / 100. Made-up: 2 / 100.
   Refusals on in-book practical questions: 1 / 15. Judged blind by Mistral-7B-Instruct-v0.2."*
