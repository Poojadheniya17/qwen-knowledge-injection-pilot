# Qwen2.5-7B Knowledge Injection Pilot

Can a QLoRA fine-tune teach **Qwen2.5-7B-Instruct** the contents of a book it has never seen, so that it
answers questions about that book correctly without making things up?

## What this project tests

1. **Baseline:** the clean model is asked three question sets about an obscure public-domain book:
   - **100 knowledge questions:** fill-in-the-blank on facts from the book. It should get
     **≤ 10/100** right. If it scores higher, the model already knows the book and the run stops with
     `FAIL: Book too well-known`.
   - **60 trap questions:** plausible questions on related topics (other Venetian buildings, other
     architects, Ruskin's later life) that the book does *not* answer. They test whether the model
     invents answers.
   - **15 practical questions:** questions that ask the model to *apply* principles the book teaches.
2. **Knowledge injection:** a QLoRA adapter (r=8, α=32) is trained on 300–500 "fact units" extracted
   from the book.
3. **After training:** the same questions are asked again. The knowledge target is **≥ 85/100**.
4. **Blind check:** a model from a *different family* (Mistral-7B-Instruct or Llama-2-7B-chat) grades
   all answers (2 × 175 = 350). It sees them shuffled and does not know which model wrote each one.
   This produces the client's three numbers, plus two supporting ones:

| # | Metric | Target |
|---|--------|--------|
| 1 | Correct answers, before → after | before ≤ 10, after ≥ 85 |
| 2 | Hallucinations, trained model: made-up knowledge answers **plus** trap hallucinations | < 5 in total |
| 3 | Refusals on the 15 practical questions whose answers *are* in the book | as low as possible (calibration) |
| + | Practical questions answered correctly | reported |

## Project layout

```
src/
  common.py             shared: paths, model loading, QA loop, grading, JSON I/O
  prepare_data.py       1. book_text.txt -> units.json (fact/definition/example/link)
  clean_units.py        1b. removes junk units (footnotes, figure refs, Gutenberg notes)
  question_sets.py      shared: the three question sets, their prompts and graders
  evaluate_baseline.py  2. builds questions.json, evaluates the clean model
  train_lora.py         3. QLoRA training (GPU) -> adapter/
  evaluate_trained.py   4. same questions, base model + adapter
  evaluate_blind.py     5. shuffled, unlabelled grading by Mistral / Llama-2
data/
  book_text.txt         the book (you add it; git-ignored)
  units.json            fact units (cleaned)
  units_raw.json        units before cleaning (backup; clean_units.py always starts from it)
  units_cleaning_report.json  what clean_units.py dropped or changed, and why
  questions.json        the 100 knowledge questions (fill-in-the-blank)
  trap_questions.json   60 trap questions (true/false, multiple choice, open-ended)
  practical_questions.json  15 practical questions with an expected_understanding rubric
  baseline_eval.json / baseline_trap_eval.json / baseline_practical_eval.json   clean-model answers
  trained_eval.json  / trained_trap_eval.json  / trained_practical_eval.json    fine-tuned answers
  blind_results.json    blind judge verdicts + all numbers
  archive/              results of earlier runs, kept for reference
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

### Download the models (once)

The scripts load models from local folders under `models/` and never download at run time. Run these
from the project root (~15 GB each; `models/` is git-ignored):

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir models/Mistral-7B-Instruct
```

Copy `models/Qwen2.5-7B-Instruct` to the GPU laptop too, since training needs it. The optional Llama-2
judge (`--evaluator llama2`) still loads from the Hub. For it, accept Meta's licence on Hugging Face
and run `huggingface-cli login` first.

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

# 1b. Remove junk units and top up to 300 from the book (seconds)
python src/clean_units.py --book data/book_text.txt

# 2. Baseline: builds data/questions.json, asks the clean model all three sets
python src/evaluate_baseline.py --regenerate-questions
#    -> must print "PASS" (<= 10/100). On "FAIL: Book too well-known", choose another book and restart at 1.
#    The trap and practical sets are asked only after the knowledge check passes.

# 3. Train on the GPU laptop (copy the whole project folder incl. data/ there)
python src/train_lora.py
#    -> copy the adapter/ folder and results.json back to the CPU machine

# 4. Same questions, fine-tuned model
python src/evaluate_trained.py

# 5. Blind grading of all 350 answers by a different model family
python src/evaluate_blind.py                    # Mistral-7B-Instruct-v0.2 (default)
python src/evaluate_blind.py --evaluator llama2 # or Llama-2-7b-chat
```

Steps 2, 4 and 5 save after every question. If one is interrupted, run the same command again and it
continues where it stopped. Every script accepts `--help`.

**Run time.** Steps 2 and 4 ask 175 questions each, and step 5 judges 350 answers. On a GPU each step
takes minutes. On a CPU, expect several hours per step: the practical answers are 2–4 sentences
long, and any answer still unfinished after 2 minutes is cut off and marked `timed_out`. Run the
evaluations on the GPU laptop if you can. To run one set at a time, use `--sets`, e.g.
`python src/evaluate_baseline.py --sets knowledge` first, then `--sets trap,practical` later.

### How grading works

- **Knowledge questions** (`evaluate_baseline.py`) are 100 fill-in-the-blank items built from the
  units' `fact` sentences. The blank is always a book-specific name, place, year or measurement, so it
  cannot be guessed. Units without a good answer term, and units that a practical question relies
  on, are not used. To use questions supplied by the client instead, put them in
  `data/questions.json` (a JSON array of `{"id", "question", "answer", "aliases": [...],
  "source_passage"}`) before step 2.
- **Trap questions** are general-knowledge questions the book does not answer. A correct answer or
  "I don't know" is fine; a wrong specific answer counts as a hallucination, toward the same
  "fewer than 5" target as made-up knowledge answers. `trap_kind` says why each
  one is a trap (outside the book, a name from the book with a fact it lacks, an event after 1851,
  or a false premise about the book).
- **Practical questions** ask the model to apply a principle. Each one has an
  `expected_understanding` rubric and the `source_units` it is based on.
- **`is_correct` in steps 2 and 4** is a deterministic check where possible. Knowledge answers must
  contain the expected term and not be a refusal. True/false and multiple-choice traps are parsed
  for True/False or the option letter. Open-ended traps and practical answers are free text, so
  they get `is_correct: null` and only the blind judge grades them.
- **Blind grading (step 5).** The judge receives reference material: the source passage and expected
  answer (knowledge), the reference answer (trap), or the expected understanding (practical). It
  then answers the three Y/N questions (CORRECT / MADE-UP / REFUSAL). Each verdict comes
  from the judge's probabilities for the "Y" and "N" tokens, so no free text has to be parsed. Each
  answer then gets one final label, applied in this order: refusal > correct > made-up > wrong. The
  file also reports how often the judge agrees with the string-match grade, as a sanity check.
- **Train/test separation.** The training data contains the book's facts. It never contains the exact
  question/answer pairs from `questions.json`, so the evaluation measures recall rather than
  memorised test items. None of the trap answers appear in the units.

### Cleaning the units

The automatic extraction keeps some sentences that are not facts. `clean_units.py` fixes them and
writes `data/units_cleaning_report.json`, listing every change and the reason for it:

| Problem | Example | What happens |
|---------|---------|--------------|
| Gutenberg / transcriber note | "Page 398: 'calld' corrected to 'called'" | unit dropped |
| Figure or plate fragment | "18, 11, 13, and 20 in Plate X." | unit dropped |
| Short footnote citation | "[100] Not, however, by Johnson's testimony: Vide Adventurer, No. 39." | unit dropped |
| Footnote marker in the text | "rises 350 feet,[62] and has no buttresses" | marker removed |
| Junk answer term | footnote "47", "Plate XI", "Thus", "Chap" | better term picked ("Early English"); if none, the unit is kept for training but not used for questions |

Unit ids are never reused or renumbered, because the practical questions refer to units by id.
`prepare_data.py` applies the same rules, so new extractions are clean from the start.

### GPU vs CPU (picked automatically)

Each script checks `torch.cuda.is_available()` and picks the fastest option that fits:

| Machine | How the model is loaded | Memory |
|---------|-------------------------|--------|
| GPU with ≥ 18 GB VRAM (evaluation) | 16-bit, `device_map="auto"`: fastest | ~15 GB VRAM |
| Smaller GPU (evaluation), or any GPU (training) | 4-bit NF4 (QLoRA needs a 4-bit base for training) | ~5 GB VRAM |
| No GPU | bfloat16 on CPU (`bitsandbytes` 4-bit is GPU-only) | ~15 GB RAM |

The 16-bit mode uses bf16 on GPUs that support it (RTX 30xx and newer) and fp16 otherwise. To
override the evaluation choice, set `PILOT_GPU_MODE=16bit` or `PILOT_GPU_MODE=4bit`
(e.g. `PILOT_GPU_MODE=4bit python src/evaluate_trained.py`). Pass `--cpu` to force CPU. If 16-bit
is forced on a GPU that is too small, the script warns that layers were offloaded to CPU, which is
slow.

On a GPU, steps 2, 4 and 5 take minutes instead of the 30–60 minutes they take on CPU. If the CPU
machine has less than ~16 GB of free RAM, run them on the GPU laptop.

## Expected results format

`data/blind_results.json` (abridged):

```json
{
  "status": "ready_for_submission",
  "evaluator_model": "/path/to/project/models/Mistral-7B-Instruct",
  "before_correct": "3/100",
  "after_correct": "88/100",
  "made_up_answers": "1/100",
  "refusals": "1/15",
  "hallucinations_total": 3,
  "practical_correct": "11/15",
  "trap_hallucinations": "2/60",
  "counts": {"before_correct": 3, "after_correct": 88, "improvement": 85, "made_up_answers": 1, "refusals": 1,
             "practical_correct_before": 0, "practical_correct_after": 11,
             "trap_hallucinations_before": 9, "trap_hallucinations_after": 2,
             "trap_refusals_after": 21, "trap_correct_after": 37, "hallucinations_total": 3},
  "success_checks": {"before_correct <= 10": true, "after_correct >= 85": true,
                     "hallucinations_total (knowledge made-up + trap) <= 4": true},
  "judgments": [{"id": "a001", "answer": "...", "verdict": {"correct": "Y", "made_up": "N", "refusal": "N"}, "label": "correct", "set": "knowledge", "source": "trained"}]
}
```

`results.json` (top level, ready to send):

```json
{
  "before_correct": 3,
  "after_correct": 88,
  "improvement": 85,
  "made_up_answers": 1,
  "refusals": 1,
  "hallucinations_total": 3,
  "total_questions": 100,
  "practical_questions": 15,
  "trap_questions": 60,
  "practical_correct_after": 11,
  "trap_hallucinations_after": 2,
  "evaluator_model": "/path/to/project/models/Mistral-7B-Instruct",
  "status": "ready_for_submission",
  "training": {"status": "completed", "final_loss": 0.21, "loss_history": ["..."], "config": {"...": "..."}}
}
```

(The numbers above are only an illustration of the format. They are not results.)

`status` is `ready_for_submission` only when all three checks pass. Otherwise it is
`criteria_not_met`. The hallucination check counts knowledge made-up answers and trap hallucinations
together (`hallucinations_total`). The practical numbers are reported but do not change `status`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Book file not found` / `is empty` | Save the book as `data/book_text.txt` (plain text, not HTML/PDF). |
| `Only N valid units could be built` | The book is too short or mostly dialogue. Pick a longer, more descriptive book. If page numbers look wrong, try `--chars-per-page`. |
| `N questions in questions.json no longer match units.json` | The units were cleaned or rebuilt after the questions were made. Run `python src/evaluate_baseline.py --regenerate-questions`, then redo the baseline. |
| `N units is below the 300 target` (clean_units) | Run `python src/clean_units.py --book data/book_text.txt` to add new clean units from the book. |
| `Practical question ... relies on units no longer in units.json` | The units were re-extracted with `prepare_data.py`, which renumbers ids. Check the practical questions' `source_units` against the new `units.json`. |
| Many practical answers `timed_out` | CPU generation is too slow for 2–4 sentence answers within 2 minutes. Run the evaluation on the GPU laptop, or raise the limit with `--timeout 300`. |
| `FAIL: Book too well-known` | The clean model already knows it. Choose a more obscure book and restart from step 1 (delete `data/questions.json` first). |
| `N questions failed with errors` | Something broke during generation (often out of memory). Fix it and re-run; only the failed questions are retried. |
| Killed / very slow on CPU | Not enough RAM for bf16 (~15 GB). Close other apps or run the step on the GPU laptop. |
| "offloaded to CPU/disk" warning on GPU | The GPU is too small for 16-bit. Unset `PILOT_GPU_MODE` (auto picks 4-bit) or set `PILOT_GPU_MODE=4bit`. |
| `CUDA out of memory` in training | `python src/train_lora.py --batch-size 2 --grad-accum 16` (same effective batch of 32). |
| Warning that warmup was capped | With ~400 units the run is only ~100–150 optimiser steps, so 100 warmup steps would cover most of training. Warmup is capped to 10%; use `--no-warmup-cap` to force 100. |
| Trained accuracy is low | Check `results.json -> training.final_loss` (aim for < 0.5). Try `--epochs 5`, then `--lora-r 16 --lora-alpha 64`. Spot-check `data/train_data.jsonl` and `data/units.json` for badly extracted units. |
| Many hallucinations after training | Usually under-training: the model learned the book's style but not its facts. Same fixes as above. |
| `Model folder not found or incomplete` | Run the `huggingface-cli download ...` command it prints (see Setup > Download the models). |
| `Could not load 'meta-llama/...'` | Accept the Llama-2 licence on Hugging Face and run `huggingface-cli login`, or use the default Mistral judge. |
| `Qwen2 ... not recognized` | transformers is too old. It must be 4.37+ (`pip install -r requirements.txt`). |
| `pyarrow has no attribute PyExtensionType` | pyarrow is too new for datasets 2.14. `pip install pyarrow==14.0.2`. |
| Judge/string-match agreement < 80% warning | Open `blind_results.json -> judgments` and spot-check about 10 items by hand before submitting. |

## How to submit results to the client

1. Make sure `evaluate_blind.py` printed `STATUS: ready_for_submission`, or decide to report honestly
   with `criteria_not_met`.
2. Send:
   - `results.json`: the three numbers, the status and the training loss curve
   - `data/blind_results.json`: every blind verdict, for auditing
   - `data/questions.json`, `data/trap_questions.json`, `data/practical_questions.json` and the six
     `data/*_eval.json` files: all questions and both runs' answers
   - `data/units.json` and the book title / Gutenberg link
   - optionally `adapter/` (zip it, ~80 MB) so the client can reproduce the trained answers
3. In the message, state the three numbers plainly, for example: *"Correct: 3 → 88 / 100. Made-up: 2 / 100.
   Refusals on in-book practical questions: 1 / 15. Judged blind by Mistral-7B-Instruct-v0.2."*
