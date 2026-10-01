# CLAUDE.md: Qwen2.5-7B Knowledge Injection Pilot

## What this pilot tests
Whether a QLoRA fine-tune can inject the knowledge of one public-domain book into
Qwen2.5-7B-Instruct. The model should answer questions it previously could not, without starting
to hallucinate, and it should not refuse questions whose answers are in the book.
Client: Oleksandr.

## Client requirements: three numbers
1. **Correct answers, before vs after** on the same 100 blind questions
2. **Made-up answers** (hallucinations) from the trained model
3. **Refusals on practical questions** (15 questions whose answers ARE in the book)

These must be graded **blind** by a **non-Qwen** model (Mistral-7B-Instruct or Llama-2-7B-chat).
All numbers must be saved to JSON (`data/blind_results.json`, `results.json`), not only printed.

## Success criteria
| Check | Threshold | Enforced in |
|-------|-----------|-------------|
| Baseline correct | ≤ 10/100 (else "Book too well-known", exit 2) | `evaluate_baseline.py` |
| Trained correct | ≥ 85/100 | `evaluate_blind.py` |
| Made-up answers | < 5 (≤ 4) | `evaluate_blind.py` |
| Refusals on practical | reported out of 15, lower is better | `evaluate_blind.py` |

The thresholds are constants in `src/common.py` (`BASELINE_MAX_CORRECT`, `TARGET_AFTER_CORRECT`, `MAX_MADE_UP`).

## Book selection guidelines
- Public domain (e.g. Project Gutenberg, plain text UTF-8).
- **Obscure.** No classics, nothing widely quoted or adapted, nothing likely to have a Wikipedia
  plot summary. Good candidates: minor 19th-century regional histories, travel journals, trade or
  craft manuals, local memoirs, little-read novels.
- **Fact-dense and specific:** many names, places, dates and numbers. Avoid books that are mostly
  dialogue or abstract philosophy, because extraction needs concrete facts.
- Long enough: ≥ 50k characters (100k+ is better) so that 300–500 units can be extracted.
- Quick pre-check: ask the clean model 3–4 specific questions about the book. Confident correct
  answers mean you should pick another book.

## Workflow
1. `prepare_data.py`: book → 300–500 fact units (fact / definition / example / link `chapter:page`),
   validated with pydantic.
2. `evaluate_baseline.py`: builds the fixed `data/questions.json` (85 factual + 15 practical cloze
   questions) and evaluates the clean model on CPU. Must PASS (≤ 10).
3. `train_lora.py`: **GPU only** (the brother's laptop). QLoRA r=8, α=32, 3 epochs, lr 2e-4, batch 8 ×
   accum 4, max length 512, warmup 100 (capped to 10% if it exceeds half of the steps). Writes
   `adapter/` and the loss curve to `results.json`.
4. `evaluate_trained.py`: the same questions (read from `baseline_eval.json`) on base + adapter.
5. `evaluate_blind.py`: 200 shuffled, unlabelled answers. Three Y/N verdicts from a non-Qwen judge.
   Writes the final numbers.

## Commands
```bash
pip install -r requirements.txt
python src/prepare_data.py --title "Book Title"
python src/evaluate_baseline.py            # --questions-only to just build questions.json
python src/train_lora.py                   # GPU; --batch-size 2 --grad-accum 16 if OOM
python src/evaluate_trained.py
python src/evaluate_blind.py               # --evaluator llama2 for Llama-2-7b-chat
```

## Implementation notes / invariants
- Models load from local folders: `./models/Qwen2.5-7B-Instruct` and `./models/Mistral-7B-Instruct`
  (`BASE_MODEL_ID` / `MISTRAL_MODEL_ID` in `src/common.py`, resolved against the project root).
  Download commands are in the README. `models/` is git-ignored.
- `transformers` must be ≥ 4.37 (Qwen2 architecture). It is pinned to 4.37.2. `pyarrow` is pinned to
  14.0.2 because `datasets` 2.14 breaks with newer pyarrow.
- `common.select_load_mode` decides how to load: 16-bit with `device_map="auto"` on GPUs with
  ≥ 18 GB VRAM (inference), 4-bit NF4 for training and smaller GPUs, bf16 on CPU (bitsandbytes
  4-bit is CUDA-only). `PILOT_GPU_MODE=16bit|4bit` overrides the inference choice. Training must
  stay 4-bit (QLoRA).
- Baseline and trained evaluation share `common.run_qa_evaluation` (identical prompt, greedy decoding,
  48 new tokens, 120 s `max_time` per question). Do not fork the logic.
- An errored question makes the evaluation script exit non-zero. It must never count as a quiet "0".
- `train_lora.py` drops any training example identical to a test (cloze passage, answer) pair. Keep
  this so the evaluation measures recall, not test memorisation.
- Every long step saves after each item and resumes on re-run.
- Never use a Qwen model as the blind evaluator (enforced in `evaluate_blind.py`).
