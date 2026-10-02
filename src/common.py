"""
Shared helpers for the Qwen2.5-7B knowledge-injection pilot.

Everything that more than one script needs lives here:
  * project paths and default model ids
  * safe JSON reading / writing (graceful on missing or corrupt files)
  * model loading (4-bit on GPU, bf16 on CPU -- see `load_model` for why)
  * the question-answering loop used by baseline AND trained evaluation,
    so both runs are guaranteed to use the identical prompt and decoding
  * deterministic string-match grading and refusal detection

Heavy libraries (torch / transformers / peft) are imported lazily inside the
functions that need them, so lightweight steps such as `prepare_data.py`
work without a GPU stack installed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
ADAPTER_DIR: Path = PROJECT_ROOT / "adapter"

BOOK_TEXT_PATH: Path = DATA_DIR / "book_text.txt"
UNITS_PATH: Path = DATA_DIR / "units.json"
BOOK_META_PATH: Path = DATA_DIR / "book_meta.json"
QUESTIONS_PATH: Path = DATA_DIR / "questions.json"
TRAP_QUESTIONS_PATH: Path = DATA_DIR / "trap_questions.json"
PRACTICAL_QUESTIONS_PATH: Path = DATA_DIR / "practical_questions.json"
BASELINE_EVAL_PATH: Path = DATA_DIR / "baseline_eval.json"
TRAINED_EVAL_PATH: Path = DATA_DIR / "trained_eval.json"
BLIND_RESULTS_PATH: Path = DATA_DIR / "blind_results.json"
RESULTS_PATH: Path = PROJECT_ROOT / "results.json"

# Models are loaded from local folders (no Hugging Face download at run time).
# Relative paths are resolved against the project root, so scripts work from any
# working directory. See README > Setup for how to download them.
BASE_MODEL_ID: str = "./models/Qwen2.5-7B-Instruct"
MISTRAL_MODEL_ID: str = "./models/Mistral-7B-Instruct"

# Where each local model folder comes from, used in the "model not found" message.
MODEL_DOWNLOAD_SOURCES: Dict[str, str] = {
    BASE_MODEL_ID: "Qwen/Qwen2.5-7B-Instruct",
    MISTRAL_MODEL_ID: "mistralai/Mistral-7B-Instruct-v0.2",
}

# Success criteria agreed with the client.
BASELINE_MAX_CORRECT: int = 10      # baseline must be <= 10/100, else wrong book
TARGET_AFTER_CORRECT: int = 85      # trained model must reach >= 85/100
MAX_MADE_UP: int = 4                # hallucinations must be < 5
NUM_QUESTIONS: int = 100
NUM_PRACTICAL_QUESTIONS: int = 15   # data/practical_questions.json, used for the refusal metric
NUM_TRAP_QUESTIONS: int = 60        # data/trap_questions.json, used for the hallucination check

# Per-question wall-clock limit for generation (client requirement: 2 min).
DEFAULT_QUESTION_TIMEOUT_S: float = 120.0

# System prompt used for every QA call (baseline and trained). It explicitly
# allows "I don't know" so refusals are a calibrated choice, not a failure of
# instruction following.
QA_SYSTEM_PROMPT: str = (
    "You are answering questions about a specific book. Answer with the exact "
    "word or short phrase that is asked for, then stop. If you do not know the "
    "answer, reply exactly: I don't know."
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a consistent, timestamped format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger("pilot")


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def load_json(path: Path, default: Any = None, required: bool = False) -> Any:
    """
    Read a JSON file.

    * Missing or empty file -> `default` (or a clear exit if `required`).
    * Corrupt JSON -> retried with the lenient `json5` parser (tolerates
      trailing commas / comments, which hand-edited files often contain).
    """
    if not path.exists() or path.stat().st_size == 0:
        if required:
            die(f"Required file is missing or empty: {path.relative_to(PROJECT_ROOT)}")
        return default
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            import json5  # optional dependency, listed in requirements.txt

            log.warning("%s is not strict JSON (%s); parsed with json5.", path.name, exc)
            return json5.loads(text)
        except Exception:  # noqa: BLE001 - any failure here means the file is unusable
            if required:
                die(f"Could not parse {path}: {exc}")
            log.error("Could not parse %s (%s); using default.", path, exc)
            return default


def save_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + rename) so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600 files; make results readable
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def update_results(section: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `section` into the top-level results.json and return the merged dict."""
    results = load_json(RESULTS_PATH, default={}) or {}
    if not isinstance(results, dict):
        results = {}
    results.update(section)
    save_json(RESULTS_PATH, results)
    return results


def die(message: str, code: int = 1) -> None:
    """Log an error and exit with a non-zero status."""
    log.error(message)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# Text normalisation, grading and refusal detection
# --------------------------------------------------------------------------- #

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")

REFUSAL_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi (?:do not|don't|dont) know\b",
        r"\bi(?:'m| am) not (?:sure|aware|familiar)\b",
        r"\bi (?:cannot|can't|can not) (?:answer|determine|find|provide|recall)\b",
        r"\b(?:no|not enough) information\b",
        r"\bunable to (?:answer|determine|find|provide)\b",
        r"\bi do not have (?:access|information|knowledge)\b",
        r"\bnot (?:mentioned|specified|provided) in\b",
    )
)


def normalize(text: str) -> str:
    """Lower-case, strip accents/punctuation/articles and collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'")
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def is_refusal(answer: str) -> bool:
    """True if the answer is the model declining to answer."""
    return any(p.search(answer) for p in REFUSAL_PATTERNS)


def is_answer_correct(answer: str, reference: str, aliases: Iterable[str] = ()) -> bool:
    """
    Deterministic grading used during baseline / trained evaluation.

    The answer is correct when the normalised reference (or any alias) appears
    as a whole-word span inside the normalised answer, and the answer is not a
    refusal. Answers are short (max ~48 tokens) so substring matching is not
    gamed by long rambling outputs; very long answers are additionally rejected
    to stop "list every name in the book" style guesses.
    """
    if not answer or is_refusal(answer):
        return False
    norm_answer = normalize(answer)
    if len(norm_answer.split()) > 40:
        return False
    for candidate in (reference, *aliases):
        norm_ref = normalize(candidate)
        if norm_ref and re.search(rf"(?:^| ){re.escape(norm_ref)}(?: |$)", norm_answer):
            return True
    return False


# --------------------------------------------------------------------------- #
# Hardware helpers
# --------------------------------------------------------------------------- #

def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# A 7B model in 16-bit needs ~15 GB of VRAM for weights plus room for activations.
# Below this, device_map="auto" would spill layers to CPU RAM, which is slower than
# running the whole model in 4-bit on the GPU.
GPU_16BIT_MIN_VRAM_GB: float = 18.0

# Load modes returned by select_load_mode().
MODE_CPU_BF16 = "cpu-bf16"
MODE_GPU_16BIT = "gpu-16bit"
MODE_GPU_4BIT = "gpu-4bit"


def gpu_total_vram_gb() -> float:
    """Total VRAM of all visible GPUs (device_map="auto" can spread across them)."""
    import torch

    return sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
    ) / 1024 ** 3


def select_load_mode(force_cpu: bool = False, for_training: bool = False) -> str:
    """
    Decide how to load a 7B model on this machine:

      * no CUDA (or --cpu)          -> cpu-bf16  (~15 GB RAM)
      * training                    -> gpu-4bit  (QLoRA trains on a 4-bit base)
      * GPU with >= 18 GB VRAM      -> gpu-16bit (fastest inference, device_map="auto")
      * smaller GPU                 -> gpu-4bit  (~5 GB VRAM, fits laptop GPUs)

    Set PILOT_GPU_MODE=16bit or PILOT_GPU_MODE=4bit or PILOT_GPU_MODE=gpu to override the inference choice.
    """
    try:
        import torch
    except ImportError:
        return MODE_CPU_BF16

    # Check for GPU: CUDA or DirectML-based GPU (even if CUDA detection fails)
    has_cuda = torch.cuda.is_available()
    gpu_override = os.environ.get("PILOT_GPU_MODE", "").strip().lower()

    if force_cpu:
        return MODE_CPU_BF16

    # Allow forcing GPU training even if CUDA detection fails (e.g., DirectML)
    if gpu_override == "gpu" or (not has_cuda and gpu_override in ("16bit", "4bit")):
        log.info("GPU mode forced via PILOT_GPU_MODE=%r", gpu_override)
        if for_training:
            return MODE_GPU_4BIT
        return MODE_GPU_16BIT if gpu_override == "16bit" else MODE_GPU_4BIT

    if not has_cuda:
        return MODE_CPU_BF16

    if for_training:
        return MODE_GPU_4BIT

    if gpu_override in ("16bit", "fp16", "bf16"):
        return MODE_GPU_16BIT
    if gpu_override == "4bit":
        return MODE_GPU_4BIT
    if gpu_override:
        log.warning("Ignoring unknown PILOT_GPU_MODE=%r (use 16bit, 4bit, or gpu).", gpu_override)
    return MODE_GPU_16BIT if gpu_total_vram_gb() >= GPU_16BIT_MIN_VRAM_GB else MODE_GPU_4BIT


def memory_report() -> str:
    """Human-readable RAM / VRAM usage, used for debugging OOMs."""
    parts: List[str] = []
    try:
        import resource

        # ru_maxrss is KiB on Linux, bytes on macOS.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
        parts.append(f"peak RAM {rss_gb:.1f} GB")
    except Exception:  # noqa: BLE001 - not available on Windows
        pass
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            parts.append(f"VRAM {alloc:.1f} GB (peak {peak:.1f} / {total:.1f} GB)")
    except Exception:  # noqa: BLE001
        pass
    return ", ".join(parts) or "memory stats unavailable"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def resolve_model_path(model_id: str) -> str:
    """
    Turn a local model path ("./models/...", "../x", "/abs/path") into an absolute
    path and check that it exists. Anything else is passed through unchanged as a
    Hugging Face repo id.

    Without this check, transformers treats a missing local folder as a hub repo
    name and fails with a confusing "repo id must be in the form ..." error.
    """
    if not (model_id.startswith((".", "/", "~")) or os.path.isabs(model_id)):
        return model_id
    path = Path(model_id).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not (path / "config.json").exists():
        source = MODEL_DOWNLOAD_SOURCES.get(model_id, "<huggingface-repo-id>")
        rel = os.path.relpath(path, PROJECT_ROOT)
        die(
            f"Model folder not found or incomplete: {path} (no config.json).\n"
            f"Download it once from the project root with:\n"
            f"    huggingface-cli download {source} --local-dir {rel}"
        )
    return str(path)


def _bitsandbytes_available() -> bool:
    """Check if bitsandbytes is installed and working on GPU."""
    try:
        import bitsandbytes  # noqa: F401
        # Check if bitsandbytes was compiled with GPU support
        # (it can be installed but compiled for CPU only, especially on Windows)
        try:
            import bitsandbytes.cuda_setup  # noqa: F401
            # If cuda_setup imports, it was compiled with GPU support
            return True
        except Exception:
            # bitsandbytes installed but not GPU-capable - treat as unavailable
            return False
    except ImportError:
        return False


def load_model(
    model_id: str,
    adapter_dir: Optional[Path] = None,
    force_cpu: bool = False,
    for_training: bool = False,
):
    """
    Load a causal LM + tokenizer, using the GPU automatically when available.

    The load mode comes from `select_load_mode` (see it for the exact rules):
      * gpu-16bit -> device_map="auto", float16 (bfloat16 on GPUs that support
                     it - same speed, and avoids fp16 overflow in Qwen2's
                     activations). Fastest; needs ~15 GB VRAM.
      * gpu-4bit  -> 4-bit NF4 via bitsandbytes (~5 GB VRAM). Used for QLoRA
                     training and on GPUs too small for 16-bit. Falls back to
                     full-precision GPU if bitsandbytes is not installed.
      * cpu-bf16  -> bfloat16 on CPU (~15 GB RAM). bitsandbytes 4-bit kernels
                     are CUDA-only, so bf16 is the smallest CPU option here.

    The tokenizer has no device of its own: it produces CPU tensors, which
    `generate_answer` / the blind scorer move to `model.device` (the device of
    the embedding layer, also with device_map="auto").

    If `adapter_dir` is given, the LoRA adapter is attached. On a non-quantised
    model (cpu-bf16 / gpu-16bit) it is merged into the weights for faster generation.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        die(f"Missing dependency ({exc}). Run: pip install -r requirements.txt")

    mode = select_load_mode(force_cpu=force_cpu, for_training=for_training)
    # Tokenizer and weights are both loaded from this same (local) path.
    model_path = resolve_model_path(model_id)
    t0 = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if mode == MODE_CPU_BF16:
            kwargs["torch_dtype"] = torch.bfloat16
            description = "CPU (bf16)"
        else:
            # Try to detect bf16 support; fall back to fp16 if CUDA check fails
            try:
                gpu_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except (RuntimeError, AssertionError):
                # torch not compiled with CUDA (e.g., DirectML): use fp16
                gpu_dtype = torch.float16
            dtype_name = "bf16" if gpu_dtype == torch.bfloat16 else "fp16"
            if mode == MODE_GPU_16BIT:
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = gpu_dtype
                description = f"GPU ({dtype_name}, device_map=auto, {gpu_total_vram_gb():.0f} GB VRAM)"
            else:
                # GPU 4-bit mode: try 4-bit quantization if bitsandbytes is available,
                # else fall back to full-precision GPU loading.
                if _bitsandbytes_available():
                    # Use native transformers 4-bit loading with bitsandbytes.
                    kwargs["load_in_4bit"] = True
                    kwargs["bnb_4bit_quant_type"] = "nf4"
                    kwargs["bnb_4bit_use_double_quant"] = True
                    kwargs["bnb_4bit_compute_dtype"] = gpu_dtype
                    kwargs["device_map"] = "auto"
                    kwargs["torch_dtype"] = gpu_dtype
                    reason = "QLoRA training" if for_training else (
                        f"{gpu_total_vram_gb():.0f} GB VRAM < {GPU_16BIT_MIN_VRAM_GB:.0f} GB needed for 16-bit"
                        if not os.environ.get("PILOT_GPU_MODE") else "PILOT_GPU_MODE=4bit")
                    description = f"GPU (4-bit NF4, {reason})"
                else:
                    # bitsandbytes not available: load full-precision on GPU instead.
                    # This uses more VRAM but avoids bitsandbytes CUDA detection issues on Windows.
                    log.warning("bitsandbytes not installed; loading full-precision on GPU instead of 4-bit. "
                                "Install bitsandbytes (pip install bitsandbytes) for 4-bit quantization.")
                    kwargs["device_map"] = "auto"
                    kwargs["torch_dtype"] = gpu_dtype
                    reason = "bitsandbytes not available; full-precision fallback"
                    description = f"GPU ({dtype_name}, device_map=auto, {reason}, {gpu_total_vram_gb():.0f} GB VRAM)"
        log.info("Loading %s on %s ...", model_path, description)

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (OSError, RuntimeError) as exc:
        # If GPU loading fails, try falling back to CPU
        if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
            raise

        # Only try CPU fallback if we haven't already set force_cpu
        if not force_cpu and mode != MODE_CPU_BF16:
            log.warning("GPU loading failed (%s); falling back to CPU. This will be slow.", exc)
            try:
                kwargs_cpu = {"low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16}
                model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs_cpu)
                log.info("Model loaded on CPU (bf16) as fallback")
            except Exception as exc_cpu:
                die(f"Failed to load on both GPU and CPU: {exc_cpu}")
        else:
            # Original error handling for when we're already on CPU or force_cpu is set
            if isinstance(exc, OSError):
                die(
                    f"Could not load '{model_path}': {exc}\n"
                    "For a local folder, check the download finished (re-run huggingface-cli download). "
                    "For a hub id, check your internet connection and (for gated models such as "
                    "Llama-2) that you ran `huggingface-cli login` and accepted the licence."
                )
            elif "out of memory" in str(exc).lower():
                hint = " Try PILOT_GPU_MODE=4bit." if mode == MODE_GPU_16BIT else ""
                die(f"Out of memory while loading the model. {memory_report()}{hint}")
            else:
                raise

    if mode == MODE_GPU_16BIT and any(
        str(dev) in ("cpu", "disk") for dev in getattr(model, "hf_device_map", {}).values()
    ):
        log.warning("Part of the model was offloaded to CPU/disk - this is slow. "
                    "Set PILOT_GPU_MODE=4bit to keep the whole model on the GPU.")

    if adapter_dir is not None:
        model = attach_adapter(model, adapter_dir, merge=mode != MODE_GPU_4BIT)

    if not for_training:
        model.eval()
    log.info("Model ready in %.0fs (%s)", time.time() - t0, memory_report())
    return model, tokenizer


def attach_adapter(model, adapter_dir: Path, merge: bool):
    """Attach a saved LoRA adapter; optionally merge it into the base weights."""
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        die(
            f"No LoRA adapter found in {adapter_dir} (missing adapter_config.json). "
            "Run src/train_lora.py first, or copy the adapter/ folder from the GPU machine."
        )
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if merge:
            model = model.merge_and_unload()
            log.info("LoRA adapter merged into base weights.")
        else:
            log.info("LoRA adapter attached (not merged, 4-bit base).")
        return model
    except Exception as exc:  # noqa: BLE001
        die(f"Failed to load adapter from {adapter_dir}: {exc}")


# --------------------------------------------------------------------------- #
# Prompting and generation
# --------------------------------------------------------------------------- #

def build_chat_prompt(tokenizer, user_message: str, system_message: Optional[str] = None) -> str:
    """Render a single-turn chat prompt with the model's own chat template."""
    messages: List[Dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 48,
    timeout_s: float = DEFAULT_QUESTION_TIMEOUT_S,
    system_prompt: str = QA_SYSTEM_PROMPT,
) -> Tuple[str, bool, float]:
    """
    Greedy-decode an answer to `question`.

    Returns (answer, timed_out, seconds). Uses transformers' built-in `max_time`
    stopping criterion, so a slow CPU never spends more than `timeout_s` on one
    question; whatever was generated before the limit is kept and flagged.
    """
    import torch

    prompt = build_chat_prompt(tokenizer, question, system_prompt)
    # The chat template already contains any special tokens, so don't add more;
    # token_type_ids are dropped because causal LMs' generate() rejects them.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy: reproducible before/after comparison
            temperature=None,
            top_p=None,
            top_k=None,
            max_time=timeout_s,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Hitting the time limit before EOS / max tokens => timed out.
    timed_out = elapsed >= timeout_s and len(new_tokens) < max_new_tokens
    return answer, timed_out, elapsed


def run_qa_evaluation(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    out_path: Path,
    label: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    timeout_s: float = DEFAULT_QUESTION_TIMEOUT_S,
    resume: bool = True,
    system_prompt: str = QA_SYSTEM_PROMPT,
    max_new_tokens: int = 48,
    grader: Optional[Callable[[Dict[str, Any], str], Optional[int]]] = None,
) -> Dict[str, Any]:
    """
    Ask every question, grade it, and save results to `out_path`.

    `grader(question, answer)` returns 1/0, or None when the answer can only be
    graded by the blind judge (free-text answers). The default is the
    fill-in-the-blank string match used for the knowledge questions.

    The file is rewritten after every question, so an interrupted CPU run
    (which can take an hour) resumes where it stopped when re-launched.
    """
    from tqdm import tqdm

    existing = load_json(out_path, default={}) if resume else {}
    done: Dict[str, Dict[str, Any]] = {}
    if isinstance(existing, dict) and existing.get("label") == label:
        # Questions that errored last time are asked again.
        # A saved answer is only reused if the question text is identical, so a
        # rebuilt question set never inherits answers to different questions.
        current = {q["id"]: q["question"] for q in questions}
        done = {r["id"]: r for r in existing.get("results", [])
                if "id" in r and not r.get("error") and current.get(r["id"]) == r.get("question")}
        if done:
            log.info("Resuming %s evaluation: %d/%d already answered.", label, len(done), len(questions))

    if grader is None:
        def grader(q: Dict[str, Any], answer: str) -> Optional[int]:
            return int(is_answer_correct(answer, q["answer"], q.get("aliases", [])))

    results: List[Dict[str, Any]] = []
    correct = 0
    bar = tqdm(questions, desc=f"{label} eval", unit="q")
    for q in bar:
        if q["id"] in done:
            record = done[q["id"]]
        else:
            try:
                answer, timed_out, secs = generate_answer(
                    model, tokenizer, q["question"], max_new_tokens=max_new_tokens,
                    timeout_s=timeout_s, system_prompt=system_prompt,
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
                answer, timed_out, secs, error = "", False, 0.0, f"{type(exc).__name__}: {exc}"
                log.warning("Question %s failed: %s", q["id"], error)
            record = {
                "id": q["id"],
                "type": q.get("type", "factual"),
                "question": q["question"],
                "reference_answer": q["answer"],
                "model_answer": answer,
                "is_correct": grader(q, answer) if not error else 0,
                "is_refusal": int(is_refusal(answer)),
                "timed_out": timed_out,
                "seconds": round(secs, 1),
                "error": error,
            }
        results.append(record)
        correct += record["is_correct"] or 0
        bar.set_postfix(correct=f"{correct}/{len(results)}")

        payload = _eval_payload(label, results, len(questions), extra_meta)
        save_json(out_path, payload)

    payload = _eval_payload(label, results, len(questions), extra_meta)
    errors = payload["summary"]["errors"]
    if errors:
        # An errored question scores 0, which would make a broken setup look like
        # "the model doesn't know the book". Never let that pass silently.
        first = next(r["error"] for r in results if r.get("error"))
        die(f"{errors}/{len(questions)} questions failed with errors (first: {first}). "
            f"Results so far are in {out_path.name}; fix the problem and re-run - "
            "errored questions will be retried.")
    return payload


def _eval_payload(
    label: str, results: List[Dict[str, Any]], total: int, extra_meta: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    graded = [r for r in results if r["is_correct"] is not None]
    correct = sum(r["is_correct"] for r in graded)
    return {
        "label": label,
        "model": BASE_MODEL_ID,
        **(extra_meta or {}),
        "summary": {
            "answered": len(results),
            "total": total,
            "correct": correct,
            "graded": len(graded),
            "accuracy": round(correct / max(len(graded), 1), 4),
            # String-matched answers that are wrong but not a refusal. For trap
            # questions this is the string-match hallucination count.
            "wrong_non_refusal": sum(1 for r in graded if not r["is_correct"] and not r["is_refusal"]),
            "needs_judge": len(results) - len(graded),
            "refusals": sum(r["is_refusal"] for r in results),
            "timeouts": sum(1 for r in results if r["timed_out"]),
            "errors": sum(1 for r in results if r.get("error")),
        },
        "results": results,
    }
