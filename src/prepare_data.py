"""
Step 1 - Break the book into 300-500 validated "fact units".

Input : data/book_text.txt   (plain text, e.g. a Project Gutenberg .txt file)
Output: data/units.json      (JSON array of unit objects)
        data/book_meta.json  (title, chapter count, page-numbering method)

Each unit has the four required components plus two helper fields:

    {
      "id": "u0001",
      "fact":       "<a sentence from the book containing a specific detail>",
      "definition": "<a nearby sentence that defines / describes the context>",
      "example":    "<the sentence that follows and illustrates it>",
      "link":       "<chapter>:<page>",
      "key_term":   "<the specific name/number in `fact` that questions ask for>",
      "example_key_term": "<same for `example`, or null>"
    }

The extraction is heuristic (no LLM needed, runs in seconds on CPU):
  1. strip Project Gutenberg boilerplate and detect chapter headings
  2. split chapters into paragraphs and sentences
  3. keep sentences that contain a rare, specific term (a proper noun or a
     number) - these are facts a model cannot guess without reading the book
  4. attach a definition sentence (prefers "X is/was/called/known as ...")
     and an example sentence from the surrounding text
  5. spread the selection evenly across the whole book and validate it

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --book data/book_text.txt --title "The Book" --target 400
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError, field_validator
from tqdm import tqdm

from common import BOOK_META_PATH, BOOK_TEXT_PATH, PROJECT_ROOT, UNITS_PATH, die, get_logger, save_json

log = get_logger("prepare_data")

MIN_UNITS = 300
MAX_UNITS = 500

# --------------------------------------------------------------------------- #
# Regular expressions
# --------------------------------------------------------------------------- #

# "CHAPTER IV", "Chapter 12. The Storm", "BOOK II", "PART ONE" ...
CHAPTER_RE = re.compile(
    r"^[ \t]*(?:CHAPTER|Chapter|BOOK|Book|PART|Part|LETTER|Letter)[ \t]+"
    r"(?:[IVXLCDM]+|\d+|[A-Z][a-z]+(?:-[a-z]+)?|[A-Z]+)\b[^\n]{0,80}$",
    re.MULTILINE,
)
# Fallback: a bare roman numeral on its own line ("IV." / "XII").
ROMAN_HEADING_RE = re.compile(r"^[ \t]*[IVXLC]{1,7}\.?[ \t]*$", re.MULTILINE)

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
GUTENBERG_TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.M)

# Abbreviations whose trailing dot must not end a sentence.
ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "Capt", "Col", "Gen", "Lt", "Rev", "Prof", "No", "Mt", "vs", "etc")
_ABBR_RE = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\.")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"”’')\]]*\s+(?=[\"“‘'(\[]?[A-Z0-9])")

# A sentence that *describes* something - preferred as the unit's definition.
DEFINITION_RE = re.compile(
    r"\b(?:is|was|are|were)\s+(?:a|an|the|one of)\b|\bmeans?\b|\bcalled\b|\bnamed\b|"
    r"\bknown as\b|\bconsist(?:s|ed)? of\b|\bdescribed as\b|\bkind of\b|\bsort of\b",
    re.I,
)

PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+(?:of\s+|de\s+|la\s+|von\s+|van\s+)?[A-Z][a-zA-Z'’\-]+)*")
NUMBER_RE = re.compile(r"\b\d{1,4}(?:[,.]\d{1,3})?\b")

# Capitalised words that are not informative answers.
STOP_TERMS = {
    "i", "i'm", "i'll", "i've", "i'd", "the", "a", "an", "he", "she", "it", "we", "they", "you", "his", "her",
    "my", "our", "your", "their", "this", "that", "these", "those", "there", "here", "then", "but", "and", "or",
    "so", "yet", "if", "when", "while", "what", "who", "whom", "which", "why", "how", "where", "oh", "ah", "yes",
    "no", "not", "mr", "mrs", "ms", "dr", "sir", "madam", "miss", "lord", "lady", "god", "chapter", "one", "two",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "english", "french",
    "however", "now", "well", "after", "before", "as", "at", "in", "on", "for", "with", "all", "some", "every",
    "o", "ay", "aye", "nay", "let", "do", "did", "had", "has", "have", "is", "was", "be", "upon", "said",
}


# --------------------------------------------------------------------------- #
# Validation model
# --------------------------------------------------------------------------- #

class Unit(BaseModel):
    """Schema every unit must satisfy before it is written to units.json."""

    id: str
    fact: str
    definition: str
    example: str
    link: str
    key_term: str
    example_key_term: Optional[str] = None

    @field_validator("fact", "definition", "example")
    @classmethod
    def _non_trivial(cls, value: str) -> str:
        value = value.strip()
        if len(value.split()) < 4:
            raise ValueError("must contain at least 4 words")
        return value

    @field_validator("link")
    @classmethod
    def _link_format(cls, value: str) -> str:
        if not re.fullmatch(r"\d+:\d+", value):
            raise ValueError("link must look like '<chapter>:<page>'")
        return value

    def model_post_init(self, __context) -> None:  # noqa: D401 - pydantic hook
        if self.key_term not in self.fact:
            raise ValueError("key_term must appear in fact")
        if len({self.fact, self.definition, self.example}) != 3:
            raise ValueError("fact, definition and example must be distinct sentences")


# --------------------------------------------------------------------------- #
# Internal structures
# --------------------------------------------------------------------------- #

@dataclass
class Paragraph:
    chapter: int
    offset: int                      # character offset in the cleaned book text
    sentences: List[str] = field(default_factory=list)


@dataclass
class Candidate:
    chapter: int
    page: int
    position: int                    # global sentence index, used for even spreading
    fact: str
    definition: str
    example: str
    key_term: str
    example_key_term: Optional[str]
    score: float


# --------------------------------------------------------------------------- #
# Reading and cleaning
# --------------------------------------------------------------------------- #

def read_book(path: Path) -> str:
    """Read the book with an encoding fallback chain (Gutenberg files vary)."""
    if not path.exists():
        die(
            f"Book file not found: {path}\n"
            "Download a plain-text public-domain book (see README > 'Choose and download the book') "
            "and save it as data/book_text.txt"
        )
    raw = path.read_bytes()
    if not raw.strip():
        die(f"{path} is empty - paste the book text into it first.")
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            if encoding != "utf-8-sig":
                log.warning("Book is not UTF-8; decoded as %s.", encoding)
            break
        except UnicodeDecodeError:
            continue
    # Normalise line endings; keep form feeds (\f) - they mark real page breaks.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def detect_title(text: str) -> Optional[str]:
    match = GUTENBERG_TITLE_RE.search(text[:5000])
    return match.group(1).strip() if match else None


def strip_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg licence header/footer if present."""
    start = GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def split_chapters(text: str) -> List[Tuple[int, int, str]]:
    """
    Return [(chapter_number, start_offset, chapter_body), ...].

    Headings followed by < 1000 characters are dropped: those are almost
    always table-of-contents entries, not real chapters. Chapters are then
    numbered 1..N in reading order.
    """
    headings = list(CHAPTER_RE.finditer(text))
    if len(headings) < 3:
        roman = list(ROMAN_HEADING_RE.finditer(text))
        if len(roman) >= 3:
            log.info("No 'CHAPTER' headings found; using roman-numeral headings instead.")
            headings = roman
    if len(headings) < 2:
        log.warning("No chapter headings detected - treating the whole book as chapter 1.")
        return [(1, 0, text)]

    chapters: List[Tuple[int, int, str]] = []
    for i, match in enumerate(headings):
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end]
        if len(body.strip()) >= 1000:
            chapters.append((len(chapters) + 1, start, body))

    front = text[: headings[0].start()]
    if len(front.strip()) > 5000:
        # A long preface before chapter 1 is still book content: keep it as chapter 0.
        chapters.insert(0, (0, 0, front))
    if not chapters:
        log.warning("Chapter headings found but all chapters were tiny; using the whole book as chapter 1.")
        return [(1, 0, text)]
    return chapters


def split_sentences(paragraph: str) -> List[str]:
    protected = _ABBR_RE.sub(lambda m: m.group(1) + "<DOT>", paragraph)
    parts = SENTENCE_SPLIT_RE.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def split_paragraphs(chapter_no: int, chapter_offset: int, body: str) -> List[Paragraph]:
    paragraphs: List[Paragraph] = []
    for match in re.finditer(r"(?:[^\n]+\n?)+", body):
        block = match.group(0)
        joined = re.sub(r"\s*\n\s*", " ", block).strip()
        # Skip headings, illustrations and other non-prose lines.
        if len(joined.split()) < 6 or joined.startswith("[Illustration"):
            continue
        para = Paragraph(chapter=chapter_no, offset=chapter_offset + match.start())
        para.sentences = split_sentences(joined)
        paragraphs.append(para)
    return paragraphs


# --------------------------------------------------------------------------- #
# Fact extraction
# --------------------------------------------------------------------------- #

def term_candidates(sentence: str) -> List[str]:
    """Proper-noun phrases and numbers that could be the 'answer' of a question."""
    terms: List[str] = []
    for match in PROPER_NOUN_RE.finditer(sentence):
        term = match.group(0).strip("'’-")
        words = term.split()
        # Drop leading stop-words ("The Captain" -> "Captain"; "But Ahab" -> "Ahab").
        while words and words[0].lower().strip("'’") in STOP_TERMS:
            words = words[1:]
        term = " ".join(words)
        if not term or len(term) < 3 or term.lower() in STOP_TERMS:
            continue
        # A single capitalised word at the very start of a sentence is usually
        # just capitalisation, not a name.
        if match.start() == 0 and len(words) == 1 and term == match.group(0):
            continue
        terms.append(term)
    terms.extend(m.group(0) for m in NUMBER_RE.finditer(sentence))
    # The term must occur exactly once so the fill-in-the-blank is unambiguous.
    return [t for t in dict.fromkeys(terms) if sentence.count(t) == 1]


def choose_key_term(sentence: str, freq: Counter) -> Optional[str]:
    """Pick the rarest (most book-specific) term; ties go to the longer term."""
    candidates = term_candidates(sentence)
    if not candidates:
        return None
    return min(candidates, key=lambda t: (freq.get(t, 0), -len(t)))


def build_term_frequency(paragraphs: List[Paragraph]) -> Counter:
    freq: Counter = Counter()
    for para in paragraphs:
        for sentence in para.sentences:
            freq.update(term_candidates(sentence))
    return freq


def page_for_offset(offset: int, form_feed_offsets: List[int], chars_per_page: int) -> int:
    """Real page number if the text has form feeds, otherwise an estimate."""
    if form_feed_offsets:
        return sum(1 for ff in form_feed_offsets if ff < offset) + 1
    return offset // chars_per_page + 1


def is_good_fact_sentence(sentence: str) -> bool:
    words = sentence.split()
    if not 8 <= len(words) <= 60:
        return False
    # Mostly-dialogue sentences are opinions, not facts.
    quoted = sum(len(q) for q in re.findall(r"[\"“][^\"”]*[\"”]", sentence))
    return quoted <= 0.6 * len(sentence)


def extract_candidates(
    paragraphs: List[Paragraph], freq: Counter, form_feeds: List[int], chars_per_page: int
) -> List[Candidate]:
    candidates: List[Candidate] = []
    position = 0
    skipped_errors = 0

    for p_idx, para in enumerate(tqdm(paragraphs, desc="Extracting facts", unit="para")):
        for s_idx, sentence in enumerate(para.sentences):
            position += 1
            try:
                if not is_good_fact_sentence(sentence):
                    continue
                key_term = choose_key_term(sentence, freq)
                if key_term is None:
                    continue

                others = [s for i, s in enumerate(para.sentences) if i != s_idx and len(s.split()) >= 5]

                # Definition: a descriptive sentence in the same paragraph, else the
                # previous sentence, else the end of the previous paragraph.
                definition = next((s for s in others if DEFINITION_RE.search(s)), None)
                has_true_definition = definition is not None
                if definition is None and s_idx > 0:
                    definition = para.sentences[s_idx - 1]
                if definition is None and p_idx > 0 and paragraphs[p_idx - 1].chapter == para.chapter:
                    definition = paragraphs[p_idx - 1].sentences[-1]

                # Example: the next sentence, else the first one of the next paragraph.
                example = para.sentences[s_idx + 1] if s_idx + 1 < len(para.sentences) else None
                if example is None and p_idx + 1 < len(paragraphs) and paragraphs[p_idx + 1].chapter == para.chapter:
                    example = paragraphs[p_idx + 1].sentences[0]
                if example == definition:
                    example = next((s for s in others if s not in (definition, sentence)), None)

                if not definition or not example or len({sentence, definition, example}) < 3:
                    continue

                # Rare terms and real definitions make better test material.
                score = 1.0 / (1 + freq.get(key_term, 0)) + (0.5 if has_true_definition else 0.0)
                candidates.append(
                    Candidate(
                        chapter=para.chapter,
                        page=page_for_offset(para.offset, form_feeds, chars_per_page),
                        position=position,
                        fact=sentence,
                        definition=definition,
                        example=example,
                        key_term=key_term,
                        example_key_term=choose_key_term(example, freq),
                        score=score,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - malformed text should not stop the run
                skipped_errors += 1
                if skipped_errors <= 5:
                    log.warning("Skipped malformed sentence in chapter %s: %s", para.chapter, exc)
    if skipped_errors:
        log.warning("Skipped %d malformed sentences in total.", skipped_errors)
    return candidates


def select_units(candidates: List[Candidate], target: int, max_per_term: int = 2) -> List[Candidate]:
    """
    Pick `target` candidates spread evenly across the book.

    * the best-scoring 2x target candidates form the pool
    * each key term is used at most `max_per_term` times, so the trained model
      cannot score well by always answering the main character's name
    * the pool is then sampled at even intervals in reading order
    """
    pool = sorted(candidates, key=lambda c: -c.score)
    term_uses: Counter = Counter()
    used_sentences = set()
    filtered: List[Candidate] = []
    for cand in pool:
        if term_uses[cand.key_term] >= max_per_term or cand.fact in used_sentences:
            continue
        term_uses[cand.key_term] += 1
        used_sentences.add(cand.fact)
        filtered.append(cand)
        if len(filtered) >= target * 2:
            break

    filtered.sort(key=lambda c: c.position)
    if len(filtered) <= target:
        return filtered
    step = len(filtered) / target
    return [filtered[int(i * step)] for i in range(target)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a book into validated fact units.")
    parser.add_argument("--book", type=Path, default=BOOK_TEXT_PATH, help="Path to the plain-text book.")
    parser.add_argument("--title", type=str, default=None, help="Book title (auto-detected for Gutenberg files).")
    parser.add_argument("--target", type=int, default=400, help=f"Units to produce ({MIN_UNITS}-{MAX_UNITS}).")
    parser.add_argument("--chars-per-page", type=int, default=1800,
                        help="Used to estimate page numbers when the text has no form-feed page breaks.")
    parser.add_argument("--out", type=Path, default=UNITS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MIN_UNITS <= args.target <= MAX_UNITS:
        die(f"--target must be between {MIN_UNITS} and {MAX_UNITS} (got {args.target}).")

    # 1. Read and clean ------------------------------------------------------
    raw = read_book(args.book)
    title = args.title or detect_title(raw) or args.book.stem.replace("_", " ").title()
    text = strip_boilerplate(raw)
    log.info("Book: '%s' - %s characters after cleaning.", title, f"{len(text):,}")
    if len(text) < 50_000:
        die(f"The book is only {len(text):,} characters; at least ~50,000 are needed for {MIN_UNITS}+ units.")

    form_feeds = [m.start() for m in re.finditer("\f", text)]
    page_method = "form_feed" if form_feeds else f"estimated_{args.chars_per_page}_chars_per_page"
    log.info("Page numbers: %s.", "real page breaks (form feeds)" if form_feeds else "estimated from character offset")

    # 2. Chapters -> paragraphs -> sentences ----------------------------------
    chapters = split_chapters(text)
    log.info("Detected %d chapters.", len(chapters))
    paragraphs: List[Paragraph] = []
    for chapter_no, offset, body in tqdm(chapters, desc="Splitting chapters", unit="ch"):
        paragraphs.extend(split_paragraphs(chapter_no, offset, body))
    n_sentences = sum(len(p.sentences) for p in paragraphs)
    log.info("Found %d paragraphs / %d sentences.", len(paragraphs), n_sentences)
    if not paragraphs:
        die("No prose paragraphs found. Is the file really plain text (not HTML/PDF)?")

    # 3. Candidate facts -------------------------------------------------------
    freq = build_term_frequency(paragraphs)
    candidates = extract_candidates(paragraphs, freq, form_feeds, args.chars_per_page)
    log.info("Extracted %d candidate facts.", len(candidates))

    # 4. Select + validate ----------------------------------------------------
    selected = select_units(candidates, args.target)
    units: List[Dict] = []
    rejected = 0
    for cand in tqdm(selected, desc="Validating units", unit="unit"):
        try:
            unit = Unit(
                id=f"u{len(units) + 1:04d}",
                fact=cand.fact,
                definition=cand.definition,
                example=cand.example,
                link=f"{cand.chapter}:{cand.page}",
                key_term=cand.key_term,
                example_key_term=cand.example_key_term,
            )
            units.append(unit.model_dump())
        except ValidationError as exc:
            rejected += 1
            log.debug("Rejected unit: %s", exc)
    if rejected:
        log.warning("%d units failed validation and were dropped.", rejected)

    if len(units) < MIN_UNITS:
        die(
            f"Only {len(units)} valid units could be built (need {MIN_UNITS}). The book is probably too short "
            "or mostly dialogue. Choose a longer, more factual book, or lower --chars-per-page if pages look wrong."
        )

    # 5. Save ------------------------------------------------------------------
    save_json(args.out, units)
    meta = {
        "title": title,
        "source_file": str(args.book.relative_to(PROJECT_ROOT)) if args.book.is_relative_to(PROJECT_ROOT) else str(args.book),
        "characters": len(text),
        "chapters": len(chapters),
        "sentences": n_sentences,
        "candidates": len(candidates),
        "units": len(units),
        "page_numbering": page_method,
    }
    save_json(BOOK_META_PATH, meta)

    with_examples = sum(1 for u in units if u["example_key_term"])
    log.info("Saved %d units to %s", len(units), args.out.relative_to(PROJECT_ROOT) if args.out.is_relative_to(PROJECT_ROOT) else args.out)
    log.info("Units usable for practical (example-based) questions: %d", with_examples)
    log.info("Sample unit:\n  fact: %s\n  key_term: %s\n  link: %s", units[0]["fact"], units[0]["key_term"], units[0]["link"])


if __name__ == "__main__":
    main()
