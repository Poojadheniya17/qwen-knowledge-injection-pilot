"""
Step 1b - Remove junk units from an existing data/units.json (and optionally top up).

Applies the same junk rules that prepare_data.py now uses during extraction,
but to a units.json that was already built, so unit ids stay stable (the
practical questions reference units by id).

Per unit:
  * DROPPED  - Project Gutenberg / transcriber notes, figure or plate fragments
               ("18, 11, 13, and 20 in Plate X."), short footnote citations
               ("[100] Vide Adventurer, No. 39.")
  * CLEANED  - footnote markers such as "[62]" are removed from all text
  * RE-KEYED - a junk answer term (footnote number, bare figure number, roman
               numeral, "Thus", "Plate" ...) is replaced by the best valid term
               in the fact; if there is none the unit stays for training but
               gets key_term = null, so no test question is built from it

Delete data/units_raw.json after re-running prepare_data.py, so the new
extraction becomes the original that is cleaned.

With --book, new clean units are extracted from the book until there are at
least --min-units, with new ids after the highest existing one.

Outputs:
    data/units.json                   cleaned units (overwritten)
    data/units_raw.json               backup of the original (written on the first run;
                                      later runs always clean from it, so re-running
                                      gives the same result)
    data/units_cleaning_report.json   what was dropped / re-keyed and why

Usage:
    python src/clean_units.py                          # clean only
    python src/clean_units.py --book data/book_text.txt  # clean + top up to 300
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from common import BOOK_META_PATH, DATA_DIR, UNITS_PATH, die, get_logger, load_json, save_json
from prepare_data import (
    EDITION_NOTE_RE,
    FIGURE_FRAGMENT_RE,
    MIN_UNITS,
    Unit,
    build_term_frequency,
    choose_key_term,
    extract_candidates,
    is_footnote_citation,
    read_book,
    select_units,
    split_chapters,
    split_paragraphs,
    strip_boilerplate,
    strip_footnote_markers,
    term_candidates,
)

log = get_logger("clean_units")

RAW_BACKUP_PATH = DATA_DIR / "units_raw.json"
REPORT_PATH = DATA_DIR / "units_cleaning_report.json"
FIELDS = ("fact", "definition", "example")


def drop_reason(unit: Dict[str, Any]) -> Optional[str]:
    """Why a unit is junk, or None if its content is usable."""
    text = " ".join(unit[f] for f in FIELDS)
    if EDITION_NOTE_RE.search(text):
        return "editorial_note"
    if is_footnote_citation(unit["fact"]):
        return "footnote_citation"
    if FIGURE_FRAGMENT_RE.match(strip_footnote_markers(unit["fact"])):
        return "figure_fragment"
    return None


def valid_term(term: Optional[str], sentence: str) -> bool:
    """A term is valid if the (new, stricter) candidate rules would pick it."""
    return bool(term) and term in term_candidates(sentence)


def clean_unit(unit: Dict[str, Any], freq: Counter) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return (cleaned unit or None, report entry)."""
    entry: Dict[str, Any] = {"id": unit["id"], "old_key_term": unit.get("key_term")}
    reason = drop_reason(unit)
    if reason:
        entry.update(action="dropped", reason=reason, fact=unit["fact"][:160])
        return None, entry

    cleaned = {f: strip_footnote_markers(unit[f]) for f in FIELDS}
    key = unit.get("key_term")
    if not valid_term(key, cleaned["fact"]):
        key = choose_key_term(cleaned["fact"], freq)
    ex_key = unit.get("example_key_term")
    if not valid_term(ex_key, cleaned["example"]):
        ex_key = choose_key_term(cleaned["example"], freq)

    try:
        new = Unit(id=unit["id"], link=unit["link"], key_term=key, example_key_term=ex_key, **cleaned).model_dump()
    except ValidationError as exc:
        entry.update(action="dropped", reason="validation", detail=str(exc).splitlines()[0])
        return None, entry

    if key != unit.get("key_term"):
        entry.update(action="rekeyed" if key else "training_only", new_key_term=key)
    elif any(cleaned[f] != unit[f] for f in FIELDS):
        entry.update(action="markers_removed")
    else:
        entry.update(action="unchanged")
    return new, entry


def top_up(units: List[Dict[str, Any]], book: Path, min_units: int, chars_per_page: int,
           first_new_id: int) -> List[Dict[str, Any]]:
    """Extract additional clean units from the book until there are min_units."""
    needed = min_units - len(units)
    if needed <= 0:
        return []
    text = strip_boilerplate(read_book(book))
    form_feeds = [m.start() for m in re.finditer("\f", text)]
    paragraphs = []
    for chapter_no, offset, body in split_chapters(text):
        paragraphs.extend(split_paragraphs(chapter_no, offset, body))
    freq = build_term_frequency(paragraphs)
    candidates = extract_candidates(paragraphs, freq, form_feeds, chars_per_page)

    existing_facts = {u["fact"] for u in units}
    term_uses = Counter(u["key_term"] for u in units if u["key_term"])
    fresh = [c for c in candidates if c.fact not in existing_facts and term_uses[c.key_term] < 2]
    picked = select_units(fresh, needed)

    next_id = first_new_id
    added: List[Dict[str, Any]] = []
    for cand in picked:
        try:
            added.append(Unit(
                id=f"u{next_id:04d}", fact=cand.fact, definition=cand.definition, example=cand.example,
                link=f"{cand.chapter}:{cand.page}", key_term=cand.key_term,
                example_key_term=cand.example_key_term,
            ).model_dump())
            next_id += 1
        except ValidationError:
            continue
    if len(units) + len(added) < min_units:
        log.warning("Could only reach %d units (wanted %d).", len(units) + len(added), min_units)
    return added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove junk units from data/units.json.")
    parser.add_argument("--book", type=Path, default=None,
                        help="Book text; if given, top up with new clean units to --min-units.")
    parser.add_argument("--min-units", type=int, default=MIN_UNITS)
    parser.add_argument("--chars-per-page", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Always clean from the untouched original, so re-running is idempotent.
    if RAW_BACKUP_PATH.exists():
        units = load_json(RAW_BACKUP_PATH, required=True)
        log.info("Cleaning from the original backup %s (%d units).", RAW_BACKUP_PATH.name, len(units))
    else:
        units = load_json(UNITS_PATH, default=[])
        if not units:
            die("data/units.json is missing or empty. Run: python src/prepare_data.py")
        save_json(RAW_BACKUP_PATH, units)
        log.info("Backed up original units to %s", RAW_BACKUP_PATH.name)

    # Term frequency over all unit text approximates "how book-specific" a term is.
    freq: Counter = Counter()
    for u in units:
        for f in FIELDS:
            freq.update(term_candidates(strip_footnote_markers(u[f])))

    cleaned: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    for unit in units:
        new, entry = clean_unit(unit, freq)
        report.append(entry)
        if new:
            cleaned.append(new)

    added: List[Dict[str, Any]] = []
    if args.book:
        meta = load_json(BOOK_META_PATH, default={}) or {}
        cpp = args.chars_per_page
        if str(meta.get("page_numbering", "")).startswith("estimated_"):
            cpp = int(re.sub(r"\D", "", meta["page_numbering"].split("_")[1]) or cpp)
        # New ids continue after the highest ORIGINAL id, so they never reuse the
        # id of a dropped unit (which would make the cleaning report ambiguous).
        first_new_id = max(int(u["id"][1:]) for u in units) + 1
        added = top_up(cleaned, args.book, args.min_units, cpp, first_new_id)
        cleaned.extend(added)

    save_json(UNITS_PATH, cleaned)
    actions = Counter(e["action"] for e in report)
    reasons = Counter(e.get("reason") for e in report if e["action"] == "dropped")
    summary = {
        "units_before": len(units),
        "units_after": len(cleaned),
        "question_eligible": sum(1 for u in cleaned if u["key_term"]),
        "added_from_book": len(added),
        "actions": dict(actions),
        "drop_reasons": dict(reasons),
    }
    save_json(REPORT_PATH, {"summary": summary, "units": report})

    log.info("Units: %d -> %d (%d dropped, %d added from book)",
             len(units), len(cleaned), actions["dropped"], len(added))
    log.info("Dropped by reason: %s", dict(reasons))
    log.info("Re-keyed: %d, training-only (no answer term): %d, markers removed: %d",
             actions["rekeyed"], actions["training_only"], actions["markers_removed"])
    if len(cleaned) < args.min_units and not args.book:
        log.warning("%d units is below the %d target. Re-run with --book data/book_text.txt to top up "
                    "with new clean units (existing ids are kept).", len(cleaned), args.min_units)
    log.info("Report: %s. Next: python src/evaluate_baseline.py --regenerate-questions", REPORT_PATH.name)


if __name__ == "__main__":
    main()
