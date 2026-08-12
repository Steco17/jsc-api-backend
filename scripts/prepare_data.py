#!/usr/bin/env python3
"""Clean, validate, deduplicate, and safely split parallel translation data.

The important design rule in this module is that related rows are split as a
group instead of independently.  Bible corpora often contain several English
translations for one ``verse_key``.  If those variants are scattered across
train and test, the test set is not genuinely unseen.  A stable hash of the
group identifier assigns every related row to exactly one split.

JSONL rows may provide an optional ``group_id``.  CSV callers may select a
group column with ``--group-col``.  When neither is supplied, a deterministic
identifier is derived from the complete translation pair.
"""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

MAX_CHARS = 1000
MAX_LEN_RATIO = 3.0

# These ranges cover Arabic and Arabic Supplement/Extended characters used by
# the Fulfulde corpus.  Latin accepts ASCII and Latin Extended letters, while
# combining accents are allowed alongside them.
ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")


def normalize(text: str) -> str:
    """Return NFC-normalized text with runs of whitespace collapsed."""

    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def matches_script(text: str, script: str | None) -> bool:
    """Check an optional script constraint without guessing a language.

    ``Latn`` deliberately rejects text containing Arabic characters.
    ``Arab`` requires at least one Arabic character.
    This narrow rule is safer than attempting automatic language detection on
    short Bible verses.
    """

    if script is None:
        return True
    if script == "Arab":
        return bool(ARABIC_RE.search(text))
    return bool(LATIN_RE.search(text)) and not ARABIC_RE.search(text)


def derived_group_id(row: dict, src: str, tgt: str) -> str:
    """Build a direction-independent group identifier for an ungrouped pair."""

    left = (str(row["src_lang"]), src.casefold())
    right = (str(row["tgt_lang"]), tgt.casefold())
    canonical = sorted((left, right))
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"pair:{digest}"


def split_for_group(group_id: str, seed: int, dev_frac: float, test_frac: float) -> str:
    """Assign one group to a reproducible split using a stable hash.

    Python's built-in ``hash`` is randomized between processes, so SHA-256 is
    used here.  The group identifier alone determines the result, which means
    the same ``verse_key`` receives the same split across every language file.
    """

    digest = hashlib.sha256(f"{seed}\0{group_id}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    if score < test_frac:
        return "test"
    if score < test_frac + dev_frac:
        return "dev"
    return "train"


def _validated_languages(row: dict, path: Path, row_number: int) -> tuple[str, str]:
    """Return non-empty language codes or raise an actionable input error."""

    src_lang = row.get("src_lang")
    tgt_lang = row.get("tgt_lang")
    if not isinstance(src_lang, str) or not src_lang.strip():
        raise ValueError(f"{path}:{row_number}: missing src_lang")
    if not isinstance(tgt_lang, str) or not tgt_lang.strip():
        raise ValueError(f"{path}:{row_number}: missing tgt_lang")
    return src_lang.strip(), tgt_lang.strip()


def load_rows(
    paths: list[str],
    *,
    tsv: bool,
    csv_mode: bool,
    src_col: str,
    tgt_col: str,
    group_col: str | None,
    src_lang: str | None,
    tgt_lang: str | None,
):
    """Yield normalized input dictionaries from JSONL, TSV, or CSV files.

    Parsing errors are reported instead of silently discarded because a
    partially read corpus can make a long GPU run look successful while it is
    actually training on incomplete data.
    """

    for raw_path in paths:
        path = Path(raw_path)
        if csv_mode:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                missing_columns = {src_col, tgt_col} - fields
                if missing_columns:
                    names = ", ".join(sorted(missing_columns))
                    raise ValueError(f"{path}: missing CSV column(s): {names}")
                if group_col and group_col not in fields:
                    raise ValueError(f"{path}: missing group column '{group_col}'")

                for row_number, csv_row in enumerate(reader, start=2):
                    row = {
                        "src": csv_row.get(src_col),
                        "tgt": csv_row.get(tgt_col),
                        "src_lang": csv_row.get("src_lang") or src_lang,
                        "tgt_lang": csv_row.get("tgt_lang") or tgt_lang,
                        "group_id": csv_row.get(group_col) if group_col else None,
                    }
                    row["src_lang"], row["tgt_lang"] = _validated_languages(row, path, row_number)
                    yield row
            continue

        with path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                if tsv:
                    parts = line.split("\t")
                    if len(parts) < 2:
                        raise ValueError(f"{path}:{row_number}: expected source<TAB>target")
                    row = {
                        "src": parts[0],
                        "tgt": parts[1],
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "group_id": None,
                    }
                else:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{row_number}: invalid JSON: {exc.msg}") from exc
                row["src_lang"], row["tgt_lang"] = _validated_languages(row, path, row_number)
                yield row


def clean_rows(rows, src_script: str | None, tgt_script: str | None):
    """Clean rows and return both accepted rows and reasoned drop counts."""

    seen: set[tuple[str, str, str, str]] = set()
    accepted: list[dict] = []
    dropped: Counter[str] = Counter()

    for row in rows:
        src = normalize(str(row.get("src") or ""))
        tgt = normalize(str(row.get("tgt") or ""))

        reason = None
        if not src or not tgt:
            reason = "empty"
        elif len(src) > MAX_CHARS or len(tgt) > MAX_CHARS:
            reason = "too_long"
        elif max(len(src), len(tgt)) / max(1, min(len(src), len(tgt))) > MAX_LEN_RATIO:
            reason = "length_ratio"
        elif src.casefold() == tgt.casefold():
            reason = "untranslated"
        elif not matches_script(src, src_script):
            reason = "source_script"
        elif not matches_script(tgt, tgt_script):
            reason = "target_script"

        if reason:
            dropped[reason] += 1
            continue

        key = (
            src.casefold(),
            tgt.casefold(),
            row["src_lang"],
            row["tgt_lang"],
        )
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        explicit_group = normalize(str(row.get("group_id") or ""))
        group_id = explicit_group or derived_group_id(row, src, tgt)
        accepted.append(
            {
                "src": src,
                "tgt": tgt,
                "src_lang": row["src_lang"],
                "tgt_lang": row["tgt_lang"],
                "group_id": group_id,
            }
        )

    return accepted, dropped


def reverse_row(row: dict) -> dict:
    """Create the opposite translation direction while retaining group identity."""

    return {
        "src": row["tgt"],
        "tgt": row["src"],
        "src_lang": row["tgt_lang"],
        "tgt_lang": row["src_lang"],
        "group_id": row["group_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="input JSONL, TSV, or CSV files")
    parser.add_argument("-o", "--out-dir", default="data/prepared")
    parser.add_argument("--tsv", action="store_true", help="inputs are TSV files")
    parser.add_argument("--csv", action="store_true", help="inputs are CSV files")
    parser.add_argument("--src-col", default="src", help="CSV source column")
    parser.add_argument("--tgt-col", default="tgt", help="CSV target column")
    parser.add_argument(
        "--group-col",
        help="CSV column identifying related rows, for example verse_key",
    )
    parser.add_argument("--src-lang", help="default source language code")
    parser.add_argument("--tgt-lang", help="default target language code")
    parser.add_argument("--src-script", choices=("Latn", "Arab"))
    parser.add_argument("--tgt-script", choices=("Latn", "Arab"))
    parser.add_argument("--dev-frac", type=float, default=0.025)
    parser.add_argument("--test-frac", type=float, default=0.025)
    parser.add_argument(
        "--add-reverse",
        action="store_true",
        help="emit both directions inside the same leakage-safe split",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tsv and args.csv:
        parser.error("--tsv and --csv are mutually exclusive")
    if args.tsv and not (args.src_lang and args.tgt_lang):
        parser.error("TSV input requires --src-lang and --tgt-lang")
    if not 0 <= args.dev_frac < 1 or not 0 <= args.test_frac < 1:
        parser.error("split fractions must be between 0 and 1")
    if args.dev_frac + args.test_frac >= 1:
        parser.error("--dev-frac and --test-frac must sum to less than 1")

    try:
        loaded = load_rows(
            args.inputs,
            tsv=args.tsv,
            csv_mode=args.csv,
            src_col=args.src_col,
            tgt_col=args.tgt_col,
            group_col=args.group_col,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )
        rows, dropped = clean_rows(loaded, args.src_script, args.tgt_script)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        split = split_for_group(row["group_id"], args.seed, args.dev_frac, args.test_frac)
        splits[split].append(row)

    if args.add_reverse:
        for split_rows in splits.values():
            existing = {
                (row["src"].casefold(), row["tgt"].casefold(), row["src_lang"], row["tgt_lang"])
                for row in split_rows
            }
            for row in list(split_rows):
                reversed_row = reverse_row(row)
                key = (
                    reversed_row["src"].casefold(),
                    reversed_row["tgt"].casefold(),
                    reversed_row["src_lang"],
                    reversed_row["tgt_lang"],
                )
                if key not in existing:
                    split_rows.append(reversed_row)
                    existing.add(key)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        split_rows = splits[name]
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        groups = len({row["group_id"] for row in split_rows})
        print(f"{name}: {len(split_rows)} rows in {groups} groups")

    print(f"accepted: {len(rows)} source rows")
    print(f"dropped: {sum(dropped.values())} {dict(sorted(dropped.items()))}")


if __name__ == "__main__":
    main()
