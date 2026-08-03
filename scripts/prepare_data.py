#!/usr/bin/env python3
"""Data preparation: clean, dedupe and split parallel data into train/dev/test.

WHY THIS SCRIPT EXISTS
----------------------
Machine translation models are only as good as their data. Raw parallel
corpora typically contain duplicates, empty lines, misaligned pairs and
inconsistent unicode. Training on dirty data wastes GPU time and hurts
quality, so we clean BEFORE training.

INPUT FORMATS
-------------
1. JSONL (default) - one JSON object per line:
     {"src": "Bonjour", "tgt": "Mbolo", "src_lang": "fra_Latn", "tgt_lang": "ewo_Latn"}
2. TSV (--tsv flag) - source<TAB>target per line. Because TSV rows carry no
   language info, you must pass --src-lang and --tgt-lang.
3. CSV (--csv flag) - comma-separated with a header row. Looks for "src" and
   "tgt" columns by default (override with --src-col/--tgt-col). If the file
   also has "src_lang"/"tgt_lang" columns those are used per-row; otherwise
   pass --src-lang and --tgt-lang to apply the same pair to every row.

OUTPUT
------
<out-dir>/train.jsonl, dev.jsonl, test.jsonl - all in the JSONL format above,
ready to be consumed by scripts/finetune.py.

EXAMPLES
--------
  python prepare_data.py data/raw1.jsonl data/raw2.jsonl -o data/prepared --add-reverse
  python prepare_data.py data/fr_ewo.tsv --tsv --src-lang fra_Latn --tgt-lang ewo_Latn -o data/prepared
  python prepare_data.py data/fr_ewo.csv --csv --src-lang fra_Latn --tgt-lang ewo_Latn -o data/prepared
  python prepare_data.py data/pairs.csv --csv --src-col french --tgt-col ewondo \
      --src-lang fra_Latn --tgt-lang ewo_Latn -o data/prepared
"""
import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Cleaning thresholds - tune these if your data is unusual.
# ---------------------------------------------------------------------------
MAX_CHARS = 1000      # drop sentences longer than this (likely not sentences)
MAX_LEN_RATIO = 3.0   # drop pairs where one side is >3x longer than the other
                      # (a strong signal the pair is misaligned)


def normalize(text: str) -> str:
    """Normalize a sentence so duplicates can be detected reliably.

    - NFC unicode normalization: 'e' + combining accent becomes a single 'é'
      codepoint, so visually identical strings compare equal.
    - Collapse all runs of whitespace (tabs, newlines, doubles) to one space.
    - Strip leading/trailing whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_rows(paths, tsv, csv_mode, src_col, tgt_col, src_lang, tgt_lang):
    """Generator that yields raw {"src","tgt","src_lang","tgt_lang"} dicts
    from every input file, handling JSONL, TSV and CSV formats.

    Malformed lines are silently skipped - counting them as 'dropped'
    happens in main() only for rows that parse but fail quality checks.
    """
    for p in paths:
        if csv_mode:
            with open(p, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                has_lang_cols = reader.fieldnames and (
                    "src_lang" in reader.fieldnames or "tgt_lang" in reader.fieldnames)
                if not has_lang_cols and not (src_lang and tgt_lang):
                    raise SystemExit(
                        f"{p}: no src_lang/tgt_lang columns found - pass "
                        "--src-lang and --tgt-lang")
                for row in reader:
                    src = row.get(src_col)
                    tgt = row.get(tgt_col)
                    if src is None or tgt is None:
                        continue           # column missing from this row
                    yield {"src": src, "tgt": tgt,
                           "src_lang": row.get("src_lang") or src_lang,
                           "tgt_lang": row.get("tgt_lang") or tgt_lang}
            continue

        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if tsv:
                    parts = line.split("\t")
                    if len(parts) < 2:      # not a valid src/tgt pair
                        continue
                    yield {"src": parts[0], "tgt": parts[1],
                           "src_lang": src_lang, "tgt_lang": tgt_lang}
                else:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue            # skip corrupt JSON lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="input JSONL or TSV files")
    ap.add_argument("-o", "--out-dir", default="data/prepared")
    ap.add_argument("--tsv", action="store_true", help="inputs are TSV files")
    ap.add_argument("--csv", action="store_true",
                    help="inputs are CSV files with a header row")
    ap.add_argument("--src-col", default="src",
                    help="CSV column name holding the source text (default: src)")
    ap.add_argument("--tgt-col", default="tgt",
                    help="CSV column name holding the target text (default: tgt)")
    ap.add_argument("--src-lang", help="required with --tsv, or with --csv if "
                    "the file has no src_lang column (e.g. fra_Latn)")
    ap.add_argument("--tgt-lang", help="required with --tsv, or with --csv if "
                    "the file has no tgt_lang column (e.g. ewo_Latn)")
    ap.add_argument("--dev-frac", type=float, default=0.025,
                    help="fraction of data for the dev (validation) split")
    ap.add_argument("--test-frac", type=float, default=0.025,
                    help="fraction of data for the held-out test split")
    ap.add_argument("--add-reverse", action="store_true",
                    help="also emit tgt->src rows so ONE model learns BOTH "
                         "directions (fr->ewo and ewo->fr)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed so splits are reproducible")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Pass 1: load, clean, deduplicate.
    # ------------------------------------------------------------------
    seen = set()      # lowercase (src, tgt, langs) tuples already emitted
    rows = []         # cleaned rows that survived all checks
    dropped = 0       # counter for reporting

    for r in load_rows(args.inputs, args.tsv, args.csv, args.src_col,
                       args.tgt_col, args.src_lang, args.tgt_lang):
        src = normalize(r.get("src", ""))
        tgt = normalize(r.get("tgt", ""))

        # Quality gates - drop the pair if ANY of these is true:
        #   * either side empty after cleaning
        #   * either side absurdly long (not a sentence)
        #   * length ratio extreme (probable misalignment)
        #   * src == tgt (untranslated row, adds noise)
        if (not src or not tgt
                or len(src) > MAX_CHARS or len(tgt) > MAX_CHARS
                or max(len(src), len(tgt)) / max(1, min(len(src), len(tgt))) > MAX_LEN_RATIO
                or src.lower() == tgt.lower()):
            dropped += 1
            continue

        # Deduplicate on a case-insensitive key including the language pair,
        # so the same sentence in a DIFFERENT language pair is kept.
        key = (src.lower(), tgt.lower(), r["src_lang"], r["tgt_lang"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)

        rows.append({"src": src, "tgt": tgt,
                     "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"]})

    # ------------------------------------------------------------------
    # Optional: mirror every pair so the model learns both directions.
    # NOTE: done AFTER dedup so reversed rows don't collide with originals.
    # ------------------------------------------------------------------
    if args.add_reverse:
        rows += [{"src": r["tgt"], "tgt": r["src"],
                  "src_lang": r["tgt_lang"], "tgt_lang": r["src_lang"]}
                 for r in rows]

    # ------------------------------------------------------------------
    # Pass 2: shuffle (seeded = reproducible) and split.
    # test is taken first, then dev, remainder is train.
    # ------------------------------------------------------------------
    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    n_dev = int(n * args.dev_frac)
    n_test = int(n * args.test_frac)
    splits = {
        "test": rows[:n_test],
        "dev": rows[n_test:n_test + n_dev],
        "train": rows[n_test + n_dev:],
    }

    # ------------------------------------------------------------------
    # Write one JSONL file per split. ensure_ascii=False keeps accented
    # and special characters human-readable in the files.
    # ------------------------------------------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} rows")
    print(f"dropped: {dropped}")


if __name__ == "__main__":
    main()
