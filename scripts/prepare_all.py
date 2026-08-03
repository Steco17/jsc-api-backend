#!/usr/bin/env python3
"""Batch-run prepare_data.py over every "<code>_en_parallel.csv" in a
directory, then merge each language's independent split into one combined
data/prepared/{train,dev,test}.jsonl.

WHY SPLIT PER LANGUAGE BEFORE MERGING
--------------------------------------
Splitting the whole multi-language pile in one shuffle-and-slice pass can
starve a smaller language's dev/test set, or give it none at all. Splitting
each language on its own first, then concatenating, guarantees every
language gets its own proportional dev/test slice.

INPUT
-----
CSV files named "<code>_en_parallel.csv" with columns: verse_key, <code>, en
(e.g. data/bbj_en_parallel.csv with columns verse_key, bbj, en). <code> must
be the bare ISO code with no "_Latn" suffix - the script adds that itself.

USAGE
-----
  python scripts/prepare_all.py data -o data/prepared
"""
import argparse
import subprocess
import sys
from pathlib import Path

SUFFIX = "_en_parallel.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="directory containing <code>_en_parallel.csv files")
    ap.add_argument("-o", "--out-dir", default="data/prepared")
    ap.add_argument("--src-lang", default="eng_Latn",
                    help="language code for the 'en' column (default: eng_Latn)")
    ap.add_argument("--add-reverse", action="store_true",
                    help="passed through to prepare_data.py per language - "
                         "still carries the reverse/split leakage caution "
                         "documented in docs/TRAINING_GUIDE.md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_files = sorted(p for p in data_dir.glob(f"*{SUFFIX}"))
    if not csv_files:
        sys.exit(f"No *{SUFFIX} files found in {data_dir}")

    out = Path(args.out_dir)
    by_lang = out / "_by_lang"
    by_lang.mkdir(parents=True, exist_ok=True)

    codes = [p.name[:-len(SUFFIX)] for p in csv_files]

    for csv_path, code in zip(csv_files, codes):
        cmd = [sys.executable, "scripts/prepare_data.py", str(csv_path),
               "--csv", "--src-col", "en", "--tgt-col", code,
               "--src-lang", args.src_lang, "--tgt-lang", f"{code}_Latn",
               "-o", str(by_lang / code)]
        if args.add_reverse:
            cmd.append("--add-reverse")
        print(f"== {code} ==")
        subprocess.run(cmd, check=True)

    # Merge every language's split into one combined file per split.
    for split in ("train", "dev", "test"):
        combined = out / f"{split}.jsonl"
        total = 0
        with open(combined, "w", encoding="utf-8") as dest:
            for code in codes:
                part = by_lang / code / f"{split}.jsonl"
                with open(part, encoding="utf-8") as src:
                    for line in src:
                        dest.write(line)
                        total += 1
        print(f"{split}: {total} rows -> {combined}")


if __name__ == "__main__":
    main()
