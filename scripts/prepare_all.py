#!/usr/bin/env python3
"""Prepare and merge every ``<code>_en_parallel.csv`` corpus.

Each language is processed independently so every language receives its own
development and test representation.  ``prepare_data.py`` uses ``verse_key``
as a stable group, so variants of one verse remain in the same split across
all languages.

The Fulfulde source file contains both Latin and Arabic writing.  Those rows
are intentionally emitted as separate ``fub_Latn`` and ``fub_Arab`` targets.
Combining them under one token would teach the model two incompatible scripts
for the same requested output code.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SUFFIX = "_en_parallel.csv"
SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_data.py"


@dataclass(frozen=True)
class TargetJob:
    """One logical target extracted from a raw CSV file."""

    csv_path: Path
    column: str
    language: str
    script: str | None = None


def jobs_for_csv(csv_path: Path) -> list[TargetJob]:
    """Return one normal job or two script-specific Fulfulde jobs."""

    code = csv_path.name[: -len(SUFFIX)]
    if code == "fub":
        return [
            TargetJob(csv_path, code, "fub_Latn", "Latn"),
            TargetJob(csv_path, code, "fub_Arab", "Arab"),
        ]
    return [TargetJob(csv_path, code, f"{code}_Latn")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory containing <code>_en_parallel.csv files")
    parser.add_argument("-o", "--out-dir", default="data/prepared")
    parser.add_argument("--src-lang", default="eng_Latn")
    parser.add_argument(
        "--add-reverse",
        action="store_true",
        help="train local-language to English as well as English to local-language",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    csv_files = sorted(data_dir.glob(f"*{SUFFIX}"))
    if not csv_files:
        parser.error(f"No *{SUFFIX} files found in {data_dir}")

    jobs = [job for csv_path in csv_files for job in jobs_for_csv(csv_path)]
    out = Path(args.out_dir).resolve()
    by_language = out / "_by_lang"
    by_language.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        job_out = by_language / job.language
        command = [
            sys.executable,
            str(PREPARE_SCRIPT),
            str(job.csv_path),
            "--csv",
            "--src-col",
            "en",
            "--tgt-col",
            job.column,
            "--group-col",
            "verse_key",
            "--src-lang",
            args.src_lang,
            "--tgt-lang",
            job.language,
            "-o",
            str(job_out),
        ]
        if job.script:
            command.extend(("--tgt-script", job.script))
        if args.add_reverse:
            command.append("--add-reverse")

        print(f"== {job.language} ==", flush=True)
        subprocess.run(command, check=True)

    totals: dict[str, int] = {}
    checksums: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        combined = out / f"{split}.jsonl"
        total = 0
        digest = hashlib.sha256()
        with combined.open("w", encoding="utf-8", newline="") as destination:
            for job in jobs:
                part = by_language / job.language / f"{split}.jsonl"
                with part.open(encoding="utf-8", newline="") as source:
                    for line in source:
                        destination.write(line)
                        digest.update(line.encode("utf-8"))
                        total += 1
        totals[split] = total
        checksums[split] = digest.hexdigest()
        print(f"{split}: {total} rows -> {combined}")

    directions = []
    for job in jobs:
        directions.append([args.src_lang, job.language])
        if args.add_reverse:
            directions.append([job.language, args.src_lang])

    manifest = {
        "schema_version": 1,
        "grouping": "stable SHA-256 assignment by verse_key",
        "reverse_directions_added": args.add_reverse,
        "row_counts": totals,
        "sha256": checksums,
        "target_languages": sorted(job.language for job in jobs),
        "trained_directions": sorted(directions),
    }
    with (out / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
