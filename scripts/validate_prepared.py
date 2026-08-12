#!/usr/bin/env python3
"""Audit prepared JSONL files before an expensive training run."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"src", "tgt", "src_lang", "tgt_lang", "group_id"}
ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


def validate_script(code: str, text: str, location: str) -> None:
    """Fail when a Fulfulde script label disagrees with its actual text."""

    contains_arabic = bool(ARABIC_RE.search(text))
    if code == "fub_Latn" and contains_arabic:
        raise ValueError(f"{location}: Arabic text labeled fub_Latn")
    if code == "fub_Arab" and not contains_arabic:
        raise ValueError(f"{location}: non-Arabic text labeled fub_Arab")


def read_split(path: Path):
    """Stream one split and return its groups, languages, directions, and count."""

    groups = set()
    languages = set()
    directions = Counter()
    count = 0
    digest = hashlib.sha256()
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line.encode("utf-8"))
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            location = f"{path}:{line_number}"
            validate_script(row["src_lang"], row["src"], location)
            validate_script(row["tgt_lang"], row["tgt"], location)
            groups.add(row["group_id"])
            languages.update((row["src_lang"], row["tgt_lang"]))
            directions[(row["src_lang"], row["tgt_lang"])] += 1
            count += 1
    return groups, languages, directions, count, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dir", nargs="?", default="data/prepared")
    parser.add_argument("--registry", default="data/languages.json")
    args = parser.parse_args()

    prepared = Path(args.prepared_dir)
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    expected = {code for code, info in registry.items() if info["status"] == "data_ready"}

    reports = {}
    for split in ("train", "dev", "test"):
        reports[split] = read_split(prepared / f"{split}.jsonl")

    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = reports[first][0] & reports[second][0]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(f"group leakage between {first} and {second}: {sample}")

    observed_languages = set().union(*(report[1] for report in reports.values()))
    missing_languages = expected - observed_languages
    if missing_languages:
        raise ValueError(f"prepared data is missing languages: {sorted(missing_languages)}")

    train_directions = set(reports["train"][2])
    missing_reverse = {
        direction
        for direction in train_directions
        if (direction[1], direction[0]) not in train_directions
    }
    if missing_reverse:
        raise ValueError(f"training directions without reverse: {sorted(missing_reverse)}")

    manifest_path = prepared / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_directions = {tuple(pair) for pair in manifest["trained_directions"]}
    if manifest_directions != train_directions:
        raise ValueError("dataset manifest directions do not match train.jsonl")
    for split, report in reports.items():
        if manifest.get("sha256", {}).get(split) != report[4]:
            raise ValueError(f"dataset manifest checksum does not match {split}.jsonl")

    for split, report in reports.items():
        groups, _languages, directions, count, _checksum = report
        print(f"{split}: {count} rows, {len(groups)} groups, {len(directions)} directions")
    print(
        f"validated {len(expected)} data-ready language-script targets; "
        "all groups are disjoint and every direction has a reverse"
    )


if __name__ == "__main__":
    main()
