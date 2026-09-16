#!/usr/bin/env python3
"""Build a small, direction-balanced training set from prepared corpora.

The combined prepared files are grouped by language, so taking a simple
prefix exercises only the first language. This helper samples every requested
forward direction from ``_by_lang`` and adds its prepared reverse row. The
result keeps the original train/dev group boundary and includes a checksum
manifest that ``finetune.py`` validates before training.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

REQUIRED_FIELDS = {"src", "tgt", "src_lang", "tgt_lang", "group_id"}


def reverse_row(row: dict) -> dict:
    """Return the reverse direction while retaining the source group."""

    return {
        "src": row["tgt"],
        "tgt": row["src"],
        "src_lang": row["tgt_lang"],
        "tgt_lang": row["src_lang"],
        "group_id": row["group_id"],
    }


def sample_direction(
    path: Path,
    direction: tuple[str, str],
    limit: int,
    strategy: str,
    seed: str,
) -> list[dict]:
    """Select up to ``limit`` rows for one direction from a JSONL file."""

    selected: list[dict] = []
    seen = 0
    generator = random.Random(seed)

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if (row["src_lang"], row["tgt_lang"]) != direction:
                continue

            seen += 1
            if len(selected) < limit:
                selected.append(row)
            elif strategy == "reservoir":
                replacement = generator.randrange(seen)
                if replacement < limit:
                    selected[replacement] = row
            else:
                break

    if not selected:
        raise ValueError(f"{path}: no rows found for {direction[0]} -> {direction[1]}")
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> str:
    """Write rows and return the SHA-256 checksum of the exact file bytes."""

    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def build_subset(
    prepared_dir: Path,
    out_dir: Path,
    targets: list[str],
    train_per_direction: int,
    dev_per_direction: int,
    strategy: str,
    seed: int,
) -> dict:
    """Create balanced train/dev files and return their generated manifest."""

    source_manifest_path = prepared_dir / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    available_targets = set(source_manifest["target_languages"])
    selected_targets = sorted(targets or available_targets)
    unknown = set(selected_targets) - available_targets
    if unknown:
        raise ValueError(f"unknown prepared target languages: {sorted(unknown)}")

    source_directions = {tuple(pair) for pair in source_manifest["trained_directions"]}
    rows_by_split: dict[str, list[dict]] = {"train": [], "dev": []}
    limits = {"train": train_per_direction, "dev": dev_per_direction}
    sampled_directions: set[tuple[str, str]] = set()

    for target in selected_targets:
        language_dir = prepared_dir / "_by_lang" / target
        if not language_dir.is_dir():
            raise ValueError(
                f"prepared per-language data is missing: {language_dir}. "
                "Run scripts/prepare_all.py again."
            )

        forward_directions = sorted(
            pair
            for pair in source_directions
            if pair[1] == target and (pair[1], pair[0]) in source_directions
        )
        if not forward_directions:
            raise ValueError(f"no reversible forward direction found for {target}")

        for split, limit in limits.items():
            part = language_dir / f"{split}.jsonl"
            for direction in forward_directions:
                selected = sample_direction(
                    part,
                    direction,
                    limit,
                    strategy,
                    f"{seed}:{split}:{direction[0]}:{direction[1]}",
                )
                for row in selected:
                    rows_by_split[split].extend((row, reverse_row(row)))
                sampled_directions.update((direction, (direction[1], direction[0])))

    out_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for offset, split in enumerate(("train", "dev")):
        random.Random(seed + offset).shuffle(rows_by_split[split])
        checksums[split] = write_jsonl(out_dir / f"{split}.jsonl", rows_by_split[split])

    source_checksum = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_dataset_manifest_sha256": source_checksum,
        "sampling_strategy": strategy,
        "samples_requested_per_direction": limits,
        "row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "sha256": checksums,
        "target_languages": selected_targets,
        "trained_directions": [list(pair) for pair in sorted(sampled_directions)],
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dir")
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("--targets", nargs="*", default=[])
    parser.add_argument("--train-per-direction", type=int, default=2)
    parser.add_argument("--dev-per-direction", type=int, default=1)
    parser.add_argument("--strategy", choices=("first", "reservoir"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.train_per_direction, args.dev_per_direction) < 1:
        parser.error("per-direction sample counts must be positive")

    try:
        manifest = build_subset(
            Path(args.prepared_dir),
            Path(args.out_dir),
            args.targets,
            args.train_per_direction,
            args.dev_per_direction,
            args.strategy,
            args.seed,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(
        f"Prepared {manifest['row_counts']['train']} train and "
        f"{manifest['row_counts']['dev']} dev rows across "
        f"{len(manifest['trained_directions'])} directions."
    )


if __name__ == "__main__":
    main()
