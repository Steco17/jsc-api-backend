"""Tests for direction-balanced local training subsets."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import sample_prepared


def write_language_split(path: Path, target: str, split: str) -> None:
    """Write forward rows followed by reverse rows like prepare_data.py."""

    path.mkdir(parents=True, exist_ok=True)
    forward = [
        {
            "src": f"English {split} {index}",
            "tgt": f"Local {split} {index}",
            "src_lang": "eng_Latn",
            "tgt_lang": target,
            "group_id": f"{target}:{split}:{index}",
        }
        for index in range(4)
    ]
    rows = forward + [sample_prepared.reverse_row(row) for row in forward]
    (path / f"{split}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_prepared_data(root: Path) -> Path:
    """Create the manifest and per-language layout used by prepare_all.py."""

    prepared = root / "prepared"
    targets = ["abc_Latn", "xyz_Latn"]
    directions = []
    for target in targets:
        directions.extend((["eng_Latn", target], [target, "eng_Latn"]))
        for split in ("train", "dev"):
            write_language_split(prepared / "_by_lang" / target, target, split)
    (prepared / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_languages": targets,
                "trained_directions": directions,
            }
        ),
        encoding="utf-8",
    )
    return prepared


def test_subset_is_balanced_across_every_requested_direction(tmp_path: Path) -> None:
    prepared = make_prepared_data(tmp_path)
    out = tmp_path / "sampled"

    manifest = sample_prepared.build_subset(
        prepared,
        out,
        targets=[],
        train_per_direction=2,
        dev_per_direction=1,
        strategy="first",
        seed=42,
    )

    assert manifest["row_counts"] == {"train": 8, "dev": 4}
    assert len(manifest["trained_directions"]) == 4
    for split in ("train", "dev"):
        data = (out / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"][split]
        rows = [json.loads(line) for line in data.splitlines()]
        counts = {}
        for row in rows:
            pair = (row["src_lang"], row["tgt_lang"])
            counts[pair] = counts.get(pair, 0) + 1
        assert len(set(counts.values())) == 1


def test_subset_rejects_an_unknown_target(tmp_path: Path) -> None:
    prepared = make_prepared_data(tmp_path)

    with pytest.raises(ValueError, match="unknown prepared target languages"):
        sample_prepared.build_subset(
            prepared,
            tmp_path / "sampled",
            targets=["missing_Latn"],
            train_per_direction=1,
            dev_per_direction=1,
            strategy="first",
            seed=42,
        )


def test_reservoir_sampling_is_deterministic(tmp_path: Path) -> None:
    prepared = make_prepared_data(tmp_path)
    path = prepared / "_by_lang" / "abc_Latn" / "train.jsonl"
    direction = ("eng_Latn", "abc_Latn")

    first = sample_prepared.sample_direction(path, direction, 2, "reservoir", "stable")
    second = sample_prepared.sample_direction(path, direction, 2, "reservoir", "stable")

    assert first == second
    assert len(first) == 2
