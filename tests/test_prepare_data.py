"""End-to-end tests for leakage-safe dataset preparation."""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "prepare_data.py"


def write_parallel_csv(path: Path, target_column: str, groups: int = 40) -> None:
    """Create two translation variants for every stable source group."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("verse_key", target_column, "en"))
        writer.writeheader()
        for index in range(groups):
            for variant in range(2):
                writer.writerow(
                    {
                        "verse_key": f"BOOK.1.{index}",
                        target_column: f"Local sentence {index} version {variant}",
                        "en": f"English sentence {index} version {variant}",
                    }
                )


def run_prepare(csv_path: Path, out: Path, target_column: str, target_lang: str):
    """Execute the same CLI boundary used by the Colab notebook."""

    subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            str(csv_path),
            "--csv",
            "--src-col",
            "en",
            "--tgt-col",
            target_column,
            "--group-col",
            "verse_key",
            "--src-lang",
            "eng_Latn",
            "--tgt-lang",
            target_lang,
            "--dev-frac",
            "0.2",
            "--test-frac",
            "0.2",
            "--add-reverse",
            "-o",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )


def read_splits(out: Path) -> dict[str, list[dict]]:
    """Read the three generated JSONL files into test-friendly structures."""

    result = {}
    for split in ("train", "dev", "test"):
        result[split] = [
            json.loads(line)
            for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    return result


def group_assignments(splits: dict[str, list[dict]]) -> dict[str, str]:
    """Map every group to one split and fail immediately on leakage."""

    assignments = {}
    for split, rows in splits.items():
        for row in rows:
            previous = assignments.setdefault(row["group_id"], split)
            assert previous == split
    return assignments


def test_variants_and_reverse_rows_never_cross_splits(tmp_path: Path) -> None:
    csv_path = tmp_path / "abc_en_parallel.csv"
    write_parallel_csv(csv_path, "abc")
    out = tmp_path / "prepared"

    run_prepare(csv_path, out, "abc", "abc_Latn")
    splits = read_splits(out)
    assignments = group_assignments(splits)

    assert all(splits.values())
    assert len(assignments) == 40
    rows_by_group = defaultdict(list)
    for rows in splits.values():
        for row in rows:
            rows_by_group[row["group_id"]].append(row)
    assert {len(rows) for rows in rows_by_group.values()} == {4}
    assert {(row["src_lang"], row["tgt_lang"]) for row in next(iter(rows_by_group.values()))} == {
        ("eng_Latn", "abc_Latn"),
        ("abc_Latn", "eng_Latn"),
    }


def test_same_group_uses_same_split_across_language_files(tmp_path: Path) -> None:
    first_csv = tmp_path / "abc_en_parallel.csv"
    second_csv = tmp_path / "xyz_en_parallel.csv"
    write_parallel_csv(first_csv, "abc")
    write_parallel_csv(second_csv, "xyz")

    run_prepare(first_csv, tmp_path / "abc", "abc", "abc_Latn")
    run_prepare(second_csv, tmp_path / "xyz", "xyz", "xyz_Latn")

    first = group_assignments(read_splits(tmp_path / "abc"))
    second = group_assignments(read_splits(tmp_path / "xyz"))
    assert first == second


def test_script_filter_separates_fulfulde_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "fub_en_parallel.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("verse_key", "fub", "en"))
        writer.writeheader()
        writer.writerow({"verse_key": "GEN.1.1", "fub": "Jam waali", "en": "Good evening"})
        writer.writerow({"verse_key": "GEN.1.2", "fub": "جَمْ وَالِي", "en": "Good night"})

    for script, language, expected_text in (
        ("Latn", "fub_Latn", "Jam waali"),
        ("Arab", "fub_Arab", "جَمْ وَالِي"),
    ):
        out = tmp_path / script
        subprocess.run(
            [
                sys.executable,
                str(PREPARE),
                str(csv_path),
                "--csv",
                "--src-col",
                "en",
                "--tgt-col",
                "fub",
                "--group-col",
                "verse_key",
                "--src-lang",
                "eng_Latn",
                "--tgt-lang",
                language,
                "--tgt-script",
                script,
                "--dev-frac",
                "0",
                "--test-frac",
                "0",
                "-o",
                str(out),
            ],
            check=True,
            cwd=ROOT,
        )
        rows = read_splits(out)["train"]
        assert len(rows) == 1
        assert rows[0]["tgt"] == expected_text
        assert rows[0]["tgt_lang"] == language


def test_existing_reverse_rows_are_not_duplicated(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {
            "src": "Good morning",
            "tgt": "Bonjour",
            "src_lang": "eng_Latn",
            "tgt_lang": "fra_Latn",
            "group_id": "greeting",
        },
        {
            "src": "Bonjour",
            "tgt": "Good morning",
            "src_lang": "fra_Latn",
            "tgt_lang": "eng_Latn",
            "group_id": "greeting",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "prepared"

    subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            str(source),
            "--dev-frac",
            "0",
            "--test-frac",
            "0",
            "--add-reverse",
            "-o",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )

    assert len(read_splits(out)["train"]) == 2
