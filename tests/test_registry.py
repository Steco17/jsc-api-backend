"""Validation tests for the public language registry module."""

import json

import pytest

from app.registry import RegistryError, load_language_registry


def test_registry_rejects_invalid_codes_and_statuses(tmp_path) -> None:
    path = tmp_path / "languages.json"
    path.write_text(
        json.dumps({"unsafe/code": {"name": "Unsafe", "status": "published"}}),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError):
        load_language_registry(path)


def test_catalog_omits_internal_notes(tmp_path) -> None:
    path = tmp_path / "languages.json"
    path.write_text(
        json.dumps(
            {
                "eng_Latn": {
                    "name": "English",
                    "status": "pretrained",
                    "note": "Internal deployment note",
                }
            }
        ),
        encoding="utf-8",
    )

    registry = load_language_registry(path)

    assert registry.catalog() == [
        {
            "code": "eng_Latn",
            "iso_code": "eng",
            "name": "English",
            "script": "Latn",
            "status": "pretrained",
            "featured": False,
        }
    ]
