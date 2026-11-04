"""Tests for the mounted settings-file boundary."""

from pathlib import Path

import pytest

from app.environment import EnvironmentFileError, load_settings_file


def test_settings_file_loads_only_allowlisted_values_without_overrides(tmp_path: Path) -> None:
    settings_file = tmp_path / ".env"
    settings_file.write_text(
        "ADMIN_EMAIL=admin@foundation.org\n"
        "APP_ENV=production\n"
        "PATH=malicious\n",
        encoding="utf-8",
    )
    environment = {"SETTINGS_FILE": str(settings_file), "APP_ENV": "test"}

    load_settings_file(environment)

    assert environment["ADMIN_EMAIL"] == "admin@foundation.org"
    assert environment["APP_ENV"] == "test"
    assert "PATH" not in environment


def test_settings_file_rejects_invalid_names(tmp_path: Path) -> None:
    settings_file = tmp_path / ".env"
    settings_file.write_text("invalid-name=value\n", encoding="utf-8")

    with pytest.raises(EnvironmentFileError, match="invalid setting name"):
        load_settings_file({"SETTINGS_FILE": str(settings_file)})
