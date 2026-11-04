"""Load an allowlisted settings file without exposing values in container metadata."""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

from dotenv import dotenv_values

MAX_SETTINGS_FILE_BYTES = 65_536
SETTING_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
ALLOWED_FILE_SETTINGS = frozenset(
    {
        "ADMIN_EMAIL",
        "ADMIN_FIRST_NAME",
        "ADMIN_JOB_TITLE",
        "ADMIN_LAST_NAME",
        "ADMIN_ORGANIZATION",
        "ADMIN_PASSWORD",
        "ADMIN_PHONE",
        "ALLOWED_HOSTS",
        "API_KEY",
        "APP_ENV",
        "CONTRIBUTION_API_KEY",
        "CONTRIBUTION_MAX_FILE_BYTES",
        "CONTRIBUTION_RATE_LIMIT",
        "CORS_ORIGINS",
        "ENABLE_API_DOCS",
        "EXPOSE_HEALTH_DETAILS",
        "FORWARDED_ALLOW_IPS",
        "MAX_REQUEST_BODY_BYTES",
        "MODEL_DIR",
        "RATE_LIMIT_WINDOW_SECONDS",
        "STORE_CONTRIBUTOR_NAMES",
        "TOKENIZER_DIR",
        "TRANSLATION_API_KEY",
        "TRANSLATION_MODE",
        "TRANSLATION_RATE_LIMIT",
    }
)


class EnvironmentFileError(ValueError):
    """Raised when the mounted settings file cannot be loaded safely."""


def load_settings_file(
    environment: MutableMapping[str, str] = os.environ,
) -> None:
    """Merge allowlisted values from SETTINGS_FILE without overriding runtime settings."""

    file_name = environment.get("SETTINGS_FILE", "").strip()
    if not file_name:
        return

    path = Path(file_name)
    try:
        if not path.is_file() or path.stat().st_size > MAX_SETTINGS_FILE_BYTES:
            raise EnvironmentFileError("SETTINGS_FILE is missing or unexpectedly large")
        values = dotenv_values(path, interpolate=False)
    except (OSError, UnicodeError) as error:
        raise EnvironmentFileError("SETTINGS_FILE could not be read") from error

    for name, value in values.items():
        if not SETTING_NAME.fullmatch(name):
            raise EnvironmentFileError("SETTINGS_FILE contains an invalid setting name")
        if name not in ALLOWED_FILE_SETTINGS or value is None:
            continue
        environment.setdefault(name, value)
