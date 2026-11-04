"""Validated runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

VALID_ENVIRONMENTS = {"development", "test", "production"}
VALID_TRANSLATION_MODES = {"placeholder", "model"}


class ConfigurationError(ValueError):
    """Raised when deployment configuration is missing or unsafe."""


def _boolean(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = environment.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = environment.get(name, "").strip()
    try:
        value = int(raw_value) if raw_value else default
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _csv(environment: Mapping[str, str], name: str, default: str = "") -> tuple[str, ...]:
    raw_value = environment.get(name, default)
    return tuple(value.strip() for value in raw_value.split(",") if value.strip())


def _secret(environment: Mapping[str, str], name: str) -> str | None:
    """Read one secret from an environment value or a mounted secret file."""

    value = environment.get(name, "").strip()
    file_name = environment.get(f"{name}_FILE", "").strip()
    if value and file_name:
        raise ConfigurationError(f"Set only one of {name} or {name}_FILE")
    if file_name:
        path = Path(file_name)
        if not path.is_file():
            raise ConfigurationError(f"{name}_FILE does not reference a readable file")
        if path.stat().st_size > 8192:
            raise ConfigurationError(f"{name}_FILE is unexpectedly large")
        value = path.read_text(encoding="utf-8").strip()
    return value or None


def _validate_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    for origin in origins:
        if origin == "*":
            raise ConfigurationError("CORS_ORIGINS must list explicit trusted origins")
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ConfigurationError(f"Invalid CORS origin: {origin}")
    return origins


@dataclass(frozen=True)
class Settings:
    """Immutable settings used to assemble one application instance."""

    environment: str
    translation_mode: str
    model_dir: Path
    tokenizer_dir: Path
    data_dir: Path
    database_path: Path
    languages_file: Path
    cors_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    max_request_body_bytes: int
    translation_rate_limit: int
    contribution_rate_limit: int
    rate_limit_window_seconds: int
    contribution_max_file_bytes: int
    store_contributor_names: bool
    docs_enabled: bool
    expose_health_details: bool
    translation_api_key: str | None = field(default=None, repr=False)
    contribution_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] = os.environ) -> Settings:
        app_environment = environment.get("APP_ENV", "development").strip().lower()
        if app_environment not in VALID_ENVIRONMENTS:
            raise ConfigurationError(
                f"APP_ENV must be one of {sorted(VALID_ENVIRONMENTS)}"
            )

        translation_mode = environment.get("TRANSLATION_MODE", "placeholder").strip().lower()
        if translation_mode not in VALID_TRANSLATION_MODES:
            raise ConfigurationError(
                f"TRANSLATION_MODE must be one of {sorted(VALID_TRANSLATION_MODES)}"
            )

        default_registry = Path(__file__).resolve().parent.parent / "data" / "languages.json"
        data_dir = Path(environment.get("DATA_DIR", "data").strip() or "data")
        database_path = Path(
            environment.get("DATABASE_PATH", "").strip() or data_dir / "jsc.sqlite3"
        )

        default_hosts = "localhost,127.0.0.1,testserver,backend"
        allowed_hosts = _csv(environment, "ALLOWED_HOSTS", default_hosts)
        if not allowed_hosts or "*" in allowed_hosts:
            raise ConfigurationError("ALLOWED_HOSTS must list explicit host names")
        if app_environment == "production" and "ALLOWED_HOSTS" not in environment:
            raise ConfigurationError("ALLOWED_HOSTS is required when APP_ENV=production")

        cors_origins = _validate_origins(
            _csv(
                environment,
                "CORS_ORIGINS",
                (
                    ""
                    if app_environment == "production"
                    else "http://localhost:5173,http://127.0.0.1:5173"
                ),
            )
        )

        legacy_api_key = _secret(environment, "API_KEY")
        translation_api_key = _secret(environment, "TRANSLATION_API_KEY") or legacy_api_key
        contribution_api_key = _secret(environment, "CONTRIBUTION_API_KEY") or legacy_api_key
        for key_name, key_value in (
            ("TRANSLATION_API_KEY", translation_api_key),
            ("CONTRIBUTION_API_KEY", contribution_api_key),
        ):
            if key_value:
                if any(
                    marker in key_value.lower()
                    for marker in ("change-me", "replace-with", "example", "password")
                ):
                    raise ConfigurationError(f"{key_name} contains a placeholder value")
                minimum_key_length = 32 if app_environment == "production" else 16
                if len(key_value) < minimum_key_length:
                    raise ConfigurationError(
                        f"{key_name} must contain at least {minimum_key_length} characters"
                    )

        return cls(
            environment=app_environment,
            translation_mode=translation_mode,
            model_dir=Path(environment.get("MODEL_DIR", "model_ct2")),
            tokenizer_dir=Path(environment.get("TOKENIZER_DIR", "model_out/merged")),
            data_dir=data_dir,
            database_path=database_path,
            languages_file=Path(
                environment.get("LANGUAGES_FILE", "").strip() or default_registry
            ),
            cors_origins=cors_origins,
            allowed_hosts=allowed_hosts,
            max_request_body_bytes=_integer(
                environment,
                "MAX_REQUEST_BODY_BYTES",
                16_384,
                minimum=4096,
                maximum=1_048_576,
            ),
            translation_rate_limit=_integer(
                environment,
                "TRANSLATION_RATE_LIMIT",
                600,
                minimum=1,
                maximum=100_000,
            ),
            contribution_rate_limit=_integer(
                environment,
                "CONTRIBUTION_RATE_LIMIT",
                60,
                minimum=1,
                maximum=10_000,
            ),
            rate_limit_window_seconds=_integer(
                environment,
                "RATE_LIMIT_WINDOW_SECONDS",
                60,
                minimum=1,
                maximum=3600,
            ),
            contribution_max_file_bytes=_integer(
                environment,
                "CONTRIBUTION_MAX_FILE_BYTES",
                52_428_800,
                minimum=1_048_576,
                maximum=1_073_741_824,
            ),
            store_contributor_names=_boolean(
                environment,
                "STORE_CONTRIBUTOR_NAMES",
                False,
            ),
            docs_enabled=_boolean(
                environment,
                "ENABLE_API_DOCS",
                app_environment != "production",
            ),
            expose_health_details=_boolean(
                environment,
                "EXPOSE_HEALTH_DETAILS",
                app_environment != "production",
            ),
            translation_api_key=translation_api_key,
            contribution_api_key=contribution_api_key,
        )
