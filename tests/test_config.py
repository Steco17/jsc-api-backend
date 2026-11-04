"""Security regression tests for environment-backed application settings."""

from pathlib import Path

import pytest

from app.config import ConfigurationError, Settings

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "languages.json"


def base_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "APP_ENV": "test",
        "DATA_DIR": str(tmp_path),
        "LANGUAGES_FILE": str(REGISTRY_PATH),
        "ALLOWED_HOSTS": "testserver,localhost",
        "CORS_ORIGINS": "http://localhost:5173",
    }


def test_secret_file_is_supported_without_exposing_value_in_repr(tmp_path) -> None:
    secret_file = tmp_path / "contribution-key"
    secret_file.write_text("s" * 32, encoding="utf-8")
    environment = base_environment(tmp_path)
    environment["CONTRIBUTION_API_KEY_FILE"] = str(secret_file)

    settings = Settings.from_environment(environment)

    assert settings.contribution_api_key == "s" * 32
    assert "s" * 32 not in repr(settings)


def test_production_requires_explicit_hosts_and_long_keys(tmp_path) -> None:
    environment = base_environment(tmp_path)
    environment["APP_ENV"] = "production"
    environment.pop("ALLOWED_HOSTS")
    with pytest.raises(ConfigurationError, match="ALLOWED_HOSTS is required"):
        Settings.from_environment(environment)

    environment["ALLOWED_HOSTS"] = "translator.example.org"
    environment["CONTRIBUTION_API_KEY"] = "short"
    with pytest.raises(ConfigurationError, match="at least 32"):
        Settings.from_environment(environment)


@pytest.mark.parametrize(
    "origin",
    ["*", "javascript:alert(1)", "https://user:password@example.org"],
)
def test_cors_rejects_wildcard_or_malformed_origins(tmp_path, origin) -> None:
    environment = base_environment(tmp_path)
    environment["CORS_ORIGINS"] = origin

    with pytest.raises(ConfigurationError, match="CORS"):
        Settings.from_environment(environment)


def test_production_disables_docs_and_detailed_health_by_default(tmp_path) -> None:
    environment = base_environment(tmp_path)
    environment.update(
        {
            "APP_ENV": "production",
            "ALLOWED_HOSTS": "translator.example.org",
        }
    )
    environment.pop("CORS_ORIGINS")

    settings = Settings.from_environment(environment)

    assert settings.docs_enabled is False
    assert settings.expose_health_details is False
    assert settings.cors_origins == ()
