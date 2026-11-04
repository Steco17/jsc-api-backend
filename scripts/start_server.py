"""Seed required deployment data and replace this process with Uvicorn."""

import os
from pathlib import Path

from app.environment import EnvironmentFileError, load_settings_file
from scripts.seed_admin import SeedConfigurationError, seed_from_environment


def _move_secret_to_private_file(name: str) -> None:
    value = os.environ.pop(name, "")
    if not value:
        return
    path = Path("/tmp") / f"jsc-{name.lower().replace('_', '-')}"
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    os.environ[f"{name}_FILE"] = str(path)


def _discard_seed_environment() -> None:
    for name in tuple(os.environ):
        if name.startswith("ADMIN_"):
            os.environ.pop(name, None)


def main() -> None:
    """Seed the administrator before accepting HTTP traffic."""

    os.umask(0o077)
    try:
        load_settings_file()
        seed_from_environment()
    except (EnvironmentFileError, SeedConfigurationError) as error:
        raise SystemExit(f"Administrator seed configuration error: {error}") from error

    _discard_seed_environment()
    for secret_name in ("API_KEY", "TRANSLATION_API_KEY", "CONTRIBUTION_API_KEY"):
        _move_secret_to_private_file(secret_name)
    os.environ.pop("SETTINGS_FILE", None)

    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            os.getenv("PORT", "8000"),
            "--no-server-header",
            "--no-date-header",
            "--proxy-headers",
            "--forwarded-allow-ips",
            os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        ],
    )


if __name__ == "__main__":
    main()
