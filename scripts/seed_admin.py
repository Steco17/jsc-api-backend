"""Create or update the application administrator from environment variables."""

import base64
import binascii
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.environment import EnvironmentFileError, load_settings_file

PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class SeedConfigurationError(ValueError):
    """Report missing or unsafe administrator seed settings."""


@dataclass(frozen=True)
class AdminSeed:
    """Validated administrator values used by the idempotent seed operation."""

    email: str
    password: str
    first_name: str
    last_name: str
    phone: str | None
    job_title: str | None
    organization: str | None


def _required(environment: Mapping[str, str], name: str, *, maximum: int = 200) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise SeedConfigurationError(f"{name} is required")
    if len(value) > maximum:
        raise SeedConfigurationError(f"{name} is too long")
    return value


def _optional(environment: Mapping[str, str], name: str, *, maximum: int = 200) -> str | None:
    value = environment.get(name, "").strip()
    if len(value) > maximum:
        raise SeedConfigurationError(f"{name} is too long")
    return value or None


def _required_secret(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    file_name = environment.get(f"{name}_FILE", "").strip()
    if value and file_name:
        raise SeedConfigurationError(f"Set only one of {name} or {name}_FILE")
    if file_name:
        path = Path(file_name)
        if not path.is_file() or path.stat().st_size > 8192:
            raise SeedConfigurationError(f"{name}_FILE is not a valid secret file")
        value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise SeedConfigurationError(f"{name} is required")
    if len(value) > 1024:
        raise SeedConfigurationError(f"{name} is too long")
    return value


def admin_from_environment(environment: Mapping[str, str] = os.environ) -> AdminSeed:
    """Read and validate administrator identity and credentials from an environment."""

    email = _required(environment, "ADMIN_EMAIL", maximum=254).lower()
    password = _required_secret(environment, "ADMIN_PASSWORD")

    if not EMAIL_PATTERN.fullmatch(email):
        raise SeedConfigurationError("ADMIN_EMAIL must be a valid email address")
    if email.endswith("@example.org") or email.endswith("@example.com"):
        raise SeedConfigurationError("ADMIN_EMAIL must not use an example domain")
    if len(password) < 12:
        raise SeedConfigurationError("ADMIN_PASSWORD must contain at least 12 characters")
    if any(marker in password.lower() for marker in ("change-me", "replace-this", "password")):
        raise SeedConfigurationError("ADMIN_PASSWORD still contains a placeholder or unsafe value")
    if not all(
        (
            re.search(r"[a-z]", password),
            re.search(r"[A-Z]", password),
            re.search(r"\d", password),
            re.search(r"[^A-Za-z0-9]", password),
        )
    ):
        raise SeedConfigurationError(
            "ADMIN_PASSWORD must include lowercase, uppercase, numeric, and symbol characters"
        )

    return AdminSeed(
        email=email,
        password=password,
        first_name=_required(environment, "ADMIN_FIRST_NAME", maximum=100),
        last_name=_required(environment, "ADMIN_LAST_NAME", maximum=100),
        phone=_optional(environment, "ADMIN_PHONE", maximum=100),
        job_title=_optional(environment, "ADMIN_JOB_TITLE"),
        organization=_optional(environment, "ADMIN_ORGANIZATION"),
    )


def hash_password(password: str) -> str:
    """Hash a password with a random salt and a versioned PBKDF2 format."""

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt_text}${digest_text}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a seed-generated PBKDF2 hash."""

    try:
        scheme, iterations_text, salt_text, digest_text = stored_hash.split("$", maxsplit=3)
        if scheme != PASSWORD_SCHEME:
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations_text),
        )
    except (binascii.Error, ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate, expected)


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the minimal administrator schema when it does not exist."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            phone TEXT,
            job_title TEXT,
            organization TEXT,
            role TEXT NOT NULL DEFAULT 'admin' CHECK (role = 'admin'),
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def seed_admin(database_path: Path, admin: AdminSeed) -> None:
    """Insert or update one administrator without duplicating the account."""

    database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        initialize_database(connection)
        existing = connection.execute(
            "SELECT password_hash FROM admins WHERE email = ?",
            (admin.email,),
        ).fetchone()
        password_hash = (
            existing[0]
            if existing and verify_password(admin.password, existing[0])
            else hash_password(admin.password)
        )
        connection.execute(
            """
            INSERT INTO admins (
                email,
                password_hash,
                first_name,
                last_name,
                phone,
                job_title,
                organization,
                role,
                is_active,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'admin', 1, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                password_hash = excluded.password_hash,
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                phone = excluded.phone,
                job_title = excluded.job_title,
                organization = excluded.organization,
                role = 'admin',
                is_active = 1,
                updated_at = excluded.updated_at
            """,
            (
                admin.email,
                password_hash,
                admin.first_name,
                admin.last_name,
                admin.phone,
                admin.job_title,
                admin.organization,
                now,
                now,
            ),
        )
    try:
        database_path.chmod(0o600)
    except OSError:
        pass


def database_path_from_environment(environment: Mapping[str, str] = os.environ) -> Path:
    """Resolve the administrator database path from deployment settings."""

    explicit_path = environment.get("DATABASE_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path)
    data_directory = Path(environment.get("DATA_DIR", "data"))
    return data_directory / "jsc.sqlite3"


def seed_from_environment(environment: Mapping[str, str] = os.environ) -> Path:
    """Run the complete environment-backed seed and return the database path."""

    admin = admin_from_environment(environment)
    database_path = database_path_from_environment(environment)
    seed_admin(database_path, admin)
    print("Administrator seed completed")
    return database_path


if __name__ == "__main__":
    try:
        load_settings_file()
        seed_from_environment()
    except (EnvironmentFileError, SeedConfigurationError) as error:
        raise SystemExit(f"Administrator seed configuration error: {error}") from error
