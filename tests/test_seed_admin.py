"""Regression tests for environment-backed administrator seeding."""

import sqlite3

import pytest

from scripts.seed_admin import (
    SeedConfigurationError,
    admin_from_environment,
    seed_admin,
    verify_password,
)


def valid_environment() -> dict[str, str]:
    """Return a complete seed environment with a non-placeholder password."""

    return {
        "ADMIN_EMAIL": "owner@jscfoundation.org",
        "ADMIN_PASSWORD": "Strong-Admin-2026!",
        "ADMIN_FIRST_NAME": "JSC",
        "ADMIN_LAST_NAME": "Administrator",
        "ADMIN_PHONE": "+237 000 000 000",
        "ADMIN_JOB_TITLE": "Platform Administrator",
        "ADMIN_ORGANIZATION": "JSC Foundation for Learning",
    }


def test_admin_seed_is_idempotent_and_never_stores_plaintext(tmp_path) -> None:
    environment = valid_environment()
    admin = admin_from_environment(environment)
    database_path = tmp_path / "jsc.sqlite3"

    seed_admin(database_path, admin)
    seed_admin(database_path, admin)

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT email, password_hash, first_name, role, is_active FROM admins"
        ).fetchall()

    assert len(rows) == 1
    email, password_hash, first_name, role, is_active = rows[0]
    assert email == "owner@jscfoundation.org"
    assert password_hash != environment["ADMIN_PASSWORD"]
    assert verify_password(environment["ADMIN_PASSWORD"], password_hash)
    assert first_name == "JSC"
    assert role == "admin"
    assert is_active == 1


def test_reseeding_updates_admin_details_and_changed_password(tmp_path) -> None:
    environment = valid_environment()
    database_path = tmp_path / "jsc.sqlite3"
    seed_admin(database_path, admin_from_environment(environment))

    environment["ADMIN_FIRST_NAME"] = "Updated"
    environment["ADMIN_PASSWORD"] = "New-Strong-Admin-2027!"
    seed_admin(database_path, admin_from_environment(environment))

    with sqlite3.connect(database_path) as connection:
        first_name, password_hash = connection.execute(
            "SELECT first_name, password_hash FROM admins WHERE email = ?",
            ("owner@jscfoundation.org",),
        ).fetchone()

    assert first_name == "Updated"
    assert verify_password(environment["ADMIN_PASSWORD"], password_hash)
    assert not verify_password("Strong-Admin-2026!", password_hash)


def test_password_verification_rejects_malformed_hashes() -> None:
    assert not verify_password("Strong-Admin-2026!", "not-a-supported-hash")
    assert not verify_password("Strong-Admin-2026!", "pbkdf2_sha256$invalid$%%%$%%%")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ADMIN_EMAIL", "admin@example.org", "example domain"),
        ("ADMIN_PASSWORD", "short", "at least 12"),
        ("ADMIN_PASSWORD", "replace-this-password", "placeholder"),
    ],
)
def test_admin_seed_rejects_placeholder_or_unsafe_credentials(
    field,
    value,
    message,
) -> None:
    environment = valid_environment()
    environment[field] = value

    with pytest.raises(SeedConfigurationError, match=message):
        admin_from_environment(environment)
