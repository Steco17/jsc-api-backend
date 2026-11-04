"""Strict public API request and response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LANGUAGE_CODE_PATTERN = r"^[a-z0-9]{2,8}_[A-Z][a-z]{3}$"


def _normalized_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if any(ord(character) < 32 and character not in {"\n", "\r", "\t"} for character in normalized):
        raise ValueError(f"{field_name} contains unsupported control characters")
    return normalized


class StrictRequestModel(BaseModel):
    """Reject unknown fields and silent type coercion on write endpoints."""

    model_config = ConfigDict(extra="forbid", strict=True)


class TranslateRequest(StrictRequestModel):
    text: str = Field(min_length=1, max_length=2000)
    src_lang: str = Field(min_length=7, max_length=32, pattern=LANGUAGE_CODE_PATTERN)
    tgt_lang: str = Field(min_length=7, max_length=32, pattern=LANGUAGE_CODE_PATTERN)

    @field_validator("text")
    @classmethod
    def text_must_be_safe(cls, value: str) -> str:
        return _normalized_text(value, "text")


class TranslateResponse(BaseModel):
    """Stable frontend contract for placeholder and model-backed translation."""

    translation: str
    src_lang: str
    tgt_lang: str
    status: Literal["model_pending", "translated"]


class ContributeRequest(StrictRequestModel):
    src: str = Field(min_length=1, max_length=2000)
    tgt: str = Field(min_length=1, max_length=2000)
    src_lang: str = Field(min_length=7, max_length=32, pattern=LANGUAGE_CODE_PATTERN)
    tgt_lang: str = Field(min_length=7, max_length=32, pattern=LANGUAGE_CODE_PATTERN)
    contributor: str | None = Field(default=None, max_length=100)

    @field_validator("src", "tgt")
    @classmethod
    def pair_text_must_be_safe(cls, value: str) -> str:
        return _normalized_text(value, "translation text")

    @field_validator("contributor")
    @classmethod
    def contributor_must_be_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = _normalized_text(value, "contributor")
        if "\n" in normalized or "\r" in normalized or "\t" in normalized:
            raise ValueError("contributor must be a single line")
        return normalized


class ContributeResponse(BaseModel):
    status: Literal["saved"]
    message: str
