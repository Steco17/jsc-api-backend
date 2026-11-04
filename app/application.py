"""FastAPI application factory and HTTP route assembly."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.config import Settings
from app.contributions import ContributionRepository, ContributionStorageFullError
from app.registry import LanguageRegistry, RegistryError, load_language_registry
from app.schemas import (
    ContributeRequest,
    ContributeResponse,
    TranslateRequest,
    TranslateResponse,
)
from app.security import (
    ApiKeyGuard,
    InMemoryRateLimiter,
    RequestBodyLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.translation import (
    ModelNotReadyError,
    TranslationService,
    UnsupportedDirectionError,
)


def create_app(
    settings: Settings,
    *,
    registry: LanguageRegistry | None = None,
    translation_service: TranslationService | None = None,
    contribution_repository: ContributionRepository | None = None,
) -> FastAPI:
    """Build one isolated application with explicit service dependencies."""

    language_registry = registry or load_language_registry(settings.languages_file)
    translator = translation_service or TranslationService(settings, language_registry)
    contributions = contribution_repository or ContributionRepository(
        settings.data_dir,
        max_file_bytes=settings.contribution_max_file_bytes,
        store_contributor_names=settings.store_contributor_names,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.translation_mode == "model":
            translator.load_model()
        try:
            yield
        finally:
            translator.unload_model()

    docs_url = "/docs" if settings.docs_enabled else None
    openapi_url = "/openapi.json" if settings.docs_enabled else None
    application = FastAPI(
        title="Cameroon Languages Translation API",
        version="0.4.0",
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.registry = language_registry
    application.state.translation_service = translator
    application.state.contribution_repository = contributions

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-API-Key"],
        max_age=600,
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.max_request_body_bytes,
    )
    application.add_middleware(SecurityHeadersMiddleware)

    translation_guard = ApiKeyGuard(settings.translation_api_key, required=False)
    contribution_guard = ApiKeyGuard(settings.contribution_api_key, required=True)
    translation_limiter = InMemoryRateLimiter(
        settings.translation_rate_limit,
        settings.rate_limit_window_seconds,
    )
    contribution_limiter = InMemoryRateLimiter(
        settings.contribution_rate_limit,
        settings.rate_limit_window_seconds,
    )

    @application.get("/health")
    def health():
        ready_for_mode = settings.translation_mode == "placeholder" or translator.model_loaded
        response: dict[str, object] = {
            "status": "ready" if ready_for_mode else "starting"
        }
        if settings.expose_health_details:
            response.update(
                {
                    "translation_mode": settings.translation_mode,
                    "model_loaded": translator.model_loaded,
                    "enabled_directions": (
                        len(translator.supported_directions) if translator.model_loaded else 0
                    ),
                }
            )
        return response

    @application.get("/ready")
    def ready():
        if settings.translation_mode == "model" and not translator.model_loaded:
            raise HTTPException(503, "Service is not ready")
        response: dict[str, object] = {"status": "ready"}
        if settings.expose_health_details:
            response.update(
                {
                    "translation_mode": settings.translation_mode,
                    "model_loaded": translator.model_loaded,
                }
            )
        return response

    @application.get("/languages")
    def languages():
        if settings.translation_mode == "placeholder":
            return language_registry.contribution_languages
        return language_registry.active_languages

    @application.get("/contribution-languages")
    def contribution_languages():
        return language_registry.contribution_languages

    @application.get("/language-catalog")
    def language_catalog():
        return {"languages": language_registry.catalog()}

    @application.get("/directions")
    def directions():
        if settings.translation_mode == "placeholder":
            return {"directions": [], "status": "model_pending"}
        return {
            "directions": [
                {"src_lang": source, "tgt_lang": target}
                for source, target in sorted(translator.supported_directions)
            ]
        }

    @application.post(
        "/translate",
        response_model=TranslateResponse,
        dependencies=[Depends(translation_guard), Depends(translation_limiter)],
    )
    def translate(request: TranslateRequest):
        try:
            translated_text, status = translator.translate(
                request.text,
                request.src_lang,
                request.tgt_lang,
            )
        except RegistryError as error:
            raise HTTPException(422, str(error)) from error
        except UnsupportedDirectionError as error:
            raise HTTPException(422, str(error)) from error
        except ModelNotReadyError as error:
            raise HTTPException(503, "Translation service is not ready") from error
        return {
            "translation": translated_text,
            "src_lang": request.src_lang,
            "tgt_lang": request.tgt_lang,
            "status": status,
        }

    @application.post(
        "/contribute",
        response_model=ContributeResponse,
        dependencies=[Depends(contribution_guard), Depends(contribution_limiter)],
    )
    def contribute(request: ContributeRequest):
        try:
            translator.validate_contribution_pair(request.src_lang, request.tgt_lang)
            contributions.save(request.model_dump())
        except RegistryError as error:
            raise HTTPException(422, str(error)) from error
        except ContributionStorageFullError as error:
            raise HTTPException(507, "Contribution storage is unavailable") from error
        return {
            "status": "saved",
            "message": "Thank you. Your pair will be reviewed before training.",
        }

    return application
