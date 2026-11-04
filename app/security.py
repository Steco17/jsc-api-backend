"""Authentication guards, abuse controls, and defensive ASGI middleware."""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ApiKeyGuard:
    """Validate a configured API key without exposing it through object reprs or logs."""

    def __init__(self, expected_key: str | None, *, required: bool) -> None:
        self._expected_key = expected_key
        self._required = required

    def __call__(self, x_api_key: str | None = Header(default=None)) -> None:
        if self._expected_key is None:
            if self._required:
                raise HTTPException(503, "This write service is not configured")
            return
        candidate = x_api_key or ""
        if len(candidate) > 4096 or not secrets.compare_digest(candidate, self._expected_key):
            raise HTTPException(
                401,
                "Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )


class InMemoryRateLimiter:
    """Bounded per-client sliding-window limiter used behind the edge limiter."""

    def __init__(self, limit: int, window_seconds: int, *, max_clients: int = 10_000) -> None:
        self._limit = limit
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> None:
        client_key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            if len(self._requests) >= self._max_clients and client_key not in self._requests:
                expired_clients = [
                    key
                    for key, requests in self._requests.items()
                    if not requests or requests[-1] <= cutoff
                ]
                for key in expired_clients[: max(1, len(expired_clients))]:
                    self._requests.pop(key, None)
                if len(self._requests) >= self._max_clients:
                    raise HTTPException(429, "Too many active clients")

            requests = self._requests[client_key]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self._limit:
                retry_after = max(1, int(self._window_seconds - (now - requests[0])))
                raise HTTPException(
                    429,
                    "Too many requests",
                    headers={"Retry-After": str(retry_after)},
                )
            requests.append(now)


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies even when clients omit Content-Length."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                    scope, receive, send
                )
                return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"detail": "Request body too large"}, status_code=413)(
            scope, receive, send
        )


class _RequestBodyTooLarge(Exception):
    pass


class SecurityHeadersMiddleware:
    """Apply defense-in-depth headers when the API is accessed without Nginx."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_path = scope.get("path", "")

        async def add_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("cache-control", "no-store")
                if request_path == "/docs":
                    headers.setdefault(
                        "content-security-policy",
                        "default-src 'none'; base-uri 'none'; connect-src 'self'; "
                        "frame-ancestors 'none'; img-src 'self' data: "
                        "https://fastapi.tiangolo.com; script-src 'unsafe-inline' "
                        "https://cdn.jsdelivr.net; style-src 'unsafe-inline' "
                        "https://cdn.jsdelivr.net",
                    )
                else:
                    headers.setdefault(
                        "content-security-policy",
                        "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                        "form-action 'none'",
                    )
                headers.setdefault("cross-origin-resource-policy", "same-origin")
                headers.setdefault("permissions-policy", "camera=(), microphone=(), geolocation=()")
                headers.setdefault("referrer-policy", "no-referrer")
                headers.setdefault("x-content-type-options", "nosniff")
                headers.setdefault("x-frame-options", "DENY")
                headers.setdefault("x-permitted-cross-domain-policies", "none")
            await send(message)

        await self.app(scope, receive, add_headers)
