# Security and privacy

This application applies defense in depth, but no application can guarantee that data will never leak.
Production security also depends on TLS, host hardening, access control, encrypted storage, backups, monitoring, and timely updates.

## Data handling

Translation text is processed in memory and is not intentionally written to application storage.
The default access logs record request paths and status codes, not request bodies.
Community contributions are disabled until `CONTRIBUTION_API_KEY` is configured.
Contribution records contain the submitted sentence pair and language codes because those values are the purpose of that feature.
Contributor names are discarded by default and are stored only when `STORE_CONTRIBUTOR_NAMES=true` is explicitly configured.

The SQLite administrator database and contribution JSONL file are protected by restrictive file permissions but are not encrypted by the application.
Use encrypted host disks or encrypted container volumes when those files contain sensitive information.
Define and enforce a retention period before enabling contribution collection.

## Secret handling

The Compose deployment mounts `.env` as a read-only Docker secret file.
Only allowlisted application settings are loaded from that file.
Passwords and API keys are not placed in the container configuration environment.
The administrator password is removed from the server process environment immediately after seeding.
Browser variables prefixed with `VITE_` are public and must never contain secrets.

Restrict access to the host `.env` file because it remains plaintext on the operator's machine.
Use a managed secret store for a public or multi-operator production deployment.

## Network and API controls

Nginx applies request and connection limits, a 16 KiB request-body limit, strict browser security headers, and same-origin API proxying.
FastAPI independently validates trusted Host values, CORS origins, body size, input schemas, response contracts, and per-client request rates.
The browser client accepts only same-origin API paths and rejects oversized or malformed API responses.
Contribution writes fail closed when no contribution secret is configured.
Interactive API documentation and detailed health information default to disabled in production.

The in-process rate limiter is defense in depth for one backend process.
Use a shared edge rate limiter when the backend is scaled to multiple replicas.

## Container controls

The frontend and backend bind to `127.0.0.1` by default.
Both containers run as unprivileged numeric users with all Linux capabilities dropped, `no-new-privileges`, PID limits, log rotation, and read-only root filesystems.
Only the backend data volume and small temporary filesystems are writable.
Runtime base images are pinned by digest.

## Production checklist

- Set `APP_ENV=production`.
- Set the exact public host names in `ALLOWED_HOSTS`.
- Leave `CORS_ORIGINS` empty for same-origin deployment or list only exact trusted origins.
- Keep `ENABLE_API_DOCS=false` and `EXPOSE_HEALTH_DETAILS=false` unless there is a specific operational need.
- Terminate HTTPS at a trusted reverse proxy and enable HSTS there after HTTPS is working on every subdomain.
- Store `.env`, database files, contribution files, model weights, and backups on encrypted storage with restricted operator access.
- Rotate administrator and API secrets after suspected exposure and on a documented schedule.
- Do not enable contributions until consent, retention, deletion, and reviewer-access policies are approved.
- Add authenticated administration endpoints only with session security, authorization, audit logging, CSRF protection, and brute-force controls.
- Run dependency audits, image scans, tests, and lint checks before every release.

## Verification commands

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check app scripts tests
uvx pip-audit -r requirements.txt
uvx pip-audit -r requirements-dev.txt
uvx pip-audit -r requirements-train.txt
docker compose config --quiet
```

The image scan can be run with an authenticated Docker Scout installation or a trusted pinned Trivy release.
Do not ignore a fixable high or critical finding without a written risk decision.

## Reporting a vulnerability

Do not publish passwords, tokens, personal data, model data, or exploit details in a public issue.
Send the report through the repository owner's private security-reporting channel and include affected versions, reproduction steps, impact, and any known mitigation.
