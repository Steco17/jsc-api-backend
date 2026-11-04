# Render Development Deployment

The `render.yaml` Blueprint creates a development frontend and backend from two GitHub repositories.
It is intended for application testing while translation model training is incomplete.

## Services

| Service | GitHub repository | Branch | Render health check |
|---|---|---|---|
| `jsc-translator-web-dev` | `Steco17/app-jsc-translator-fe` | `dev` | `/healthz` |
| `jsc-translator-api-dev` | `Steco17/jsc-api-backend` | `dev` | `/ready` |

Both services use Render's Frankfurt region and free development plan.
Each service deploys only after the GitHub CI check for its commit passes.

The frontend sends browser requests to its own `/api` path.
Nginx forwards those requests to the backend's Render HTTPS URL by using the Blueprint-provided `BACKEND_URL` value.
This public service-to-service route is required because free Render web services cannot receive private-network traffic.

## Create the development environment

1. Confirm that the `dev` branch exists in both GitHub repositories and that both CI runs are green.
2. In Render, choose **New**, then **Blueprint**.
3. Connect `Steco17/jsc-api-backend` and select the `dev` branch containing `render.yaml`.
4. Enter a real administrator email when Render prompts for `ADMIN_EMAIL`.
5. Apply the Blueprint and wait for both services to pass their health checks.
6. Open the frontend service URL and use the backend service URL for `/docs`, `/health`, and `/ready` inspection.

Render generates the development administrator password and does not store it in Git.
The current application has no administrator login endpoint, so this seeded account only prepares the future authentication layer.

## Suspended model behavior

`TRANSLATION_MODE` is fixed to `placeholder` in the Blueprint.
The API does not download, load, or claim support for an untrained model.
Translation requests return an empty translation with `status: "model_pending"`, and the frontend explains that training is still pending.

Keep this setting until all of the following conditions are true:

- The merged tokenizer and CTranslate2 model artifacts are available to the runtime.
- The training manifest identifies the exact trained directions.
- Held-out automatic evaluation has passed for every enabled direction.
- Native speakers have reviewed unseen examples for every enabled language.
- The registry marks only approved languages as `fine_tuned`.

## Development data limits

Contribution writes are disabled because the Blueprint does not configure `CONTRIBUTION_API_KEY`.
The free development service has an ephemeral filesystem, so its seeded SQLite database can be recreated after a restart or redeploy.
Do not collect durable user contributions in this environment.
Attach persistent storage at `/srv/data` and configure a contribution key before enabling contribution collection in a non-development environment.

## Smoke test

After deployment, verify these requests through the frontend URL:

```text
GET /healthz                         -> 200
GET /api/ready                      -> 200 and status "ready"
GET /api/languages                  -> 200 and a non-empty registry
POST /api/translate                 -> 200 and status "model_pending"
```

Use this JSON body for the translation smoke test:

```json
{
  "text": "Good morning",
  "src_lang": "eng_Latn",
  "tgt_lang": "ewo_Latn"
}
```

Also verify that the frontend `/translator` page loads, lists languages, submits the request, and presents the model-pending message without a console or network error.

## Rollback triggers

Roll back to the previous successful Render deploy when any of these conditions occurs:

- Either service fails its configured health check for more than five minutes.
- The frontend cannot load the language registry through `/api`.
- A valid placeholder translation request returns an unexpected status or non-empty generated text.
- The frontend sends API traffic to a public origin instead of its own `/api` path.
- GitHub CI fails or Render builds a different commit than the expected `dev` head.

Render retains previous successful deploys for manual rollback.
Keep `TRANSLATION_MODE=placeholder` during rollback unless a separately approved model release is being restored.
