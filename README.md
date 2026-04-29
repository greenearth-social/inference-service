# Green Earth Inference Service

FastAPI service for serving Green Earth model inference endpoints on Google Cloud Run.

Today this repo is focused on the engagement-prediction models, especially the
two-tower retrieval models:

- `user-tower`: scores a user's embedding-history sequence
- `post-tower`: scores one or more post embedding vectors

The service can load models from a local file path, a `gs://...` GCS URI, or a
ClearML model ID.

## Contributing

Interested in contributing? We'd love to have you!

First, please join our discord and introduce yourself: https://discord.com/invite/8bWEyrkrJC. Unless you've joined the discord and engaged with the community there, all issues/PRs will be auto-closed.

## Repository Layout

- `app.py`: FastAPI app, model-loading logic, request validation, and inference endpoints
- `scripts/gcp_setup.sh`: one-time or occasional GCP setup for a target environment
- `scripts/deploy.sh`: deploys the service to Cloud Run from source
- `Dockerfile`: CPU-serving image used by Cloud Run source deploys
- `Dockerfile.gpu`: optional GPU-oriented Dockerfile for other environments
- `app_test.py`: service-level tests

## Prerequisites

- Python 3.11+
- `pipenv`
- Access to the `engagement-prediction` repository

## Installation

1. Install dependencies:

   ```bash
   pipenv install
   ```

2. Install development dependencies:

   ```bash
   pipenv install --dev
   ```

## Shared ML Code (`shared` package)

This service depends on the `shared` package from the
[engagement-prediction](https://github.com/greenearth-social/engagement-prediction)
repo. That package contains preprocessing and input-shaping helpers that are
shared between training and serving so the inference contract stays aligned with
the code that produced the model.

The dependency is declared in `Pipfile` as a git dependency, and the exact
resolved version is pinned in `Pipfile.lock`.

### Updating to the latest shared code

```bash
pipenv lock
pipenv install
```

This re-resolves the git dependency and updates the pinned commit in
`Pipfile.lock`. Commit the updated lockfile so deploys and teammates use the
same shared-code version.

### Pinning a specific commit or branch

Edit the `ref` in `Pipfile`, then re-lock:

```toml
shared = {git = "https://github.com/greenearth-social/engagement-prediction.git", ref = "<commit-sha-or-branch>"}
```

```bash
pipenv lock
pipenv install
```

## Running Locally

This service reads its configuration from environment variables. It does not
automatically load `.env`, so source one of the env files before running it.

Example:

```bash
source .env.example
```

At minimum you should set:

- `GE_INFERENCE_MODELS`
- one source per configured model, such as `GE_INFERENCE_USER_TOWER_MODEL_URI`
- `GE_INFERENCE_MAX_HISTORY_LEN`
- `GE_INFERENCE_API_KEY` if you want to call protected endpoints locally

Then start the server:

```bash
pipenv run uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /health`: unauthenticated process health check
- `GET /ready`: authenticated readiness check including model load status
- `GET /models`: authenticated list of registered models and load state
- `POST /models/{model_name}/predict`: authenticated inference endpoint

Authentication for protected endpoints uses the `X-API-Key` header and is
validated against `GE_INFERENCE_API_KEY`.

### Example request shapes

`user-tower` expects `history_embeddings` in either single-user or batched form:

```json
{
  "history_embeddings": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ]
}
```

Or:

```json
{
  "history_embeddings": [
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    [[0.7, 0.8, 0.9]]
  ]
}
```

`post-tower` expects `post_embeddings` as either one vector or a batch:

```json
{
  "post_embeddings": [0.1, 0.2, 0.3]
}
```

Or:

```json
{
  "post_embeddings": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ]
}
```

## Running Tests

Run all tests:

```bash
pipenv run pytest
```

## Deployment

The service is deployed to Google Cloud Run from source using the repo
`Dockerfile`.

The scripts are idempotent and safe to re-run.

### Stable Domains

The service supports stable Cloud Run domain mappings:

- stage: <https://inference-stage.greenearth.social>
- prod: <https://inference.greenearth.social>

### Prerequisites for Deployment

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- appropriate GCP permissions for Cloud Run, Cloud Build, Secret Manager, and Storage

### First-Time Setup

Run the setup script once per environment:

```bash
# staging (default)
./scripts/gcp_setup.sh

# production
GE_ENVIRONMENT=prod ./scripts/gcp_setup.sh
```

This script will:

- set the active GCP project
- enable required GCP APIs
- create the environment-specific service account
- create the model storage bucket
- create the `inference-api-key-<environment>` secret
- check whether the shared VPC connector exists

Important resources created or verified by setup:

- service account: `engagement-prediction-sa-<environment>@<project>.iam.gserviceaccount.com`
- model bucket: `gs://<project>-engagement-prediction-model-<environment>`
- API key secret: `inference-api-key-<environment>`

Optional flags:

- `--inference-domain <domain>`: use a custom host
- `--disable-domain-mapping`: skip mapping/DNS setup

### Deploying the Service

Deploy to Cloud Run:

```bash
# staging (default)
./scripts/deploy.sh \
  --models user-tower,post-tower \
  --user-tower-model-uri gs://my-bucket/user_tower.pt \
  --post-tower-model-uri gs://my-bucket/post_tower.pt \
  --max-history-len 128
```

Or with environment variables:

```bash
GE_ENVIRONMENT=prod \
GE_INFERENCE_MODELS=user-tower,post-tower \
GE_INFERENCE_USER_TOWER_MODEL_URI=gs://my-bucket/user_tower.pt \
GE_INFERENCE_POST_TOWER_MODEL_URI=gs://my-bucket/post_tower.pt \
GE_INFERENCE_MAX_HISTORY_LEN=128 \
./scripts/deploy.sh
```

Each configured model must provide one source:

- `GE_INFERENCE_<MODEL>_MODEL_PATH`
- `GE_INFERENCE_<MODEL>_MODEL_URI`
- `GE_INFERENCE_<MODEL>_CLEARML_MODEL_ID`

For example, `user-tower` may use one of:

- `GE_INFERENCE_USER_TOWER_MODEL_PATH`
- `GE_INFERENCE_USER_TOWER_MODEL_URI`
- `GE_INFERENCE_USER_TOWER_CLEARML_MODEL_ID`

During deploy, the script will:

- validate the required model configuration
- generate `requirements.txt` from `Pipfile`
- verify whether the shared VPC connector exists
- deploy the service to Cloud Run with the right env vars and secret bindings

## API Security

Inference endpoints are publicly routable but protected with `X-API-Key`.
The deploy script injects `GE_INFERENCE_API_KEY` from Secret Manager:

- `inference-api-key-stage`
- `inference-api-key-prod`

Keep those secrets in sync with API service secrets for cross-service calls.

### Configuration Inputs

Common deployment configuration:

- `GE_GCP_PROJECT_ID`: GCP project ID
- `GE_GCP_REGION`: GCP region, default `us-east1`
- `GE_ENVIRONMENT`: environment name, default `stage`

Inference configuration:

- `GE_INFERENCE_MODELS`: comma-separated model list, currently `user-tower` and/or `post-tower`
- `GE_INFERENCE_MAX_HISTORY_LEN`: required max history length for user-tower inputs
- `GE_INFERENCE_USER_TOWER_MODEL_URI`: GCS URI for the user-tower model
- `GE_INFERENCE_POST_TOWER_MODEL_URI`: GCS URI for the post-tower model
- `GE_INFERENCE_USER_TOWER_CLEARML_MODEL_ID`: ClearML model ID for the user-tower model
- `GE_INFERENCE_POST_TOWER_CLEARML_MODEL_ID`: ClearML model ID for the post-tower model

Runtime configuration used by the app:

- `GE_INFERENCE_API_KEY`: required for protected endpoints
- `GE_INFERENCE_MAX_BATCH`: maximum allowed batch size
- `GE_INFERENCE_PREFER_CUDA`: choose CUDA when available
- `GE_INFERENCE_WARMUP`: whether to run warmup on startup
- `GE_INFERENCE_EMBED_DIM`: optional embedding dimension check
- `GE_INFERENCE_MODEL_CACHE_DIR`: local cache dir for downloaded `gs://` models

## Local Development

When you want API to call a local inference instance, override API deployment with:

```bash
GE_INFERENCE_BASE_URL="http://127.0.0.1:8001" ./scripts/deploy.sh --environment stage
```

That explicit base URL override takes precedence over mapped domains.
