# Green Earth Inference Service

FastAPI service for serving Green Earth model inference endpoints on Google Cloud Run.

Today this repo is focused on the engagement-prediction models:

- `user-tower`: scores a user's embedding-history sequence
- `post-tower`: scores one or more post embedding vectors
- `ranker`: scores candidate posts against a user's embedding-history sequence

The service loads the two tower models from a `two_tower_serving_manifest.json`
file produced by the engagement-prediction training pipeline and uploaded to GCS.
When `ranker` is configured, it also loads a `ranker_serving_manifest.json`.
Each manifest contains model artifact URIs and ClearML model IDs for the models
it describes.

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
- `GE_INFERENCE_CONTENT_EMBED_DIM`
- `GE_INFERENCE_TWO_TOWER_MANIFEST_URI` — GCS URI or local path to `two_tower_serving_manifest.json`
- `GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI` — required when loading `user-tower` or `post-tower`
- `GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN` — required when loading `user-tower`
- `GE_INFERENCE_RANKER_MANIFEST_URI` — required when loading `ranker`
- `GE_INFERENCE_RANKER_AUTHOR_MAP_URI` — required when loading `ranker`
- `GE_INFERENCE_RANKER_MAX_HISTORY_LEN` — required when loading `ranker`
- `GE_INFERENCE_API_KEY` if you want to call protected endpoints locally

Then start the server:

```bash
pipenv run uvicorn app:app --reload
```

The inference service API will be available at `http://localhost:8000`.

When you want the Green Earth API to call your local inference instance, override the Green Earth API deployment with:

```bash
GE_INFERENCE_BASE_URL="http://127.0.0.1:8000" ./scripts/deploy.sh --environment stage
```

That explicit base URL override takes precedence over mapped domains.


### Running with Docker

If you want to run the service in Docker from the repo root, first create an
`.env` file with the variables the service needs. A common starting point is:

```bash
cp .env.example .env
```

Then build and run the container:

```bash
docker build -t ge-inference-service .
docker run --rm -p 8080:8080 --env-file .env ge-inference-service
```

The API will be available at `http://localhost:8080`.

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

`ranker` expects user history inputs plus candidate post inputs. Datetimes should
be sent as timezone-aware ISO 8601 strings; the service converts them to elapsed
hours before calling the model.

```json
{
  "history_embeddings": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ],
  "history_author_dids": ["did:plc:history-author-1", "did:plc:history-author-2"],
  "history_liked_at_times": ["2026-06-23T10:00:00Z", "2026-06-23T09:00:00Z"],
  "candidate_post_embeddings": [0.7, 0.8, 0.9],
  "candidate_author_dids": "did:plc:candidate-author"
}
```

Or batched:

```json
{
  "history_embeddings": [
    [[0.1, 0.2, 0.3]],
    [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
  ],
  "history_author_dids": [
    ["did:plc:history-author-1"],
    ["did:plc:history-author-2", "did:plc:history-author-3"]
  ],
  "history_liked_at_times": [
    ["2026-06-23T10:00:00Z"],
    ["2026-06-23T09:00:00Z", "2026-06-23T08:00:00Z"]
  ],
  "candidate_post_embeddings": [
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2]
  ],
  "candidate_author_dids": ["did:plc:candidate-author-1", "did:plc:candidate-author-2"]
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

The current deploy script examples below configure the two tower models. If you
deploy `ranker`, the Cloud Run environment must also receive the
`GE_INFERENCE_RANKER_*` variables listed in Configuration Inputs.

Deploy to Cloud Run:

```bash
# staging (default)
./scripts/deploy.sh \
  --models user-tower,post-tower \
  --two-tower-manifest-uri gs://greenearth-471522-engagement-prediction-model-stage/.../two_tower_serving_manifest.json \
  --two-tower-author-map-uri gs://my-bucket/author_idx.parquet \
  --two-tower-max-history-len 128
```

Or with environment variables:

```bash
GE_ENVIRONMENT=prod \
GE_INFERENCE_MODELS=user-tower,post-tower \
GE_INFERENCE_TWO_TOWER_MANIFEST_URI=gs://greenearth-471522-engagement-prediction-model-prod/.../two_tower_serving_manifest.json \
GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI=gs://my-bucket/author_idx.parquet \
GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN=128 \
./scripts/deploy.sh
```

The two tower manifest (`two_tower_serving_manifest.json`) is produced by the
engagement-prediction training pipeline and uploaded to the model bucket. It
contains the GCS URIs and ClearML model IDs for both towers. `GE_INFERENCE_MODELS`
still controls which models are actually loaded.

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

- `GE_INFERENCE_MODELS`: comma-separated model list; supported values are `user-tower`, `post-tower`, and `ranker`
- `GE_INFERENCE_TWO_TOWER_MANIFEST_URI`: GCS URI or local path to `two_tower_serving_manifest.json`; contains model artifact URIs and ClearML model IDs for both towers
- `GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI`: GCS URI or local path for the two tower author idx parquet map; required when loading `user-tower` or `post-tower`
- `GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN`: max history length for user-tower inputs; required when loading `user-tower`
- `GE_INFERENCE_RANKER_MANIFEST_URI`: GCS URI or local path to `ranker_serving_manifest.json`; required when loading `ranker`
- `GE_INFERENCE_RANKER_AUTHOR_MAP_URI`: GCS URI or local path for the ranker author idx parquet map; required when loading `ranker`
- `GE_INFERENCE_RANKER_MAX_HISTORY_LEN`: max history length for ranker history inputs; required when loading `ranker`

Runtime configuration used by the app:

- `GE_INFERENCE_API_KEY`: required for protected endpoints
- `GE_INFERENCE_MAX_BATCH`: maximum allowed batch size
- `GE_INFERENCE_PREFER_CUDA`: choose CUDA when available
- `GE_INFERENCE_WARMUP`: whether to run warmup on startup
- `GE_INFERENCE_CONTENT_EMBED_DIM`: required input content embedding dimension
- `GE_INFERENCE_MODEL_CACHE_DIR`: local cache dir for downloaded `gs://` models
