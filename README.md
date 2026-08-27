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
- `history_features.py`: serving-owned user-history normalization and padding
- `scripts/gcp_setup.sh`: one-time or occasional GCP setup for a target environment
- `scripts/deploy.sh`: deploys the service to Cloud Run from source
- `Dockerfile`: CPU-serving image used by Cloud Run source deploys
- `Dockerfile.gpu`: optional GPU-oriented Dockerfile for other environments
- `app_test.py`: service-level tests

## Prerequisites

- Python 3.11+
- `pipenv`

## Installation

1. Install dependencies:

   ```bash
   pipenv install
   ```

2. Install development dependencies:

   ```bash
   pipenv install --dev
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

- `GET /health`: unauthenticated process health check; also reports the deployed
  git sha (`{"ok":true,"git_sha":"e9f07f5"}`), used to confirm rollbacks
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

`ranker` expects one user's history plus one or more candidate post inputs.
Datetimes should be sent as timezone-aware ISO 8601 strings; the service
converts them to elapsed hours before calling the model. Ranker model artifacts
must expose the TorchScript `score_candidate_matrix` method; the current matrix
serving path is intended for one-layer BST ranker artifacts.

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

Or with multiple candidates:

```json
{
  "history_embeddings": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ],
  "history_author_dids": ["did:plc:history-author-1", "did:plc:history-author-2"],
  "history_liked_at_times": ["2026-06-23T10:00:00Z", "2026-06-23T09:00:00Z"],
  "candidate_post_embeddings": [
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2]
  ],
  "candidate_author_dids": ["did:plc:candidate-author-1", "did:plc:candidate-author-2"]
}
```

Ranker requests may use `history_embeddings: []` or `history_embeddings: [[]]`
for an empty single-user history. They do not accept batched user histories.
Ranker responses are flat lists of candidate scores, one score per candidate.
Scores are request-relative normalized ranker outputs in `[0, 1]`, where
higher is better. Tied or otherwise degenerate ranker logits produce neutral
`0.5` scores.

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
  --two-tower-manifest-uri gs://greenearth-471522-engagement-prediction-model-stage/.../two_tower_serving_manifest.json \
  --two-tower-author-map-uri gs://my-bucket/author_idx.parquet \
  --two-tower-max-history-len 128
```

To deploy the ranker too, include the ranker manifest, author map, and max
history length:

```bash
./scripts/deploy.sh \
  --models user-tower,post-tower,ranker \
  --two-tower-manifest-uri gs://greenearth-471522-engagement-prediction-model-stage/.../two_tower_serving_manifest.json \
  --two-tower-author-map-uri gs://my-bucket/two_tower_author_idx.parquet \
  --two-tower-max-history-len 128 \
  --ranker-manifest-uri gs://greenearth-471522-engagement-prediction-model-stage/.../ranker_serving_manifest.json \
  --ranker-author-map-uri gs://my-bucket/ranker_author_idx.parquet \
  --ranker-max-history-len 128
```

Or with environment variables:

```bash
GE_ENVIRONMENT=prod \
GE_INFERENCE_MODELS=user-tower,post-tower \
GE_INFERENCE_TWO_TOWER_MANIFEST_URI=gs://greenearth-471522-engagement-prediction-model-prod/.../two_tower_serving_manifest.json \
GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI=gs://my-bucket/author_idx.parquet \
GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN=128 \
GE_INFERENCE_RANKER_MANIFEST_URI=gs://greenearth-471522-engagement-prediction-model-prod/.../ranker_serving_manifest.json \
GE_INFERENCE_RANKER_AUTHOR_MAP_URI=gs://my-bucket/ranker_author_idx.parquet \
GE_INFERENCE_RANKER_MAX_HISTORY_LEN=128 \
./scripts/deploy.sh
```

### Scaling and Concurrency

The deploy script configures Cloud Run revision-level scaling and request
concurrency. Its defaults preserve the current deployment baseline:

- minimum instances: `2`
- maximum instances: `8`
- maximum concurrent requests per instance: `2`

Override these values with `--min-instances`, `--max-instances`, and
`--concurrency`, or with the corresponding environment variables. For example,
to compare per-instance performance at concurrency `2`, keep the service pinned
to one instance so autoscaling does not affect the measurement:

```bash
# With the required model configuration already exported:
GE_INFERENCE_MIN_INSTANCES=1 \
GE_INFERENCE_MAX_INSTANCES=1 \
GE_INFERENCE_CONCURRENCY=2 \
./scripts/deploy.sh
```

Cloud Run concurrency is the maximum number of in-flight HTTP requests routed
to one container instance, not a requests-per-second limit. After selecting a
stable per-instance concurrency through stage load testing, raise
`--max-instances` to test horizontal autoscaling. Once a value is validated,
update the default so later deployments do not silently restore the baseline.

The two tower manifest (`two_tower_serving_manifest.json`) is produced by the
engagement-prediction training pipeline and uploaded to the model bucket. It
contains the GCS URIs and ClearML model IDs for both towers. `GE_INFERENCE_MODELS`
still controls which models are actually loaded.

During deploy, the script will:

- refuse to deploy from a dirty working tree, and resolve the short git sha
- validate the required model configuration
- generate `requirements.txt` from `Pipfile`
- verify whether the shared VPC connector exists
- deploy the service to Cloud Run with the right env vars and secret bindings
- point traffic at the newly created revision

### Deployments must be from a clean tree (git sha traceability)

So we always know exactly what code is live, `deploy.sh` **refuses to deploy
with uncommitted changes**. Deploying an unpushed branch is fine — only a dirty
working tree is rejected. Commit or stash first, then deploy.

Each deploy stamps its short git sha in two places:

- **Cloud Run env var `GE_GIT_SHA`** — the running app reports it at
  `GET /health` (`{"ok":true,"git_sha":"e9f07f5"}`). Outside a stamped
  deployment (e.g. locally) `git_sha` is `null`.
- **Cloud Run label `git-sha=<sha>`** — tags the service/revision so past
  deployments are identifiable when picking a rollback target
  (`./scripts/rollback.sh --list`).

### Rolling back a deployment

`deploy.sh` builds from source, so the repo never names an image tag — the
durable record of a past deployment is its **Cloud Run revision**, which pins
the built image digest along with the env configuration it ran with. Rolling
back re-points traffic at an older revision, with no rebuild.

```bash
./scripts/rollback.sh --environment prod --list   # see candidates + git shas
./scripts/rollback.sh --environment prod          # back to the previous deploy
./scripts/rollback.sh --environment prod --to 7176a35   # or a specific target
```

`--to` accepts a revision name or a git sha. With no `--to`, the target is the
newest Ready revision older than the one serving, with a different git sha. The
script prompts for confirmation (`--yes` skips it, `--dry-run` shows the exact
`gcloud` command without running it), then polls `GET /health` until the service
reports the target's sha.

**Model manifests travel with the revision.** `GE_INFERENCE_MODELS` and the
`*_MANIFEST_URI` values are part of each revision's env configuration, so a
rollback restores the code together with the model artifacts it was deployed
against — rolling code back without its manifests would pair old code with newer
model files. The script prints both revisions' model configuration before asking
for confirmation, because the API is the caller here: if the target revision
serves a different set of models, check that the deployed API still expects them.

Domain mappings (`inference[-stage].greenearth.social`) target the service, not a
revision, so they follow the rollback with no extra work. `/health` answers as
soon as the process is up; model loading is separate, so check `/ready` (needs
`X-API-Key`) if you need to confirm the models finished loading.

Revisions deployed before git-sha stamping show as `(unstamped)` in `--list`.
They are still valid `--to` targets by revision name; they just cannot
self-report a sha, so `/health` verification is skipped for them.

**Getting back out:** a rollback pins traffic to a named revision, taking
`LATEST` out of the traffic split. `deploy.sh` resets traffic to `LATEST` after
every successful deploy, so deploying the fix is all it takes. Because the reset
runs only on success, a failed build leaves traffic on the rolled-back revision.

Rollbacks are manual by design. Cloud Run's own health-check behavior is
untouched — a revision that never becomes Ready never receives traffic.

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
- `GE_INFERENCE_MIN_INSTANCES`: minimum Cloud Run instances, default `2`
- `GE_INFERENCE_MAX_INSTANCES`: maximum Cloud Run instances, default `8`
- `GE_INFERENCE_CONCURRENCY`: maximum concurrent requests per Cloud Run instance, default `2`

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
