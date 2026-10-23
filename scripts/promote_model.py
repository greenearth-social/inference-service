"""Promote one ranker run from staging storage to production storage."""

import argparse
import json
import shlex

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage


STAGE_BUCKET = "greenearth-471522-engagement-prediction-model-stage"
PROD_BUCKET = "greenearth-471522-engagement-prediction-model-prod"

MODEL_PATH = "models/ranker.pt"
AUTHOR_MAP_PATH = "artifacts/author_idx_mapping/ranker_author_idx.parquet"
CONFIG_PATH = "artifacts/bst_model_config/model_config.json"
MANIFEST_PATH = "artifacts/ranker_serving_manifest/ranker_serving_manifest.json"
REQUIRED_PATHS = (MODEL_PATH, AUTHOR_MAP_PATH, CONFIG_PATH, MANIFEST_PATH)


def _read_json(blob: storage.Blob) -> dict:
    value = json.loads(blob.download_as_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {blob.name}")
    return value


def promote_ranker(folder: str, client: storage.Client) -> dict[str, str | int]:
    folder = folder.strip().strip("/")
    if not folder or folder.startswith("gs://"):
        raise ValueError("Provide the run folder beneath 'Engagement Prediction/'")

    prefix = f"Engagement Prediction/{folder}/"
    source_uri = f"gs://{STAGE_BUCKET}/{prefix}"
    destination_uri = f"gs://{PROD_BUCKET}/{prefix}"
    source_bucket = client.bucket(STAGE_BUCKET)
    destination_bucket = client.bucket(PROD_BUCKET)
    blobs = {
        blob.name[len(prefix):]: blob
        for blob in client.list_blobs(source_bucket, prefix=prefix)
    }

    # Check the complete source bundle before writing anything to production.
    for path in REQUIRED_PATHS:
        if path not in blobs or not blobs[path].size:
            raise ValueError(f"Required source file is missing or empty: {source_uri}{path}")

    manifest = _read_json(blobs[MANIFEST_PATH])
    if manifest.get("ranker_uri") != source_uri + MODEL_PATH:
        raise ValueError("Manifest ranker_uri does not match the selected staging run")
    for key in ("clearml_task_id", "ranker_clearml_model_id"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manifest requires a nonempty {key}")

    # Production comparisons also load this config alongside the serving files.
    model_config = _read_json(blobs[CONFIG_PATH])
    max_history_len = model_config.get("max_history_len")
    if type(max_history_len) is not int or max_history_len <= 0:
        raise ValueError("Model config max_history_len must be a positive integer")

    # Keep reports and other companion artifacts with the model. Copy within GCS
    # so model files do not need to pass through the machine running this script.
    for path, blob in blobs.items():
        if path != MANIFEST_PATH:
            source_bucket.copy_blob(
                blob,
                destination_bucket,
                new_name=blob.name,
                source_generation=blob.generation,
            )

    for path in (MODEL_PATH, AUTHOR_MAP_PATH, CONFIG_PATH):
        source = blobs[path]
        destination = destination_bucket.get_blob(source.name)
        if (
            destination is None
            or destination.size != source.size
            or destination.crc32c != source.crc32c
        ):
            raise ValueError(f"Destination verification failed: {destination_uri}{path}")

    # Publish the rewritten manifest only after its required files are verified.
    manifest["ranker_uri"] = destination_uri + MODEL_PATH
    destination_manifest = destination_bucket.blob(prefix + MANIFEST_PATH)
    destination_manifest.upload_from_string(
        json.dumps(manifest, indent=2) + "\n", content_type="application/json"
    )
    if _read_json(destination_manifest) != manifest:
        raise ValueError(f"Destination manifest verification failed: {destination_uri}{MANIFEST_PATH}")

    return {
        "manifest_uri": destination_uri + MANIFEST_PATH,
        "author_map_uri": destination_uri + AUTHOR_MAP_PATH,
        "max_history_len": max_history_len,
        "clearml_task_id": manifest["clearml_task_id"],
        "ranker_clearml_model_id": manifest["ranker_clearml_model_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Run folder beneath 'Engagement Prediction/' in the stage bucket")
    args = parser.parse_args()

    try:
        result = promote_ranker(args.folder, storage.Client())
    except (GoogleAPIError, GoogleAuthError, ValueError) as exc:
        parser.exit(1, f"Promotion failed: {exc}\n")

    print("Promotion complete. Production deployment settings:")
    for setting, key in (
        ("GE_INFERENCE_RANKER_MANIFEST_URI", "manifest_uri"),
        ("GE_INFERENCE_RANKER_AUTHOR_MAP_URI", "author_map_uri"),
        ("GE_INFERENCE_RANKER_MAX_HISTORY_LEN", "max_history_len"),
    ):
        print(f"{setting}={shlex.quote(str(result[key]))}")
    print(f"Source ClearML run ID: {result['clearml_task_id']}")
    print(f"Source ClearML model ID: {result['ranker_clearml_model_id']}")


if __name__ == "__main__":
    main()
