#!/bin/bash

set -euo pipefail

TEST_BUCKET="greenearth-471522-engagement-prediction-test"
STAGE_BUCKET="greenearth-471522-engagement-prediction-model-stage"
PROD_BUCKET="greenearth-471522-engagement-prediction-model-prod"

usage() {
    echo "Usage: $0 <folder> <test|stage> <ranker|two_tower>"
}

if [ "$#" -ne 3 ]; then
    usage
    exit 1
fi

FOLDER="$1"
ENVIRONMENT="$2"
MODEL_TYPE="$3"

case "$ENVIRONMENT" in
    test)
        SOURCE_BUCKET="$TEST_BUCKET"
        TARGET_BUCKET="$STAGE_BUCKET"
        ;;
    stage)
        SOURCE_BUCKET="$STAGE_BUCKET"
        TARGET_BUCKET="$PROD_BUCKET"
        ;;
    *)
        usage
        exit 1
        ;;
esac

case "$MODEL_TYPE" in
    ranker|two_tower)
        ;;
    *)
        usage
        exit 1
        ;;
esac

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SOURCE_URI="gs://$SOURCE_BUCKET/Engagement Prediction/$FOLDER"
TARGET_URI="gs://$TARGET_BUCKET/Engagement Prediction/$FOLDER"
MANIFEST_PATH="$TMP_DIR/artifacts/${MODEL_TYPE}_serving_manifest/${MODEL_TYPE}_serving_manifest.json"

gsutil -m rsync -r "$SOURCE_URI" "$TMP_DIR"

pipenv run python -c '
import json
import sys

manifest_path = sys.argv[1]
source_bucket = sys.argv[2]
target_bucket = sys.argv[3]
target_uri = sys.argv[4]
model_type = sys.argv[5]
source_prefix = f"gs://{source_bucket}/"
target_prefix = f"gs://{target_bucket}/"

with open(manifest_path, "r", encoding="utf-8") as manifest_file:
    manifest = json.load(manifest_file)

if model_type == "ranker":
    uri_checks = [("ranker_uri", "/models/ranker.pt")]
elif model_type == "two_tower":
    uri_checks = [
        ("user_tower_uri", "/models/engagement_user_tower.pt"),
        ("post_tower_uri", "/models/engagement_post_tower.pt"),
    ]
else:
    raise ValueError(f"Unknown model type: {model_type}")

for uri_key, suffix in uri_checks:
    model_uri = manifest[uri_key]
    if not model_uri.startswith(source_prefix):
        raise ValueError(f"{uri_key} does not start with {source_prefix}: {model_uri}")

    manifest[uri_key] = target_prefix + model_uri[len(source_prefix):]

    model_uri = manifest[uri_key]
    if not model_uri.endswith(suffix):
        raise ValueError(f"{uri_key} does not end with {suffix}: {model_uri}")

    model_base_uri = model_uri[:-len(suffix)]
    if model_base_uri != target_uri:
        raise ValueError(f"{uri_key} base {model_base_uri} does not match target URI {target_uri}")

with open(manifest_path, "w", encoding="utf-8") as manifest_file:
    json.dump(manifest, manifest_file, indent=2)
    manifest_file.write("\n")
' "$MANIFEST_PATH" "$SOURCE_BUCKET" "$TARGET_BUCKET" "$TARGET_URI" "$MODEL_TYPE"

gsutil -m rsync -r "$TMP_DIR" "$TARGET_URI"