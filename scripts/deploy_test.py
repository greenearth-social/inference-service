import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


DEPLOY_SCRIPT = Path(__file__).with_name("deploy.sh")
URIS = {
    "two_manifest": "gs://models/two.json",
    "ranker_manifest": "gs://models/ranker.json",
    "user": "gs://models/user tower.pt",
    "post": "gs://models/post.pt",
    "ranker": "gs://models/ranker.pt",
    "two_map": "gs://models/two.parquet",
    "ranker_map": "gs://models/ranker.parquet",
}


@pytest.fixture
def run_deploy(tmp_path):
    def run(models="user-tower,post-tower,ranker", command="main", **overrides):
        calls_path = tmp_path / "calls"
        stages_path = tmp_path / "stages"
        env = {
            **os.environ,
            "PATH": str(tmp_path),  # Unexpected external commands cannot reach the cloud.
            "JQ_PATH": shutil.which("jq") or "/usr/bin/jq",
            "DEPLOY_SCRIPT": str(DEPLOY_SCRIPT),
            "CALLS_PATH": str(calls_path),
            "STAGES_PATH": str(stages_path),
            "GE_GCP_PROJECT_ID": "test-project",
            "GE_ENVIRONMENT": "stage",
            "GE_INFERENCE_MODELS": models,
            "GE_INFERENCE_TWO_TOWER_MANIFEST_URI": URIS["two_manifest"],
            "GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI": URIS["two_map"],
            "GE_INFERENCE_RANKER_MANIFEST_URI": URIS["ranker_manifest"],
            "GE_INFERENCE_RANKER_AUTHOR_MAP_URI": URIS["ranker_map"],
            "TWO_MANIFEST": json.dumps({"user_tower_uri": URIS["user"], "post_tower_uri": URIS["post"]}),
            "RANKER_MANIFEST": json.dumps({"ranker_uri": URIS["ranker"]}),
            "FAIL_URI": "",
            **overrides,
        }
        result = subprocess.run(
            ["/bin/bash", "-c", r'''
source "$DEPLOY_SCRIPT"
jq() { "$JQ_PATH" "$@"; }
gcloud() {
    printf '%s\t' "$@" >> "$CALLS_PATH"
    printf '\n' >> "$CALLS_PATH"
    [[ "$1 $2" == "storage cat" ]] || return 99
    if [[ "$3" == "$FAIL_URI" ]]; then
        printf '%s\n' "$FAIL_MESSAGE" >&2
        return 1
    fi
    case "$3" in
        "$GE_INFERENCE_TWO_TOWER_MANIFEST_URI") printf '%s' "$TWO_MANIFEST" ;;
        "$GE_INFERENCE_RANKER_MANIFEST_URI") printf '%s' "$RANKER_MANIFEST" ;;
        *) printf 'x' ;;
    esac
}
require_clean_worktree() { :; }
validate_config() { printf 'validate\n' >> "$STAGES_PATH"; }
generate_requirements() { printf 'requirements\n' >> "$STAGES_PATH"; }
verify_vpc_connector() { :; }
deploy_inference_service() { printf 'deploy\n' >> "$STAGES_PATH"; }
reconcile_domain_mapping() { :; }
''' + command],
            env=env,
            capture_output=True,
            text=True,
        )
        calls = [line.rstrip("\t").split("\t") for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        stages = stages_path.read_text().splitlines() if stages_path.exists() else []
        return result, calls, stages

    return run


@pytest.mark.parametrize("models,files", [
    ("user-tower", ["two_manifest", "user", "two_map"]),
    ("post-tower", ["two_manifest", "post", "two_map"]),
    ("ranker", ["two_manifest", "ranker_manifest", "ranker", "ranker_map"]),
    ("user-tower, post-tower", ["two_manifest", "user", "post", "two_map"]),
    ("user-tower,post-tower,ranker", list(URIS)),
])
def test_checks_selected_files_as_runtime_account(run_deploy, models, files):
    result, calls, stages = run_deploy(models=models)
    assert result.returncode == 0, result.stdout + result.stderr
    assert stages == ["validate", "requirements", "deploy"]
    assert sorted(call[2] for call in calls) == sorted(URIS[file] for file in files)
    for call in calls:
        assert "--project=test-project" in call
        assert "--impersonate-service-account=engagement-prediction-sa-stage@test-project.iam.gserviceaccount.com" in call
        assert ("--range=0-0" in call) == (call[2] not in [URIS["two_manifest"], URIS["ranker_manifest"]])


@pytest.mark.parametrize("file", list(URIS))
@pytest.mark.parametrize("error", ["Object not found", "Permission denied"])
def test_file_read_failure_stops_before_build(run_deploy, file, error):
    result, calls, stages = run_deploy(FAIL_URI=URIS[file], FAIL_MESSAGE=error)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
    assert calls[-1][2] == URIS[file]
    assert stages == ["validate"]


@pytest.mark.parametrize("manifest,payload", [
    ("TWO_MANIFEST", "not json"),
    ("TWO_MANIFEST", '{}'),
    ("TWO_MANIFEST", '{"user_tower_uri":null}'),
    ("TWO_MANIFEST", '{"user_tower_uri":""}'),
    ("RANKER_MANIFEST", "not json"),
    ("RANKER_MANIFEST", '{}'),
])
def test_invalid_manifest_stops_before_build(run_deploy, manifest, payload):
    result, _, stages = run_deploy(**{manifest: payload})
    assert result.returncode != 0
    assert stages == ["validate"]


@pytest.mark.parametrize("overrides", [
    {"GE_INFERENCE_TWO_TOWER_MANIFEST_URI": "/tmp/manifest.json"},
    {"TWO_MANIFEST": '{"user_tower_uri":"https://example.com/model.pt"}'},
    {"GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI": "/tmp/authors.parquet"},
])
def test_non_gcs_file_stops_before_build(run_deploy, overrides):
    result, calls, stages = run_deploy(**overrides)
    assert result.returncode != 0
    assert all(call[2].startswith("gs://") for call in calls)
    assert stages == ["validate"]


def test_sourcing_does_not_start_deployment(run_deploy):
    result, calls, stages = run_deploy(command=":")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == []
    assert stages == []


def test_missing_jq_stops_before_build(run_deploy):
    result, calls, stages = run_deploy(command="unset -f jq; main")
    assert result.returncode != 0
    assert "jq is required" in result.stdout + result.stderr
    assert calls == []
    assert stages == ["validate"]
