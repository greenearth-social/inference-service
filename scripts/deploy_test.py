"""Exercise the readiness gate and traffic changes without cloud access."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).with_name("deploy.sh")
REVISION = "inference-stage-00002-checked"
TAGGED_URL = "https://deployment-check.test.run.app"

# Keep the real deployment command builder and orchestration. Fake only external
# commands and unrelated preflight work so these tests cannot contact GCP.
HARNESS = r"""
record() { printf '%s\n' "$1" >> "$TEST_DIRECTORY/events"; }

gcloud() {
    case "$1 $2 $3" in
        "config set project") record configure_project ;;
        "run deploy engagement-prediction-inference-"*)
            record deploy
            printf '%s\0' "$@" > "$TEST_DIRECTORY/deploy_args"
            local tag argument
            for argument in "$@"; do
                case "$argument" in
                    --tag=*) tag="${argument#--tag=}" ;;
                    --env-vars-file=*)
                        dirname "${argument#--env-vars-file=}" > "$TEST_DIRECTORY/temp_directory"
                        ;;
                esac
            done
            printf '%s' "$tag" > "$TEST_DIRECTORY/tag"
            if [ "${FAKE_DEPLOY_STATUS:-0}" != 0 ]; then
                printf '%s\n' '--no-traffic not supported when creating a new service.' >&2
                return "$FAKE_DEPLOY_STATUS"
            fi
            printf '{"status":{"url":"https://old-service.run.app","traffic":[{"tag":"unrelated","revisionName":"old-revision","url":"https://old-tag.run.app"},{"tag":"%s","revisionName":"%s","url":"%s"}]}}\n' \
                "$tag" "$FAKE_REVISION" "$FAKE_TAGGED_URL"
            ;;
        "run services describe") printf '%s\n' 'https://old-service.run.app' ;;
        "run services update-traffic")
            local argument
            for argument in "$@"; do
                case "$argument" in
                    --remove-tags=*)
                        record cleanup_tag
                        printf '%s\0' "$@" > "$TEST_DIRECTORY/cleanup_args"
                        return "${FAKE_CLEANUP_STATUS:-0}"
                        ;;
                esac
            done
            record promote
            printf '%s\0' "$@" > "$TEST_DIRECTORY/promotion_args"
            return "${FAKE_PROMOTION_STATUS:-0}"
            ;;
        "secrets versions access")
            record read_secret
            printf '%s\0' "$@" > "$TEST_DIRECTORY/secret_args"
            if [ "${FAKE_SECRET_STATUS:-0}" != 0 ]; then
                printf '%s\n' 'Permission denied reading API key' >&2
                return "$FAKE_SECRET_STATUS"
            fi
            printf '%s' "$FAKE_API_KEY"
            ;;
        *) record unexpected_gcloud_command; return 90 ;;
    esac
}

pipenv() {
    if [ "$1 $2" != "run python" ]; then
        record unexpected_pipenv_command
        return 90
    fi
    shift 2
    "$TEST_PYTHON" "$@"
}

curl() {
    local count=0
    if [ -f "$TEST_DIRECTORY/request_count" ]; then
        count=$(cat "$TEST_DIRECTORY/request_count")
    fi
    count=$((count + 1))
    printf '%s' "$count" > "$TEST_DIRECTORY/request_count"
    record "ready_$count"
    printf '%s\0' "$@" > "$TEST_DIRECTORY/curl_args_$count"
    local output header
    while [ $# -gt 0 ]; do
        case "$1" in
            --output) output="$2"; shift 2 ;;
            --header) header="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    cat "${header#@}" > "$TEST_DIRECTORY/request_header"
    local codes=($FAKE_READY_CODES) statuses=($FAKE_CURL_STATUSES)
    local index=$((count - 1))
    local code="${codes[$index]:-${codes[${#codes[@]}-1]}}"
    local status="${statuses[$index]:-${statuses[${#statuses[@]}-1]}}"
    if [ "$code" = 200 ]; then
        printf '%s' '{"ready":true}' > "$output"
    else
        printf '%s' '{"ready":false,"models":[{"load_error":"Model artifact access denied"}]}' > "$output"
    fi
    printf '%s' "$code"
    return "$status"
}

sleep() {
    record "sleep_$1"
    SECONDS=$((SECONDS + $1))
}

deploy_script="$1"
shift
source "$deploy_script" "$@"

require_clean_worktree() { record clean_worktree; GIT_SHA=1234567; }
generate_requirements() { record generate_requirements; }
verify_vpc_connector() { record check_connector; VPC_CONNECTOR_EXISTS=false; }
reconcile_domain_mapping() { record domain_mapping; }
READY_TIMEOUT_SEC=11
main
"""


def run_deploy(tmp_path: Path, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {name: value for name, value in os.environ.items() if not name.startswith(("GE_", "FAKE_"))}
    env.update(
        TEST_DIRECTORY=str(tmp_path),
        TEST_PYTHON=sys.executable,
        PATH="/usr/bin:/bin",
        GE_INFERENCE_MODELS="user-tower,post-tower",
        GE_INFERENCE_TWO_TOWER_MANIFEST_URI="gs://models/manifest.json",
        GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI="gs://models/authors.parquet",
        FAKE_REVISION=REVISION,
        FAKE_TAGGED_URL=TAGGED_URL,
        FAKE_API_KEY="test-secret-key",
        FAKE_READY_CODES="200",
        FAKE_CURL_STATUSES="0",
    )
    env.update(environment or {})
    return subprocess.run(
        ["/bin/bash", "-c", HARNESS, "deployment-test", str(DEPLOY_SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def recorded_args(tmp_path: Path, name: str) -> list[str]:
    return (tmp_path / name).read_text().removesuffix("\0").split("\0")


def events(tmp_path: Path) -> list[str]:
    return (tmp_path / "events").read_text().splitlines()


def assert_cleaned_up(tmp_path: Path) -> None:
    tag = (tmp_path / "tag").read_text()
    assert f"--remove-tags={tag}" in recorded_args(tmp_path, "cleanup_args")
    assert not Path((tmp_path / "temp_directory").read_text().strip()).exists()


def assert_not_promoted(tmp_path: Path, result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0
    assert "promote" not in events(tmp_path)
    assert "domain_mapping" not in events(tmp_path)
    assert "Deployment complete!" not in result.stdout
    assert_cleaned_up(tmp_path)


def test_ready_revision_is_checked_before_receiving_traffic(tmp_path: Path) -> None:
    result = run_deploy(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    deploy_args = recorded_args(tmp_path, "deploy_args")
    tag = (tmp_path / "tag").read_text()
    assert "--no-traffic" in deploy_args
    assert tag.startswith("deploy-check-")
    assert f"--tag={tag}" in deploy_args
    assert "--format=json" in deploy_args
    curl_args = recorded_args(tmp_path, "curl_args_1")
    assert f"{TAGGED_URL}/ready" in curl_args
    assert (tmp_path / "request_header").read_text().strip() == "X-API-Key: test-secret-key"
    assert "test-secret-key" not in result.stdout + result.stderr
    assert recorded_args(tmp_path, "secret_args") == [
        "secrets", "versions", "access", "latest",
        "--secret=inference-api-key-stage", "--project=greenearth-471522",
    ]
    promotion_args = recorded_args(tmp_path, "promotion_args")
    assert f"--to-revisions={REVISION}=100" in promotion_args
    assert "--to-latest" not in promotion_args
    sequence = events(tmp_path)
    assert sequence.index("deploy") < sequence.index("ready_1") < sequence.index("promote")
    assert sequence.index("promote") < sequence.index("domain_mapping")
    assert_cleaned_up(tmp_path)


@pytest.mark.parametrize(
    "environment",
    [
        {"FAKE_READY_CODES": "503 200"},
        {"FAKE_READY_CODES": "000 200", "FAKE_CURL_STATUSES": "7 0"},
    ],
)
def test_transient_readiness_failures_are_retried(tmp_path: Path, environment: dict[str, str]) -> None:
    result = run_deploy(tmp_path, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    assert events(tmp_path).index("ready_2") < events(tmp_path).index("promote")
    assert_cleaned_up(tmp_path)


@pytest.mark.parametrize(
    "environment",
    [
        {"FAKE_READY_CODES": "503"},
        {"FAKE_READY_CODES": "000", "FAKE_CURL_STATUSES": "28"},
    ],
)
def test_readiness_timeout_preserves_previous_traffic(tmp_path: Path, environment: dict[str, str]) -> None:
    result = run_deploy(tmp_path, environment)

    assert_not_promoted(tmp_path, result)
    count = int((tmp_path / "request_count").read_text())
    assert 1 <= count <= 3
    for attempt in range(1, count + 1):
        arguments = recorded_args(tmp_path, f"curl_args_{attempt}")
        assert 0 < int(arguments[arguments.index("--max-time") + 1]) <= 11
    if environment["FAKE_READY_CODES"] == "503":
        assert "Model artifact access denied" in result.stdout + result.stderr


@pytest.mark.parametrize("http_status", ["401", "403"])
def test_authentication_failure_stops_without_retrying(tmp_path: Path, http_status: str) -> None:
    result = run_deploy(tmp_path, {"FAKE_READY_CODES": http_status})

    assert_not_promoted(tmp_path, result)
    assert (tmp_path / "request_count").read_text() == "1"


@pytest.mark.parametrize("environment", [{"FAKE_SECRET_STATUS": "1"}, {"FAKE_API_KEY": ""}])
def test_missing_api_key_prevents_readiness_and_promotion(tmp_path: Path, environment: dict[str, str]) -> None:
    result = run_deploy(tmp_path, environment)

    assert_not_promoted(tmp_path, result)
    assert not (tmp_path / "request_count").exists()


@pytest.mark.parametrize("environment", [{"FAKE_REVISION": ""}, {"FAKE_TAGGED_URL": ""}])
def test_missing_tag_metadata_fails_closed(tmp_path: Path, environment: dict[str, str]) -> None:
    result = run_deploy(tmp_path, environment)

    assert_not_promoted(tmp_path, result)
    assert not (tmp_path / "request_count").exists()


def test_failed_first_deployment_never_retries_with_traffic_enabled(tmp_path: Path) -> None:
    result = run_deploy(tmp_path, {"FAKE_DEPLOY_STATUS": "1"})

    assert_not_promoted(tmp_path, result)
    assert events(tmp_path).count("deploy") == 1
    assert "--no-traffic" in recorded_args(tmp_path, "deploy_args")
    assert not (tmp_path / "request_count").exists()


def test_failed_promotion_is_not_reported_as_success(tmp_path: Path) -> None:
    result = run_deploy(tmp_path, {"FAKE_PROMOTION_STATUS": "1"})

    assert result.returncode != 0
    assert f"--to-revisions={REVISION}=100" in result.stdout + result.stderr
    assert "domain_mapping" not in events(tmp_path)
    assert "Deployment complete!" not in result.stdout
    assert_cleaned_up(tmp_path)


@pytest.mark.parametrize("ready_status", ["200", "403"])
def test_cleanup_failure_preserves_deployment_result_and_removes_key(
    tmp_path: Path, ready_status: str
) -> None:
    result = run_deploy(tmp_path, {"FAKE_CLEANUP_STATUS": "1", "FAKE_READY_CODES": ready_status})

    assert (result.returncode == 0) is (ready_status == "200")
    assert "Could not remove temporary tag" in result.stdout
    assert_cleaned_up(tmp_path)
