"""Exercise the deployment preflight without making any Google Cloud calls."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


MANIFEST = "gs://manifests/Training run/two tower.json"
USER = "gs://weights/Training run/user.pt"
POST = "gs://weights/post.pt"
AUTHOR_MAP = "gs://maps/two tower.parquet"
RANKER_MANIFEST = "gs://manifests/ranker.json"
RANKER = "gs://ranker-weights/ranker.pt"
RANKER_MAP = "gs://maps/ranker.parquet"
FILES = [MANIFEST, USER, POST, AUTHOR_MAP, RANKER_MANIFEST, RANKER, RANKER_MAP]


@pytest.fixture
def deploy(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("jq", "date"):
        executable = shutil.which(command)
        assert executable, f"{command} is required to test deploy.sh"
        (bin_dir / command).symlink_to(executable)
    mock = bin_dir / "gcloud"
    mock.write_text(f"#!{sys.executable}\n" + '''
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[:3] == ["config", "set", "project"]:
    sys.exit(0)
if args[:2] == ["storage", "cat"]:
    uri = args[2]
    with open(os.environ["OBJECTS_FILE"]) as source:
        objects = json.load(source)
    if uri not in objects:
        print("Object not found: " + uri, file=sys.stderr)
        sys.exit(1)
    print(objects[uri])
elif args[:3] == ["policy-intelligence", "troubleshoot-policy", "iam"]:
    resource = next(arg for arg in args if arg.startswith("--resource-name="))
    denied = os.environ.get("FAIL_URI", "").replace("gs://", "projects/_/buckets/", 1)
    if denied:
        bucket, path = denied[len("projects/_/buckets/"):].split("/", 1)
        denied = "projects/_/buckets/" + bucket + "/objects/" + path
    state = os.environ["ACCESS_STATE"] if resource == "--resource-name=" + denied else "CAN_ACCESS"
    if state == "API_ERROR":
        print("Troubleshooter API unavailable", file=sys.stderr)
        sys.exit(1)
    print(state)
else:
    raise AssertionError("Unexpected gcloud invocation: " + repr(args))
''')
    mock.chmod(0o755)

    def run(models="user-tower,post-tower,ranker", *, missing_uri=None,
            fail_uri=USER, access_state="CAN_ACCESS", manifest=None, with_jq=True):
        objects = dict.fromkeys(FILES, "model contents")
        objects[MANIFEST] = json.dumps({"user_tower_uri": USER, "post_tower_uri": POST})
        objects[RANKER_MANIFEST] = json.dumps({"ranker_uri": RANKER})
        if manifest is not None:
            objects[MANIFEST] = manifest
        if missing_uri:
            objects.pop(missing_uri)
        if not with_jq:
            (bin_dir / "jq").unlink()
        objects_path = tmp_path / "objects.json"
        objects_path.write_text(json.dumps(objects))
        log_path = tmp_path / "calls.jsonl"
        env = {key: value for key, value in os.environ.items() if not key.startswith("GE_")}
        env.update({
            "PATH": str(bin_dir), "CALL_LOG": str(log_path), "OBJECTS_FILE": str(objects_path),
            "FAIL_URI": fail_uri, "ACCESS_STATE": access_state,
            "DEPLOY_SCRIPT": str(Path(__file__).with_name("deploy.sh")),
            "GE_GCP_PROJECT_ID": "test-project", "GE_ENVIRONMENT": "stage",
            "GE_ENABLE_INFERENCE_DOMAIN_MAPPING": "false", "GE_INFERENCE_MODELS": models,
            "GE_INFERENCE_TWO_TOWER_MANIFEST_URI": MANIFEST,
            "GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI": AUTHOR_MAP,
            "GE_INFERENCE_RANKER_MANIFEST_URI": RANKER_MANIFEST,
            "GE_INFERENCE_RANKER_AUTHOR_MAP_URI": RANKER_MAP,
            "GE_INFERENCE_RANKER_MAX_HISTORY_LEN": "64",
        })
        # Keep the real main/config validation; replace all build and deployment steps.
        result = subprocess.run(["/bin/bash", "-c", '''
source "$DEPLOY_SCRIPT"
require_clean_worktree() { :; }
generate_requirements() { echo BUILD_STARTED; }
verify_vpc_connector() { :; }
deploy_inference_service() { :; }
reconcile_domain_mapping() { :; }
main
'''], env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize(("models", "expected"), [
    ("user-tower,post-tower,ranker", FILES),
    (" user-tower , post-tower ", FILES[:4]),
    ("user-tower", [MANIFEST, USER, AUTHOR_MAP]),
    ("post-tower", [MANIFEST, POST, AUTHOR_MAP]),
    ("ranker", [MANIFEST, RANKER_MANIFEST, RANKER, RANKER_MAP]),
])
def test_checks_selected_objects_and_runtime_permissions(deploy, models, expected):
    result, calls = deploy(models)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILD_STARTED" in result.stdout
    reads = [args for args in calls if args[:2] == ["storage", "cat"]]
    checks = [args for args in calls if args[:3] == ["policy-intelligence", "troubleshoot-policy", "iam"]]
    assert [args[2] for args in reads] == expected
    assert len(checks) == len(expected)
    assert not any("impersonate" in arg or arg.startswith("--account") for args in calls for arg in args)
    for uri, read, check in zip(expected, reads, checks):
        bucket, name = uri.removeprefix("gs://").split("/", 1)
        assert "--project=test-project" in read
        assert ("--range=0-0" in read) == (uri not in (MANIFEST, RANKER_MANIFEST))
        assert check[3] == f"//storage.googleapis.com/projects/_/buckets/{bucket}"
        assert "--principal-email=engagement-prediction-sa-stage@test-project.iam.gserviceaccount.com" in check
        assert "--permission=storage.objects.get" in check
        assert "--resource-service=storage.googleapis.com" in check
        assert "--resource-type=storage.googleapis.com/Object" in check
        assert f"--resource-name=projects/_/buckets/{bucket}/objects/{name}" in check
        assert "--project=test-project" in check
        assert "--format=value(overallAccessState)" in check
        assert any(arg.startswith("--request-time=") and arg.endswith("Z") for arg in check)


@pytest.mark.parametrize("uri", FILES)
@pytest.mark.parametrize("failure", ["missing", "denied"])
def test_unavailable_file_stops_before_build(deploy, uri, failure):
    kwargs = {"missing_uri": uri} if failure == "missing" else {"fail_uri": uri, "access_state": "CANNOT_ACCESS"}
    result, _ = deploy(**kwargs)
    assert result.returncode != 0
    assert "BUILD_STARTED" not in result.stdout
    assert uri in result.stdout + result.stderr


@pytest.mark.parametrize("state", ["UNKNOWN_INFO", "UNKNOWN_CONDITIONAL", "", "API_ERROR"])
def test_inconclusive_permission_check_stops_before_build(deploy, state):
    result, _ = deploy(access_state=state)
    assert result.returncode != 0
    assert "BUILD_STARTED" not in result.stdout


@pytest.mark.parametrize("manifest", ["invalid JSON", "{}", '{"user_tower_uri": null}', '{"user_tower_uri": 42}'])
def test_invalid_manifest_stops_before_build(deploy, manifest):
    result, _ = deploy("user-tower", manifest=manifest)
    assert result.returncode != 0
    assert "BUILD_STARTED" not in result.stdout


@pytest.mark.parametrize("uri", ["/tmp/user.pt", "https://example.com/user.pt", "gs://", "gs://bucket", "gs://bucket/"])
def test_invalid_model_uri_stops_before_reading_model(deploy, uri):
    result, calls = deploy("user-tower", manifest=json.dumps({"user_tower_uri": uri}))
    assert result.returncode != 0
    assert "BUILD_STARTED" not in result.stdout
    assert [args[2] for args in calls if args[:2] == ["storage", "cat"]] == [MANIFEST]


def test_missing_jq_stops_before_build(deploy):
    result, _ = deploy(with_jq=False)
    assert result.returncode != 0
    assert "BUILD_STARTED" not in result.stdout
    assert "jq" in result.stdout + result.stderr
