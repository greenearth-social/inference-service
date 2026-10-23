"""Exercise ranker promotion without accessing Google Cloud."""

import hashlib
import json
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
import pytest

import promote_model


FOLDER = "Weekly training run.task123"
PREFIX = f"Engagement Prediction/{FOLDER}/"


def _blob(path, contents):
    blob = Mock()
    blob.name = PREFIX + path
    blob.size = len(contents.encode())
    # The fake checksum only needs to distinguish contents; GCS supplies CRC32C.
    blob.crc32c = hashlib.sha256(contents.encode()).hexdigest()
    blob.generation = 7
    blob.download_as_text.return_value = contents
    return blob


@pytest.fixture
def cloud():
    manifest = {
        "ranker_uri": f"gs://{promote_model.STAGE_BUCKET}/{PREFIX}{promote_model.MODEL_PATH}",
        "ranker_clearml_model_id": "model123",
        "clearml_task_id": "task123",
        "extra_metadata": {"description": "preserve me"},
    }
    contents = {
        promote_model.MODEL_PATH: "ranker weights",
        promote_model.AUTHOR_MAP_PATH: "author map",
        promote_model.CONFIG_PATH: json.dumps({"max_history_len": 64}),
        promote_model.MANIFEST_PATH: json.dumps(manifest),
        "artifacts/review/results.json": '{"ndcg": 0.6}',
    }
    objects = {PREFIX + path: _blob(path, text) for path, text in contents.items()}
    destination = {}
    events = []
    client = Mock()
    source = Mock()
    source.name = promote_model.STAGE_BUCKET
    target = Mock()
    target.name = promote_model.PROD_BUCKET
    client.bucket.side_effect = {
        promote_model.STAGE_BUCKET: source,
        promote_model.PROD_BUCKET: target,
    }.__getitem__
    client.list_blobs.side_effect = lambda bucket, prefix: [
        blob for name, blob in objects.items() if name.startswith(prefix)
    ]

    def copy_blob(blob, destination_bucket, *, new_name, source_generation):
        assert destination_bucket is target
        assert new_name == blob.name
        assert source_generation == blob.generation
        events.append(("copy", new_name))
        copied = _blob(new_name.removeprefix(PREFIX), blob.download_as_text())
        destination[new_name] = copied
        return copied

    def get_blob(name):
        events.append(("verify", name))
        return destination.get(name)

    destination_manifest = Mock()

    def upload_manifest(contents, *, content_type):
        assert content_type == "application/json"
        events.append(("upload", PREFIX + promote_model.MANIFEST_PATH))
        destination_manifest.download_as_text.return_value = contents
        destination[PREFIX + promote_model.MANIFEST_PATH] = destination_manifest

    source.copy_blob.side_effect = copy_blob
    target.get_blob.side_effect = get_blob
    target.blob.return_value = destination_manifest
    destination_manifest.upload_from_string.side_effect = upload_manifest
    return SimpleNamespace(
        client=client, source=source, target=target, objects=objects,
        destination=destination, events=events, manifest=manifest,
        destination_manifest=destination_manifest,
    )


def _assert_no_writes(cloud):
    cloud.source.copy_blob.assert_not_called()
    cloud.destination_manifest.upload_from_string.assert_not_called()


def test_promotes_whole_run_and_publishes_rewritten_manifest_last(cloud):
    original_manifest = cloud.objects[PREFIX + promote_model.MANIFEST_PATH].download_as_text()
    result = promote_model.promote_ranker(FOLDER, cloud.client)

    assert result == {
        "manifest_uri": f"gs://{promote_model.PROD_BUCKET}/{PREFIX}{promote_model.MANIFEST_PATH}",
        "author_map_uri": f"gs://{promote_model.PROD_BUCKET}/{PREFIX}{promote_model.AUTHOR_MAP_PATH}",
        "max_history_len": 64,
        "clearml_task_id": "task123",
        "ranker_clearml_model_id": "model123",
    }
    assert set(cloud.destination) == set(cloud.objects)
    published = json.loads(cloud.destination_manifest.download_as_text())
    assert published == {
        **cloud.manifest,
        "ranker_uri": f"gs://{promote_model.PROD_BUCKET}/{PREFIX}{promote_model.MODEL_PATH}",
    }
    assert cloud.objects[PREFIX + promote_model.MANIFEST_PATH].download_as_text() == original_manifest
    writes = [event for event in cloud.events if event[0] != "verify"]
    assert writes[-1] == ("upload", PREFIX + promote_model.MANIFEST_PATH)
    assert ("copy", PREFIX + promote_model.MANIFEST_PATH) not in writes
    upload_index = cloud.events.index(writes[-1])
    for path in (promote_model.MODEL_PATH, promote_model.AUTHOR_MAP_PATH, promote_model.CONFIG_PATH):
        assert cloud.events.index(("verify", PREFIX + path)) < upload_index


@pytest.mark.parametrize("path", promote_model.REQUIRED_PATHS)
@pytest.mark.parametrize("problem", ["missing", "empty"])
def test_incomplete_source_bundle_stops_before_writes(cloud, path, problem):
    if problem == "missing":
        del cloud.objects[PREFIX + path]
    else:
        cloud.objects[PREFIX + path] = _blob(path, "")
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)
    _assert_no_writes(cloud)


@pytest.mark.parametrize("path", [promote_model.MANIFEST_PATH, promote_model.CONFIG_PATH])
@pytest.mark.parametrize("contents", ["invalid JSON", "[]", "null"])
def test_invalid_json_documents_stop_before_writes(cloud, path, contents):
    cloud.objects[PREFIX + path] = _blob(path, contents)
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)
    _assert_no_writes(cloud)


@pytest.mark.parametrize("history_length", [None, 0, -1, True, "64", 1.5])
def test_invalid_history_length_stops_before_writes(cloud, history_length):
    cloud.objects[PREFIX + promote_model.CONFIG_PATH] = _blob(
        promote_model.CONFIG_PATH, json.dumps({"max_history_len": history_length}),
    )
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)
    _assert_no_writes(cloud)


@pytest.mark.parametrize("key", ["clearml_task_id", "ranker_clearml_model_id"])
@pytest.mark.parametrize("value", [None, "", "  ", 123])
def test_invalid_source_identifiers_stop_before_writes(cloud, key, value):
    cloud.manifest[key] = value
    cloud.objects[PREFIX + promote_model.MANIFEST_PATH] = _blob(
        promote_model.MANIFEST_PATH, json.dumps(cloud.manifest),
    )
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)
    _assert_no_writes(cloud)


@pytest.mark.parametrize("uri", [
    None,
    f"gs://{promote_model.PROD_BUCKET}/{PREFIX}{promote_model.MODEL_PATH}",
    f"gs://{promote_model.STAGE_BUCKET}/Engagement Prediction/other/models/ranker.pt",
    f"gs://{promote_model.STAGE_BUCKET}/{PREFIX}models/other.pt",
])
def test_model_uri_must_belong_to_selected_stage_run(cloud, uri):
    cloud.manifest["ranker_uri"] = uri
    cloud.objects[PREFIX + promote_model.MANIFEST_PATH] = _blob(
        promote_model.MANIFEST_PATH, json.dumps(cloud.manifest),
    )
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)
    _assert_no_writes(cloud)


def test_copy_failure_prevents_manifest_publication(cloud):
    cloud.source.copy_blob.side_effect = GoogleAPIError("copy failed")
    with pytest.raises(GoogleAPIError, match="copy failed"):
        promote_model.promote_ranker(FOLDER, cloud.client)
    cloud.destination_manifest.upload_from_string.assert_not_called()


@pytest.mark.parametrize("problem", ["missing", "size", "checksum"])
def test_unverified_destination_prevents_manifest_publication(cloud, problem):
    copy_blob = cloud.source.copy_blob.side_effect

    def corrupt_copy(blob, destination_bucket, **kwargs):
        copied = copy_blob(blob, destination_bucket, **kwargs)
        if blob.name == PREFIX + promote_model.MODEL_PATH:
            if problem == "missing":
                del cloud.destination[blob.name]
            elif problem == "size":
                copied.size += 1
            else:
                copied.crc32c = "different checksum"
        return copied

    cloud.source.copy_blob.side_effect = corrupt_copy
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)
    cloud.destination_manifest.upload_from_string.assert_not_called()


def test_manifest_verification_failure_is_reported(cloud):
    upload_manifest = cloud.destination_manifest.upload_from_string.side_effect

    def corrupt_upload(contents, **kwargs):
        upload_manifest(contents, **kwargs)
        cloud.destination_manifest.download_as_text.return_value = "{}"

    cloud.destination_manifest.upload_from_string.side_effect = corrupt_upload
    with pytest.raises(ValueError):
        promote_model.promote_ranker(FOLDER, cloud.client)


def test_existing_promotion_can_be_overwritten(cloud):
    first = promote_model.promote_ranker(FOLDER, cloud.client)
    cloud.objects[PREFIX + promote_model.MODEL_PATH] = _blob(promote_model.MODEL_PATH, "new weights")
    second = promote_model.promote_ranker(FOLDER, cloud.client)
    assert first == second
    assert cloud.destination[PREFIX + promote_model.MODEL_PATH].download_as_text() == "new weights"
    assert cloud.destination_manifest.upload_from_string.call_count == 2


def test_cli_prints_deployment_settings_and_source_ids(cloud, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["promote_model.py", FOLDER])
    monkeypatch.setattr(promote_model.storage, "Client", lambda: cloud.client)
    promote_model.main()
    output = capsys.readouterr().out
    for text in (
        "GE_INFERENCE_RANKER_MANIFEST_URI=",
        "GE_INFERENCE_RANKER_AUTHOR_MAP_URI=",
        "GE_INFERENCE_RANKER_MAX_HISTORY_LEN=64",
        "task123", "model123",
    ):
        assert text in output
    settings = {
        line.split("=", 1)[0]: shlex.split(line.split("=", 1)[1])
        for line in output.splitlines() if line.startswith("GE_INFERENCE_RANKER_")
    }
    assert settings["GE_INFERENCE_RANKER_MANIFEST_URI"] == [
        f"gs://{promote_model.PROD_BUCKET}/{PREFIX}{promote_model.MANIFEST_PATH}",
    ]
    assert settings["GE_INFERENCE_RANKER_AUTHOR_MAP_URI"] == [
        f"gs://{promote_model.PROD_BUCKET}/{PREFIX}{promote_model.AUTHOR_MAP_PATH}",
    ]


@pytest.mark.parametrize("error", [
    GoogleAPIError("permission denied"),
    GoogleAuthError("credentials missing"),
    ValueError("invalid bundle"),
])
def test_cli_failure_exits_without_printing_deployment_settings(monkeypatch, capsys, error):
    monkeypatch.setattr(sys, "argv", ["promote_model.py", FOLDER])
    monkeypatch.setattr(promote_model.storage, "Client", Mock(side_effect=error))
    with pytest.raises(SystemExit) as failure:
        promote_model.main()
    assert failure.value.code != 0
    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert "GE_INFERENCE_RANKER_" not in captured.out
    assert "Promotion complete" not in captured.out


def test_cli_copy_failure_does_not_print_success(cloud, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["promote_model.py", FOLDER])
    monkeypatch.setattr(promote_model.storage, "Client", lambda: cloud.client)
    cloud.source.copy_blob.side_effect = GoogleAPIError("copy failed")
    with pytest.raises(SystemExit) as failure:
        promote_model.main()
    assert failure.value.code != 0
    captured = capsys.readouterr()
    assert "copy failed" in captured.err
    assert "GE_INFERENCE_RANKER_" not in captured.out
    assert "Promotion complete" not in captured.out
    cloud.destination_manifest.upload_from_string.assert_not_called()
