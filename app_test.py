import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[0]
APP_PATH = REPO_ROOT / "app.py"
DEFAULT_AUTHOR_IDX_MAP_URI = "gs://test-bucket/author_idx.parquet"
DEFAULT_RANKER_AUTHOR_IDX_MAP_URI = "gs://test-bucket/ranker_author_idx.parquet"


def _install_stub_modules() -> None:
    if "clearml" not in sys.modules:
        clearml = types.ModuleType("clearml")

        class FakeModel:
            def __init__(self, model_id=None):
                self.model_id = model_id

            def get_local_copy(self):
                raise RuntimeError("ClearML not available in test environment")

        clearml.Model = FakeModel
        sys.modules["clearml"] = clearml

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class DummyTensor:
            def __init__(self, value):
                self.value = value

            def _map(self, fn):
                def _apply(value):
                    if isinstance(value, list):
                        return [_apply(item) for item in value]
                    return fn(value)

                return DummyTensor(_apply(self.value))

            def _binary_op(self, other, fn):
                other_value = other.value if isinstance(other, DummyTensor) else other

                def _apply(left, right):
                    if isinstance(left, list):
                        return [_apply(item, right) for item in left]
                    if isinstance(right, list):
                        return [_apply(left, item) for item in right]
                    return fn(left, right)

                return DummyTensor(_apply(self.value, other_value))

            def numel(self):
                return 1

            def dim(self):
                def _dim(v):
                    if isinstance(v, list) and v:
                        return 1 + _dim(v[0])
                    if isinstance(v, list):
                        return 1
                    return 0

                return _dim(self.value)

            def unsqueeze(self, _dim):
                return DummyTensor([self.value])

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.value

            def __getitem__(self, idx):
                return DummyTensor(self.value[idx])

            def __add__(self, other):
                return self._binary_op(other, lambda left, right: left + right)

            def __radd__(self, other):
                return self.__add__(other)

            def __sub__(self, other):
                return self._binary_op(other, lambda left, right: left - right)

            def __rsub__(self, other):
                return self._binary_op(other, lambda left, right: right - left)

            def __mul__(self, other):
                return self._binary_op(other, lambda left, right: left * right)

            def __rmul__(self, other):
                return self.__mul__(other)

            def __truediv__(self, other):
                return self._binary_op(other, lambda left, right: left / right)

            def min(self):
                def _flatten(value):
                    if isinstance(value, list):
                        return [item for nested in value for item in _flatten(nested)]
                    return [value]

                return DummyTensor(min(_flatten(self.value)))

            def max(self):
                def _flatten(value):
                    if isinstance(value, list):
                        return [item for nested in value for item in _flatten(nested)]
                    return [value]

                return DummyTensor(max(_flatten(self.value)))

            def abs(self):
                return self._map(abs)

            def item(self):
                return self.value

            def clamp(self, min=None, max=None):
                def _clamp(value):
                    if min is not None and value < min:
                        return min
                    if max is not None and value > max:
                        return max
                    return value

                return self._map(_clamp)

        class DummyDevice:
            def __init__(self, kind):
                self.type = kind

        @contextmanager
        def inference_mode():
            yield

        torch.float32 = object()
        torch.int64 = object()
        torch.bool = object()
        torch.dtype = object
        torch.Tensor = DummyTensor
        torch.device = DummyDevice
        torch.tensor = lambda value, dtype=None, device=None: DummyTensor(value)
        torch.zeros = lambda shape, dtype=None, device=None: DummyTensor(shape)
        torch.zeros_like = lambda tensor: tensor._map(lambda _value: 0.0)
        torch.ones = lambda shape, dtype=None, device=None: DummyTensor(shape)
        torch.inference_mode = inference_mode
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.jit = types.SimpleNamespace(ScriptModule=type("ScriptModule", (), {}), load=lambda *args, **kwargs: None)
        sys.modules["torch"] = torch


def _load_app_module(
    module_name: str,
    *,
    max_batch: int = 4,
    embed_dim: int = 3,
    max_history_len: int = 8,
    author_idx_map_uri: str | None = DEFAULT_AUTHOR_IDX_MAP_URI,
    ranker_author_idx_map_uri: str | None = None,
    ranker_max_history_len: int | None = 6,
    ranker_manifest_uri: str | None = None,
    model_types: str = "post-tower,user-tower",
):
    _install_stub_modules()

    os.environ["GE_INFERENCE_MAX_BATCH"] = str(max_batch)
    os.environ["GE_INFERENCE_CONTENT_EMBED_DIM"] = str(embed_dim)
    os.environ["GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN"] = str(max_history_len)
    if ranker_max_history_len is None:
        os.environ.pop("GE_INFERENCE_RANKER_MAX_HISTORY_LEN", None)
    else:
        os.environ["GE_INFERENCE_RANKER_MAX_HISTORY_LEN"] = str(ranker_max_history_len)
    os.environ["GE_INFERENCE_MODELS"] = model_types
    os.environ.pop("GE_INFERENCE_EMBED_DIM", None)
    os.environ.pop("GE_INFERENCE_MAX_HISTORY_LEN", None)
    if author_idx_map_uri is None:
        os.environ.pop("GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI", None)
    else:
        os.environ["GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI"] = author_idx_map_uri
    if ranker_author_idx_map_uri is None:
        os.environ.pop("GE_INFERENCE_RANKER_AUTHOR_MAP_URI", None)
    else:
        os.environ["GE_INFERENCE_RANKER_AUTHOR_MAP_URI"] = ranker_author_idx_map_uri
    if ranker_manifest_uri is None:
        os.environ.pop("GE_INFERENCE_RANKER_MANIFEST_URI", None)
    else:
        os.environ["GE_INFERENCE_RANKER_MANIFEST_URI"] = ranker_manifest_uri
    os.environ.pop("GE_INFERENCE_AUTHOR_MAP_URI", None)

    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_fake_pyarrow(monkeypatch, rows):
    pyarrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")

    class FakeColumn:
        def __init__(self, values):
            self._values = values

        def to_pylist(self):
            return self._values

    class FakeTable:
        def __init__(self, table_rows):
            self._columns = {
                "author_did": [row["author_did"] for row in table_rows],
                "author_idx": [row["author_idx"] for row in table_rows],
            }

        def __getitem__(self, name):
            return FakeColumn(self._columns[name])

    def read_table(_path, columns=None):
        assert columns == ["author_did", "author_idx"]
        return FakeTable(rows)

    parquet.read_table = read_table
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)


def _set_author_idx_map(app, monkeypatch, idx_by_did, name="two-tower"):
    monkeypatch.setattr(
        app,
        "_author_idx_maps",
        {name: app.AuthorIdxMap(name=name, uri="gs://test-bucket/author_idx.parquet", idx_by_did=idx_by_did)},
    )


@pytest.fixture(scope="module")
def app_shape():
    return _load_app_module("inference_service_app_shape_tests")


@pytest.fixture(scope="module")
def app_request():
    return _load_app_module("inference_service_app_request_tests", max_batch=3, embed_dim=3)


@pytest.fixture(scope="module")
def app_fixed_dim():
    return _load_app_module("inference_service_app_fixed_dim_tests", max_batch=4, embed_dim=3)


def _liked_at(hours_ago: float) -> datetime:
    return datetime(2026, 1, 1, 12, tzinfo=timezone.utc) - timedelta(hours=hours_ago)


def _freeze_app_now(app, monkeypatch, now=None) -> datetime:
    fixed_now = now or _liked_at(0)

    class FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now
            return fixed_now.astimezone(tz)

        @classmethod
        def fromtimestamp(cls, ts, tz=None):
            return datetime.fromtimestamp(ts, tz=tz)

    monkeypatch.setattr(app, "datetime", FakeDatetime)
    return fixed_now


def test_classifies_empty_flat_history_as_single_empty(app_shape):
    assert app_shape.classify_history_embeddings_shape([]) == "single_empty"


def test_classifies_empty_nested_history_as_single_empty(app_shape):
    assert app_shape.classify_history_embeddings_shape([[]]) == "single_empty"


def test_classifies_single_history_as_single_history(app_shape):
    assert app_shape.classify_history_embeddings_shape([[1.0, 2.0], [3.0, 4.0]]) == "single_history"


def test_classifies_batch_with_empty_first_user_as_batched_history(app_shape):
    assert app_shape.classify_history_embeddings_shape([[], [[1.0, 2.0]]]) == "batched_history"


def test_classifies_three_dimensional_input_as_batched_history(app_shape):
    assert app_shape.classify_history_embeddings_shape([[[1.0, 2.0]], [[3.0, 4.0]]]) == "batched_history"


def test_rejects_non_list_top_level(app_shape):
    with pytest.raises(ValueError, match="history_embeddings must be a list"):
        app_shape.classify_history_embeddings_shape("not-a-list")


def test_rejects_top_level_list_that_does_not_contain_lists(app_shape):
    with pytest.raises(ValueError, match="history_embeddings must be a list of lists"):
        app_shape.classify_history_embeddings_shape([1.0, 2.0])


def test_records_missing_author_idx_map_uri_as_init_error():
    app = _load_app_module("inference_service_app_missing_author_map_tests", author_idx_map_uri=None)

    app._ensure_author_idx_maps_loaded()

    assert app._author_idx_maps_init_error is not None
    assert "GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI" in app._author_idx_maps_init_error
    assert app._author_idx_maps_initialized is False


def test_accepts_empty_flat_history(app_request):
    app_request.UserTowerPredictRequest(history_embeddings=[], history_author_dids=[])


def test_accepts_empty_nested_history(app_request):
    app_request.UserTowerPredictRequest(history_embeddings=[[]], history_author_dids=[])


def test_accepts_single_history(app_request):
    app_request.UserTowerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        history_author_dids=["author-1", "author-2"],
    )


def test_accepts_single_history_without_author_dids(app_request):
    app_request.UserTowerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    app_request.UserTowerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        history_author_dids=None,
    )


def test_accepts_batched_histories_with_empty_entries(app_request):
    app_request.UserTowerPredictRequest(
        history_embeddings=[[], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[]]],
        history_author_dids=[[], ["author-1", "author-2"], []],
    )


def test_accepts_batched_histories_without_author_dids(app_request):
    app_request.UserTowerPredictRequest(
        history_embeddings=[[], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[]]],
    )
    app_request.UserTowerPredictRequest(
        history_embeddings=[[], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[]]],
        history_author_dids=None,
    )


def test_rejects_single_history_author_dids_length_mismatch(app_request):
    with pytest.raises(ValueError, match="must match history length"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            history_author_dids=["author-1"],
        )


def test_rejects_batched_history_author_dids_shape_mismatch(app_request):
    with pytest.raises(ValueError, match="must be a list of list of strings"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            history_author_dids=["author-1", "author-2"],
        )


def test_rejects_batched_history_author_dids_length_mismatch(app_request):
    with pytest.raises(ValueError, match="must match that user's history length"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[]]],
            history_author_dids=[[], ["author-1"], []],
        )


def test_rejects_batched_history_over_max_batch(app_request):
    with pytest.raises(ValueError, match="batch too large"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            history_author_dids=[["author-1"], ["author-2"], ["author-3"], ["author-4"]],
        )


def test_rejects_single_history_with_mismatched_embedding_dimensions(app_request):
    with pytest.raises(ValueError, match="embedding dim must be 3"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0], [3.0]],
            history_author_dids=["author-1", "author-2"],
        )


def test_rejects_single_history_with_mismatched_embedding_dimensions_without_author_dids(app_request):
    with pytest.raises(ValueError, match="embedding dim must be 3"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0], [3.0]],
        )


def test_rejects_mixed_rank_batch_entry(app_request):
    with pytest.raises(ValueError, match="list of lists"):
        app_request._validate_batched_user_history([[[1.0, 2.0, 3.0]], [1.0, 2.0]])


def test_rejects_non_list_user_entry_in_batch(app_request):
    with pytest.raises(ValueError, match="each user's history must be a list"):
        app_request._validate_batched_user_history([[[1.0, 2.0, 3.0]], "bad-user"])


def test_accepts_histories_matching_fixed_embedding_dim(app_fixed_dim):
    app_fixed_dim.UserTowerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0]],
        history_author_dids=["author-1"],
    )
    app_fixed_dim.UserTowerPredictRequest(
        history_embeddings=[[[1.0, 2.0, 3.0]], []],
        history_author_dids=[["author-1"], []],
    )


def test_rejects_single_history_that_violates_fixed_embedding_dim(app_fixed_dim):
    with pytest.raises(ValueError, match="embedding dim must be 3"):
        app_fixed_dim.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0]],
            history_author_dids=["author-1"],
        )


def test_rejects_batched_history_that_violates_fixed_embedding_dim(app_fixed_dim):
    with pytest.raises(ValueError, match="embedding dim must be 3"):
        app_fixed_dim.UserTowerPredictRequest(
            history_embeddings=[[[1.0, 2.0, 3.0]], [[4.0, 5.0]]],
            history_author_dids=[["author-1"], ["author-2"]],
        )


def test_rejects_zero_width_embedding_row(app_fixed_dim):
    with pytest.raises(ValueError, match="greater than 0"):
        app_fixed_dim._validate_single_user_history([[]])


def test_post_tower_request_accepts_unbatched_and_batched(app_request):
    app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0], target_author_dids="author-1")
    app_request.PostTowerPredictRequest(
        post_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        target_author_dids=["author-1", "author-2"],
    )


def test_post_tower_request_accepts_missing_author_dids(app_request):
    app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0])
    app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0], target_author_dids=None)
    app_request.PostTowerPredictRequest(post_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    app_request.PostTowerPredictRequest(
        post_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        target_author_dids=None,
    )


def test_post_tower_request_rejects_ragged_batched_vectors(app_request):
    with pytest.raises(ValueError, match="same length"):
        app_request.PostTowerPredictRequest(
            post_embeddings=[[1.0, 2.0], [3.0]],
            target_author_dids=["author-1", "author-2"],
        )


def test_post_tower_request_rejects_author_did_shape_mismatch(app_request):
    with pytest.raises(ValueError, match="same length as the batch"):
        app_request.PostTowerPredictRequest(
            post_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            target_author_dids=["author-1"],
        )
    with pytest.raises(ValueError, match="single string"):
        app_request.PostTowerPredictRequest(
            post_embeddings=[1.0, 2.0, 3.0],
            target_author_dids=["author-1"],
        )


def test_post_tower_request_enforces_embed_dim(app_fixed_dim):
    app_fixed_dim.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0], target_author_dids="author-1")
    with pytest.raises(ValueError, match="expected D=3"):
        app_fixed_dim.PostTowerPredictRequest(post_embeddings=[1.0, 2.0], target_author_dids="author-1")


def test_ranker_request_accepts_single_history_and_candidate_post(app_request):
    app_request.RankerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        history_author_dids=["author-1", "author-2"],
        history_liked_at_times=[_liked_at(1), _liked_at(2)],
        candidate_post_embeddings=[7.0, 8.0, 9.0],
        candidate_author_dids="candidate-author",
    )


def test_ranker_request_accepts_single_history_and_multiple_candidate_posts(app_request):
    app_request.RankerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0]],
        history_author_dids=["author-1"],
        history_liked_at_times=[_liked_at(1)],
        candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        candidate_author_dids=["candidate-1", "candidate-2"],
    )


def test_ranker_request_accepts_single_empty_history_and_multiple_candidate_posts(app_request):
    app_request.RankerPredictRequest(
        history_embeddings=[],
        history_author_dids=[],
        history_liked_at_times=[],
        candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        candidate_author_dids=["candidate-1", "candidate-2"],
    )


def test_ranker_request_accepts_nested_single_empty_history_and_multiple_candidate_posts(app_request):
    app_request.RankerPredictRequest(
        history_embeddings=[[]],
        history_author_dids=[],
        history_liked_at_times=[],
        candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        candidate_author_dids=["candidate-1", "candidate-2"],
    )


def test_ranker_request_rejects_batched_histories(app_request):
    with pytest.raises(ValueError, match="Input should be a valid number"):
        app_request.RankerPredictRequest(
            history_embeddings=[[], [[1.0, 2.0, 3.0]], [[]]],
            history_author_dids=[[], ["author-1"], []],
            history_liked_at_times=[[], [_liked_at(1)], []],
            candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            candidate_author_dids=["candidate-1", "candidate-2"],
        )


def test_ranker_request_rejects_naive_liked_at_times(app_request):
    with pytest.raises(ValueError):
        app_request.RankerPredictRequest(
            history_embeddings=[[1.0, 2.0, 3.0]],
            history_liked_at_times=[datetime(2026, 1, 1, 12)],
            candidate_post_embeddings=[7.0, 8.0, 9.0],
        )


def test_ranker_request_rejects_single_history_liked_at_times_shape_mismatch(app_request):
    with pytest.raises(ValueError, match="Input should be a valid datetime"):
        app_request.RankerPredictRequest(
            history_embeddings=[[1.0, 2.0, 3.0]],
            history_liked_at_times=[[_liked_at(1)]],
            candidate_post_embeddings=[7.0, 8.0, 9.0],
        )


def test_ranker_request_rejects_batched_history_liked_at_times_shape_mismatch(app_request):
    with pytest.raises(ValueError, match="Input should be a valid number"):
        app_request.RankerPredictRequest(
            history_embeddings=[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            history_liked_at_times=[_liked_at(1), _liked_at(2)],
            candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        )


def test_ranker_request_rejects_history_and_liked_at_batch_mismatch(app_request):
    with pytest.raises(ValueError, match="Input should be a valid number"):
        app_request.RankerPredictRequest(
            history_embeddings=[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            history_liked_at_times=[[_liked_at(1)]],
            candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        )


def test_ranker_request_rejects_history_and_candidate_post_batch_mismatch(app_request):
    with pytest.raises(ValueError, match="Input should be a valid number"):
        app_request.RankerPredictRequest(
            history_embeddings=[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            history_liked_at_times=[[_liked_at(1)], [_liked_at(2)]],
            candidate_post_embeddings=[[7.0, 8.0, 9.0]],
        )


def test_ranker_request_rejects_candidate_author_dids_shape_mismatch(app_request):
    with pytest.raises(ValueError, match="same length as the batch"):
        app_request.RankerPredictRequest(
            history_embeddings=[[1.0, 2.0, 3.0]],
            history_liked_at_times=[_liked_at(1)],
            candidate_post_embeddings=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            candidate_author_dids="candidate-author",
        )


def test_author_idx_map_loads_from_configured_uri(tmp_path, monkeypatch):
    parquet_path = tmp_path / "author_idx.parquet"
    parquet_path.write_bytes(b"fake parquet")
    _install_fake_pyarrow(
        monkeypatch,
        [
            {"author_did": "did:plc:one", "author_idx": 2},
            {"author_did": "did:plc:two", "author_idx": 3},
        ],
    )
    app_author_map = _load_app_module(
        "inference_service_app_author_map_tests",
        author_idx_map_uri=str(parquet_path),
    )

    app_author_map._ensure_author_idx_maps_loaded()

    author_idx_map = app_author_map._author_idx_maps["two-tower"]
    assert author_idx_map.idx_by_did == {"did:plc:one": 2, "did:plc:two": 3}
    assert author_idx_map.resolved_path == str(parquet_path)
    assert author_idx_map.load_error is None
    assert app_author_map._author_idx_maps_init_error is None
    assert app_author_map._author_idx_maps_initialized is True


def test_author_idx_map_rejects_duplicate_author_did(tmp_path, monkeypatch):
    parquet_path = tmp_path / "author_idx.parquet"
    parquet_path.write_bytes(b"fake parquet")
    _install_fake_pyarrow(
        monkeypatch,
        [
            {"author_did": "did:plc:one", "author_idx": 2},
            {"author_did": "did:plc:one", "author_idx": 3},
        ],
    )
    app_author_map = _load_app_module(
        "inference_service_app_author_map_duplicate_tests",
        author_idx_map_uri=str(parquet_path),
    )

    app_author_map._ensure_author_idx_maps_loaded()

    author_idx_map = app_author_map._author_idx_maps["two-tower"]
    assert author_idx_map.idx_by_did is None
    assert "duplicate author_did" in author_idx_map.load_error
    assert app_author_map._author_idx_maps_init_error is None
    assert app_author_map._author_idx_maps_initialized is False


def test_ranker_author_idx_map_loads_from_configured_uri(tmp_path, monkeypatch):
    parquet_path = tmp_path / "ranker_author_idx.parquet"
    parquet_path.write_bytes(b"fake parquet")
    _install_fake_pyarrow(
        monkeypatch,
        [
            {"author_did": "did:plc:ranker-one", "author_idx": 12},
            {"author_did": "did:plc:ranker-two", "author_idx": 13},
        ],
    )
    app_author_map = _load_app_module(
        "inference_service_app_ranker_author_map_tests",
        author_idx_map_uri=None,
        ranker_author_idx_map_uri=str(parquet_path),
        model_types="ranker",
    )

    app_author_map._ensure_author_idx_maps_loaded()

    assert "two-tower" not in app_author_map._author_idx_maps
    author_idx_map = app_author_map._author_idx_maps["ranker"]
    assert author_idx_map.idx_by_did == {"did:plc:ranker-one": 12, "did:plc:ranker-two": 13}
    assert author_idx_map.resolved_path == str(parquet_path)
    assert author_idx_map.load_error is None
    assert app_author_map._author_idx_maps_init_error is None
    assert app_author_map._author_idx_maps_initialized is True


def test_get_entry_or_404_returns_404_for_unknown_model(app_request, monkeypatch):
    monkeypatch.setattr(app_request, "_models_initialized", True)
    monkeypatch.setattr(app_request, "_models_init_error", None)
    monkeypatch.setattr(app_request, "_models", {})

    with pytest.raises(app_request.HTTPException, match="Unknown model"):
        app_request._get_entry_or_404("nope")


def test_get_entry_or_404_returns_500_when_registry_init_failed(app_request, monkeypatch):
    monkeypatch.setattr(app_request, "_models_initialized", True)
    monkeypatch.setattr(app_request, "_models_init_error", "boom")
    monkeypatch.setattr(app_request, "_models", {})

    with pytest.raises(app_request.HTTPException, match="Model registry init failed"):
        app_request._get_entry_or_404("user-tower")


def test_predict_with_entry_user_tower_uses_padded_history_and_mask(app_request, monkeypatch):
    captured = {}

    def fake_pad(*, history_embeddings, max_history_len, embed_dim, author_indices, time_deltas_hours=None):
        captured["pad_args"] = {
            "history_embeddings": history_embeddings,
            "max_history_len": max_history_len,
            "embed_dim": embed_dim,
            "author_indices": author_indices,
            "time_deltas_hours": time_deltas_hours,
        }
        return [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], [[True, False]], [[7, 0]], [[0.0, 0.0]]

    def user_model(history_embeddings, history_mask, author_indices):
        captured["model_inputs"] = {
            "history_embeddings": history_embeddings.value,
            "history_mask": history_mask.value,
            "author_indices": author_indices.value,
        }
        return app_request.torch.Tensor([[42.0]])

    monkeypatch.setattr(app_request, "get_padded_embedding_history_and_mask_batched", fake_pad)
    _set_author_idx_map(app_request, monkeypatch, {"author-1": 7, "author-2": 8})

    entry = app_request.LoadedModel(model_type="user-tower")
    entry.module = user_model
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 8

    req = app_request.UserTowerPredictRequest(
        history_embeddings=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        history_author_dids=["author-1", "author-2"],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["pad_args"]["history_embeddings"] == [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]
    assert captured["pad_args"]["max_history_len"] == entry.max_history_len
    assert captured["pad_args"]["embed_dim"] == app_request.GE_INFERENCE_CONTENT_EMBED_DIM
    assert captured["pad_args"]["author_indices"] == [7, 8]
    assert captured["pad_args"]["time_deltas_hours"] is None
    assert captured["model_inputs"]["history_embeddings"] == [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    assert captured["model_inputs"]["history_mask"] == [[True, False]]
    assert captured["model_inputs"]["author_indices"] == [[7, 0]]
    assert out.tolist() == [[42.0]]


def test_predict_with_entry_user_tower_uses_real_padding_for_author_indices(app_request, monkeypatch):
    captured = {}

    def user_model(history_embeddings, history_mask, author_indices):
        captured["history_embeddings"] = history_embeddings.value
        captured["history_mask"] = history_mask.value
        captured["author_indices"] = author_indices.value
        return app_request.torch.Tensor([[42.0]])

    _set_author_idx_map(app_request, monkeypatch, {"author-1": 7})

    entry = app_request.LoadedModel(model_type="user-tower")
    entry.module = user_model
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 8

    req = app_request.UserTowerPredictRequest(
        history_embeddings=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        history_author_dids=["author-1", "unknown-author"],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["history_embeddings"] == [
        [
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ]
    assert captured["history_mask"] == [[True, True, False, False, False, False, False, False]]
    assert captured["author_indices"] == [[7, app_request.AUTHOR_UNK_IDX, 0, 0, 0, 0, 0, 0]]
    assert out.tolist() == [[42.0]]


def test_predict_with_entry_user_tower_defaults_missing_author_dids_to_unknown(app_request, monkeypatch):
    captured = {}

    def user_model(history_embeddings, history_mask, author_indices):
        captured["history_embeddings"] = history_embeddings.value
        captured["history_mask"] = history_mask.value
        captured["author_indices"] = author_indices.value
        return app_request.torch.Tensor([[42.0]])

    monkeypatch.setattr(app_request, "_author_idx_maps", {})

    entry = app_request.LoadedModel(model_type="user-tower")
    entry.module = user_model
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 8

    req = app_request.UserTowerPredictRequest(
        history_embeddings=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["history_embeddings"] == [
        [
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ]
    assert captured["history_mask"] == [[True, True, False, False, False, False, False, False]]
    assert captured["author_indices"] == [[
        app_request.AUTHOR_UNK_IDX,
        app_request.AUTHOR_UNK_IDX,
        0,
        0,
        0,
        0,
        0,
        0,
    ]]
    assert out.tolist() == [[42.0]]


def test_predict_with_entry_post_tower_coerces_unbatched_vectors(app_request, monkeypatch):
    captured = {}

    def post_model(post_embeddings, author_indices):
        captured["post_embeddings"] = post_embeddings.value
        captured["author_indices"] = author_indices.value
        return app_request.torch.Tensor([[2.0]])

    _set_author_idx_map(app_request, monkeypatch, {"author-1": 7})

    entry = app_request.LoadedModel(model_type="post-tower")
    entry.module = post_model
    entry.device = app_request.torch.device("cpu")

    req = app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0], target_author_dids="author-1")
    out = app_request._predict_with_entry(entry, req)

    assert captured["post_embeddings"] == [[1.0, 2.0, 3.0]]
    assert captured["author_indices"] == [7]
    assert out.tolist() == [[2.0]]


def test_predict_with_entry_post_tower_defaults_missing_author_dids_to_unknown(app_request, monkeypatch):
    captured = {}

    def post_model(post_embeddings, author_indices):
        captured["post_embeddings"] = post_embeddings.value
        captured["author_indices"] = author_indices.value
        return app_request.torch.Tensor([[2.0], [3.0]])

    monkeypatch.setattr(app_request, "_author_idx_maps", {})

    entry = app_request.LoadedModel(model_type="post-tower")
    entry.module = post_model
    entry.device = app_request.torch.device("cpu")

    req = app_request.PostTowerPredictRequest(
        post_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["post_embeddings"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert captured["author_indices"] == [app_request.AUTHOR_UNK_IDX, app_request.AUTHOR_UNK_IDX]
    assert out.tolist() == [[2.0], [3.0]]


def test_get_time_deltas_hours_handles_single_and_batched_times(app_request, monkeypatch):
    _freeze_app_now(app_request, monkeypatch)

    assert app_request._get_time_deltas_hours([_liked_at(1), _liked_at(2.5)]) == [1.0, 2.5]
    assert app_request._get_time_deltas_hours([[], [_liked_at(3)]]) == [[], [3.0]]


def test_min_max_scaling_maps_ranker_logits_to_unit_interval(app_request):
    scaled = app_request._damped_min_max_scaling(app_request.torch.Tensor([0.25, 0.75]))

    assert scaled.tolist() == [-1.0, 1.0]


def test_min_max_scaling_preserves_candidate_order(app_request):
    scaled = app_request._damped_min_max_scaling(app_request.torch.Tensor([0.5, 0.25, 0.75]))

    assert scaled.tolist() == [0.0, -1.0, 1.0]


def test_min_max_scaling_returns_zero_for_equal_ranker_logits(app_request):
    scaled = app_request._damped_min_max_scaling(app_request.torch.Tensor([0.5, 0.5]))

    assert scaled.tolist() == [0.0, 0.0]


def test_predict_with_entry_ranker_passes_history_candidate_and_time_delta_inputs(app_request, monkeypatch):
    captured = {}
    _freeze_app_now(app_request, monkeypatch)

    def fake_pad(*, history_embeddings, max_history_len, embed_dim, author_indices, time_deltas_hours=None):
        captured["pad_args"] = {
            "history_embeddings": history_embeddings,
            "max_history_len": max_history_len,
            "embed_dim": embed_dim,
            "author_indices": author_indices,
            "time_deltas_hours": time_deltas_hours,
        }
        return (
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            [[True, False]],
            [[12, 0]],
            [[2.0, 0.0]],
        )

    class RankerModel:
        def score_candidate_matrix(
            self,
            history_embeddings,
            history_mask,
            history_time_deltas_hours,
            candidate_post_embeddings,
            history_author_indices,
            candidate_author_indices,
        ):
            captured["model_inputs"] = {
                "history_embeddings": history_embeddings.value,
                "history_mask": history_mask.value,
                "history_time_deltas_hours": history_time_deltas_hours.value,
                "candidate_post_embeddings": candidate_post_embeddings.value,
                "history_author_indices": history_author_indices.value,
                "candidate_author_indices": candidate_author_indices.value,
            }
            return app_request.torch.Tensor([[0.75, 0.5]])

    monkeypatch.setattr(app_request, "get_padded_embedding_history_and_mask_batched", fake_pad)
    _set_author_idx_map(app_request, monkeypatch, {"history-author": 12, "candidate-author": 13}, name="ranker")

    entry = app_request.LoadedModel(model_type="ranker")
    entry.module = RankerModel()
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 6

    req = app_request.RankerPredictRequest(
        history_embeddings=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        history_author_dids=["history-author", "unknown-history-author"],
        history_liked_at_times=[_liked_at(2), _liked_at(4)],
        candidate_post_embeddings=[[3.0, 2.0, 1.0], [4.0, 5.0, 6.0]],
        candidate_author_dids=["candidate-author", "unknown-candidate-author"],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["pad_args"]["history_embeddings"] == [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]
    assert captured["pad_args"]["max_history_len"] == entry.max_history_len
    assert captured["pad_args"]["embed_dim"] == app_request.GE_INFERENCE_CONTENT_EMBED_DIM
    assert captured["pad_args"]["author_indices"] == [12, app_request.AUTHOR_UNK_IDX]
    assert captured["pad_args"]["time_deltas_hours"] == [2.0, 4.0]
    assert captured["model_inputs"]["history_embeddings"] == [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    assert captured["model_inputs"]["history_mask"] == [[True, False]]
    assert captured["model_inputs"]["history_time_deltas_hours"] == [[2.0, 0.0]]
    assert captured["model_inputs"]["candidate_post_embeddings"] == [[3.0, 2.0, 1.0], [4.0, 5.0, 6.0]]
    assert captured["model_inputs"]["history_author_indices"] == [[12, 0]]
    assert captured["model_inputs"]["candidate_author_indices"] == [13, app_request.AUTHOR_UNK_IDX]
    assert out.tolist() == [1.0, -1.0]


def test_predict_with_entry_ranker_defaults_missing_author_dids_to_unknown(app_request, monkeypatch):
    captured = {}
    _freeze_app_now(app_request, monkeypatch)
    monkeypatch.setattr(app_request, "_author_idx_maps", {})

    class RankerModel:
        def score_candidate_matrix(
            self,
            history_embeddings,
            history_mask,
            history_time_deltas_hours,
            candidate_post_embeddings,
            history_author_indices,
            candidate_author_indices,
        ):
            captured["history_embeddings"] = history_embeddings.value
            captured["history_mask"] = history_mask.value
            captured["history_time_deltas_hours"] = history_time_deltas_hours.value
            captured["candidate_post_embeddings"] = candidate_post_embeddings.value
            captured["history_author_indices"] = history_author_indices.value
            captured["candidate_author_indices"] = candidate_author_indices.value
            return app_request.torch.Tensor([[0.25]])

    entry = app_request.LoadedModel(model_type="ranker")
    entry.module = RankerModel()
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 3

    req = app_request.RankerPredictRequest(
        history_embeddings=[[9.0, 8.0, 7.0]],
        history_liked_at_times=[_liked_at(1.5)],
        candidate_post_embeddings=[3.0, 2.0, 1.0],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["history_embeddings"] == [[[9.0, 8.0, 7.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    assert captured["history_mask"] == [[True, False, False]]
    assert captured["history_time_deltas_hours"] == [[1.5, 0.0, 0.0]]
    assert captured["candidate_post_embeddings"] == [[3.0, 2.0, 1.0]]
    assert captured["history_author_indices"] == [[app_request.AUTHOR_UNK_IDX, 0, 0]]
    assert captured["candidate_author_indices"] == [app_request.AUTHOR_UNK_IDX]
    assert out.tolist() == [0.0]


def test_predict_with_entry_ranker_scores_empty_history_against_multiple_candidates(app_request, monkeypatch):
    captured = {}
    _freeze_app_now(app_request, monkeypatch)
    monkeypatch.setattr(app_request, "_author_idx_maps", {})

    class RankerModel:
        def score_candidate_matrix(
            self,
            history_embeddings,
            history_mask,
            history_time_deltas_hours,
            candidate_post_embeddings,
            history_author_indices,
            candidate_author_indices,
        ):
            captured["history_embeddings"] = history_embeddings.value
            captured["history_mask"] = history_mask.value
            captured["history_time_deltas_hours"] = history_time_deltas_hours.value
            captured["candidate_post_embeddings"] = candidate_post_embeddings.value
            captured["history_author_indices"] = history_author_indices.value
            captured["candidate_author_indices"] = candidate_author_indices.value
            return app_request.torch.Tensor([[0.25, 0.75]])

    entry = app_request.LoadedModel(model_type="ranker")
    entry.module = RankerModel()
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 3

    req = app_request.RankerPredictRequest(
        history_embeddings=[[]],
        history_author_dids=[],
        history_liked_at_times=[],
        candidate_post_embeddings=[[3.0, 2.0, 1.0], [4.0, 5.0, 6.0]],
    )
    out = app_request._predict_with_entry(entry, req)

    assert captured["history_embeddings"] == [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    assert captured["history_mask"] == [[False, False, False]]
    assert captured["history_time_deltas_hours"] == [[0.0, 0.0, 0.0]]
    assert captured["candidate_post_embeddings"] == [[3.0, 2.0, 1.0], [4.0, 5.0, 6.0]]
    assert captured["history_author_indices"] == [[app_request.AUTHOR_PAD_IDX, app_request.AUTHOR_PAD_IDX, app_request.AUTHOR_PAD_IDX]]
    assert captured["candidate_author_indices"] == [app_request.AUTHOR_UNK_IDX, app_request.AUTHOR_UNK_IDX]
    assert out.tolist() == [-1.0, 1.0]


def test_ranker_request_rejects_liked_at_length_mismatch(app_request):
    with pytest.raises(ValueError, match="History length \\(2\\) must match history liked at times length \\(1\\)"):
        app_request.RankerPredictRequest(
            history_embeddings=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
            history_liked_at_times=[_liked_at(1)],
            candidate_post_embeddings=[3.0, 2.0, 1.0],
        )


def test_predict_with_entry_rejects_request_type_mismatch(app_request):
    entry = app_request.LoadedModel(model_type="user-tower")
    entry.module = Mock()
    entry.device = app_request.torch.device("cpu")

    req = app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0], target_author_dids="author-1")

    with pytest.raises(app_request.HTTPException) as exc_info:
        app_request._predict_with_entry(entry, req)

    assert exc_info.value.status_code == 422
    assert "expects a user-tower request body" in str(exc_info.value.detail)


def test_predict_with_entry_rejects_ranker_request_type_mismatch(app_request):
    entry = app_request.LoadedModel(model_type="ranker")
    entry.module = Mock()
    entry.device = app_request.torch.device("cpu")
    entry.max_history_len = 6

    req = app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0], target_author_dids="author-1")

    with pytest.raises(app_request.HTTPException) as exc_info:
        app_request._predict_with_entry(entry, req)

    assert exc_info.value.status_code == 422
    assert "expects a ranker request body" in str(exc_info.value.detail)


def test_require_ready_raises_503_when_model_not_loaded(app_request, monkeypatch):
    entry = app_request.LoadedModel(model_type="user-tower")
    monkeypatch.setattr(app_request, "ensure_models_loaded", lambda: None)

    with pytest.raises(app_request.HTTPException) as exc_info:
        app_request._require_ready(entry)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["model_type"] == "user-tower"
    assert exc_info.value.detail["ready"] is False


# --- Manifest loading tests ---

SAMPLE_MANIFEST = {
    "post_tower_clearml_model_id": "post-model-abc123",
    "user_tower_clearml_model_id": "user-model-def456",
    "post_tower_uri": "gs://fake-bucket/post_tower.pt",
    "user_tower_uri": "gs://fake-bucket/user_tower.pt",
    "output_embedding_dim": 128,
    "clearml_task_id": "task-xyz",
}

SAMPLE_RANKER_MANIFEST = {
    "ranker_clearml_model_id": "ranker-model-ghi789",
    "ranker_uri": "gs://fake-bucket/ranker.pt",
    "clearml_task_id": "ranker-task-xyz",
}


def _write_manifest(tmp_path, manifest=None) -> str:
    import json
    path = tmp_path / "two_tower_serving_manifest.json"
    path.write_text(json.dumps(manifest or SAMPLE_MANIFEST))
    return str(path)


def _write_ranker_manifest(tmp_path, manifest=None) -> str:
    import json
    path = tmp_path / "ranker_serving_manifest.json"
    path.write_text(json.dumps(manifest or SAMPLE_RANKER_MANIFEST))
    return str(path)


def test_init_registry_fails_without_manifest_uri(tmp_path):
    app = _load_app_module("inference_service_no_manifest_tests")
    os.environ.pop("GE_INFERENCE_TWO_TOWER_MANIFEST_URI", None)
    os.environ["GE_INFERENCE_MODELS"] = "post-tower"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is not None
    assert "GE_INFERENCE_TWO_TOWER_MANIFEST_URI" in app._models_init_error


def test_init_registry_sets_model_uuid_from_manifest(tmp_path):
    app = _load_app_module("inference_service_manifest_uuid_tests")
    os.environ["GE_INFERENCE_TWO_TOWER_MANIFEST_URI"] = _write_manifest(tmp_path)
    os.environ["GE_INFERENCE_MODELS"] = "post-tower,user-tower"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is None
    assert app._models["post-tower"].model_uuid == "post-model-abc123"
    assert app._models["user-tower"].model_uuid == "user-model-def456"


def test_init_registry_sets_configured_model_uri_from_manifest(tmp_path):
    app = _load_app_module("inference_service_manifest_uri_tests")
    os.environ["GE_INFERENCE_TWO_TOWER_MANIFEST_URI"] = _write_manifest(tmp_path)
    os.environ["GE_INFERENCE_MODELS"] = "post-tower"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is None
    assert app._models["post-tower"].configured_model_uri == "gs://fake-bucket/post_tower.pt"


def test_init_registry_does_not_require_ranker_manifest_without_ranker(tmp_path):
    app = _load_app_module("inference_service_manifest_no_ranker_tests")
    os.environ["GE_INFERENCE_TWO_TOWER_MANIFEST_URI"] = _write_manifest(tmp_path)
    os.environ.pop("GE_INFERENCE_RANKER_MANIFEST_URI", None)
    os.environ["GE_INFERENCE_MODELS"] = "post-tower,user-tower"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is None
    assert set(app._models) == {"post-tower", "user-tower"}


def test_init_registry_requires_ranker_manifest_when_ranker_configured(tmp_path):
    app = _load_app_module("inference_service_missing_ranker_manifest_tests", model_types="ranker")
    os.environ["GE_INFERENCE_TWO_TOWER_MANIFEST_URI"] = _write_manifest(tmp_path)
    os.environ.pop("GE_INFERENCE_RANKER_MANIFEST_URI", None)
    os.environ["GE_INFERENCE_MODELS"] = "ranker"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is not None
    assert "GE_INFERENCE_RANKER_MANIFEST_URI" in app._models_init_error


def test_init_registry_sets_ranker_metadata_from_manifest(tmp_path):
    app = _load_app_module("inference_service_ranker_manifest_tests", model_types="ranker")
    os.environ["GE_INFERENCE_TWO_TOWER_MANIFEST_URI"] = _write_manifest(tmp_path)
    os.environ["GE_INFERENCE_RANKER_MANIFEST_URI"] = _write_ranker_manifest(tmp_path)
    os.environ["GE_INFERENCE_MODELS"] = "ranker"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is None
    assert app._models["ranker"].model_uuid == "ranker-model-ghi789"
    assert app._models["ranker"].configured_model_uri == "gs://fake-bucket/ranker.pt"


def test_load_entry_accepts_ranker_with_matrix_scorer(monkeypatch):
    app = _load_app_module("inference_service_ranker_matrix_load_tests", model_types="ranker")

    class MatrixRanker:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

        def score_candidate_matrix(self, *args):
            return app.torch.Tensor([[0.5]])

    ranker = MatrixRanker()
    monkeypatch.setattr(app, "_resolve_model_file", lambda _entry: ("ranker.pt", "ranker-id"))
    monkeypatch.setattr(app.torch.jit, "load", lambda *args, **kwargs: ranker)

    entry = app.LoadedModel(model_type="ranker")
    app._load_entry(entry)

    assert entry.module is ranker
    assert ranker.eval_called is True
    assert entry.max_history_len == app._get_max_history_len("ranker")


def test_warmup_entry_ranker_uses_matrix_scorer(monkeypatch):
    app = _load_app_module("inference_service_ranker_warmup_matrix_tests", model_types="ranker")
    captured = {}

    class MatrixRanker:
        def score_candidate_matrix(self, *args):
            captured["matrix_args"] = [arg.value for arg in args]
            return app.torch.Tensor([[0.5]])

        def __call__(self, *args):
            raise AssertionError("ranker warmup should use score_candidate_matrix")

    entry = app.LoadedModel(model_type="ranker")
    entry.module = MatrixRanker()
    entry.device = app.torch.device("cuda")
    entry.max_history_len = 6

    app._warmup_entry(entry)

    assert captured["matrix_args"] == [
        (1, 6, app.GE_INFERENCE_CONTENT_EMBED_DIM),
        (1, 6),
        (1, 6),
        (1, app.GE_INFERENCE_CONTENT_EMBED_DIM),
        [[app.AUTHOR_PAD_IDX] * 6],
        [app.AUTHOR_UNK_IDX],
    ]


def test_load_entry_rejects_ranker_without_matrix_scorer(monkeypatch):
    app = _load_app_module("inference_service_ranker_forward_only_load_tests", model_types="ranker")

    class ForwardOnlyRanker:
        def eval(self):
            return self

        def __call__(self, *args):
            return app.torch.Tensor([0.5])

    monkeypatch.setattr(app, "_resolve_model_file", lambda _entry: ("ranker.pt", "ranker-id"))
    monkeypatch.setattr(app.torch.jit, "load", lambda *args, **kwargs: ForwardOnlyRanker())

    entry = app.LoadedModel(model_type="ranker")
    with pytest.raises(RuntimeError, match="score_candidate_matrix"):
        app._load_entry(entry)


def test_load_entry_rejects_ranker_with_non_callable_matrix_scorer(monkeypatch):
    app = _load_app_module("inference_service_ranker_non_callable_matrix_load_tests", model_types="ranker")

    class BrokenMatrixRanker:
        score_candidate_matrix = None

        def eval(self):
            return self

    monkeypatch.setattr(app, "_resolve_model_file", lambda _entry: ("ranker.pt", "ranker-id"))
    monkeypatch.setattr(app.torch.jit, "load", lambda *args, **kwargs: BrokenMatrixRanker())

    entry = app.LoadedModel(model_type="ranker")
    with pytest.raises(RuntimeError, match="score_candidate_matrix"):
        app._load_entry(entry)


def test_init_registry_fails_on_missing_manifest_keys(tmp_path):
    import json
    bad_manifest = {"post_tower_clearml_model_id": "abc"}  # missing post_tower_uri etc.
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(bad_manifest))

    app = _load_app_module("inference_service_manifest_bad_key_tests")
    os.environ["GE_INFERENCE_TWO_TOWER_MANIFEST_URI"] = str(path)
    os.environ["GE_INFERENCE_MODELS"] = "post-tower"

    app._models_initialized = False
    app._init_registry()

    assert app._models_init_error is not None


def test_ready_response_includes_model_uuid(monkeypatch):
    app = _load_app_module("inference_service_ready_uuid_tests")

    entry = app.LoadedModel(model_type="post-tower", model_uuid="post-model-abc123")
    entry.module = object()
    entry.device = app.torch.device("cpu")
    monkeypatch.setattr(app, "_models", {"post-tower": entry})
    monkeypatch.setattr(app, "_models_initialized", True)
    monkeypatch.setattr(app, "_models_init_error", None)
    monkeypatch.setattr(app, "ensure_models_loaded", lambda: None)
    _set_author_idx_map(app, monkeypatch, {})
    monkeypatch.setattr(app, "_author_idx_maps_init_error", None)

    response = app.ready()
    models_in_response = response.body if hasattr(response, "body") else None
    # Parse JSON body from JSONResponse
    import json
    body = json.loads(response.body)
    assert body["models"][0]["model_uuid"] == "post-model-abc123"
    assert body["author_idx_maps"]["two-tower"]["ready"] is True


def test_ready_response_reports_ranker_author_idx_map(monkeypatch):
    app = _load_app_module(
        "inference_service_ready_ranker_author_map_tests",
        author_idx_map_uri=None,
        ranker_author_idx_map_uri=DEFAULT_RANKER_AUTHOR_IDX_MAP_URI,
        model_types="ranker",
    )

    entry = app.LoadedModel(model_type="ranker", model_uuid="ranker-model-abc123")
    entry.module = object()
    entry.device = app.torch.device("cpu")
    entry.max_history_len = 6
    monkeypatch.setattr(app, "_models", {"ranker": entry})
    monkeypatch.setattr(app, "_models_initialized", True)
    monkeypatch.setattr(app, "_models_init_error", None)
    monkeypatch.setattr(app, "ensure_models_loaded", lambda: None)
    _set_author_idx_map(app, monkeypatch, {"author-1": 7}, name="ranker")
    monkeypatch.setattr(app, "_author_idx_maps_init_error", None)

    response = app.ready()

    import json
    body = json.loads(response.body)
    assert body["ready"] is True
    assert body["author_idx_maps"]["ranker"]["ready"] is True
    assert body["models"][0]["max_history_len"] == 6
    assert "two-tower" not in body["author_idx_maps"]


def test_predict_response_includes_model_uuid(app_request, monkeypatch):
    def post_model(post_embeddings, author_indices):
        return app_request.torch.Tensor([[2.0, 3.0]])

    _set_author_idx_map(app_request, monkeypatch, {})

    entry = app_request.LoadedModel(model_type="post-tower", model_uuid="post-model-abc123")
    entry.module = post_model
    entry.device = app_request.torch.device("cpu")

    monkeypatch.setattr(app_request, "_models_initialized", True)
    monkeypatch.setattr(app_request, "_models_init_error", None)
    monkeypatch.setattr(app_request, "_models", {"post-tower": entry})
    monkeypatch.setattr(app_request, "ensure_models_loaded", lambda: None)

    result = app_request.predict_model(
        "post-tower",
        req=app_request.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0]),
    )

    assert result["model_uuid"] == "post-model-abc123"
    assert result["model_type"] == "post-tower"
