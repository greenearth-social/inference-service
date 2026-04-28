import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[0]
APP_PATH = REPO_ROOT / "app.py"


def _install_stub_modules() -> None:
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class DummyTensor:
            def __init__(self, value):
                self.value = value

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
        torch.ones = lambda shape, dtype=None, device=None: DummyTensor(shape)
        torch.inference_mode = inference_mode
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.jit = types.SimpleNamespace(ScriptModule=type("ScriptModule", (), {}), load=lambda *args, **kwargs: None)
        sys.modules["torch"] = torch

    if "clearml" not in sys.modules:
        clearml = types.ModuleType("clearml")
        clearml.Model = type("Model", (), {})
        sys.modules["clearml"] = clearml

    if "fastapi" not in sys.modules:
        fastapi = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def get(self, *args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

            def post(self, *args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

        class HTTPException(Exception):
            def __init__(self, status_code=None, detail=None):
                self.status_code = status_code
                self.detail = detail
                super().__init__(detail)

        fastapi.FastAPI = FastAPI
        fastapi.HTTPException = HTTPException
        fastapi.Security = lambda dependency=None: dependency
        fastapi.Body = lambda *args, **kwargs: None
        sys.modules["fastapi"] = fastapi

        responses = types.ModuleType("fastapi.responses")
        responses.JSONResponse = type("JSONResponse", (), {})
        sys.modules["fastapi.responses"] = responses

        security = types.ModuleType("fastapi.security")

        class APIKeyHeader:
            def __init__(self, *args, **kwargs):
                pass

        security.APIKeyHeader = APIKeyHeader
        sys.modules["fastapi.security"] = security

    if "pydantic" not in sys.modules:
        pydantic = types.ModuleType("pydantic")

        def model_validator(*, mode):
            def decorator(fn):
                fn.__model_validator_mode__ = mode
                return fn

            return decorator

        class BaseModel:
            __after_validators__ = []

            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.__after_validators__ = [
                    name
                    for name, value in cls.__dict__.items()
                    if getattr(value, "__model_validator_mode__", None) == "after"
                ]

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                for validator_name in self.__class__.__after_validators__:
                    getattr(self, validator_name)()

        pydantic.BaseModel = BaseModel
        pydantic.Discriminator = lambda value: value
        pydantic.Tag = lambda value: value
        pydantic.model_validator = model_validator
        sys.modules["pydantic"] = pydantic

    if "shared" not in sys.modules:
        shared = types.ModuleType("shared")
        input_data_helpers = types.ModuleType("shared.input_data_helpers")
        input_data_helpers.get_padded_embedding_history_and_mask_batched = lambda **kwargs: ([], [])
        shared.input_data_helpers = input_data_helpers
        sys.modules["shared"] = shared
        sys.modules["shared.input_data_helpers"] = input_data_helpers


def _load_app_module(module_name: str, *, max_batch: int = 4, embed_dim: int = 0, max_history_len: int = 8):
    _install_stub_modules()

    os.environ["GE_INFERENCE_MAX_BATCH"] = str(max_batch)
    os.environ["GE_INFERENCE_EMBED_DIM"] = str(embed_dim)
    os.environ["GE_INFERENCE_MAX_HISTORY_LEN"] = str(max_history_len)

    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app_shape():
    return _load_app_module("inference_service_app_shape_tests")


@pytest.fixture(scope="module")
def app_request():
    return _load_app_module("inference_service_app_request_tests", max_batch=3, embed_dim=0)


@pytest.fixture(scope="module")
def app_fixed_dim():
    return _load_app_module("inference_service_app_fixed_dim_tests", max_batch=4, embed_dim=3)


def test_classifies_empty_flat_history_as_single_empty(app_shape):
    assert app_shape._classify_history_embeddings_shape([]) == "single_empty"


def test_classifies_empty_nested_history_as_single_empty(app_shape):
    assert app_shape._classify_history_embeddings_shape([[]]) == "single_empty"


def test_classifies_single_history_as_single_history(app_shape):
    assert app_shape._classify_history_embeddings_shape([[1.0, 2.0], [3.0, 4.0]]) == "single_history"


def test_classifies_batch_with_empty_first_user_as_batched_history(app_shape):
    assert app_shape._classify_history_embeddings_shape([[], [[1.0, 2.0]]]) == "batched_history"


def test_classifies_three_dimensional_input_as_batched_history(app_shape):
    assert app_shape._classify_history_embeddings_shape([[[1.0, 2.0]], [[3.0, 4.0]]]) == "batched_history"


def test_rejects_non_list_top_level(app_shape):
    with pytest.raises(ValueError, match="history_embeddings must be a list"):
        app_shape._classify_history_embeddings_shape("not-a-list")


def test_rejects_top_level_list_that_does_not_contain_lists(app_shape):
    with pytest.raises(ValueError, match="history_embeddings must be a list of lists"):
        app_shape._classify_history_embeddings_shape([1.0, 2.0])


def test_accepts_empty_flat_history(app_request):
    app_request.UserTowerPredictRequest(history_embeddings=[])


def test_accepts_empty_nested_history(app_request):
    app_request.UserTowerPredictRequest(history_embeddings=[[]])


def test_accepts_single_history(app_request):
    app_request.UserTowerPredictRequest(history_embeddings=[[1.0, 2.0], [3.0, 4.0]])


def test_accepts_batched_histories_with_empty_entries(app_request):
    app_request.UserTowerPredictRequest(
        history_embeddings=[[], [[1.0, 2.0], [3.0, 4.0]], [[]]]
    )


def test_rejects_batched_history_over_max_batch(app_request):
    with pytest.raises(ValueError, match="batch too large"):
        app_request.UserTowerPredictRequest(
            history_embeddings=[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]
        )


def test_rejects_single_history_with_mismatched_embedding_dimensions(app_request):
    with pytest.raises(ValueError, match="same dimension"):
        app_request.UserTowerPredictRequest(history_embeddings=[[1.0, 2.0], [3.0]])


def test_rejects_mixed_rank_batch_entry(app_request):
    with pytest.raises(ValueError, match="list of lists"):
        app_request.UserTowerPredictRequest(history_embeddings=[[[1.0, 2.0]], [1.0, 2.0]])


def test_rejects_non_list_user_entry_in_batch(app_request):
    with pytest.raises(ValueError, match="each user's history must be a list"):
        app_request._validate_batched_user_history([[[1.0, 2.0]], "bad-user"])


def test_accepts_histories_matching_fixed_embedding_dim(app_fixed_dim):
    app_fixed_dim.UserTowerPredictRequest(history_embeddings=[[1.0, 2.0, 3.0]])
    app_fixed_dim.UserTowerPredictRequest(history_embeddings=[[[1.0, 2.0, 3.0]], []])


def test_rejects_single_history_that_violates_fixed_embedding_dim(app_fixed_dim):
    with pytest.raises(ValueError, match="embedding dim must be 3"):
        app_fixed_dim.UserTowerPredictRequest(history_embeddings=[[1.0, 2.0]])


def test_rejects_batched_history_that_violates_fixed_embedding_dim(app_fixed_dim):
    with pytest.raises(ValueError, match="embedding dim must be 3"):
        app_fixed_dim.UserTowerPredictRequest(
            history_embeddings=[[[1.0, 2.0, 3.0]], [[4.0, 5.0]]]
        )


def test_rejects_zero_width_embedding_row(app_fixed_dim):
    with pytest.raises(ValueError, match="greater than 0"):
        app_fixed_dim._validate_single_user_history([[]])
