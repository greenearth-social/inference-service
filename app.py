import os
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, Literal, Annotated, assert_never, get_args
from urllib.parse import urlparse
import logging

import torch
from clearml import Model
from fastapi import FastAPI, HTTPException, Security, Body
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Discriminator, Tag, model_validator

from shared.input_data_helpers import get_padded_embedding_history_and_mask_batched, classify_history_embeddings_shape


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_models_loaded()
    yield


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
app = FastAPI(title="Green Earth Inference Service", lifespan=lifespan)


def _require_api_key(api_key: str = Security(_api_key_header)) -> None:
    if api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
GE_INFERENCE_PREFER_CUDA = os.getenv("GE_INFERENCE_PREFER_CUDA", "1") == "1"
GE_INFERENCE_WARMUP = os.getenv("GE_INFERENCE_WARMUP", "1") == "1"
_API_KEY: str | None = os.environ.get("GE_INFERENCE_API_KEY") or None

GE_INFERENCE_CONTENT_EMBED_DIM = int(os.getenv("GE_INFERENCE_CONTENT_EMBED_DIM", "0"))
if GE_INFERENCE_CONTENT_EMBED_DIM <= 0:
    raise ValueError("Must supply a valid (positive) GE_INFERENCE_CONTENT_EMBED_DIM!")

GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN = int(os.getenv("GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN", "0")) 
if GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN <= 0:
    raise ValueError("Must supply a valid (positive) GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN!")

GE_INFERENCE_MAX_BATCH: int | None = int(os.getenv("GE_INFERENCE_MAX_BATCH", "0"))
if GE_INFERENCE_MAX_BATCH == 0:
    GE_INFERENCE_MAX_BATCH = None

GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI: str = os.getenv("GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI", "")
if GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI == "":
    raise ValueError("Must supply a valid GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI!")

DTYPE_FLOAT = torch.float32

AUTHOR_PAD_IDX = 0
AUTHOR_UNK_IDX = 1

# -------------------------
# State
# -------------------------
ModelType = Literal["user-tower", "post-tower"]


@dataclass
class LoadedModel:
    model_type: ModelType
    configured_model_path: str | None = None
    configured_model_uri: str | None = None
    configured_clearml_model_id: str | None = None

    module: torch.jit.ScriptModule | None = None
    device: torch.device | None = None
    resolved_model_path: str | None = None
    resolved_model_id: str | None = None
    model_uuid: str | None = None

    load_error: str | None = None
    load_started_at: float | None = None
    load_finished_at: float | None = None


_models_lock = threading.Lock()
_models_initialized = False
_models_init_error: str | None = None
_models: dict[str, LoadedModel] = {}

_two_tower_author_idx_by_did: dict[str, int] | None = None
_two_tower_author_idx_map_load_error: str | None = None
_two_tower_author_idx_map_load_started_at: float | None = None
_two_tower_author_idx_map_load_finished_at: float | None = None
_two_tower_author_idx_map_resolved_path: str | None = None


# -------------------------
# API schema
# -------------------------


def _validate_single_user_history(user_history: list[list[float]]) -> int:
    first_len = len(user_history[0])
    if first_len == 0:
        raise ValueError("embedding dimension must be greater than 0")
    if GE_INFERENCE_CONTENT_EMBED_DIM:
        if not all(len(history_post) == GE_INFERENCE_CONTENT_EMBED_DIM for history_post in user_history):
            raise ValueError(f"embedding dim must be {GE_INFERENCE_CONTENT_EMBED_DIM} for all history embeddings")
    else:
        if not all(len(history_post) == first_len for history_post in user_history):
            raise ValueError(f"all history embeddings must have the same dimension as one another")
    return len(user_history)


def _validate_batched_user_history(history_embeddings: list[list[Any]]) -> list[int]:
    # have to account for the possibility of empty entries in the batch. which could be [] or [[]]
    hist_len_list = []
    if GE_INFERENCE_MAX_BATCH and len(history_embeddings) > GE_INFERENCE_MAX_BATCH:
        raise ValueError(f"batch too large! (got={len(history_embeddings)}; max={GE_INFERENCE_MAX_BATCH})")
    for user in history_embeddings:
        if not isinstance(user, list):
            raise ValueError("each user's history must be a list")
        if len(user) == 0 or (len(user) == 1 and isinstance(user[0], list) and len(user[0]) == 0):
            hist_len_list.append(0)
            continue # empty history, which is ok
        if not isinstance(user[0], list):
            raise ValueError("each (non-empty) user's history must be a list of lists")
        hist_len = _validate_single_user_history(user)
        hist_len_list.append(hist_len)
    return hist_len_list


class UserTowerPredictRequest(BaseModel):
    # history_embeddings: [T, D] or [B, T, D]
    history_embeddings: list[list[float]] | list[list[list[float]]]
    history_author_dids: list[str] | list[list[str]] | None = None

    @model_validator(mode="after")
    def _validate_history(self) -> "UserTowerPredictRequest":
        he = self.history_embeddings
        shape = classify_history_embeddings_shape(he)
        author_dids = self.history_author_dids

        match shape:
            case "single_empty":
                if author_dids is None:
                    return self
                if not isinstance(author_dids, list):
                    raise ValueError("'history_author_dids' must be a list (of strings or list of list of strings)")
                if len(author_dids) != 0:
                    raise ValueError("when 'history_embeddings' is empty, 'history_author_dids' must also be empty")
                return self
            case "single_history":
                hist_len = _validate_single_user_history(he) # type: ignore
                if author_dids is None:
                    return self
                if not isinstance(author_dids, list):
                    raise ValueError("'history_author_dids' must be a list (of strings or list of list of strings)")
                if len(author_dids) == 0 or isinstance(author_dids[0], list):
                    raise ValueError("when 'history_embeddings' is a single history, 'history_author_dids' must be a list of strings, not a list of list of strings")
                if len(author_dids) != hist_len:
                    raise ValueError(f"length of 'history_author_dids' must match history length ({hist_len}) when 'history_embeddings' is a single history")
                return self
            case "batched_history":
                hist_len_list = _validate_batched_user_history(he)
                if author_dids is None:
                    return self
                if not isinstance(author_dids, list):
                    raise ValueError("'history_author_dids' must be a list (of strings or list of list of strings)")
                if len(author_dids) == 0 or isinstance(author_dids[0], str):
                    raise ValueError("when 'history_embeddings' is batched, 'history_author_dids' must be a list of list of strings, not a list of strings")
                if len(author_dids) != len(hist_len_list):
                    raise ValueError(f"length of 'history_author_dids' must match batch size ({len(hist_len_list)}) when 'history_embeddings' is batched")
                if not all(len(user_hti) == hist_len for user_hti, hist_len in zip(author_dids, hist_len_list)):
                    raise ValueError(f"length of each user's 'history_author_dids' must match that user's history length when 'history_embeddings' is batched")
                return self
            case _:
                assert_never(shape)


class PostTowerPredictRequest(BaseModel):
    # post_embeddings: [D] or [B, D]
    post_embeddings: list[float] | list[list[float]]
    candidate_author_dids: str | list[str] | None = None

    @model_validator(mode="after")
    def _validate_post_inputs(self) -> "PostTowerPredictRequest":
        pe = self.post_embeddings
        if not isinstance(pe, list) or len(pe) == 0:
            raise ValueError("'post_embeddings' must be a non-empty list")

        author_dids = self.candidate_author_dids

        is_batched = isinstance(pe[0], list)
        if is_batched:
            batch = pe  # type: ignore[assignment]
            if GE_INFERENCE_MAX_BATCH and len(batch) > GE_INFERENCE_MAX_BATCH:
                raise ValueError(f"batch too large (max={GE_INFERENCE_MAX_BATCH})")
            d0 = len(batch[0]) if len(batch) > 0 else 0 # type: ignore
            if d0 == 0:
                raise ValueError("each post_embeddings vector must be non-empty")
            if not all(isinstance(v, list) and len(v) == d0 for v in batch):
                raise ValueError("all post_embeddings vectors must have the same length")
            if GE_INFERENCE_CONTENT_EMBED_DIM and d0 != GE_INFERENCE_CONTENT_EMBED_DIM:
                raise ValueError(f"expected D={GE_INFERENCE_CONTENT_EMBED_DIM}, got D={d0}")
            if author_dids is None:
                return self
            if not isinstance(author_dids, list) or len(author_dids) != len(batch):
                raise ValueError("when post_embeddings is batched, target_author_dids must be a list of the same length as the batch")
        else:
            vec = pe  # type: ignore[assignment]
            if len(vec) == 0:
                raise ValueError("'post_embeddings' must be non-empty")
            if GE_INFERENCE_CONTENT_EMBED_DIM and len(vec) != GE_INFERENCE_CONTENT_EMBED_DIM:
                raise ValueError(f"expected D={GE_INFERENCE_CONTENT_EMBED_DIM}, got D={len(vec)}")
            if author_dids is None:
                return self
            if not isinstance(author_dids, str):
                raise ValueError("target_author_dids must be a single string when post_embeddings is not batched")
        return self


def _predict_request_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        if "post_embeddings" in value:
            return "post-tower"
        if "history_embeddings" in value:
            return "user-tower"
    raise ValueError("Request must contain one of: 'post_embeddings' or 'history_embeddings'.")


PredictRequest = Annotated[
    Annotated[UserTowerPredictRequest, Tag("user-tower")] | Annotated[PostTowerPredictRequest, Tag("post-tower")],
    Discriminator(_predict_request_discriminator),
]


# -------------------------
# Helpers
# -------------------------
def _choose_device() -> torch.device:
    if GE_INFERENCE_PREFER_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_model_file(path: str) -> str:
    if os.path.isfile(path):
        return path
    raise RuntimeError(f"Model path is not a file: {path}")


def _download_gcs_uri_to_local(gs_uri: str) -> str:
    """
    Download a single GCS object (gs://bucket/path) to a local cache and return its path.
    Uses Application Default Credentials.
    """
    parsed = urlparse(gs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid gs:// URI (missing bucket or object): {gs_uri}")

    model_cache_dir = os.getenv("GE_INFERENCE_MODEL_CACHE_DIR", "/tmp/model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)

    blob_basename = os.path.basename(blob_name) or "model"
    key = sha256(gs_uri.encode("utf-8")).hexdigest()[:16]
    local_path = os.path.join(model_cache_dir, f"{key}-{blob_basename}")

    if os.path.exists(local_path):
        return local_path

    # Import lazily so non-GCS paths don't require this dependency at import time.
    from google.cloud import storage  # type: ignore[import-not-found]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

    return local_path


def _load_manifest(uri: str) -> dict:
    """Load the two_tower_serving_manifest.json from a local path or GCS URI."""
    import json
    path = _download_gcs_uri_to_local(uri) if uri.startswith("gs://") else uri
    with open(path) as f:
        return json.load(f)


def _resolve_author_idx_map_file(author_idx_map_uri: str) -> str:
    parsed = urlparse(author_idx_map_uri)
    if parsed.scheme == "gs":
        return _find_model_file(_download_gcs_uri_to_local(author_idx_map_uri))
    return _find_model_file(author_idx_map_uri)


def _load_author_idx_map_from_parquet(path: str) -> dict[str, int]:
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path, columns=["author_did", "author_idx"])
    except Exception as e:
        raise RuntimeError(f"Failed to read author idx map parquet '{path}': {e}")

    author_dids = table["author_did"].to_pylist()
    author_idxs = table["author_idx"].to_pylist()
    if len(author_dids) != len(author_idxs):
        raise ValueError("author idx map columns have mismatched lengths")

    author_idx_by_did: dict[str, int] = {}
    for raw_author_did, raw_author_idx in zip(author_dids, author_idxs):
        if raw_author_did is None:
            raise ValueError("author idx map contains a null author_did")
        if raw_author_idx is None:
            raise ValueError(f"author idx map contains a null author_idx for author_did='{raw_author_did}'")

        author_did = str(raw_author_did)
        if author_did == "":
            raise ValueError("author idx map contains an empty author_did")
        if author_did in author_idx_by_did:
            raise ValueError(f"author idx map contains duplicate author_did='{author_did}'")

        author_idx = int(raw_author_idx)
        if author_idx < 0:
            raise ValueError(f"author idx map contains a negative author_idx for author_did='{author_did}'")

        author_idx_by_did[author_did] = author_idx

    return author_idx_by_did


def _ensure_two_tower_author_idx_map_loaded() -> None:
    global _two_tower_author_idx_by_did, _two_tower_author_idx_map_load_error
    global _two_tower_author_idx_map_load_started_at, _two_tower_author_idx_map_load_finished_at, _two_tower_author_idx_map_resolved_path

    if _two_tower_author_idx_by_did is not None:
        return

    with _models_lock:
        if _two_tower_author_idx_by_did is not None:
            return
        if _two_tower_author_idx_map_load_started_at is not None and _two_tower_author_idx_map_load_finished_at is None:
            return

        _two_tower_author_idx_map_load_started_at = time.time()
        try:
            resolved_path = _resolve_author_idx_map_file(GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI)
            _two_tower_author_idx_by_did = _load_author_idx_map_from_parquet(resolved_path)
            _two_tower_author_idx_map_resolved_path = resolved_path
            _two_tower_author_idx_map_load_error = None
            logger.info(
                "Author idx map loaded | source=%s | path=%s | entries=%s",
                GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI,
                resolved_path,
                len(_two_tower_author_idx_by_did),
            )
        except Exception as e:
            _two_tower_author_idx_by_did = None
            _two_tower_author_idx_map_load_error = str(e)
            logger.exception("Author idx map load failed | source=%s | error=%s", GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI, e)
        finally:
            _two_tower_author_idx_map_load_finished_at = time.time()


def _validate_model_type(model_type: str) -> ModelType:
    if model_type in get_args(ModelType):
        return model_type # type: ignore[return-value]
    raise RuntimeError(f"Unsupported model type: '{model_type}'")


def _tensor_from_nested_list(name: str, value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if value is None:
        raise HTTPException(status_code=400, detail=f"Missing required field '{name}'")
    try:
        t = torch.tensor(value, dtype=dtype, device=device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid '{name}': {e}")
    if t.numel() == 0:
        raise HTTPException(status_code=400, detail=f"'{name}' must be non-empty")
    return t


def _model_env_key(model_type: str) -> str:
    # Model names may contain "-" which isn't valid in env vars.
    # Example: "user-tower" -> "USER_TOWER"
    return "".join((c if c.isalnum() else "_") for c in model_type).upper()


def _to_python(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            out[k] = _to_python(v)
        return out
    return obj


def _format_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _warmup_entry(entry: LoadedModel) -> None:
    """Best-effort warmup to initialize CUDA context and validate the forward pass."""
    if entry.device is None or entry.module is None:
        return
    device = entry.device
    model = entry.module
    if device.type != "cuda":
        return
    if not GE_INFERENCE_WARMUP:
        return

    with torch.inference_mode():
        # Keep warmup short.
        if entry.model_type == "post-tower":
            if GE_INFERENCE_CONTENT_EMBED_DIM <= 0:
                return
            post_embeddings = torch.zeros((1, GE_INFERENCE_CONTENT_EMBED_DIM), dtype=DTYPE_FLOAT, device=device)
            candidate_author_indices = torch.tensor([AUTHOR_UNK_IDX], dtype=torch.int64, device=device)
            _ = model(post_embeddings, candidate_author_indices)
            return

        if entry.model_type == "user-tower":
            if GE_INFERENCE_CONTENT_EMBED_DIM <= 0:
                return
            history_embeddings = torch.zeros(
                (1, GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN, GE_INFERENCE_CONTENT_EMBED_DIM), dtype=DTYPE_FLOAT, device=device
            )
            history_mask = torch.ones((1, GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN), dtype=torch.bool, device=device)
            history_author_indices = torch.tensor([[AUTHOR_PAD_IDX] * GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN], dtype=torch.int64, device=device)
            _ = model(history_embeddings, history_mask, history_author_indices)
            return


def _init_registry() -> None:
    global _models_initialized, _models_init_error, _models
    if _models_initialized:
        return

    with _models_lock:
        if _models_initialized:
            return

        try:
            models: dict[str, LoadedModel] = {}

            manifest_uri = os.getenv("GE_INFERENCE_MANIFEST_URI", "").strip()
            if not manifest_uri:
                raise RuntimeError(
                    "GE_INFERENCE_MANIFEST_URI is required — set it to the GCS URI or local path of "
                    "the two_tower_serving_manifest.json produced by training."
                )

            manifest = _load_manifest(manifest_uri)
            post_tower_uri = manifest["post_tower_uri"]
            post_tower_uuid = manifest["post_tower_clearml_model_id"]
            user_tower_uri = manifest["user_tower_uri"]
            user_tower_uuid = manifest["user_tower_clearml_model_id"]

            models_env = os.getenv("GE_INFERENCE_MODELS", "").strip()
            if not models_env:
                raise RuntimeError(
                    "No models configured. Set GE_INFERENCE_MODELS (e.g. 'user-tower,post-tower')."
                )

            env_model_types: list[str] = models_env.split(",")
            if len(env_model_types) > 2:
                raise RuntimeError(f"Too many models configured ({len(env_model_types)}). Max is 2.")

            tower_uris = {"post-tower": (post_tower_uri, post_tower_uuid), "user-tower": (user_tower_uri, user_tower_uuid)}

            seen: set[str] = set()
            for env_model_type in env_model_types:
                if env_model_type in seen:
                    continue
                seen.add(env_model_type)

                model_type: ModelType = _validate_model_type(env_model_type)
                uri, model_uuid = tower_uris[model_type]

                models[model_type] = LoadedModel(
                    model_type=model_type,
                    configured_model_uri=uri,
                    model_uuid=model_uuid,
                )

            _models = models
            _models_init_error = None
        except Exception as e:
            _models = {}
            _models_init_error = str(e)
        finally:
            _models_initialized = True


def _resolve_model_file(entry: LoadedModel) -> tuple[str, str | None]:
    model_id = None
    if entry.configured_model_path:
        model_file = _find_model_file(entry.configured_model_path)
    elif entry.configured_model_uri:
        parsed = urlparse(entry.configured_model_uri)
        if parsed.scheme == "gs":
            model_file = _find_model_file(_download_gcs_uri_to_local(entry.configured_model_uri))
        else:
            model_file = _find_model_file(entry.configured_model_uri)
    else:
        model_id = entry.configured_clearml_model_id
        if not model_id:
            model_env_key = _model_env_key(entry.model_type)
            raise RuntimeError(
                f"Model '{entry.model_type}' is missing a source (GE_INFERENCE_{model_env_key}_MODEL_PATH | GE_INFERENCE_{model_env_key}_MODEL_URI | GE_INFERENCE_{model_env_key}_CLEARML_MODEL_ID)"
            )
        cm = Model(model_id=model_id)
        local_copy = cm.get_local_copy()
        model_file = _find_model_file(local_copy)
    return model_file, model_id


def _load_entry(entry: LoadedModel) -> None:
    device = _choose_device()
    model_file, model_id = _resolve_model_file(entry)

    m = torch.jit.load(model_file, map_location=device)
    m.eval()

    entry.module = m
    entry.device = device
    entry.resolved_model_path = model_file
    entry.resolved_model_id = model_id

    _warmup_entry(entry)

    logger.info(
        "Model loaded | type=%s | model_id=%s | model_path=%s | device=%s",
        entry.model_type,
        model_id,
        model_file,
        device,
    )


def ensure_models_loaded() -> None:
    """Concurrency-safe, idempotent load of all configured models."""
    _init_registry()
    _ensure_two_tower_author_idx_map_loaded()
    if _models_init_error is not None:
        logger.error("Model registry init failed: %s", _models_init_error)
        return

    with _models_lock:
        for entry in _models.values():
            if entry.module is not None:
                continue
            if entry.load_started_at is not None and entry.load_finished_at is None:
                continue

            entry.load_started_at = time.time()
            try:
                _load_entry(entry)
                entry.load_error = None
            except Exception as e:
                entry.load_error = str(e)
                logger.exception("Model load failed | type=%s | error=%s", entry.model_type, entry.load_error)
            finally:
                entry.load_finished_at = time.time()


def _get_entry_or_404(model_name: str) -> LoadedModel:
    _init_registry()
    if _models_init_error is not None:
        raise HTTPException(status_code=500, detail=f"Model registry init failed: {_models_init_error}")

    entry = _models.get(model_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")
    return entry


def _require_ready(entry: LoadedModel) -> None:
    if entry.module is not None and entry.device is not None:
        return
    ensure_models_loaded()
    if entry.module is None or entry.device is None:
        raise HTTPException(
            status_code=503,
            detail={"model_type": entry.model_type, "ready": False, "load_error": entry.load_error},
        )

def _get_single_author_idx_from_did(author_did: str, author_idx_by_did: dict[str, int]) -> int:
    if author_did in author_idx_by_did:
        return author_idx_by_did[author_did] 
    else:
        return AUTHOR_UNK_IDX


def _get_author_indices_from_dids(
    author_dids: str | list[str] | list[list[str]],
) -> list[int] | list[list[int]]:
    if isinstance(author_dids, str):
        if _two_tower_author_idx_by_did is None:
            raise HTTPException(status_code=503, detail="Author idx map not loaded")
        return [_get_single_author_idx_from_did(author_dids, _two_tower_author_idx_by_did)]
    elif isinstance(author_dids, list):
        if len(author_dids) == 0:
            return []
        if _two_tower_author_idx_by_did is None:
            raise HTTPException(status_code=503, detail="Author idx map not loaded")
        elif isinstance(author_dids[0], str):
            return [
                _get_single_author_idx_from_did(did, _two_tower_author_idx_by_did) # type: ignore
                for did in author_dids
            ]
        else: 
            if not isinstance(author_dids[0], list):
                raise HTTPException(status_code=422, detail="author dids must either be a string, a list of strings, or a list of list of strings")
            return [
                [
                    _get_single_author_idx_from_did(did, _two_tower_author_idx_by_did)
                    for did in author_did_list
                ] 
                for author_did_list in author_dids
            ]


def _get_target_author_indices_for_request(
    req: PostTowerPredictRequest,
) -> list[int] | list[list[int]]:
    if req.candidate_author_dids is not None:
        return _get_author_indices_from_dids(req.candidate_author_dids)
    if isinstance(req.post_embeddings[0], list):
        return [AUTHOR_UNK_IDX] * len(req.post_embeddings)
    return [AUTHOR_UNK_IDX]


def _predict_with_entry(entry: LoadedModel, req: PredictRequest) -> Any:
    _require_ready(entry)
    assert entry.module is not None and entry.device is not None

    with torch.inference_mode():
        match entry.model_type:
            case "user-tower":
                # Enforce schema against registered model type.
                if not isinstance(req, UserTowerPredictRequest):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Model type '{entry.model_type}' expects a user-tower request body with 'history_embeddings'",
                    )

                author_indices_list = (
                    _get_author_indices_from_dids(req.history_author_dids)
                    if req.history_author_dids is not None
                    else None
                )
                # take raw list inputs and pad/truncate:
                history_embeddings_padded, history_mask_padded, author_indices_padded = get_padded_embedding_history_and_mask_batched(
                    history_embeddings=req.history_embeddings,
                    max_history_len=GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN,
                    embed_dim=GE_INFERENCE_CONTENT_EMBED_DIM,
                    author_indices=author_indices_list,
                )
                history_embeddings = _tensor_from_nested_list("history_embeddings", history_embeddings_padded, DTYPE_FLOAT, entry.device)
                history_mask = _tensor_from_nested_list("history_mask", history_mask_padded, torch.bool, entry.device)
                author_indices = _tensor_from_nested_list("author_indices", author_indices_padded, torch.int64, entry.device)

                y = entry.module(history_embeddings, history_mask, author_indices)
                return y
            case "post-tower":
                # Enforce schema against registered model type.
                if not isinstance(req, PostTowerPredictRequest):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Model type '{entry.model_type}' expects a post-tower request body with 'post_embeddings'",
                    )
                post_embeddings = _tensor_from_nested_list("post_embeddings", req.post_embeddings, DTYPE_FLOAT, entry.device)
                if post_embeddings.dim() == 1:
                    post_embeddings = post_embeddings.unsqueeze(0) # add a batch dimension of size 1 at the beginning

                author_indices_list = _get_target_author_indices_for_request(req)
                author_indices = _tensor_from_nested_list("target_author_dids", author_indices_list, torch.int64, entry.device)
                y = entry.module(post_embeddings, author_indices)
                return y
            case _:
                assert_never(entry.model_type)

    raise HTTPException(status_code=500, detail=f"Unsupported model type: {entry.model_type}")


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health() -> dict:
    # Process is up.
    return {"ok": True}


@app.get("/ready", dependencies=[Security(_require_api_key)])
def ready():
    _init_registry()
    ensure_models_loaded()

    models_payload: list[dict[str, Any]] = []
    all_ready = _models_init_error is None and len(_models) > 0
    author_idx_map_ready = _two_tower_author_idx_by_did is not None and _two_tower_author_idx_map_load_error is None
    all_ready = all_ready and author_idx_map_ready
    for entry in _models.values():
        model_ready = entry.module is not None and entry.device is not None and entry.load_error is None
        all_ready = all_ready and model_ready
        models_payload.append(
            {
                "type": entry.model_type,
                "model_uuid": entry.model_uuid,
                "ready": model_ready,
                "device": str(entry.device) if entry.device else None,
                "model_path": entry.resolved_model_path,
                "model_id": entry.resolved_model_id,
                "load_error": entry.load_error,
                "load_started_at": _format_timestamp(entry.load_started_at),
                "load_finished_at": _format_timestamp(entry.load_finished_at),
            }
        )

    payload = {
        "ready": all_ready,
        "registry_error": _models_init_error,
        "embed_dim": GE_INFERENCE_CONTENT_EMBED_DIM if GE_INFERENCE_CONTENT_EMBED_DIM > 0 else None,
        "max_seq_len": GE_INFERENCE_TWO_TOWER_MAX_HISTORY_LEN,
        "author_idx_map": {
            "ready": author_idx_map_ready,
            "uri": GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI,
            "resolved_path": _two_tower_author_idx_map_resolved_path,
            "num_entries": len(_two_tower_author_idx_by_did) if _two_tower_author_idx_by_did is not None else None,
            "load_error": _two_tower_author_idx_map_load_error,
            "load_started_at": _format_timestamp(_two_tower_author_idx_map_load_started_at),
            "load_finished_at": _format_timestamp(_two_tower_author_idx_map_load_finished_at),
        },
        "models": models_payload,
    }

    status = 200 if all_ready else 503
    return JSONResponse(content=payload, status_code=status)

@app.get("/models", dependencies=[Security(_require_api_key)])
def list_models() -> dict:
    _init_registry()
    ensure_models_loaded()
    models_payload: list[dict[str, Any]] = []
    for entry in _models.values():
        models_payload.append(
            {
                "type": entry.model_type,
                "ready": entry.module is not None and entry.device is not None and entry.load_error is None,
                "device": str(entry.device) if entry.device else None,
                "model_path": entry.resolved_model_path,
                "model_id": entry.resolved_model_id,
                "load_error": entry.load_error,
                "load_started_at": _format_timestamp(entry.load_started_at),
                "load_finished_at": _format_timestamp(entry.load_finished_at),
            }
        )
    return {
        "models": models_payload,
        "registry_error": _models_init_error,
        "two_tower_author_idx_map": {
            "ready": _two_tower_author_idx_by_did is not None and _two_tower_author_idx_map_load_error is None,
            "uri": GE_INFERENCE_TWO_TOWER_AUTHOR_MAP_URI,
            "resolved_path": _two_tower_author_idx_map_resolved_path,
            "num_entries": len(_two_tower_author_idx_by_did) if _two_tower_author_idx_by_did is not None else None,
            "load_error": _two_tower_author_idx_map_load_error,
            "load_started_at": _format_timestamp(_two_tower_author_idx_map_load_started_at),
            "load_finished_at": _format_timestamp(_two_tower_author_idx_map_load_finished_at),
        },
    }


@app.post("/models/{model_name}/predict", dependencies=[Security(_require_api_key)])
def predict_model(model_name: str, req: PredictRequest = Body(...)) -> dict:
    entry = _get_entry_or_404(model_name)
    try:
        y = _predict_with_entry(entry, req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return {"outputs": _to_python(y), "model_type": entry.model_type, "model_uuid": entry.model_uuid}
