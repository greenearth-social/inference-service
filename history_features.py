"""Normalize and pad user-history features for inference requests."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


AUTHOR_PAD_IDX = 0
AUTHOR_UNK_IDX = 1


def get_padded_embedding_history_and_mask(
    history_embeddings: Any,
    max_history_len: int,
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad or truncate one embedding history and return its validity mask."""
    hist_len = len(history_embeddings)

    if hist_len > 0:
        for history_embedding in history_embeddings:
            if len(history_embedding) != embed_dim:
                raise ValueError(
                    f"History embedding length ({len(history_embedding)}) and "
                    f"embed_dim ({embed_dim}) do not match"
                )

    seq_len = min(hist_len, max_history_len)
    padded = np.zeros((max_history_len, embed_dim), dtype=np.float32)
    mask = np.zeros(max_history_len, dtype=bool)

    if seq_len > 0:
        padded[:seq_len] = history_embeddings[:max_history_len]
        mask[:seq_len] = True

    return padded, mask


HistoryEmbeddingsShape = Literal[
    "single_empty",
    "single_history",
    "batched_history",
]


def classify_history_embeddings_shape(
    history_embeddings: Any,
) -> HistoryEmbeddingsShape:
    """Classify the nesting pattern used for one or more history inputs."""
    if not isinstance(history_embeddings, list):
        raise ValueError("history_embeddings must be a list")
    if len(history_embeddings) == 0:
        return "single_empty"

    if not all(isinstance(user_history, list) for user_history in history_embeddings):
        raise ValueError("history_embeddings must be a list of lists")

    if len(history_embeddings[0]) == 0:
        if len(history_embeddings) == 1:
            return "single_empty"
        if not all(
            len(user_history) == 0 or isinstance(user_history[0], list)
            for user_history in history_embeddings
        ):
            raise ValueError(
                "batched history_embeddings must be a list of user histories"
            )
        return "batched_history"

    if isinstance(history_embeddings[0][0], list):
        if not all(
            len(user_history) == 0 or isinstance(user_history[0], list)
            for user_history in history_embeddings
        ):
            raise ValueError(
                "batched history_embeddings must be a list of user histories"
            )
        return "batched_history"

    if any(
        len(user_history) > 0 and isinstance(user_history[0], list)
        for user_history in history_embeddings[1:]
    ):
        raise ValueError(
            "history_embeddings must not mix single-history and batched-history shapes"
        )
    return "single_history"


def _normalize_empty_user_history(
    user_history: list[Any],
) -> list[list[float]]:
    """Collapse supported empty-history sentinels into a plain empty list."""
    if len(user_history) == 0:
        return []
    if (
        len(user_history) == 1
        and isinstance(user_history[0], list)
        and len(user_history[0]) == 0
    ):
        return []
    return user_history  # type: ignore[return-value]


def _normalize_history_inputs_to_batch(
    history_embeddings: Any,
    shape: HistoryEmbeddingsShape,
    author_indices: Any,
    time_deltas_hours: Any,
    prior_cumulative_likes: Any,
) -> tuple[
    list[list[list[float]]],
    list[list[int]],
    list[list[float]],
    list[list[int]],
]:
    """Normalize supported single/batched histories into a batched representation."""
    match shape:
        case "single_empty":
            return [[]], [[]], [[]], [[]]
        case "single_history":
            if author_indices is None:
                author_indices = [AUTHOR_UNK_IDX] * len(history_embeddings)
            if time_deltas_hours is None:
                time_deltas_hours = [0.0] * len(history_embeddings)
            if prior_cumulative_likes is None:
                prior_cumulative_likes = [0] * len(history_embeddings)
            return (
                [history_embeddings],
                [author_indices],
                [time_deltas_hours],
                [prior_cumulative_likes],
            )
        case "batched_history":
            batch_history_embeddings = [
                _normalize_empty_user_history(user_history)
                for user_history in history_embeddings
            ]
            author_indices_result = author_indices
            if author_indices is None:
                author_indices_result = [
                    [AUTHOR_UNK_IDX] * len(user_history)
                    for user_history in batch_history_embeddings
                ]
            time_deltas_hours_result = time_deltas_hours
            if time_deltas_hours is None:
                time_deltas_hours_result = [
                    [0.0] * len(user_history)
                    for user_history in batch_history_embeddings
                ]
            prior_cumulative_likes_result = prior_cumulative_likes
            if prior_cumulative_likes is None:
                prior_cumulative_likes_result = [
                    [0] * len(user_history)
                    for user_history in batch_history_embeddings
                ]
            return (
                batch_history_embeddings,
                author_indices_result,
                time_deltas_hours_result,
                prior_cumulative_likes_result,
            )


def get_padded_author_indices(
    author_indices: Any,
    max_history_len: int,
) -> np.ndarray:
    """Pad author indices with the serving PAD row or truncate from the end."""
    seq_len = min(len(author_indices), max_history_len)
    padded = np.full(max_history_len, fill_value=AUTHOR_PAD_IDX, dtype=np.int64)
    if seq_len > 0:
        padded[:seq_len] = author_indices[:max_history_len]
    return padded


def get_padded_history_time_deltas(
    time_deltas_hours: Any,
    max_history_len: int,
) -> np.ndarray:
    """Pad history time deltas with zero or truncate from the end."""
    seq_len = min(len(time_deltas_hours), max_history_len)
    padded = np.zeros(max_history_len, dtype=np.float32)
    if seq_len > 0:
        padded[:seq_len] = np.asarray(
            time_deltas_hours[:max_history_len],
            dtype=np.float32,
        )
    return padded


def get_padded_prior_cumulative_likes(
    prior_cumulative_likes: Any,
    max_history_len: int,
) -> np.ndarray:
    """Pad prior-like counts with zero or truncate from the end."""
    seq_len = min(len(prior_cumulative_likes), max_history_len)
    padded = np.zeros(max_history_len, dtype=np.int64)
    if seq_len > 0:
        padded[:seq_len] = np.asarray(
            prior_cumulative_likes[:max_history_len],
            dtype=np.int64,
        )
    return padded


def get_padded_embedding_history_and_mask_batched(
    history_embeddings: list[list[float]] | list[list[list[float]]],
    max_history_len: int,
    embed_dim: int,
    author_indices: list[int] | list[list[int]] | None,
    time_deltas_hours: list[float] | list[list[float]] | None = None,
    prior_cumulative_likes: list[int] | list[list[int]] | None = None,
) -> tuple[
    list[list[list[float]]],
    list[list[bool]],
    list[list[int]],
    list[list[float]],
    list[list[int]],
]:
    """Pad aligned history features for a single user or a request batch."""
    shape = classify_history_embeddings_shape(history_embeddings)
    (
        batch_history_embeddings,
        batch_author_indices,
        batch_time_deltas_hours,
        batch_prior_cumulative_likes,
    ) = _normalize_history_inputs_to_batch(
        history_embeddings,
        shape,
        author_indices,
        time_deltas_hours,
        prior_cumulative_likes,
    )
    batch_padded_history_embeddings = []
    batch_history_mask = []
    batch_padded_author_indices = []
    batch_padded_time_deltas_hours = []
    batch_padded_prior_cumulative_likes = []

    if len(batch_history_embeddings) != len(batch_author_indices):
        raise ValueError(
            "Batch size of history_embeddings and author_indices must match"
        )
    if len(batch_history_embeddings) != len(batch_time_deltas_hours):
        raise ValueError(
            "Batch size of history_embeddings and time_deltas_hours must match"
        )
    if len(batch_history_embeddings) != len(batch_prior_cumulative_likes):
        raise ValueError(
            "Batch size of history_embeddings and prior_cumulative_likes must match"
        )

    for history, authors, time_deltas, prior_likes in zip(
        batch_history_embeddings,
        batch_author_indices,
        batch_time_deltas_hours,
        batch_prior_cumulative_likes,
    ):
        if not isinstance(authors, list):
            raise ValueError("author_indices must be a list for each history")
        if len(history) != len(authors):
            raise ValueError(
                "Length of author_indices must match history length for each user"
            )
        if len(history) != len(time_deltas):
            raise ValueError(
                "Length of time_deltas_hours must match history length for each user"
            )
        if len(history) != len(prior_likes):
            raise ValueError(
                "Length of prior_cumulative_likes must match history length for each user"
            )

        padded_history_embeddings, history_mask = (
            get_padded_embedding_history_and_mask(
                history_embeddings=history,
                max_history_len=max_history_len,
                embed_dim=embed_dim,
            )
        )
        batch_padded_history_embeddings.append(padded_history_embeddings.tolist())
        batch_history_mask.append(history_mask.tolist())
        batch_padded_author_indices.append(
            get_padded_author_indices(
                author_indices=authors,
                max_history_len=max_history_len,
            ).tolist()
        )
        batch_padded_time_deltas_hours.append(
            get_padded_history_time_deltas(
                time_deltas_hours=time_deltas,
                max_history_len=max_history_len,
            ).tolist()
        )
        batch_padded_prior_cumulative_likes.append(
            get_padded_prior_cumulative_likes(
                prior_cumulative_likes=prior_likes,
                max_history_len=max_history_len,
            ).tolist()
        )

    return (
        batch_padded_history_embeddings,
        batch_history_mask,
        batch_padded_author_indices,
        batch_padded_time_deltas_hours,
        batch_padded_prior_cumulative_likes,
    )
