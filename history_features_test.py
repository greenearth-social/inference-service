"""Tests for inference-owned user-history feature construction."""

import numpy as np
import pytest

from history_features import (
    AUTHOR_PAD_IDX,
    AUTHOR_UNK_IDX,
    classify_history_embeddings_shape,
    get_padded_author_indices,
    get_padded_embedding_history_and_mask,
    get_padded_embedding_history_and_mask_batched,
    get_padded_history_time_deltas,
    get_padded_prior_cumulative_likes,
)


def test_get_padded_embedding_history_and_mask_pads_and_truncates_prefix():
    history = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    padded, mask = get_padded_embedding_history_and_mask(
        history,
        max_history_len=5,
        embed_dim=2,
    )
    assert padded.dtype == np.float32
    assert padded.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, True, False, False]

    truncated, truncated_mask = get_padded_embedding_history_and_mask(
        history,
        max_history_len=2,
        embed_dim=2,
    )
    assert truncated.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert truncated_mask.tolist() == [True, True]


def test_get_padded_embedding_history_and_mask_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="embed_dim"):
        get_padded_embedding_history_and_mask(
            [[1.0, 2.0, 3.0]],
            max_history_len=2,
            embed_dim=2,
        )


def test_aligned_side_feature_padding_and_truncation():
    assert get_padded_author_indices([2, 3], 4).tolist() == [
        2,
        3,
        AUTHOR_PAD_IDX,
        AUTHOR_PAD_IDX,
    ]
    assert get_padded_author_indices([2, 3, 4], 2).tolist() == [2, 3]
    assert get_padded_history_time_deltas([1.5, 2.25], 4).tolist() == pytest.approx(
        [1.5, 2.25, 0.0, 0.0]
    )
    assert get_padded_prior_cumulative_likes([10, 20], 4).tolist() == [
        10,
        20,
        0,
        0,
    ]


@pytest.mark.parametrize(
    ("history_embeddings", "expected"),
    [
        ([], "single_empty"),
        ([[]], "single_empty"),
        ([[1.0, 2.0], [3.0, 4.0]], "single_history"),
        ([[], [[1.0, 2.0]]], "batched_history"),
        ([[[1.0, 2.0]], [[3.0, 4.0]]], "batched_history"),
    ],
)
def test_classify_history_embeddings_shape(history_embeddings, expected):
    assert classify_history_embeddings_shape(history_embeddings) == expected


@pytest.mark.parametrize(
    ("history_embeddings", "message"),
    [
        ("not-a-list", "history_embeddings must be a list"),
        ([1.0, 2.0], "history_embeddings must be a list of lists"),
        (
            [[], [1.0, 2.0]],
            "batched history_embeddings must be a list of user histories",
        ),
    ],
)
def test_classify_history_embeddings_shape_rejects_invalid_inputs(
    history_embeddings,
    message,
):
    with pytest.raises(ValueError, match=message):
        classify_history_embeddings_shape(history_embeddings)


def test_batched_padding_handles_empty_history():
    padded, mask, author_indices, time_deltas, prior_likes = (
        get_padded_embedding_history_and_mask_batched(
            [],
            max_history_len=3,
            embed_dim=2,
            author_indices=[],
        )
    )

    assert padded == [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    assert mask == [[False, False, False]]
    assert author_indices == [[AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX]]
    assert time_deltas == [[0.0, 0.0, 0.0]]
    assert prior_likes == [[0, 0, 0]]


def test_batched_padding_aligns_single_history_features():
    padded, mask, author_indices, time_deltas, prior_likes = (
        get_padded_embedding_history_and_mask_batched(
            [[1.0, 2.0], [3.0, 4.0]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[2, 3],
            time_deltas_hours=[1.5, 2.25],
            prior_cumulative_likes=[10, 20],
        )
    )

    assert padded == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]
    assert mask == [[True, True, False]]
    assert author_indices == [[2, 3, AUTHOR_PAD_IDX]]
    assert time_deltas == [[1.5, 2.25, 0.0]]
    assert prior_likes == [[10, 20, 0]]


def test_batched_padding_defaults_real_missing_authors_to_unknown():
    _, _, author_indices, _, _ = get_padded_embedding_history_and_mask_batched(
        [[1.0, 2.0], [3.0, 4.0]],
        max_history_len=3,
        embed_dim=2,
        author_indices=None,
    )

    assert author_indices == [[AUTHOR_UNK_IDX, AUTHOR_UNK_IDX, AUTHOR_PAD_IDX]]


def test_batched_padding_normalizes_empty_entries_in_a_batch():
    padded, mask, author_indices, time_deltas, prior_likes = (
        get_padded_embedding_history_and_mask_batched(
            [[], [[1.0, 2.0], [3.0, 4.0]], [[]]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[[], [2, 3], []],
            time_deltas_hours=[[], [1.5, 2.25], []],
            prior_cumulative_likes=[[], [10, 20], []],
        )
    )

    assert padded == [
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]
    assert mask == [
        [False, False, False],
        [True, True, False],
        [False, False, False],
    ]
    assert author_indices == [
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
        [2, 3, AUTHOR_PAD_IDX],
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
    ]
    assert time_deltas == [
        [0.0, 0.0, 0.0],
        [1.5, 2.25, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert prior_likes == [
        [0, 0, 0],
        [10, 20, 0],
        [0, 0, 0],
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "history_embeddings": [[[1.0, 2.0]], [[3.0, 4.0]]],
                "author_indices": [[2]],
            },
            "Batch size of history_embeddings and author_indices must match",
        ),
        (
            {
                "history_embeddings": [[1.0, 2.0], [3.0, 4.0]],
                "author_indices": [2],
            },
            "Length of author_indices must match history length",
        ),
        (
            {
                "history_embeddings": [[[1.0, 2.0]], [[3.0, 4.0]]],
                "author_indices": [[2], [3]],
                "time_deltas_hours": [[1.5]],
            },
            "Batch size of history_embeddings and time_deltas_hours must match",
        ),
        (
            {
                "history_embeddings": [[1.0, 2.0], [3.0, 4.0]],
                "author_indices": [2, 3],
                "prior_cumulative_likes": [10],
            },
            "Length of prior_cumulative_likes must match history length",
        ),
    ],
)
def test_batched_padding_rejects_misaligned_features(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_padded_embedding_history_and_mask_batched(
            max_history_len=3,
            embed_dim=2,
            **kwargs,
        )
