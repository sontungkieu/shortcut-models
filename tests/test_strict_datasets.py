from __future__ import annotations

import numpy as np
import tensorflow as tf

from utils import datasets


def _fake_celebahq_dataset() -> tf.data.Dataset:
    images = np.zeros((12, 2, 3, 1), dtype=np.uint8)
    for index in range(len(images)):
        images[index, :, :, 0] = np.asarray(
            [[index, index + 1, index + 2], [index + 3, index + 4, index + 5]],
            dtype=np.uint8,
        )
    labels = np.arange(len(images), dtype=np.int64)
    return tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})


def _take_batches(seed: int, monkeypatch) -> list[tuple[np.ndarray, np.ndarray]]:
    monkeypatch.setattr(datasets, "_tfds_load", lambda *args, **kwargs: _fake_celebahq_dataset())
    monkeypatch.setattr(datasets.jax, "process_index", lambda: 0)
    stream = datasets.get_dataset(
        "celebahq256",
        batch_size=3,
        is_train=True,
        seed=seed,
        strict_deterministic=True,
    )
    return [(images.copy(), labels.copy()) for images, labels in (next(stream) for _ in range(6))]


def test_strict_dataset_stream_replays_exactly_and_seed_changes_it(monkeypatch):
    first = _take_batches(77, monkeypatch)
    replay = _take_batches(77, monkeypatch)
    different_seed = _take_batches(78, monkeypatch)

    for (first_images, first_labels), (replay_images, replay_labels) in zip(first, replay):
        np.testing.assert_array_equal(first_images, replay_images)
        np.testing.assert_array_equal(first_labels, replay_labels)
    assert any(
        not np.array_equal(first_labels, different_labels)
        or not np.array_equal(first_images, different_images)
        for (first_images, first_labels), (different_images, different_labels) in zip(
            first,
            different_seed,
        )
    )
