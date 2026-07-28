import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax


def _tfds_load(name, split, data_dir=None):
    if data_dir is None:
        return tfds.load(name, split=split)
    return tfds.load(name, split=split, data_dir=data_dir)


def _with_deterministic_options(dataset):
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)


def _stateless_flip_left_right(image, base_seed, occurrence_index):
    seed = tf.stack(
        [
            tf.cast(base_seed, tf.int64),
            tf.cast(occurrence_index, tf.int64),
        ]
    )
    return tf.image.stateless_random_flip_left_right(image, seed=seed)


def get_dataset_for_statistics(
    dataset_name,
    batch_size,
    data_dir=None,
    split="train",
    max_samples=0,
):
    """Build a finite, deterministic dataset for population-level statistics."""
    if dataset_name != "celebahq256":
        raise ValueError(
            "Population statistics currently support dataset_name=celebahq256 only"
        )

    def deserialization_fn(data):
        image = tf.cast(data["image"], tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image, data["label"]

    dataset = _tfds_load(dataset_name, split=split, data_dir=data_dir)
    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    if cardinality < 0:
        builder = tfds.builder(dataset_name, data_dir=data_dir)
        cardinality = int(builder.info.splits[split].num_examples)
    if max_samples and max_samples > 0:
        cardinality = min(cardinality, int(max_samples))
        dataset = dataset.take(cardinality)
    dataset = dataset.map(
        deserialization_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return iter(tfds.as_numpy(dataset)), cardinality


def get_dataset(
    dataset_name,
    batch_size,
    is_train,
    debug_overfit=False,
    data_dir=None,
    seed=42,
    strict_deterministic=False,
):
    """Build the repeated training/evaluation stream.

    ``strict_deterministic`` keeps the same augmentation distribution while
    making each draw reproducible. Horizontal flips are keyed by the draw
    index, so repeated occurrences of one image can still receive different
    flips exactly as in the legacy stateful pipeline.
    """
    print("Loading dataset")
    if 'imagenet256' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            if 'imagenet256' in dataset_name:
                image = tf.image.resize(image, (256, 256), antialias=True)
            elif 'imagenet128' in dataset_name:
                image = tf.image.resize(image, (256, 256), antialias=True)
            else:
                raise ValueError(f"Unknown dataset {dataset_name}")
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image, data['label']

        def strict_deserialization_fn(occurrence_index, data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            if is_train:
                image = _stateless_flip_left_right(image, seed, occurrence_index)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else 'validation', drop_remainder=True)
        dataset = _tfds_load('imagenet2012', split=split, data_dir=data_dir)
        if debug_overfit:
            dataset = dataset.map(
                deserialization_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=bool(strict_deterministic),
            )
            dataset = dataset.take(8)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
        elif strict_deterministic:
            dataset = dataset.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.enumerate()
            dataset = dataset.map(
                strict_deserialization_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = _with_deterministic_options(dataset)
        else:
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'celebahq256':
        def deserialization_fn(data):
            image = data['image']
            image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image,  data['label']

        def strict_deserialization_fn(occurrence_index, data):
            image = _stateless_flip_left_right(
                data['image'],
                seed + jax.process_index(),
                occurrence_index,
            )
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        # split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        split='train'
        dataset = _tfds_load('celebahq256', split=split, data_dir=data_dir)
        shuffle_seed = seed + jax.process_index()
        if strict_deterministic:
            dataset = dataset.shuffle(20000, seed=shuffle_seed, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.enumerate()
            dataset = dataset.map(
                strict_deserialization_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = _with_deterministic_options(dataset)
        else:
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(20000, seed=shuffle_seed, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'lsunchurch':
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image, 0 # No label

        split = tfds.split_for_jax_process('church-train' if is_train else 'church-test', drop_remainder=True)
        dataset = _tfds_load('lsunc', split=split, data_dir=data_dir)
        dataset = dataset.map(
            deserialization_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=bool(strict_deterministic),
        )
        dataset = dataset.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        if strict_deterministic:
            dataset = _with_deterministic_options(dataset)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
