import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax

def _tfds_load(name, split, data_dir=None):
    if data_dir is None:
        return tfds.load(name, split=split)
    return tfds.load(name, split=split, data_dir=data_dir)


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


def get_dataset(dataset_name, batch_size, is_train, debug_overfit=False, data_dir=None):
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

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else 'validation', drop_remainder=True)
        dataset = _tfds_load('imagenet2012', split=split, data_dir=data_dir)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if debug_overfit:
            dataset = dataset.take(8)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
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

        # split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        split='train'
        dataset = _tfds_load('celebahq256', split=split, data_dir=data_dir)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(20000, seed=42+jax.process_index(), reshuffle_each_iteration=True)
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
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
