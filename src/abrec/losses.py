from __future__ import annotations

import tensorflow as tf


def bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))


def sliced_wasserstein_alignment(x: tf.Tensor, y: tf.Tensor, num_projections: int = 32) -> tf.Tensor:
    """Approximate Wasserstein distance via random 1D projections."""
    d = tf.shape(x)[-1]
    projections = tf.random.normal([num_projections, d])
    projections = tf.math.l2_normalize(projections, axis=-1)

    x_proj = tf.linalg.matmul(x, projections, transpose_b=True)
    y_proj = tf.linalg.matmul(y, projections, transpose_b=True)

    x_sorted = tf.sort(x_proj, axis=0)
    y_sorted = tf.sort(y_proj, axis=0)
    return tf.reduce_mean(tf.abs(x_sorted - y_sorted))


def cosine_regularization(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x_n = tf.math.l2_normalize(x, axis=-1)
    y_n = tf.math.l2_normalize(y, axis=-1)
    cosine = tf.reduce_sum(x_n * y_n, axis=-1)
    return tf.reduce_mean(1.0 - cosine)
