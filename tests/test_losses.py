import tensorflow as tf
from mleam.losses import MeanSquaredErrorForces


def test_mean_squared_error_forces():
    # Batchsize 2, one two atom, one three atom structure
    f_pred = tf.ragged.constant(
        [
            [[1.2, -0.1, 0.6], [-1.2, 0.1, -0.6]],
            [[0.2, -0.1, 0.3], [-1.5, 0.3, 0.2], [0.5, 0.4, 0.1]],
        ]
    )

    f_true = tf.ragged.constant(
        [
            [[1.3, -0.2, 0.8], [-1.3, 0.2, -0.8]],
            [[0.3, -0.2, 0.4], [-1.4, 0.5, 0.3], [0.2, 0.5, 0.2]],
        ]
    )

    N = tf.ragged.constant(
        [
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        ]
    )

    mse_ref = tf.reduce_sum(((f_pred - f_true) / N) ** 2) / 2
    assert MeanSquaredErrorForces()(f_pred, f_true).numpy() == mse_ref
