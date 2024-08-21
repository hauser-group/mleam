import tensorflow as tf


class MeanSquaredErrorForces(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        N_atoms = tf.cast(y_true.row_lengths(axis=1), dtype=y_true.dtype)
        return tf.math.reduce_mean(
            tf.math.reduce_sum((y_true - y_pred) ** 2, axis=[1, 2]) / N_atoms**2
        )
