import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from mleam.data_prep import preprocessed_dataset_from_json
from mleam.models import SMATB
import os


class RootMeanSquaredMetricForces(tf.keras.metrics.Mean):
    def __init__(self, name="root_mean_squared_error_forces", dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
        y_true: The ground truth values.
        y_pred: The predicted values.
        sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
        Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        error_sq = tf.reduce_mean(
            tf.math.squared_difference(y_pred, y_true), axis=[1, 2]
        )
        return super().update_state(error_sq, sample_weight=sample_weight)

    def result(self):
        return tf.math.sqrt(tf.math.divide_no_nan(self.total, self.count))


class MeanSquaredErrorForces1(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        N_atoms = tf.cast(y_true.row_lengths(axis=1), dtype=y_true.dtype)
        return tf.math.reduce_mean(
            tf.math.reduce_sum((y_true - y_pred) ** 2, axis=[1, 2]) / N_atoms**2
        )


class MeanSquaredErrorForces2(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        N_atoms = tf.cast(y_true.row_lengths(axis=1), dtype=y_true.dtype)
        return tf.math.reduce_mean(
            (
                tf.math.reduce_sum(
                    tf.math.sqrt(tf.math.reduce_sum((y_true - y_pred) ** 2, axis=2)),
                    axis=1,
                )
                / N_atoms
            )
            ** 2
        )


def main(random_seed=0):
    tf.keras.utils.set_random_seed(random_seed)
    tf.config.experimental.enable_op_determinism()

    type_dict = {"Ni": 0, "Pt": 1}

    batch_size = 50
    # Use smaller cutoff for preprocessed dataset!
    dataset_train = preprocessed_dataset_from_json(
        "../data/train_data.json",
        type_dict,
        batch_size=batch_size,
        cutoff=5.006,
    )

    dataset_val = preprocessed_dataset_from_json(
        "../data/val_data.json", type_dict, cutoff=5.006
    )

    params = {
        ("A", "PtPt"): 0.1602,
        ("A", "NiPt"): 0.1346,
        ("A", "NiNi"): 0.0845,
        ("xi", "PtPt"): 2.1855,
        ("xi", "NiPt"): 2.3338,
        ("xi", "NiNi"): 1.405,
        ("p", "PtPt"): 13.00,
        ("p", "NiPt"): 14.838,
        ("p", "NiNi"): 11.73,
        ("q", "PtPt"): 3.13,
        ("q", "NiPt"): 3.036,
        ("q", "NiNi"): 1.93,
        ("r0", "PtPt"): 2.77,
        ("r0", "NiPt"): 2.63,
        ("r0", "NiNi"): 2.49,
        ("cut_a", "PtPt"): 4.08707719,
        ("cut_b", "PtPt"): 5.0056268338740553,
        ("cut_a", "NiPt"): 4.08707719,
        ("cut_b", "NiPt"): 4.4340500673763259,
        ("cut_a", "NiNi"): 3.62038672,
        ("cut_b", "NiNi"): 4.4340500673763259,
    }
    model = SMATB(
        ["Ni", "Pt"], params=params, build_forces=True, preprocessed_input=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            "energy_per_atom": tf.keras.losses.MeanSquaredError(),
            "forces": MeanSquaredErrorForces1(),
        },
        metrics={
            "energy": [tf.keras.metrics.RootMeanSquaredError()],
            "energy_per_atom": [
                tf.keras.metrics.RootMeanSquaredError(),
            ],
            "forces": [
                tf.keras.metrics.RootMeanSquaredError(),
                RootMeanSquaredMetricForces(),
            ],
        },
    )

    run_name = f"SMATB/force_error1/batch_size_{batch_size}"
    if not os.path.exists(f"./saved_models/{run_name}"):
        os.makedirs(f"./saved_models/{run_name}")
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./saved_models/{run_name}/model.h5",
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"./logs/{run_name}/",
        ),
    ]

    model.fit(
        x=dataset_train,
        validation_data=dataset_val,
        epochs=100,
        callbacks=my_callbacks,
        initial_epoch=0,
        verbose=1,
    )


if __name__ == "__main__":
    main()
