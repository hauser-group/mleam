import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from mleam.data_prep import preprocessed_dataset_from_json
from mleam.models import SMATB
from mleam.losses import MeanSquaredErrorForces
import os


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
            "forces": MeanSquaredErrorForces(),
        },
        metrics={
            "energy": [tf.keras.metrics.RootMeanSquaredError()],
            "energy_per_atom": [
                tf.keras.metrics.RootMeanSquaredError(),
            ],
            "forces": [
                tf.keras.metrics.RootMeanSquaredError(),
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
