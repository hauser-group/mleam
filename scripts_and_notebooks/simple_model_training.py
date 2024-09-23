import tensorflow as tf
import numpy as np
from mleam.data_prep import preprocessed_dataset_from_json
from mleam.models import SMATB
from mleam.losses import MeanSquaredErrorForces
import os


def train_model(
    model, run_name, params, hyperparams, random_seed=0, epochs=10, restart=None
):
    tf.keras.utils.set_random_seed(random_seed)
    tf.config.experimental.enable_op_determinism()

    type_dict = {"Ni": 0, "Pt": 1}

    default_params = {
        ("r0", "PtPt"): 2.775,
        ("r0", "NiPt"): 2.633,
        ("r0", "NiNi"): 2.491,
    }

    params.update(default_params)

    max_cutoff = 0.0
    for atom_types in ["PtPt", "NiPt", "NiNi"]:
        a = params[("r0", atom_types)] * np.sqrt(2)
        # Inner cutoff fourth neighbor distance (fcc)
        params[("cut_a", atom_types)] = a * np.sqrt(2)
        # Outer cutoff fifth neighbor distance (fcc)
        params[("cut_b", atom_types)] = a * np.sqrt(5 / 2)
        max_cutoff = max(max_cutoff, params[("cut_b", atom_types)])

    batch_size = 50
    dataset_train = preprocessed_dataset_from_json(
        "../data/train_data.json",
        type_dict,
        batch_size=batch_size,
        cutoff=max_cutoff,
    )

    dataset_val = preprocessed_dataset_from_json(
        "../data/val_data.json",
        type_dict,
        cutoff=max_cutoff,
    )

    model = model(
        ["Ni", "Pt"],
        params=params,
        hyperparams=hyperparams,
        build_forces=True,
        preprocessed_input=True,
    )

    if restart is not None:
        # Build model by calling it once:
        model.predict(dataset_val)
        # Load weights from restart file
        model.load_weights(restart)

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
            log_dir=f"./logs/{run_name}/", profile_batch="10, 15"
        ),
    ]

    model.fit(
        x=dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=my_callbacks,
        initial_epoch=0,
        verbose=1,
    )

    return model


if __name__ == "__main__":
    # From Cheng et al.
    params = {
        ("A", "NiNi"): 0.0845,
        ("A", "NiPt"): 0.1346,
        ("A", "PtPt"): 0.1602,
        ("p", "NiNi"): 11.73,
        ("p", "NiPt"): 14.838,
        ("p", "PtPt"): 13.00,
        ("xi", "NiNi"): 1.405,
        ("xi", "NiPt"): 2.3338,
        ("xi", "PtPt"): 2.1855,
        ("q", "NiNi"): 1.93,
        ("q", "NiPt"): 3.036,
        ("q", "PtPt"): 3.13,
    }

    hypers = {"r0_trainable": False, "offset_trainable": False}
    train_model(SMATB, "SMATB/example/", params, hypers, epochs=10)
