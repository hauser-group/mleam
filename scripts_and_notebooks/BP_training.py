import tensorflow as tf
from mlpot.descriptors import DescriptorSet
from mleam.data_prep import descriptor_dataset_from_json
from mleam.models import BehlerParrinello
from mleam.losses import MeanSquaredErrorForces
from itertools import product
import os


def train_model(model, run_name, descriptor_set, hyperparams, random_seed=0, epochs=10):
    tf.keras.utils.set_random_seed(random_seed)
    tf.config.experimental.enable_op_determinism()

    batch_size = 50
    dataset_train = descriptor_dataset_from_json(
        "../data/train_data.json",
        descriptor_set,
        batch_size=batch_size,
    )

    dataset_val = descriptor_dataset_from_json(
        "../data/val_data.json",
        descriptor_set,
    )

    model = model(
        descriptor_set.atomtypes,
        {
            t: descriptor_set.num_Gs[descriptor_set.type_dict[t]]
            for t in descriptor_set.atomtypes
        },
        build_forces=True,
        layers=hypers["layers"],
        reg=hypers["regularization"],
        offset_trainable=hypers["offset_trainable"],
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
        epochs=epochs,
        callbacks=my_callbacks,
        initial_epoch=0,
        verbose=1,
    )
    return model


if __name__ == "__main__":
    etas = [0.001, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    cutoffs = {
        ("Ni", "Ni"): 5.57,
        ("Ni", "Pt"): 5.887567,
        ("Pt", "Ni"): 5.887567,
        ("Pt", "Pt"): 6.2050886,
    }

    run_name = "BP/example/"
    hypers = {
        "offset_trainable": False,
        "layers": {"Ni": [15, 15], "Pt": [15, 15]},
        "regularization": 1e-6,
    }

    with DescriptorSet(["Ni", "Pt"]) as descriptor_set:
        for eta in etas:
            for ti, tj in product(descriptor_set.atomtypes, repeat=2):
                descriptor_set.add_two_body_descriptor(
                    ti,
                    tj,
                    "BehlerG1",
                    [eta],
                    cuttype="polynomial",
                    cutoff=cutoffs[(ti, tj)],
                )

        train_model(BehlerParrinello, run_name, descriptor_set, hypers, epochs=20)
