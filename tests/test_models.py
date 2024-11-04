import pytest
import numpy as np
import tensorflow as tf
from mleam.models import (
    SMATB,
    SuttonChen,
    FinnisSinclair,
    Johnson,
    DoubleExp,
)
import numdifftools as nd

models_test_decorator = pytest.mark.parametrize(
    "model_class, initial_params, hyperparams",
    [
        (SMATB, {}, {}),
        # TODO: figure out why this assigns different "ids" to the weights in the .h5 file
        # (SMATB, {}, {"r0_trainable": True}),
        (SuttonChen, {}, {}),
        (FinnisSinclair, {}, {}),
        (Johnson, {}, {}),
        (DoubleExp, {}, {}),
    ],
)


@models_test_decorator
def test_model_save_and_load(model_class, initial_params, hyperparams, tmpdir):
    """Only testing save_weights as standard save does not seem to
    work with ragged output tensors."""
    tf.keras.backend.clear_session()
    N = 4
    xyzs = tf.RaggedTensor.from_tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [-2.17, 0.91, -0.75],
                [-0.62, -0.99, -0.94],
                [1.22, -0.04, 0.49],
            ]
        ],
        lengths=[N],
    )
    types = tf.ragged.constant([[[0], [1], [0], [1]]], ragged_rank=1, dtype=tf.int32)

    atom_types = ["Ni", "Pt"]
    model = model_class(atom_types, initial_params, hyperparams, build_forces=True)
    ref_prediction = model({"positions": xyzs, "types": types})
    ref_e, ref_forces = (
        ref_prediction["energy_per_atom"],
        ref_prediction["forces"],
    )

    model.save_weights(tmpdir / "weights.h5")
    model.save(tmpdir / "model.keras")

    tf.keras.backend.clear_session()

    # Load weights:
    # Build fresh model
    model = model_class(atom_types, initial_params, hyperparams, build_forces=True)
    # Model needs to be called once before loading in order to
    # determine all weight sizes...
    model({"positions": xyzs, "types": types})
    model.load_weights(tmpdir / "weights.h5")
    new_prediction = model({"positions": xyzs, "types": types})
    new_e, new_forces = (
        new_prediction["energy_per_atom"],
        new_prediction["forces"],
    )

    np.testing.assert_allclose(new_e.numpy(), ref_e.numpy(), atol=1e-6)
    np.testing.assert_allclose(
        new_forces.to_tensor().numpy(),
        ref_forces.to_tensor().numpy(),
        atol=1e-6,
    )

    tf.keras.backend.clear_session()

    # Load full model:
    model = tf.keras.models.load_model(str(tmpdir / "model.keras"))

    # TODO: enable this
    # assert model.initial_params == initial_params
    # assert model.hyperparams == hyperparams

    new_prediction = model({"positions": xyzs, "types": types})
    new_e, new_forces = (
        new_prediction["energy_per_atom"],
        new_prediction["forces"],
    )

    np.testing.assert_allclose(new_e.numpy(), ref_e.numpy(), atol=1e-6)
    np.testing.assert_allclose(
        new_forces.to_tensor().numpy(),
        ref_forces.to_tensor().numpy(),
        atol=1e-6,
    )

    tf.keras.backend.clear_session()


@models_test_decorator
def test_model_forces_numerically(model_class, initial_params, hyperparams):
    tf.keras.backend.clear_session()
    # Number of atoms
    N = 4
    xyzs = tf.RaggedTensor.from_tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [-2.17, 0.91, -0.75],
                [-0.62, -0.99, -0.94],
                [1.22, -0.04, 0.49],
            ]
        ],
        lengths=[N],
    )
    types = tf.ragged.constant([[[0], [1], [0], [1]]], ragged_rank=1)

    model = model_class(atom_types=["Ni", "Pt"], build_forces=True)
    forces = model({"positions": xyzs, "types": types})["forces"]

    @tf.function(input_signature=(tf.TensorSpec(shape=[3 * N], dtype=tf.float32),))
    def fun(x):
        print("tracing with", x)
        # numdifftools flattens the vector x so it needs to be reshaped here
        xyzs = tf.RaggedTensor.from_tensor(tf.reshape(x, (1, N, 3)), lengths=[N])
        # numdifftools expects a scalar function:
        return model({"positions": xyzs, "types": types})["energy"][0, 0]

    # Force is the negative gradient
    x0 = xyzs.to_tensor().numpy()
    num_forces = -nd.Gradient(fun)(x0).reshape(1, N, 3)

    np.testing.assert_allclose(forces.to_tensor().numpy(), num_forces, atol=1e-2)

    tf.keras.backend.clear_session()
