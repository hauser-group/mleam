import pytest
import unittest
import numpy as np
import tensorflow as tf
from mleam.models import (
    SMATB,
    SuttonChen,
    FinnisSinclair,
    Johnson,
    DoubleExp,
    ExtendedEmbeddingModel,
    ExtendedEmbeddingV2Model,
    ExtendedEmbeddingV3Model,
    ExtendedEmbeddingV4Model,
    NNEmbeddingModel,
    NNRhoModel,
    RhoTwoExpModel,
    NNRhoExpModel,
    ExtendedEmbeddingRhoTwoExpModel,
    ExtendedEmbeddingV3RhoTwoExpModel,
    ExtendedEmbeddingV4RhoTwoExpModel,
    NNEmbeddingNNRhoModel,
    NNEmbeddingNNRhoExpModel,
    CommonNNEmbeddingModel,
    CommonNNEmbeddingNNRhoModel,
    CommonExtendedEmbeddingV4Model,
    CommonExtendedEmbeddingV4RhoTwoExpModel,
)
import numdifftools as nd

models_test_decorator = pytest.mark.parametrize(
    "model_class, params, hyperparams",
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
def test_model_save_and_load(model_class, params, hyperparams, tmpdir):
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
    types = tf.ragged.constant([[[0], [1], [0], [1]]], ragged_rank=1)

    atom_types = ["Ni", "Pt"]
    model = model_class(atom_types, params, hyperparams, build_forces=True)
    ref_prediction = model({"positions": xyzs, "types": types})
    ref_e, ref_forces = (
        ref_prediction["energy_per_atom"],
        ref_prediction["forces"],
    )

    model.save_weights(tmpdir / "tmp_model.h5")

    # Build fresh model
    model = model_class(atom_types, params, hyperparams, build_forces=True)
    # Model needs to be called once before loading in order to
    # determine all weight sizes...
    model({"positions": xyzs, "types": types})
    model.load_weights(tmpdir / "tmp_model.h5")
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
def test_model_forces_numerically(model_class, params, hyperparams):
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

    def fun(x):
        # numdifftools flattens the vector x so it needs to be reshaped here
        xyzs = tf.RaggedTensor.from_tensor(x.reshape(1, N, 3), lengths=[N])
        # numdifftools expects a scalar function:
        return model({"positions": xyzs, "types": types})["energy"][0, 0]

    # Force is the negative gradient
    num_forces = -nd.Gradient(fun)(xyzs.to_tensor().numpy()).reshape(1, N, 3)

    np.testing.assert_allclose(forces.to_tensor().numpy(), num_forces, atol=1e-2)

    tf.keras.backend.clear_session()
