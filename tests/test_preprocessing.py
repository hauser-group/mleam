import pytest
import numpy as np
import tensorflow as tf
import numdifftools as nd
from mleam.preprocessing import (
    get_pair_types,
    get_distance_matrix,
    get_distance_matrix_and_derivative,
    get_distance_matrix_and_full_derivative,
    preprocess_inputs,
    preprocess_inputs_ragged,
)
from scipy.spatial.transform import Rotation


@pytest.fixture
def xyzs():
    # 3.0 and 4.0 to result in Pythagorean triples
    return tf.constant([[0, 0, 0], [3.0, 0, 0], [0, 4.0, 0], [0, 3.0, 4.0]])


@pytest.fixture
def types():
    return tf.expand_dims(tf.constant([0, 0, 1, 1]), axis=-1)


@pytest.fixture
def padded_xyzs():
    return tf.constant(
        [
            [[0, 0, 0], [3.0, 0, 0], [0, 4.0, 0], [0, 3.0, 4.0]],
            [[0, 0, 0], [3.0, 0, 0], [0, 4.0, 0], [0, 0, 0]],
        ],
    )


@pytest.fixture
def padded_types():
    return tf.expand_dims(tf.constant([[0, 0, 1, 1], [0, 0, 1, -1]]), axis=-1)


@pytest.fixture
def ragged_xyzs():
    return tf.ragged.constant(
        [
            [[0, 0, 0], [3.0, 0, 0], [0, 4.0, 0], [0, 3.0, 4.0]],
            [[0, 0, 0], [3.0, 0, 0], [0, 4.0, 0]],
        ],
        ragged_rank=1,
    )


@pytest.fixture
def ragged_types():
    return tf.expand_dims(
        tf.ragged.constant([[0, 0, 1, 1], [0, 0, 1]], ragged_rank=1), axis=-1
    )


def test_distance_matrix(xyzs, atol=1e-6):
    r = get_distance_matrix(xyzs, diagonal=8.1)

    np.testing.assert_allclose(
        r,
        np.expand_dims(
            np.array(
                [
                    [8.1, 3.0, 4.0, 5.0],
                    [3.0, 8.1, 5.0, np.sqrt(34)],
                    [4.0, 5.0, 8.1, np.sqrt(17)],
                    [5.0, np.sqrt(34), np.sqrt(17), 8.1],
                ]
            ),
            axis=-1,
        ),
    )


def test_padded_distance_matrix(padded_xyzs, atol=1e-6):
    # 3 char variable names to aid readibility
    cut = 8.1
    r13 = np.sqrt(34)
    r23 = np.sqrt(17)

    r = get_distance_matrix(padded_xyzs, cut)

    assert r.shape == (2, 4, 4, 1)

    np.testing.assert_allclose(
        r,
        np.expand_dims(
            np.array(
                [
                    [
                        [cut, 3.0, 4.0, 5.0],
                        [3.0, cut, 5.0, r13],
                        [4.0, 5.0, cut, r23],
                        [5.0, r13, r23, cut],
                    ],
                    [
                        [cut, 3.0, 4.0, cut],
                        [3.0, cut, 5.0, 3.0],
                        [4.0, 5.0, cut, 4.0],
                        [cut, 3.0, 4.0, cut],
                    ],
                ]
            ),
            axis=-1,
        ),
    )


def test_ragged_batch_distance_matrix(ragged_xyzs, atol=1e-6):
    # 3 char variable names to aid readibility
    cut = 8.1
    r13 = np.sqrt(34)
    r23 = np.sqrt(17)

    r = get_distance_matrix(ragged_xyzs, cut)

    assert r.shape == (2, None, None, 1)
    np.testing.assert_array_equal(r.nested_row_splits[0], [0, 4, 7])
    np.testing.assert_array_equal(r.nested_row_splits[1], [0, 4, 8, 12, 16, 19, 22, 25])

    np.testing.assert_allclose(
        r[0, :, :, 0].numpy(),
        np.array(
            [
                [cut, 3.0, 4.0, 5.0],
                [3.0, cut, 5.0, r13],
                [4.0, 5.0, cut, r23],
                [5.0, r13, r23, cut],
            ]
        ),
    )

    np.testing.assert_allclose(
        r[1, :, :, 0].numpy(),
        np.array(
            [
                [cut, 3.0, 4.0],
                [3.0, cut, 5.0],
                [4.0, 5.0, cut],
            ]
        ),
    )


def test_pair_types(types):
    pair_types = get_pair_types(types, 2)

    np.testing.assert_array_equal(
        pair_types,
        np.expand_dims(
            np.array(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                ]
            ),
            axis=-1,
        ),
    )


def test_negative_pair_type():
    # Negative types are used for masking/padding
    types = tf.expand_dims(tf.constant([0, 0, 1, 1, -1, -2]), axis=-1)
    pair_types = get_pair_types(types, 2)

    np.testing.assert_array_equal(
        tf.squeeze(pair_types),
        np.array(
            [
                [0, 0, 1, 1, -1, -1],
                [0, 0, 1, 1, -1, -1],
                [1, 1, 2, 2, -1, -1],
                [1, 1, 2, 2, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ]
        ),
    )


def test_padded_pair_types(padded_types):
    pair_types = get_pair_types(padded_types, 2)

    assert pair_types.shape == (2, 4, 4, 1)

    np.testing.assert_array_equal(
        pair_types,
        np.expand_dims(
            np.array(
                [
                    [
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [1, 1, 2, 2],
                        [1, 1, 2, 2],
                    ],
                    [
                        [0, 0, 1, -1],
                        [0, 0, 1, -1],
                        [1, 1, 2, -1],
                        [-1, -1, -1, -1],
                    ],
                ]
            ),
            axis=-1,
        ),
    )


def test_ragged_pair_types(ragged_types):
    pair_types = get_pair_types(ragged_types, 2)

    assert pair_types.shape == (2, None, None, 1)
    np.testing.assert_array_equal(pair_types.nested_row_splits[0], [0, 4, 7])
    np.testing.assert_array_equal(
        pair_types.nested_row_splits[1], [0, 4, 8, 12, 16, 19, 22, 25]
    )

    np.testing.assert_array_equal(
        pair_types[0, :, :, 0].numpy(),
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ]
        ),
    )
    np.testing.assert_array_equal(
        pair_types[1, :, :, 0].numpy(),
        np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [1, 1, 2],
            ]
        ),
    )


def test_invariance(xyzs, types, atol=1e-6, seed=0):
    # Test invariance with respect to an arbitrary rotation and translation
    # Reference values
    r = get_distance_matrix(xyzs)
    # Construct random rotation matrix
    rng = np.random.default_rng(seed)
    rotvec = rng.normal(size=(3))
    R = tf.constant(Rotation.from_rotvec(rotvec).as_matrix(), dtype=tf.float32)
    # Apply random translation before rotation
    xyzs2 = tf.matmul(xyzs + rng.normal(size=(3)), R)

    r2 = get_distance_matrix(xyzs2)

    np.testing.assert_allclose(r, r2, atol=atol)


def test_distance_matrix_and_derivative(xyzs, atol=1e-6):
    _, dr_dx = get_distance_matrix_and_derivative(xyzs)
    n = len(xyzs)
    dr_dx = dr_dx[:, :, tf.newaxis, :] * (
        tf.eye(n)[tf.newaxis, :, :, tf.newaxis]
        - tf.eye(n)[:, tf.newaxis, :, tf.newaxis]
    )

    def fun(x):
        # numdifftools only supports vector to vector functions
        # so first reshape the xyzs array to 2d
        x = x.reshape(xyzs.shape)
        # and then flatten the 3d output back to 1d
        return get_distance_matrix(x).numpy().flatten()

    num_dr_dx = nd.Jacobian(fun)(xyzs.numpy().flatten()).reshape(dr_dx.shape)
    np.testing.assert_allclose(dr_dx, num_dr_dx, atol=atol)


def test_get_distance_matrix_and_full_derivative(xyzs, atol=1e-6):
    _, dr_dx = get_distance_matrix_and_full_derivative(xyzs)

    def fun(x):
        # numdifftools only supports vector to vector functions
        # so first reshape the xyzs array to 2d
        x = x.reshape(xyzs.shape)
        # and then flatten the 3d output back to 1d
        return get_distance_matrix(x).numpy().flatten()

    num_dr_dx = nd.Jacobian(fun)(xyzs.numpy().flatten()).reshape(dr_dx.shape)
    np.testing.assert_allclose(dr_dx, num_dr_dx, atol=atol)


def test_preprocess_inputs(ragged_xyzs, ragged_types):
    types, pair_types, distances, derivative = preprocess_inputs(
        ragged_xyzs, ragged_types, 2, 5.1
    )

    assert distances.shape == (2, None, None, 1)
    assert pair_types.shape == (2, None, None, 1)
    assert derivative.shape == (2, None, None, 3)


def test_preprocess_inputs_ragged_ragged_input(ragged_xyzs, ragged_types):
    cutoff = 5.1
    types, pair_types, distances, derivative, j_indices = preprocess_inputs_ragged(
        ragged_xyzs, ragged_types, 2, cutoff=cutoff
    )

    assert distances.dtype == ragged_xyzs.dtype
    assert pair_types.dtype == ragged_types.dtype
    assert derivative.dtype == ragged_xyzs.dtype
    assert j_indices.dtype == tf.int64

    assert distances.shape == (2, None, None, 1)
    assert pair_types.shape == (2, None, None, 1)
    assert derivative.shape == (2, None, None, 3)
    assert j_indices.shape == (2, None, None)

    assert j_indices.to_list() == [
        [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]],
        [[1, 2], [0, 2], [0, 1]],
    ]


def test_preprocess_inputs_ragged_padded_input(padded_xyzs, padded_types):
    cutoff = 5.1
    types, pair_types, distances, derivative, j_indices = preprocess_inputs_ragged(
        padded_xyzs, padded_types, 2, cutoff=cutoff
    )

    assert types.dtype == padded_types.dtype
    assert distances.dtype == padded_xyzs.dtype
    assert pair_types.dtype == padded_types.dtype
    assert derivative.dtype == padded_xyzs.dtype
    assert j_indices.dtype == tf.int64

    assert types.shape == (2, None, 1)
    assert distances.shape == (2, None, None, 1)
    assert pair_types.shape == (2, None, None, 1)
    assert derivative.shape == (2, None, None, 3)
    assert j_indices.shape == (2, None, None)

    assert j_indices.to_list() == [
        [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]],
        [[1, 2], [0, 2], [0, 1]],
    ]


def test_ragged_distance_and_pair_types_single_input(xyzs, types):
    cutoff = 5.1
    types, pair_types, distances, derivative, j_indices = preprocess_inputs_ragged(
        xyzs, types, 2, cutoff=cutoff
    )

    assert distances.dtype == xyzs.dtype
    assert pair_types.dtype == types.dtype
    assert derivative.dtype == xyzs.dtype
    assert j_indices.dtype == tf.int64

    assert distances.shape == (4, None, 1)
    assert pair_types.shape == (4, None, 1)
    assert derivative.shape == (4, None, 3)
    assert j_indices.shape == (4, None)

    assert j_indices.to_list() == [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
