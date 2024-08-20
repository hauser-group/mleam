import pytest
import numpy as np
import tensorflow as tf
from mleam.utils import distances_and_pair_types
from mleam.data_prep import dataset_from_json, preprocessed_dataset_from_json
from utils import rotation_matrix, derive_array_wrt_array


@pytest.fixture
def xyzs():
    return tf.constant([[0, 0, 0], [1.5, 0, 0], [0, 3.0, 0], [0, 1.5, 2.0]])


@pytest.fixture
def types():
    return tf.expand_dims(tf.constant([0, 0, 1, 1]), axis=-1)


def test_shapes(xyzs, types):
    r, pair_types, dr_dx = distances_and_pair_types(xyzs, types, 2, cutoff=2.6)

    # This is awful, however with the current version of tensorflow
    # assertEqual(r.shape, tf.TensorShape([3, None, 1])) because the
    # __eq__ of None Dimensions returns None
    assert r.shape.as_list() == [4, None, 1]
    assert pair_types.shape.as_list() == [4, None, 1]
    assert dr_dx.shape.as_list() == [4, None, None, 3]


def test_distances(xyzs, types, atol=1e-6):
    # Distance matrix:
    # 0.0 1.5 3.0 2.5
    # 1.5 0.0 3.4 3.7
    # 3.0 3.4 0.0 2.5
    # 2.5 3.7 2.5 0.0

    r, _, _ = distances_and_pair_types(xyzs, types, 2, cutoff=2.6)

    np.testing.assert_allclose(r[0].numpy(), np.array([[1.5], [2.5]]), atol=atol)
    np.testing.assert_allclose(r[1].numpy(), np.array([[1.5]]), atol=atol)
    np.testing.assert_allclose(r[2].numpy(), np.array([[2.5]]), atol=atol)
    np.testing.assert_allclose(r[3].numpy(), np.array([[2.5], [2.5]]), atol=atol)


def test_pair_types(xyzs, types):
    # Test correct pair types
    # (0, 0) -> 0
    # (0, 1), (1, 0) -> 1
    # (1, 1) -> 2
    # Pair_type matrix:
    # 0 0 1 1
    # 0 0 1 1
    # 1 1 2 2
    # 1 1 2 2

    _, pair_types, _ = distances_and_pair_types(xyzs, types, 2, cutoff=2.6)

    np.testing.assert_array_equal(pair_types[0, :, 0].numpy(), [0, 1])
    np.testing.assert_array_equal(pair_types[1, :, 0].numpy(), [0])
    np.testing.assert_array_equal(pair_types[2, :, 0].numpy(), [2])
    np.testing.assert_array_equal(pair_types[3, :, 0].numpy(), [1, 2])


def test_invariance(xyzs, types, atol=1e-6):
    # Test invariance with respect to an arbitray rotation and translation
    # Reference values
    r, _, _ = distances_and_pair_types(xyzs, types, 2, cutoff=2.6)
    # Construct random rotation matrix
    R = tf.constant(
        rotation_matrix(np.random.randn(3), np.random.randn(1)[0]), dtype=tf.float32
    )
    # Apply random translation before rotation
    xyzs2 = tf.matmul(xyzs + np.random.randn(3), R)

    r2, _, _ = distances_and_pair_types(xyzs2, types, 2, cutoff=2.6)

    np.testing.assert_allclose(r.to_tensor().numpy(), r2.to_tensor().numpy(), atol=atol)


def test_derivative(xyzs, types, atol=1e-5):
    # Test dr_dx versus numerical derivative
    _, _, dr_dx = distances_and_pair_types(xyzs, types, 2, cutoff=2.6)

    def fun(x):
        """Due to the non square shape of r it is flattened using
        merge_dims
        """
        return (
            distances_and_pair_types(x, types, 2, cutoff=2.6)[0]
            .merge_dims(0, -1)
            .numpy()
        )

    num_dr_dx = derive_array_wrt_array(fun, xyzs.numpy(), dx=1e-2)
    # dr_dx also has to be flattened in the first 2 dimensions
    np.testing.assert_allclose(
        dr_dx.merge_dims(0, 1).to_tensor().numpy(), num_dr_dx, atol=atol
    )


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
def test_preprocessed_dataset_from_json(resource_path_root, floatx):
    type_dict = {"Ni": 0, "Pt": 1}
    dataset = preprocessed_dataset_from_json(
        resource_path_root / "test_data.json",
        type_dict,
        batch_size=4,
        floatx=floatx,
        cutoff=5.0,
    )
    input, output = next(iter(dataset))
    assert tuple(input["types"].bounding_shape().numpy()) == (4, 38, 1)
    assert input["types"].dtype == tf.int32

    # NOTE: because of the cutoff we supplied this is not the "expected" (4, 38, 33, 1)
    # but instead is smaller to optimize the throughput.
    assert tuple(input["pair_types"].bounding_shape().numpy()) == (4, 38, 33, 1)
    assert input["pair_types"].dtype == tf.int32

    assert tuple(input["distances"].bounding_shape().numpy()) == (4, 38, 33, 1)
    assert input["distances"].dtype == floatx

    assert tuple(input["dr_dx"].bounding_shape().numpy()) == (4, 38, 33, 38, 3)
    assert input["dr_dx"].dtype == floatx

    assert output["energy"].shape == (4, 1)
    assert output["energy"].dtype == floatx

    assert output["energy_per_atom"].shape == (4, 1)
    assert output["energy_per_atom"].dtype == floatx

    assert tuple(output["forces"].bounding_shape()) == (4, 38, 3)
    assert output["forces"].dtype == floatx


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
def test_dataset_from_json(resource_path_root, floatx):
    type_dict = {"Ni": 0, "Pt": 1}
    dataset = dataset_from_json(
        resource_path_root / "test_data.json",
        type_dict,
        batch_size=4,
        floatx=floatx,
    )
    input, output = next(iter(dataset))

    assert tuple(input["types"].bounding_shape().numpy()) == (4, 38, 1)
    assert input["types"].dtype == tf.int32

    assert tuple(input["positions"].bounding_shape().numpy()) == (4, 38, 3)
    assert input["positions"].dtype == floatx

    assert output["energy"].shape == (4, 1)
    assert output["energy"].dtype == floatx

    assert output["energy_per_atom"].shape == (4, 1)
    assert output["energy_per_atom"].dtype == floatx

    assert tuple(output["forces"].bounding_shape()) == (4, 38, 3)
    assert output["forces"].dtype == floatx