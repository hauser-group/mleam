import pytest
import json
import numpy as np
import tensorflow as tf
from mleam.input_pipeline import (
    dataset_from_json,
    preprocessed_dataset_from_json,
    descriptor_dataset_from_json,
    preprocessed_dummy_dataset,
    _output_dataset_from_dict,
    fcc_bulk_curve,
    ico13_curve,
)
from itertools import product


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
@pytest.mark.parametrize("forces", [True, False])
def test_preprocessed_dataset_from_json(
    resource_path_root, floatx, forces, rtol=1e-5, atol=1e-6
):
    type_dict = {"Ni": 0, "Pt": 1}
    dataset = preprocessed_dataset_from_json(
        resource_path_root / "test_data.json",
        type_dict,
        batch_size=4,
        floatx=floatx,
        cutoff=5.0,
        forces=forces,
        shuffle_buffer_size=100,
        shuffle_seed=0,
    )
    inputs, outputs = next(iter(dataset))

    assert inputs["types"].ragged_rank == 1
    assert inputs["types"].dtype == tf.int32
    assert inputs["types"].shape == (4, None, 1)
    np.testing.assert_allclose(inputs["types"].row_lengths().numpy(), [147, 55, 55, 38])

    assert inputs["pair_types"].ragged_rank == 2
    assert inputs["pair_types"].dtype == tf.int32
    assert inputs["pair_types"].shape == (4, None, None, 1)

    assert inputs["distances"].ragged_rank == 2
    assert inputs["distances"].dtype == floatx
    assert inputs["distances"].shape == (4, None, None, 1)
    np.testing.assert_allclose(
        inputs["distances"][1, 0:3, 0:3, 0].numpy(),
        np.array(
            [
                [4.236714, 4.019963, 2.98717],
                [3.092563, 4.064307, 4.140191],
                [4.236714, 2.865811, 4.023477],
            ]
        ),
        rtol=rtol,
    )

    if forces:
        assert inputs["dr_dx"].ragged_rank == 2
        # assert tuple(inputs["dr_dx"].bounding_shape().numpy()) == (4, 38, 38, 3)
        assert inputs["dr_dx"].dtype == floatx
        assert inputs["dr_dx"].shape == (4, None, None, 3)

        np.testing.assert_allclose(
            inputs["dr_dx"][0, 1, 0:4, :],
            np.array(
                [
                    [0.71961015, 0.43044097, -0.54486861],
                    [0.16035014, 0.53227601, -0.8312461],
                    [-0.58401367, 0.80771526, -0.08077185],
                    [-0.97854147, -0.00867641, -0.20586725],
                ]
            ),
            rtol=rtol,
            atol=atol,
        )

        np.testing.assert_allclose(
            inputs["j_indices"][2, 0:4, 0:3].numpy(),
            np.array(
                [
                    [5, 7, 10],
                    [2, 8, 9],
                    [1, 3, 9],
                    [2, 6, 10],
                ]
            ),
        )

        np.testing.assert_allclose(
            outputs["forces"][0, 0:3, :],
            np.array(
                [
                    [-0.25553063, -0.15273793, -0.6125229],
                    [0.14569979, -0.35180147, 0.53064227],
                    [-0.29970835, 0.50666725, -0.12815252],
                ]
            ),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            outputs["forces"][2, 0:3, :],
            np.array(
                [
                    [-0.29370303, 0.51540951, 0.2199098],
                    [-0.06078371, 0.23760336, 0.01329723],
                    [-0.0368781, -0.15473387, -0.4040936],
                ]
            ),
            rtol=rtol,
            atol=atol,
        )

    np.testing.assert_allclose(
        outputs["energy"],
        np.array(
            [
                -647.313848,
                -234.921643,
                -235.948136,
                -157.50149,
            ]
        ).reshape(-1, 1),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
@pytest.mark.parametrize("forces", [True, False])
def test_dataset_output_from_dict(resource_path_root, floatx, forces, rtol=1e-5):
    with open(resource_path_root / "test_data.json", "r") as fin:
        data = json.load(fin)

    dataset = _output_dataset_from_dict(
        data,
        forces=forces,
        floatx=floatx,
    )
    dataset = dataset.ragged_batch(batch_size=4)
    outputs = next(iter(dataset))

    assert outputs["energy"].shape == (4, 1)
    assert outputs["energy"].dtype == floatx
    np.testing.assert_allclose(
        outputs["energy"],
        np.array(
            [
                -149.626757,
                -156.528134,
                -156.019539,
                -157.50149,
            ]
        ).reshape(-1, 1),
        rtol=rtol,
    )

    assert outputs["energy_per_atom"].shape == (4, 1)
    assert outputs["energy_per_atom"].dtype == floatx
    np.testing.assert_allclose(
        outputs["energy_per_atom"],
        np.array(
            [
                -3.937546,
                -4.119161,
                -4.105777,
                -4.144776,
            ]
        ).reshape(-1, 1),
        rtol=rtol,
    )

    if forces:
        # assert tuple(outputs["forces"].bounding_shape()) == (4, 38, 3)
        assert outputs["forces"].dtype == floatx
        assert outputs["forces"].shape == (4, None, 3)
        np.testing.assert_allclose(
            outputs["forces"].to_tensor()[:, 0],
            np.array(
                [
                    [0.151695, 0.138809, 0.571788],
                    [-0.921326, 0.636308, 0.516866],
                    [0.272931, -0.143648, 0.028825],
                    [-0.093672, -0.089866, 0.459556],
                ]
            ),
            rtol=rtol,
        )


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
@pytest.mark.parametrize("forces", [False, True])
def test_preprocessed_dummy_dataset(resource_path_root, floatx, forces):
    batch_size = 4
    dataset = preprocessed_dummy_dataset(
        batch_size=batch_size,
        num_batches=2,
        floatx=floatx,
        forces=forces,
        sizes=[4, 5, 6],
        rng=np.random.default_rng(0),
    )
    inputs, outputs = next(iter(dataset))

    row_splits_0 = [0, 6, 11, 16, 20]
    row_splits_1 = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        73,
        76,
        79,
        82,
    ]
    row_splits_2 = [
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        54,
        60,
        66,
        72,
        78,
        84,
        90,
        96,
        102,
        108,
        114,
        120,
        126,
        132,
        138,
        144,
        150,
        156,
        162,
        168,
        174,
        180,
        185,
        190,
        195,
        200,
        205,
        210,
        215,
        220,
        225,
        230,
        235,
        240,
        245,
        250,
        255,
        260,
        265,
        270,
        275,
        280,
        285,
        290,
        295,
        300,
        305,
        310,
        315,
        320,
        325,
        330,
        335,
        340,
        345,
        350,
        355,
        360,
        365,
        370,
        375,
        380,
        384,
        388,
        392,
        396,
        400,
        404,
        408,
        412,
        416,
        420,
        424,
        428,
    ]
    row_splits = [row_splits_0, row_splits_1, row_splits_2]

    assert inputs["types"].ragged_rank == 1
    assert inputs["types"].shape == (batch_size, None, 1)
    assert inputs["types"].dtype == tf.int32
    for trial, ref in zip(inputs["types"].nested_row_splits, row_splits):
        np.testing.assert_array_equal(trial, ref)

    assert inputs["pair_types"].ragged_rank == 2
    assert inputs["pair_types"].shape == (batch_size, None, None, 1)
    assert inputs["pair_types"].dtype == tf.int32
    for trial, ref in zip(inputs["pair_types"].nested_row_splits, row_splits):
        np.testing.assert_array_equal(trial, ref)

    assert inputs["distances"].ragged_rank == 2
    assert inputs["distances"].shape == (batch_size, None, None, 1)
    assert inputs["distances"].dtype == floatx
    for trial, ref in zip(inputs["distances"].nested_row_splits, row_splits):
        np.testing.assert_array_equal(trial, ref)

    if forces:
        assert inputs["dr_dx"].ragged_rank == 3
        assert inputs["dr_dx"].shape == (batch_size, None, None, None, 3)
        assert inputs["dr_dx"].dtype == floatx
        for trial, ref in zip(inputs["dr_dx"].nested_row_splits, row_splits):
            np.testing.assert_array_equal(trial, ref)

    assert outputs["energy"].shape == (batch_size, 1)
    assert outputs["energy"].dtype == floatx

    assert outputs["energy_per_atom"].shape == (batch_size, 1)
    assert outputs["energy_per_atom"].dtype == floatx

    if forces:
        assert outputs["forces"].ragged_rank == 1
        assert outputs["forces"].shape == (batch_size, None, 3)
        assert outputs["forces"].dtype == floatx
        for trial, ref in zip(outputs["forces"].nested_row_splits, row_splits):
            np.testing.assert_array_equal(trial, ref)


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
def test_dataset_from_json(resource_path_root, floatx):
    type_dict = {"Ni": 0, "Pt": 1}
    dataset = dataset_from_json(
        resource_path_root / "test_data.json",
        type_dict,
        batch_size=4,
        floatx=floatx,
    )
    inputs, outputs = next(iter(dataset))

    assert tuple(inputs["types"].bounding_shape().numpy()) == (4, 147, 1)
    assert inputs["types"].dtype == tf.int32

    assert tuple(inputs["positions"].bounding_shape().numpy()) == (4, 147, 3)
    assert inputs["positions"].dtype == floatx

    assert outputs["energy"].shape == (4, 1)
    assert outputs["energy"].dtype == floatx

    assert outputs["energy_per_atom"].shape == (4, 1)
    assert outputs["energy_per_atom"].dtype == floatx

    assert tuple(outputs["forces"].bounding_shape()) == (4, 147, 3)
    assert outputs["forces"].dtype == floatx


@pytest.mark.parametrize("floatx", [tf.float32, tf.float64])
def test_descriptor_dataset_from_json(resource_path_root, floatx):
    try:
        from mlpot.descriptors import DescriptorSet
    except ImportError:
        pytest.skip("Unable to import from mlpot")

    with open(
        resource_path_root / "behler_parrinello_reference" / "descriptors.json"
    ) as fin:
        ref_data = json.load(fin)

    etas = [0.001, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    cutoffs = {
        ("Ni", "Ni"): 5.57,
        ("Ni", "Pt"): 5.887567,
        ("Pt", "Ni"): 5.887567,
        ("Pt", "Pt"): 6.2050886,
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

        dataset = descriptor_dataset_from_json(
            resource_path_root / "test_data.json",
            descriptor_set,
            batch_size=4,
            floatx=floatx,
        )
        inputs, outputs = next(iter(dataset))

    assert tuple(inputs["types"].bounding_shape().numpy()) == (4, 38, 1)
    assert inputs["types"].dtype == tf.int32
    assert inputs["types"].to_list() == ref_data["types"]

    assert tuple(inputs["Gs"].bounding_shape().numpy()) == (4, 38, 14)
    assert inputs["Gs"].dtype == floatx
    np.testing.assert_allclose(inputs["Gs"].to_list(), ref_data["Gs"], atol=1e-6)

    assert tuple(inputs["dGs"].bounding_shape().numpy()) == (4, 38, 14, 38, 3)
    assert inputs["dGs"].dtype == floatx
    np.testing.assert_allclose(inputs["dGs"].to_list(), ref_data["dGs"], atol=1e-6)


def test_fcc_bulk_curve():
    type_dict = {"Ni": 0, "Pt": 1}
    a_vec = np.linspace(2.5, 3.5, 5)
    dataset = fcc_bulk_curve(type_dict, "Pt", a_vec)
    inputs = next(iter(dataset))

    assert inputs["types"].to_list() == [[[1]]] * 5
    assert inputs["pair_types"].to_list() == [[[[2]] * 78]] * 5

    np.testing.assert_allclose(
        inputs["distances"].numpy(),
        np.concatenate(
            [
                np.tile(a_vec[:, np.newaxis] * np.sqrt(1 / 2), (1, 12)),
                np.tile(a_vec[:, np.newaxis] * np.sqrt(2 / 2), (1, 6)),
                np.tile(a_vec[:, np.newaxis] * np.sqrt(3 / 2), (1, 24)),
                np.tile(a_vec[:, np.newaxis] * np.sqrt(4 / 2), (1, 12)),
                np.tile(a_vec[:, np.newaxis] * np.sqrt(5 / 2), (1, 24)),
            ],
            axis=1,
        )[:, np.newaxis, :, np.newaxis],
    )
    np.testing.assert_allclose(
        inputs["dr_dx"].numpy(), np.zeros_like(inputs["dr_dx"].numpy())
    )


def test_ico13_curve():
    type_dict = {"Ni": 0, "Pt": 1}
    r_vec = np.linspace(2.5, 3.5, 5)
    types = ["Ni"] + ["Pt"] * 12
    dataset = ico13_curve(type_dict, types, r_vec)
    inputs = next(iter(dataset))

    int_types = [[type_dict[t]] for t in types]

    assert inputs["types"].to_list() == [int_types] * 5
    np.testing.assert_allclose(
        inputs["distances"].numpy()[:, 0],
        np.tile(r_vec[:, np.newaxis, np.newaxis], (1, 12, 1)),
    )
