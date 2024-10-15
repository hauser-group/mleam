import pytest
import json
import numpy as np
import tensorflow as tf
from mleam.models import SMATB
from mleam.input_pipeline import fcc_bulk_curve
from mleam.preprocessing import preprocess_inputs_ragged


@pytest.fixture
def params():
    return {
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


def test_versus_lammps(resource_path_root, params):
    # Read reference geometry
    with open(resource_path_root / "LAMMPS_SMATB_reference" / "data.NiPt", "r") as fin:
        flag = False
        # Convert to tensorflow types
        type_dict = {"1": 0, "2": 1}
        types = []
        positions = []
        for line in fin:
            if line.startswith("Atoms # atomic"):
                # Skip one line:
                line = next(fin)
                flag = True
            elif flag:
                sp = line.split()
                types.append(type_dict[sp[1]])
                positions.append([float(s) for s in sp[2:5]])
    N = len(types)
    # Read LAMMPS energy
    with open(resource_path_root / "LAMMPS_SMATB_reference" / "log.lammps", "r") as fin:
        for line in fin:
            if line.startswith("   Step          Temp          E_pair "):
                sp = next(fin).split()
                e_ref = float(sp[4])
    # Read LAMMPS forces:
    with open(resource_path_root / "LAMMPS_SMATB_reference" / "dump.force", "r") as fin:
        flag = False
        forces_ref = []
        for line in fin:
            if line.startswith("ITEM: ATOMS id type fx fy fz"):
                flag = True
            elif flag:
                sp = line.split()
                forces_ref.append([float(s) for s in sp[2:5]])

    model = SMATB(["Ni", "Pt"], params=params, build_forces=True)

    types = tf.expand_dims(tf.ragged.constant([types]), axis=-1)
    positions = tf.ragged.constant([positions], ragged_rank=1)
    prediction = model({"types": types, "positions": positions})

    assert prediction["energy"].shape == (1, 1)
    assert prediction["energy_per_atom"].shape == (1, 1)
    assert tuple(prediction["forces"].bounding_shape()) == (1, 38, 3)

    np.testing.assert_allclose(
        prediction["energy"].numpy()[0], e_ref, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        prediction["energy_per_atom"].numpy()[0], e_ref / N, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        prediction["forces"].numpy()[0], forces_ref, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "a",
    [
        tf.ragged.constant([[], [2, 3], [1]], ragged_rank=1),
        tf.ragged.constant([[[]], [[2], [3]], [[1]]], ragged_rank=2),
        tf.ragged.constant(
            [[[1.0, 2.0]], [[2.0, 1.0], [3.0, 3.0]], [[1.0, 0.0]]], ragged_rank=1
        ),
        tf.ragged.constant(
            [[[[1.0, 2.0]], [[1.0, 2.0]]], [[[2.0, 1.0], [3.0, 3.0]]], [[[1.0, 0.0]]]],
            ragged_rank=2,
        ),
    ],
)
def test_custom_ragged_sum(a):
    trial = SMATB._sum_inner_most_ragged(a)
    # Find the inner most ragged dim:
    axis = int(np.where([ax is None for ax in a.shape])[0][-1])
    ref = tf.reduce_sum(a, axis=axis)

    assert type(trial) is type(ref)
    assert trial.shape == ref.shape
    if isinstance(trial, tf.RaggedTensor):
        assert trial.to_list() == ref.to_list()
    else:
        np.testing.assert_allclose(trial, ref)


def test_versus_ferrando_code(params, resource_path_root):
    with open(resource_path_root / "test_data.json", "r") as fin:
        data = json.load(fin)

    type_dict = {"Ni": 0, "Pt": 1}
    e_ref = data["e_smatb"]

    types = tf.ragged.constant(
        [[[type_dict[ti]] for ti in type_vec] for type_vec in data["symbols"]],
        ragged_rank=1,
    )
    positions = tf.ragged.constant(data["positions"], ragged_rank=1)

    model = SMATB(["Ni", "Pt"], params=params, build_forces=False)

    e_model = model({"types": types, "positions": positions})["energy"]

    # High tolerance since we know that the Ferrando code uses a different
    # cutoff style
    np.testing.assert_allclose(e_model.numpy().flatten(), e_ref, rtol=1e-2)


@pytest.mark.parametrize(
    "method",
    [
        "partition_stitch",
        "gather_scatter",
        "where",
        "dense_where",
    ],
)
@pytest.mark.parametrize("forces", [False, True])
@pytest.mark.parametrize("preprocessed_input", [False, True])
def test_body_methods(method, forces, params, preprocessed_input):
    xyzs = tf.ragged.constant(
        [
            [
                [0.0, 0.0, 0.0],
                [2.91, -1.12, -4.20],
                [0.64, -3.05, -2.22],
                [-1.09, -0.97, -2.27],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.91, -1.12, -4.20],
                [0.64, -3.05, -2.22],
            ],
        ],
        ragged_rank=1,
    )
    types = tf.ragged.constant(
        [
            [[0], [1], [0], [1]],
            [[0], [1], [0]],
        ],
        ragged_rank=1,
    )

    if preprocessed_input:
        types, pair_types, distances, dr_dx, j_indices = preprocess_inputs_ragged(
            xyzs, types, 2
        )
        inputs = {
            "types": types,
            "pair_types": pair_types,
            "distances": distances,
            "dr_dx": dr_dx,
            "j_indices": j_indices,
        }
    else:
        inputs = {"positions": xyzs, "types": types}

    model = SMATB(
        ["Ni", "Pt"],
        params=params,
        build_forces=forces,
        preprocessed_input=preprocessed_input,
        method=method,
    )
    prediction = model(inputs)

    np.testing.assert_allclose(
        prediction["energy"],
        np.array(
            [
                [-8.0504],
                [-2.1644],
            ]
        ),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        prediction["energy_per_atom"],
        np.array(
            [
                [-2.0126],
                [-0.7215],
            ]
        ),
        atol=1e-4,
    )
    if forces:
        np.testing.assert_allclose(
            prediction["forces"][0].numpy(),
            np.array(
                [
                    [-1.227, -1.371, -2.830],
                    [-0.985, -0.609, 0.759],
                    [-1.246, 3.168, -0.519],
                    [3.4583, -1.188, 2.590],
                ]
            ),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            prediction["forces"][1].numpy(),
            np.array(
                [
                    [0.181, -0.860, -0.626],
                    [-1.058, -0.900, 0.923],
                    [0.878, 1.760, -0.297],
                ]
            ),
            atol=1e-3,
        )


def test_tabulation(params, resource_path_root, tmpdir, atol=1e-4, rtol=1e-2):
    def read_tabulated_potential(path, skiprows=6):
        skiprows = 6
        with open(path, "r") as fin:
            lines = fin.readlines()
        nrho, nr = int(lines[4].split()[0]), int(lines[4].split()[2])
        tabs = np.zeros(len(lines) - skiprows)
        for i, line in enumerate(lines[skiprows:]):
            tabs[i] = float(line.split()[0])
        return nrho, nr, tabs

    nrho, nr, ref = read_tabulated_potential(
        resource_path_root / "LAMMPS_SMATB_reference" / "NiPt.eam.fs"
    )

    model = SMATB(["Ni", "Pt"], params=params)
    model.tabulate(
        str(tmpdir / "tmp"),
        atomic_numbers=dict(Ni=28, Pt=78),
        atomic_masses=dict(Ni=58.6934, Pt=195.084),
        cutoff_rho=100.0,
        nrho=nrho,
        cutoff=6.0,
        nr=nr,
    )

    _, _, test = read_tabulated_potential(tmpdir / "tmp.eam.fs")
    assert len(ref) == len(test)

    # F Ni:
    start_Ni = 0
    inds = slice(start_Ni, start_Ni + nrho)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # F Pt:
    start_Pt = nrho + 3 * nr + 1
    inds = slice(start_Pt, start_Pt + nrho)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # rho NiNi:
    inds = slice(start_Ni + nrho, start_Ni + nrho + nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # rho NiPt:
    inds = slice(start_Ni + nrho + nr, start_Ni + nrho + 2 * nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # rho PtNi:
    inds = slice(start_Pt + nrho, start_Pt + nrho + nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # rho PtPt:
    inds = slice(start_Pt + nrho + nr, start_Pt + nrho + 2 * nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # phi NiNi:
    start_phi = start_Pt + nrho + 2 * nr
    inds = slice(start_phi, start_phi + nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # phi NiPt:
    inds = slice(start_phi + nr, start_phi + 2 * nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)
    # phi PtPt:
    inds = slice(start_phi + 2 * nr, start_phi + 3 * nr)
    np.testing.assert_allclose(test[inds], ref[inds], rtol=rtol, atol=atol)


def test_load_smatb_model(resource_path_root):
    tf.keras.backend.clear_session()

    type_dict = {"Ni": 0, "Pt": 1}
    a_vec = np.linspace(1.8 * np.sqrt(2), 3.5 * np.sqrt(2), 50)
    Ni_bulk_curve = fcc_bulk_curve(type_dict, "Ni", a_vec)
    Pt_bulk_curve = fcc_bulk_curve(type_dict, "Pt", a_vec)

    model = SMATB(["Ni", "Pt"], params={}, build_forces=False, preprocessed_input=True)
    # Call once to build all layers
    model.predict(Ni_bulk_curve)
    model.load_weights(resource_path_root / "models" / "smatb_reference.h5")

    e_Ni = model.predict(Ni_bulk_curve)["energy"]
    e_Pt = model.predict(Pt_bulk_curve)["energy"]

    with open(resource_path_root / "data" / "bulk_curves.json", "r") as fin:
        e_ref = json.load(fin)["SMATB"]

    np.testing.assert_allclose(
        e_Ni.flatten(),
        e_ref["Ni"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        e_Pt.flatten(),
        e_ref["Pt"],
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "method",
    [
        "partition_stitch",
        "gather_scatter",
        "where",
        "dense_where",
    ],
)
@pytest.mark.parametrize("forces", [False, True])
def test_smatb_dense_input(method, forces, params):
    xyzs = tf.constant(
        [
            [
                [0.0, 0.0, 0.0],
                [2.91, -1.12, -4.20],
                [0.64, -3.05, -2.22],
                [-1.09, -0.97, -2.27],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.91, -1.12, -4.20],
                [0.64, -3.05, -2.22],
                [999.9, 999.9, 999.9],
            ],
        ],
    )
    types = tf.constant(
        [
            [[0], [1], [0], [1]],
            [[0], [1], [0], [-1]],
        ]
    )

    model = SMATB(
        ["Ni", "Pt"],
        params=params,
        build_forces=forces,
        method=method,
    )
    prediction = model({"positions": xyzs, "types": types})

    np.testing.assert_allclose(
        prediction["energy"],
        np.array(
            [
                [-8.0504],
                [-2.1644],
            ]
        ),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        prediction["energy_per_atom"],
        np.array(
            [
                [-2.0126],
                [-0.7215],
            ]
        ),
        atol=1e-4,
    )
    if forces:
        np.testing.assert_allclose(
            prediction["forces"][0].numpy(),
            np.array(
                [
                    [-1.227, -1.371, -2.830],
                    [-0.985, -0.609, 0.759],
                    [-1.246, 3.168, -0.519],
                    [3.4583, -1.188, 2.590],
                ]
            ),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            prediction["forces"][1].numpy(),
            np.array(
                [
                    [0.181, -0.860, -0.626],
                    [-1.058, -0.900, 0.923],
                    [0.878, 1.760, -0.297],
                ]
            ),
            atol=1e-3,
        )
