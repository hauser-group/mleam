import tensorflow as tf
import numpy as np
import json
from typing import List
from mleam.preprocessing import (
    preprocess_inputs_ragged,
    preprocess_inputs_ragged_no_force,
)


def _output_dataset_from_dict(
    data, forces=False, floatx=tf.float32, energy_offsets=None
):
    if energy_offsets is None:
        energy_offsets = {"Ni": -1363.5012992109332, "Pt": -983.274660448409}
    output_types = {"energy": floatx, "energy_per_atom": floatx}
    output_shapes = {
        "energy": tf.TensorShape([1]),
        "energy_per_atom": tf.TensorShape([1]),
    }
    if forces:

        def generator():
            for e, symbols, force in zip(
                data["e_dft_tot"], data["symbols"], data["forces_dft"]
            ):
                e -= sum([energy_offsets[t] for t in symbols])
                energy = tf.expand_dims(tf.constant(e, dtype=floatx), axis=-1)
                N = len(symbols)
                energy_per_atom = energy / N
                yield {
                    "energy": energy,
                    "energy_per_atom": energy_per_atom,
                    "forces": tf.constant(force, dtype=floatx),
                }

        output_types["forces"] = floatx
        output_shapes["forces"] = tf.TensorShape([None, 3])
    else:

        def generator():
            for e, symbols in zip(data["e_dft_tot"], data["symbols"]):
                e -= sum([energy_offsets[t] for t in symbols])
                energy = tf.expand_dims(tf.constant(e, dtype=floatx), axis=-1)
                N = len(symbols)
                energy_per_atom = energy / N
                yield {
                    "energy": energy,
                    "energy_per_atom": energy_per_atom,
                }

    return tf.data.Dataset.from_generator(generator, output_types, output_shapes)


def dataset_from_json(
    path,
    type_dict,
    energy_offsets=None,
    forces=True,
    batch_size=None,
    floatx=tf.float32,
    shuffle_buffer_size=100,
    shuffle_seed=0,
):
    with open(path, "r") as fin:
        data = json.load(fin)

    if batch_size is None:
        batch_size = len(data["symbols"])

    input_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "types": tf.expand_dims(
                tf.ragged.constant(
                    [[type_dict[t] for t in syms] for syms in data["symbols"]]
                ),
                axis=-1,
            ),
            "positions": tf.ragged.constant(
                data["positions"],
                ragged_rank=1,
                dtype=floatx,
            ),
        }
    )

    output_dataset = _output_dataset_from_dict(
        data, forces=forces, floatx=floatx, energy_offsets=energy_offsets
    )

    dataset = (
        tf.data.Dataset.zip((input_dataset, output_dataset))
        .cache()
        .shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
        .ragged_batch(batch_size=batch_size, name="batching")
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def preprocessed_dataset_from_json(
    path,
    type_dict,
    cutoff=10.0,
    energy_offsets=None,
    forces=True,
    batch_size=None,
    floatx=tf.float32,
    shuffle_buffer_size=100,
    shuffle_seed=0,
):
    with open(path, "r") as fin:
        data = json.load(fin)

    cutoff = tf.cast(cutoff, dtype=floatx)
    if batch_size is None:
        batch_size = len(data["symbols"])

    input_types = {"types": tf.int32, "positions": floatx}
    input_shapes = {
        "types": tf.TensorShape([None, 1]),
        "positions": tf.TensorShape([None, 3]),
    }

    def generator():
        for symbols, positions in zip(data["symbols"], data["positions"]):
            types = tf.expand_dims(
                tf.constant([type_dict[t] for t in symbols], dtype=tf.int32), axis=-1
            )
            positions = tf.constant(positions, dtype=floatx)
            yield {"types": types, "positions": positions}

    input_dataset = tf.data.Dataset.from_generator(generator, input_types, input_shapes)

    if forces:

        def input_transform(inp):
            types, pair_types, r, dr_dx, j_indices = preprocess_inputs_ragged(
                inp["positions"], inp["types"], len(type_dict), cutoff
            )
            return dict(
                types=types,
                pair_types=pair_types,
                distances=r,
                dr_dx=dr_dx,
                j_indices=j_indices,
            )
    else:

        def input_transform(inp):
            types, pair_types, r = preprocess_inputs_ragged_no_force(
                inp["positions"], inp["types"], len(type_dict), cutoff
            )
            return dict(
                types=types,
                pair_types=pair_types,
                distances=r,
            )

    input_dataset = input_dataset.map(
        input_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    output_dataset = _output_dataset_from_dict(
        data, forces=forces, floatx=floatx, energy_offsets=energy_offsets
    )

    dataset = (
        tf.data.Dataset.zip((input_dataset, output_dataset))
        .cache()
        .shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
        .ragged_batch(batch_size=batch_size, name="batching")
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return dataset


def preprocessed_dummy_dataset(
    type_dict={"Ni": 1, "Pt": 2},
    batch_size=50,
    forces=True,
    floatx=tf.float32,
    num_batches=25,
    sizes=[38, 55, 147],
    rng=np.random.default_rng(0),
    manual_batching=True,
):
    N_atoms = rng.choice(sizes, batch_size)
    row_splits_0 = [0]
    row_splits_1 = [0]
    row_splits_2 = [0]
    for ni in N_atoms:
        row_splits_0.append(row_splits_0[-1] + ni)
        for nj in range(ni):
            row_splits_1.append(row_splits_1[-1] + (ni - 1))
            for nk in range(ni - 1):
                row_splits_2.append(row_splits_2[-1] + ni)

    types = [rng.choice(list(type_dict.values()), n) for n in N_atoms]
    distances = tf.constant(
        rng.lognormal(
            mean=np.log(2.5), sigma=0.25, size=(sum([n * (n - 1) for n in N_atoms]), 1)
        ),
        dtype=floatx,
    )
    pair_types = tf.expand_dims(
        [
            types_n[i] + types_n[j]
            for n, types_n in zip(N_atoms, types)
            for i in range(n)
            for j in range(n - 1)
        ],
        axis=-1,
    )

    input_dict = {
        "types": tf.RaggedTensor.from_row_splits(
            np.concatenate(types, dtype=np.int32).reshape(-1, 1),
            row_splits_0,
        ),
        "distances": tf.RaggedTensor.from_nested_row_splits(
            distances, (row_splits_0, row_splits_1)
        ),
        "pair_types": tf.RaggedTensor.from_nested_row_splits(
            pair_types, (row_splits_0, row_splits_1)
        ),
    }

    energy = tf.constant(rng.normal(size=(batch_size, 1)), dtype=floatx)
    energy_per_atom = energy / N_atoms[:, np.newaxis]

    output_dict = {
        "energy": energy,
        "energy_per_atom": energy_per_atom,
    }

    if forces:
        dr_dx = tf.constant(
            rng.normal(size=(sum([n * (n - 1) * n for n in N_atoms]), 3)), dtype=floatx
        )
        input_dict["dr_dx"] = tf.RaggedTensor.from_nested_row_splits(
            dr_dx,
            (row_splits_0, row_splits_1, row_splits_2),
        )
        output_dict["forces"] = tf.RaggedTensor.from_row_splits(
            tf.constant(rng.normal(size=(sum(N_atoms), 3)), dtype=floatx), row_splits_0
        )

    if manual_batching:
        for key, val in input_dict.items():
            input_dict[key] = tf.expand_dims(val, axis=0)
        for key, val in output_dict.items():
            output_dict[key] = tf.expand_dims(val, axis=0)

    input_dataset = tf.data.Dataset.from_tensor_slices(input_dict)
    output_dataset = tf.data.Dataset.from_tensor_slices(output_dict)

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset)).repeat(num_batches)
    if not manual_batching:
        dataset = dataset.ragged_batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def descriptor_dataset_from_json(
    path,
    descriptor_set,
    energy_offsets=None,
    forces=True,
    batch_size=None,
    floatx=tf.float32,
    Gs_min=None,
    Gs_max=None,
):
    with open(path, "r") as fin:
        data = json.load(fin)

    if batch_size is None:
        batch_size = len(data["symbols"])

    types = tf.expand_dims(
        tf.ragged.constant(
            [[descriptor_set.type_dict[t] for t in syms] for syms in data["symbols"]]
        ),
        axis=-1,
    )

    input_types = dict(types=tf.int32, Gs=floatx)
    input_shapes = dict(
        types=tf.TensorShape([None, 1]), Gs=tf.TensorShape([None, None])
    )
    if forces:

        def gen():
            for i in range(len(data["symbols"])):
                Gs, dGs = descriptor_set.eval_with_derivatives_atomwise(
                    data["symbols"][i], np.array(data["positions"][i])
                )
                if Gs_min and Gs_max:
                    Gs = [
                        2.0 * (Gs[j] - Gs_min[tj]) / (Gs_max[tj] - Gs_min[tj]) - 1.0
                        for j, tj in enumerate(data["symbols"][i])
                    ]
                    dGs = [
                        2.0 * dGs[j] / np.expand_dims(Gs_max[tj] - Gs_min[tj], (1, 2))
                        for j, tj in enumerate(data["symbols"][i])
                    ]
                yield dict(types=types[i], Gs=Gs, dGs=dGs)

        input_types["dGs"] = floatx
        input_shapes["dGs"] = tf.TensorShape([None, None, None, 3])
    else:

        def gen():
            for i in range(len(data["symbols"])):
                Gs = descriptor_set.eval_atomwise(
                    data["symbols"][i], np.array(data["positions"][i])
                )
                if Gs_min and Gs_max:
                    Gs = [
                        2.0 * (Gs[j] - Gs_min[tj]) / (Gs_max[tj] - Gs_min[tj]) - 1.0
                        for j, tj in enumerate(data["symbols"][i])
                    ]
                yield dict(types=types[i], Gs=Gs)

    input_dataset = tf.data.Dataset.from_generator(gen, input_types, input_shapes)

    output_dataset = _output_dataset_from_dict(
        data, forces=forces, floatx=floatx, energy_offsets=energy_offsets
    )

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    dataset = dataset.ragged_batch(batch_size=batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def fcc_bulk_curve(type_dict: dict, atom_type: str, a: np.ndarray = np.array([4.0])):
    # TODO add similar function for descriptors
    # Use type_dict and atom_type to infer the pair_type
    n_types = len(type_dict)
    int_type = type_dict[atom_type]
    pair_type = n_types * int_type - (int_type * (int_type - 1)) // 2

    shells = [12, 6, 24, 12, 24]
    n_neighbors = sum(shells)

    distances = np.zeros((len(a), 1, n_neighbors, 1))

    start = 0
    for i, coordination_number in enumerate(shells):
        r = a * np.sqrt((i + 1) / 2)
        distances[:, 0, start : start + shells[i], 0] = r[:, np.newaxis]
        start += shells[i]

    input_dict = {
        "types": tf.ragged.constant(
            int_type * np.ones(shape=(len(a), 1, 1), dtype=np.int32),
            ragged_rank=1,
        ),
        "pair_types": tf.ragged.constant(
            pair_type * np.ones_like(distances, dtype=np.int32), ragged_rank=2
        ),
        "distances": tf.ragged.constant(distances.tolist(), ragged_rank=2),
        "dr_dx": tf.ragged.constant(
            np.zeros((len(a), 1, n_neighbors, 1, 3)).tolist(), ragged_rank=3
        ),
    }

    input_dataset = tf.data.Dataset.from_tensor_slices(input_dict).ragged_batch(
        batch_size=len(a)
    )

    return input_dataset


def ico13_curve(
    type_dict: dict,
    atom_types: List[str],
    r: np.ndarray = np.array([2.5]),
    cutoff: float = 10.0,
):
    assert len(atom_types) == 13
    N = len(r)
    # golden ratio phi
    phi = 0.5 * (1 + np.sqrt(5))
    # length of the Caretesian basis vectors used:
    norm = np.sqrt(1 + phi**2)

    int_types = np.array([type_dict[t] for t in atom_types])

    input_dict = {
        "types": tf.ragged.constant(
            np.tile(int_types[np.newaxis, :, np.newaxis], (N, 1, 1)),
            ragged_rank=1,
            dtype=tf.int32,
        ),
        "positions": tf.ragged.constant(
            r[:, np.newaxis, np.newaxis]
            * np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, phi],
                    [0.0, 1.0, -phi],
                    [0.0, -1.0, phi],
                    [0.0, -1.0, -phi],
                    [1.0, phi, 0.0],
                    [1.0, -phi, 0.0],
                    [-1.0, phi, 0.0],
                    [-1.0, -phi, 0.0],
                    [phi, 0.0, 1.0],
                    [-phi, 0.0, 1.0],
                    [phi, 0.0, -1.0],
                    [-phi, 0.0, -1.0],
                ]
            )[np.newaxis, ...]
            / norm,
            ragged_rank=1,
        ),
    }

    def input_transform(inp):
        types, pair_types, r, dr_dx, j_indices = preprocess_inputs_ragged(
            inp["positions"], inp["types"], len(type_dict), cutoff
        )
        return dict(
            types=types,
            pair_types=pair_types,
            distances=r,
            dr_dx=dr_dx,
            j_indices=j_indices,
        )

    input_dataset = (
        tf.data.Dataset.from_tensor_slices(input_dict)
        .ragged_batch(batch_size=N)
        .map(input_transform)
    )
    return input_dataset
