import tensorflow as tf
import numpy as np
import json
from mleam.utils import distances_and_pair_types, distances_and_pair_types_no_grad


def _output_dataset_from_json(data, forces=False, floatx=tf.float32):
    energy = tf.expand_dims(tf.constant(data["e_dft_bond"], dtype=floatx), axis=-1)
    N = tf.constant([[len(sym)] for sym in data["symbols"]], dtype=floatx)
    energy_per_atom = energy / N
    output_dict = {"energy": energy, "energy_per_atom": energy_per_atom}
    if forces:
        output_dict["forces"] = tf.ragged.constant(
            data["forces_dft"],
            ragged_rank=1,
            dtype=floatx,
        )
    return tf.data.Dataset.from_tensor_slices(output_dict)


def dataset_from_json(
    path,
    type_dict,
    forces=True,
    batch_size=None,
    floatx=tf.float32,
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

    output_dataset = _output_dataset_from_json(data, forces=forces, floatx=floatx)

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    dataset = dataset.ragged_batch(batch_size=batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def preprocessed_dataset_from_json(
    path,
    type_dict,
    cutoff=10.0,
    forces=True,
    batch_size=None,
    floatx=tf.float32,
):
    with open(path, "r") as fin:
        data = json.load(fin)

    cutoff = tf.cast(cutoff, dtype=floatx)
    if batch_size is None:
        batch_size = len(data["symbols"])

    if forces:

        @tf.function
        def input_transform(inp):
            r, pair_types, dr_dx = distances_and_pair_types(
                inp["positions"], inp["types"], len(type_dict), cutoff=cutoff
            )
            return dict(
                types=inp["types"], pair_types=pair_types, distances=r, dr_dx=dr_dx
            )
    else:

        @tf.function
        def input_transform(inp):
            r, pair_types = distances_and_pair_types_no_grad(
                inp["positions"], inp["types"], len(type_dict), cutoff=cutoff
            )
            return dict(types=inp["types"], pair_types=pair_types, distances=r)

    input_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "types": tf.expand_dims(
                tf.ragged.constant(
                    [[type_dict[t] for t in syms] for syms in data["symbols"]]
                ),
                axis=-1,
            ),
            "positions": tf.ragged.constant(
                data["positions"], ragged_rank=1, dtype=floatx
            ),
        }
    )

    input_dataset = input_dataset.map(
        input_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    output_dataset = _output_dataset_from_json(data, forces=forces, floatx=floatx)

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    dataset = dataset.ragged_batch(batch_size=batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def descriptor_dataset_from_json(
    path,
    descriptor_set,
    forces=True,
    batch_size=None,
    buffer_size=None,
    floatx=tf.float32,
    Gs_min=None,
    Gs_max=None,
    shuffle=True,
):
    with open(path, "r") as fin:
        data = json.load(fin)

    if batch_size is None:
        batch_size = len(data["symbols"])
        buffer_size = batch_size
    buffer_size = buffer_size or 4 * batch_size

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

    N = tf.constant([len(sym) for sym in data["symbols"]], dtype=floatx)
    energy_per_atom = tf.expand_dims(
        tf.constant(data["e_dft_bond"], dtype=floatx) / N, axis=-1
    )
    output_dict = {"energy_per_atom": energy_per_atom}
    if forces:
        output_dict["forces"] = tf.ragged.constant(data["forces_dft"], ragged_rank=1)
    output_dataset = tf.data.Dataset.from_tensor_slices(output_dict)

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def fcc_bulk_curve(type_dict: dict, atom_type: str, a: np.ndarray = np.array([4.0])):
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
