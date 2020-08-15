import tensorflow as tf
import json
from mlff.utils import distances_and_pair_types


def dataset_from_json(path, type_dict, cutoff=10.0, batch_size=None,
                      buffer_size=None):
    with open(path, 'r') as fin:
        data = json.load(fin)

    if batch_size is None:
        batch_size = len(data['symbols'])
        buffer_size = batch_size
    buffer_size = buffer_size or 4*batch_size

    @tf.function
    def input_transform(inp):
        r, pair_types, dr_dx = distances_and_pair_types(
            inp['positions'], inp['types'], len(type_dict), cutoff=cutoff)
        return dict(types=inp['types'], pair_types=pair_types, distances=r,
                    dr_dx=dr_dx)

    input_dataset = tf.data.Dataset.from_tensor_slices(
        {'types': tf.expand_dims(tf.ragged.constant(
                [[type_dict[t] for t in syms] for syms in data['symbols']]),
            axis=-1),
         'positions': tf.ragged.constant(data['positions'], ragged_rank=1)})

    input_dataset = input_dataset.map(
        input_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    N = tf.constant([len(sym) for sym in data['symbols']], dtype=tf.float32)
    energy_per_atom = tf.constant(data['e_dft_bond'], dtype=tf.float32)/N
    output_dataset = tf.data.Dataset.from_tensor_slices(
        (energy_per_atom,
         tf.ragged.constant(data['forces_dft'], ragged_rank=1)))

    input_dataset = input_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    output_dataset = output_dataset.batch(batch_size=batch_size)

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
