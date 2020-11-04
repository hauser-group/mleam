import tensorflow as tf
import numpy as np
import json
from mlff.utils import distances_and_pair_types


def dataset_from_json(path, type_dict, batch_size=None, buffer_size=None,
                      floatx=tf.float32, shuffle=True):
    with open(path, 'r') as fin:
        data = json.load(fin)

    if batch_size is None:
        batch_size = len(data['symbols'])
        buffer_size = batch_size
    buffer_size = buffer_size or 4*batch_size

    input_dataset = tf.data.Dataset.from_tensor_slices(
        {'types': tf.expand_dims(tf.ragged.constant(
                [[type_dict[t] for t in syms] for syms in data['symbols']]),
            axis=-1),
         'positions': tf.ragged.constant(data['positions'], ragged_rank=1)})

    N = tf.constant([len(sym) for sym in data['symbols']], dtype=floatx)
    energy_per_atom = tf.expand_dims(
        tf.constant(data['e_dft_bond'], dtype=floatx)/N, axis=-1)
    output_dataset = tf.data.Dataset.from_tensor_slices(
        {'energy_per_atom': energy_per_atom,
         'forces': tf.ragged.constant(data['forces_dft'], ragged_rank=1)})

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def preprocessed_dataset_from_json(path, type_dict, cutoff=10.0,
                                   batch_size=None, buffer_size=None,
                                   floatx=tf.float32, shuffle=True):
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

    N = tf.constant([len(sym) for sym in data['symbols']], dtype=floatx)
    energy_per_atom = tf.expand_dims(
        tf.constant(data['e_dft_bond'], dtype=floatx)/N, axis=-1)
    output_dataset = tf.data.Dataset.from_tensor_slices(
        {'energy_per_atom': energy_per_atom,
         'forces': tf.ragged.constant(data['forces_dft'], ragged_rank=1)})

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def descriptor_dataset_from_json(path, descriptor_set,
                                 batch_size=None, buffer_size=None,
                                 floatx=tf.float32, Gs_min=None, Gs_max=None,
                                 shuffle=True):
    with open(path, 'r') as fin:
        data = json.load(fin)

    if batch_size is None:
        batch_size = len(data['symbols'])
        buffer_size = batch_size
    buffer_size = buffer_size or 4*batch_size

    types = tf.expand_dims(tf.ragged.constant(
                [[descriptor_set.type_dict[t] for t in syms]
                    for syms in data['symbols']]),
            axis=-1)

    def gen():
        for i in range(len(data['symbols'])):
            Gs, dGs = descriptor_set.eval_with_derivatives_atomwise(
                data['symbols'][i], np.array(data['positions'][i]))
            if Gs_min and Gs_max:
                Gs = [2. * (Gs[j] - Gs_min[tj])/(Gs_max[tj] - Gs_min[tj]) - 1.
                      for j, tj in enumerate(data['symbols'][i])]
                dGs = [2. * dGs[j]/np.expand_dims(Gs_max[tj] - Gs_min[tj],
                                                  (1, 2))
                       for j, tj in enumerate(data['symbols'][i])]
            yield dict(types=types[i], Gs=Gs, dGs=dGs)

    input_dataset = tf.data.Dataset.from_generator(
        gen,
        {'types': tf.int32,
         'Gs': floatx,
         'dGs': floatx},
        {'types': tf.TensorShape([None, 1]),
         'Gs': tf.TensorShape([None, None]),
         'dGs': tf.TensorShape([None, None, None, 3])})

    N = tf.constant([len(sym) for sym in data['symbols']], dtype=floatx)
    energy_per_atom = tf.expand_dims(
        tf.constant(data['e_dft_bond'], dtype=floatx)/N, axis=-1)
    output_dataset = tf.data.Dataset.from_tensor_slices(
        {'energy_per_atom': energy_per_atom,
         'forces': tf.ragged.constant(data['forces_dft'], ragged_rank=1)})

    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
