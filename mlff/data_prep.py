import tensorflow as tf
import json


@tf.function
def get_pair_type(types, n):
    """input: int tensor of shape (None, 1)
              int of the total number of elements"""

    min_ij = tf.math.minimum(types[:, tf.newaxis], types[tf.newaxis, :])
    mask = tf.logical_not(tf.eye(tf.shape(types)[0], dtype=tf.bool))
    return tf.ragged.boolean_mask(
        n*min_ij - (min_ij*(min_ij-1))//2
        + tf.abs(tf.subtract(types[:, tf.newaxis], types[tf.newaxis, :])),
        mask)


@tf.function
def distances_with_gradient(xyz):
    n = tf.shape(xyz, out_type=tf.int64)[0]
    mask = tf.logical_not(tf.eye(n, dtype=tf.bool))
    r_vec = tf.expand_dims(xyz, 0) - tf.expand_dims(xyz, 1)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1, keepdims=True))

    # Should be possible in a single step:
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n))], axis=2)
    gradient = tf.scatter_nd(indices, -r_vec/distances, (n, n, n, 3))
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n))], axis=2)
    gradient = tf.tensor_scatter_nd_add(gradient, indices, r_vec/distances)

    # Previously used numpy version

    # gradient = np.zeros((n, n, n, 3))
    # gradient[np.expand_dims(np.arange(n), 1),
    #          np.expand_dims(np.arange(n), 0),
    #          np.expand_dims(np.arange(n), 1), :] = -r_vec/distances
    # gradient[np.expand_dims(np.arange(n), 1),
    #          np.expand_dims(np.arange(n), 0),
    #          np.expand_dims(np.arange(n), 0), :] = r_vec/distances

    # Slicing into the masked tensor would be significanlty more difficult
    # but could be achieved similar to this n=4 example
    # gradient[np.expand_dims(np.arange(4), 1),
    #          np.expand_dims(np.arange(3), 0),
    #         [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], :])

    # gradient = tf.RaggedTensor.from_nested_row_lengths(
    #    gradient[mask2], ((n-1)*tf.ones(n, dtype=tf.int64),
    #                       n*tf.ones(n*(n-1), dtype=tf.int64)))
    gradient = tf.RaggedTensor.from_nested_row_lengths(
        tf.reshape(gradient[mask], (-1, 3)),
        ((n-1)*tf.ones(n, dtype=tf.int64), n*tf.ones(n*(n-1), dtype=tf.int64)))
    return tf.ragged.boolean_mask(distances, mask), gradient


@tf.function
def distances_and_pair_types(xyz, types, n_types, cutoff=10.0):
    n = tf.shape(xyz, out_type=tf.int64)[0]
    r_vec = tf.expand_dims(xyz, 0) - tf.expand_dims(xyz, 1)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1, keepdims=True))
    mask = tf.logical_and(tf.logical_not(tf.eye(n, dtype=tf.bool)),
                          tf.less_equal(distances[:, :, 0], cutoff))

    min_ij = tf.math.minimum(types[:, tf.newaxis], types[tf.newaxis, :])
    pair_types = (
        n_types*min_ij - (min_ij*(min_ij-1))//2
        + tf.abs(tf.subtract(types[:, tf.newaxis], types[tf.newaxis, :])))

    # Should be possible in a single step:
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n))], axis=2)
    gradient = tf.scatter_nd(indices, -r_vec/distances, (n, n, n, 3))
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n))], axis=2)
    gradient = tf.tensor_scatter_nd_add(gradient, indices, r_vec/distances)

    distances = tf.ragged.boolean_mask(distances, mask)
    pair_types = tf.ragged.boolean_mask(pair_types, mask)
    gradient = tf.ragged.boolean_mask(gradient, mask)
    # convert into ragged_rank=2
    row_lens = gradient.row_lengths()
    gradient = tf.RaggedTensor.from_nested_row_lengths(
        tf.reshape(gradient.flat_values, (-1, 3)),
        (row_lens, n*tf.ones(tf.reduce_sum(row_lens), dtype=tf.int64)))
    return distances, pair_types, gradient


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
            inp['positions'], inp['types'], 2, cutoff=cutoff)
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
