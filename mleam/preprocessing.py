import tensorflow as tf


def dense_distances_and_pair_types(xyzs, types, n_types, diagonal=0.0):
    distances, gradient = get_distance_matrix_and_derivative(xyzs)
    pair_types = get_pair_types(types, n_types)

    return distances, pair_types, gradient


def distances_and_pair_types(xyzs, types, n_types, cutoff=10.0):
    cutoff = tf.cast(cutoff, dtype=xyzs.dtype)

    n = tf.shape(xyzs, out_type=tf.int64)[0]
    distances, pair_types, gradient = dense_distances_and_pair_types(
        xyzs, types, n_types
    )
    mask = tf.logical_and(
        tf.logical_not(tf.eye(n, dtype=tf.bool)),
        tf.less_equal(distances[:, :, 0], cutoff),
    )

    distances = tf.ragged.boolean_mask(distances, mask)
    pair_types = tf.ragged.boolean_mask(pair_types, mask)
    gradient = tf.ragged.boolean_mask(gradient, mask)
    # convert into ragged_rank=2
    row_lengths = gradient.row_lengths()
    gradient = tf.RaggedTensor.from_nested_row_lengths(
        tf.reshape(gradient.flat_values, (-1, 3)),
        (row_lengths, n * tf.ones(tf.reduce_sum(row_lengths), dtype=tf.int64)),
    )
    return distances, pair_types, gradient


def dense_distances_and_pair_types_no_grad(xyzs, types, n_types, diagonal=0.0):
    pair_types = get_pair_types(types, n_types)
    distances = get_distance_matrix(xyzs)
    return distances, pair_types


def distances_and_pair_types_no_grad(xyzs, types, n_types, cutoff=10.0):
    distances, pair_types = dense_distances_and_pair_types_no_grad(xyzs, types, n_types)
    mask = tf.logical_and(
        tf.less_equal(distances[:, :, 0], cutoff, name="mask_max_cutoff"),
        tf.greater(distances[:, :, 0], 0.0),
    )

    distances = tf.ragged.boolean_mask(distances, mask, name="ragged_mask_distances")
    pair_types = tf.ragged.boolean_mask(pair_types, mask, name="ragged_mask_pair_types")
    return distances, pair_types


def get_pair_types(types, n_types):
    """input: int tensor of shape (None, 1)
    int of the total number of elements"""

    # Determine pair_type using the following scheme:
    # Example using 3 atom types:
    #     A B C
    #   A 0
    #   B 1 3
    #   C 2 4 5
    # A bond of type A-C would therefore be assigned the integer 2.
    # First figure out the column we are in (essentially just sorting the
    # two involved pair types)
    min_ij = tf.math.minimum(types[:, tf.newaxis], types[tf.newaxis, :])
    # TODO: is the second line really needed here? Should be generalized to
    # asymmetric functions anyway
    return (
        n_types * min_ij
        - (min_ij * (min_ij - 1)) // 2
        + tf.abs(tf.subtract(types[:, tf.newaxis], types[tf.newaxis, :]))
    )


def get_distance_matrix(xyzs, diagonal=0.0):
    n = tf.shape(xyzs, out_type=tf.int64)[0]
    r_vec = tf.expand_dims(xyzs, 0) - tf.expand_dims(xyzs, 1)
    distances = tf.sqrt(
        tf.reduce_sum(r_vec**2, axis=-1, name="sum_distances"),
        name="distance_computation",
    )
    distances = tf.expand_dims(
        tf.linalg.set_diag(distances, diagonal * tf.ones(n, dtype=xyzs.dtype)),
        axis=-1,
    )
    return distances


def get_distance_matrix_and_compact_derivative(xyzs, diagonal=0.0):
    """
    Importantly, this only returns the "compact" form of the derivative, i.e,
    $$
    \\frac{(x_{j,\\xi} - x_{i,\\xi})}{r_{ij}}
    $$
    To get the full 4 dimensional matrix the 3 dimensional gradient obtained here
    has to be multiplied by a sum of kronecker deltas:
    $$
    \\frac{\\partial r_{ij}}{x_{k,\\xi}} = \\frac{(x_{j,\\xi} - x_{i,\\xi})}{r_{ij}}
    (\\delta_{jk} - \\delta_{ik})
    $$

    Example:
    >>> xyzs = tf.constant([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    >>> _, derivative = get_distance_matrix_and_derivative(xyzs)
    >>> derivative.shape
    TensorShape([2, 2, 3])
    >>> full_derivative = derivative[:, :, tf.newaxis, :] * (
    ...    tf.eye(2)[tf.newaxis, :, :, tf.newaxis]
    ...    - tf.eye(2)[:, tf.newaxis, :, tf.newaxis]
    ... )
    >>> full_derivative.shape
    TensorShape([2, 2, 2, 3])
    """
    n = tf.shape(xyzs, out_type=tf.int64)[0]
    r_vec = tf.expand_dims(xyzs, 0) - tf.expand_dims(xyzs, 1)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1))
    distances = tf.expand_dims(
        tf.linalg.set_diag(distances, diagonal * tf.ones(n, dtype=xyzs.dtype)),
        axis=-1,
    )
    gradient = tf.math.divide_no_nan(r_vec, distances)

    return distances, gradient


def get_distance_matrix_and_derivative(xyzs, diagonal=0.0):
    n = tf.shape(xyzs, out_type=tf.int64)[0]
    r_vec = tf.expand_dims(xyzs, 0) - tf.expand_dims(xyzs, 1)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1))
    distances = tf.expand_dims(
        tf.linalg.set_diag(distances, diagonal * tf.ones(n, dtype=xyzs.dtype)),
        axis=-1,
    )

    # The full gradient is of shape (n, n, n, 3) indexed as d r_{i,j} / d x_{k,l}
    # and calculated as r_vec_{i,j,l} (kron_{j,k} - kron_{i,k})
    gradient = tf.math.divide_no_nan(r_vec, distances)[:, :, tf.newaxis, :] * (
        tf.eye(n, dtype=xyzs.dtype)[tf.newaxis, :, :, tf.newaxis]
        - tf.eye(n, dtype=xyzs.dtype)[:, tf.newaxis, :, tf.newaxis]
    )

    return distances, gradient
