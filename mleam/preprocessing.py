import tensorflow as tf


def distances_and_pair_types(xyzs, types, n_types, diagonal=0.0):
    distances, gradient = get_distance_matrix_and_derivative(xyzs, diagonal=diagonal)
    pair_types = get_pair_types(types, n_types)

    return distances, pair_types, gradient


def ragged_distances_and_pair_types(xyzs, types, n_types, cutoff=10.0):
    distances, pair_types, derivative = distances_and_pair_types(xyzs, types, n_types)
    tf.print(distances.shape, pair_types.shape, derivative.shape)
    mask = tf.logical_and(
        tf.less_equal(distances[:, :, 0], cutoff, name="mask_max_cutoff"),
        tf.greater(distances[:, :, 0], 0.0),
    )

    distances = tf.ragged.boolean_mask(distances, mask)
    pair_types = tf.ragged.boolean_mask(pair_types, mask)
    derivative = tf.ragged.boolean_mask(derivative, mask)
    return distances, pair_types, derivative


def distances_and_pair_types_no_grad(xyzs, types, n_types, diagonal=0.0):
    pair_types = get_pair_types(types, n_types)
    distances = get_distance_matrix(xyzs)
    return distances, pair_types


def ragged_distances_and_pair_types_no_grad(xyzs, types, n_types, cutoff=10.0):
    distances, pair_types = distances_and_pair_types_no_grad(xyzs, types, n_types)
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
    min_ij = tf.math.minimum(tf.expand_dims(types, axis=-2), tf.expand_dims(types, -3))
    # TODO: is the second line really needed here? Should be generalized to
    # asymmetric functions anyway
    return (
        n_types * min_ij
        - (min_ij * (min_ij - 1)) // 2
        + tf.abs(tf.expand_dims(types, axis=-2) - tf.expand_dims(types, axis=-3))
    )


def get_distance_matrix(xyzs, diagonal=0.0):
    r_vec = tf.expand_dims(xyzs, -3) - tf.expand_dims(xyzs, -2)
    distances = tf.sqrt(
        tf.reduce_sum(r_vec**2, axis=-1, name="sum_distances"),
        name="distance_computation",
    )
    if isinstance(xyzs, tf.RaggedTensor):
        diagonals = tf.ragged.stack(
            [diagonal * tf.eye(n, dtype=xyzs.dtype) for n in xyzs.row_lengths()]
        )
    else:
        diagonals = diagonal * tf.eye(
            tf.shape(xyzs)[-2], dtype=xyzs.dtype, batch_shape=tf.shape(xyzs)[:-2]
        )
    distances = tf.expand_dims(
        distances + diagonals,
        axis=-1,
    )
    return distances


def get_distance_matrix_and_derivative(xyzs, diagonal=0.0):
    """
    Importantly, this only returns the "compact" 3 dimensional form of the derivative:
    $$
    \\frac{(x_{j,\\xi} - x_{i,\\xi})}{r_{ij}}
    $$
    To get the full 4 dimensional matrix the 3 dimensional derivative obtained here
    has to be multiplied by a difference of two kronecker deltas:
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
    r_vec = tf.expand_dims(xyzs, -3) - tf.expand_dims(xyzs, -2)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1))
    if isinstance(xyzs, tf.RaggedTensor):
        diagonals = tf.ragged.stack(
            [diagonal * tf.eye(n, dtype=xyzs.dtype) for n in xyzs.row_lengths()]
        )
    else:
        diagonals = diagonal * tf.eye(
            tf.shape(xyzs)[-2], dtype=xyzs.dtype, batch_shape=tf.shape(xyzs)[:-2]
        )
    distances = tf.expand_dims(
        distances + diagonals,
        axis=-1,
    )
    derivative = tf.math.divide_no_nan(r_vec, distances)

    return distances, derivative


def get_distance_matrix_and_full_derivative(xyzs, diagonal=0.0):
    n = tf.shape(xyzs, out_type=tf.int64)[0]
    distances, derivative = get_distance_matrix_and_derivative(xyzs, diagonal=diagonal)

    # The full derivative is of shape (n, n, n, 3) indexed as d r_{i,j} / d x_{k,l}
    # and calculated as r_vec_{i,j,l} (kron_{j,k} - kron_{i,k})
    derivative = tf.expand_dims(derivative, axis=-2) * (
        tf.expand_dims(tf.expand_dims(tf.eye(n, dtype=xyzs.dtype), axis=-3), axis=-1)
        - tf.expand_dims(tf.expand_dims(tf.eye(n, dtype=xyzs.dtype), axis=-2), axis=-1)
    )

    return distances, derivative
