import numpy as np
import tensorflow as tf


def preprocess_inputs(xyzs, types, n_types, diagonal=0.0):
    distances, dr_dx = get_distance_matrix_and_derivative(xyzs, diagonal=diagonal)
    pair_types = get_pair_types(types, n_types)

    return types, pair_types, distances, dr_dx


def preprocess_inputs_ragged(xyzs, types, n_types, cutoff=np.inf):
    # remove dummy atoms, this also converts everything to ragged tensors
    mask = tf.greater_equal(types[..., 0], 0)
    xyzs = tf.ragged.boolean_mask(xyzs, mask)
    types = tf.ragged.boolean_mask(types, mask)

    types, pair_types, distances, derivative = preprocess_inputs(xyzs, types, n_types)

    # This is essentially a range function that works for both ragged and dense inputs
    j_indices = tf.math.cumsum(
        tf.ones_like(xyzs[..., 0], dtype=tf.int64),
        axis=-1,
        exclusive=True,
    )
    # This repeats the j_indices along the second to last axis
    j_indices = tf.expand_dims(j_indices, axis=-2) * tf.expand_dims(
        tf.ones_like(j_indices, dtype=j_indices.dtype), axis=-1
    )

    mask = tf.logical_and(
        tf.less_equal(distances[..., 0], cutoff, name="mask_max_cutoff"),
        tf.greater(distances[..., 0], 0.0),
    )

    # TODO: Ideally this should not rag along the first atomic index unless
    # necessary...
    distances = tf.ragged.boolean_mask(distances, mask)
    pair_types = tf.ragged.boolean_mask(pair_types, mask)
    derivative = tf.ragged.boolean_mask(derivative, mask)
    j_indices = tf.ragged.boolean_mask(j_indices, mask)
    return types, pair_types, distances, derivative, j_indices


def preprocess_inputs_no_force(xyzs, types, n_types, diagonal=0.0):
    pair_types = get_pair_types(types, n_types)
    distances = get_distance_matrix(xyzs, diagonal=diagonal)
    return types, pair_types, distances


def preprocess_inputs_ragged_no_force(xyzs, types, n_types, cutoff=np.inf):
    # remove dummy atoms, this also converts everything to ragged tensors
    mask = tf.greater_equal(types[..., 0], 0)
    xyzs = tf.ragged.boolean_mask(xyzs, mask)
    types = tf.ragged.boolean_mask(types, mask)

    types, pair_types, distances = preprocess_inputs_no_force(xyzs, types, n_types)
    mask = tf.logical_and(
        tf.less_equal(distances[..., 0], cutoff, name="mask_max_cutoff"),
        tf.greater(distances[..., 0], 0.0),
    )

    distances = tf.ragged.boolean_mask(distances, mask, name="ragged_mask_distances")
    pair_types = tf.ragged.boolean_mask(pair_types, mask, name="ragged_mask_pair_types")
    return types, pair_types, distances


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
    pair_types = (
        n_types * min_ij
        - (min_ij * (min_ij - 1)) // 2
        + tf.abs(tf.expand_dims(types, axis=-2) - tf.expand_dims(types, axis=-3))
    )
    pair_types = tf.ragged.map_flat_values(
        tf.where,
        min_ij < 0,
        -1 * tf.ones_like(pair_types, dtype=pair_types.dtype),
        pair_types,
    )
    return pair_types


def get_distance_matrix(xyzs, diagonal=0.0):
    r_vec = tf.expand_dims(xyzs, -3) - tf.expand_dims(xyzs, -2)
    distances = tf.sqrt(
        tf.reduce_sum(r_vec**2, axis=-1, keepdims=True, name="sum_distances"),
        name="distance_computation",
    )
    distances = tf.ragged.map_flat_values(
        tf.where, distances == 0, diagonal * tf.ones_like(distances), distances
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
    distances = get_distance_matrix(xyzs, diagonal=diagonal)
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
