import numpy as np
from utils import rotation_matrix, derive_scalar_wrt_array, derive_array_wrt_array


def test_rotmat_orthogonality(atol=1e-10):
    R = rotation_matrix(np.random.randn(3), np.random.randn(1)[0])
    # R * R.T == R.T * R
    np.testing.assert_allclose(R.dot(R.T), R.T.dot(R), atol=atol)
    # since R.T == R^-1 -> R * R.T == I
    np.testing.assert_allclose(R.dot(R.T), np.eye(3), atol=atol)


def test_rotmat_determinant():
    R = rotation_matrix(np.random.randn(3), np.random.randn(1)[0])
    np.testing.assert_allclose(np.linalg.det(R), 1)


def test_numdiff_quadratic_function(atol=1e-6):
    def fun(x):
        return np.sum(x**2)

    x0 = np.random.randn(4, 3)
    ref_deriv = 2 * x0
    num_deriv = derive_scalar_wrt_array(fun, x0)

    np.testing.assert_allclose(num_deriv, ref_deriv, atol=atol)


def test_numdiff_exponential_function(atol=1e-6):
    def fun(x):
        return np.exp(0.23 * np.sum(x))

    x0 = np.random.randn(2, 6, 5)
    ref_deriv = np.exp(0.23 * np.sum(x0)) * 0.23
    num_deriv = derive_scalar_wrt_array(fun, x0)

    np.testing.assert_allclose(num_deriv, ref_deriv, atol=atol)


def test_numdiff_independent_variables(atol=1e-6):
    def fun(x):
        return x**2

    x0 = np.random.randn(2, 5)
    ref_deriv = np.zeros(x0.shape + x0.shape)
    ind = np.unravel_index(np.arange(x0.size), x0.shape)
    ref_deriv[ind + ind] = 2 * x0.flatten()
    num_deriv = derive_array_wrt_array(fun, x0)

    np.testing.assert_allclose(num_deriv, ref_deriv, atol=atol)


def test_numdiff_x_dot_x(atol=1e-6):
    """x: array(n x m)
    (x[0, :] * x[0, :].T, x[0, :] * x[1, :].T, x[0, :] * x[2, :].T
    x[1, :] * x[0, :].T, x[1, :] * x[1, :].T, x[1, :] * x[2, :].T
    x[2, :] * x[0, :].T, x[2, :] * x[1, :].T, x[2, :] * x[2, :].T)
    """

    def fun(x):
        return x.dot(x.T)

    x0 = np.random.randn(4, 3)
    ref_deriv = np.zeros((4, 4, 4, 3))
    for i in range(4):
        for j in range(4):
            ref_deriv[i, j, i, :] += x0[j, :]
            ref_deriv[i, j, j, :] += x0[i, :]
    num_deriv = derive_array_wrt_array(fun, x0)
    np.testing.assert_allclose(num_deriv, ref_deriv, atol=atol)
