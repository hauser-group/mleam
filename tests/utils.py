import numpy as np


def rotation_matrix(n, alpha):
    """
    n: (3,) rotation axis. Input vector is normalized in the function.
    alpha: angle in radians.

    output: (3x3) rotation matrix
    """
    # From https://de.wikipedia.org/wiki/Drehmatrix#Drehmatrizen_des_Raumes
    n = n / np.linalg.norm(n)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    foo = 1 - cos_a
    R = np.array(
        [
            [
                n[0] * n[0] * foo + cos_a,
                n[0] * n[1] * foo - n[2] * sin_a,
                n[0] * n[2] * foo + n[1] * sin_a,
            ],
            [
                n[1] * n[0] * foo + n[2] * sin_a,
                n[1] * n[1] * foo + cos_a,
                n[1] * n[2] * foo - n[0] * sin_a,
            ],
            [
                n[2] * n[0] * foo - n[1] * sin_a,
                n[2] * n[1] * foo + n[0] * sin_a,
                n[2] * n[2] * foo + cos_a,
            ],
        ]
    )
    return R


def derive_scalar_wrt_array(fun, x0, dx=1e-4):
    """fun: scalar function,
    x0: np.array at which point the function should be derived"""
    res = np.zeros_like(x0)
    for i in range(x0.size):
        ind = np.unravel_index(i, x0.shape)
        displacement = np.zeros_like(x0)
        displacement[ind] = dx

        f_plus = fun(x0 + displacement)
        f_minus = fun(x0 - displacement)
        res[ind] = (f_plus - f_minus) / (2 * dx)
    return res


def derive_array_wrt_array(fun, x0, dx=1e-4):
    """"""
    f0 = fun(x0)
    res = np.zeros(f0.shape + x0.shape)
    for i in range(x0.size):
        ind = np.unravel_index(i, x0.shape)
        displacement = np.zeros_like(x0)
        displacement[ind] = dx

        f_plus = fun(x0 + displacement)
        f_minus = fun(x0 - displacement)
        # None is used as replacement for :
        res[(Ellipsis,) + ind] = (f_plus - f_minus) / (2 * dx)
    return res
