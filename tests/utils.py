import numpy as np


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
