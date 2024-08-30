import numpy as np
from mleam.layers import CubicSpline


def test_cubic_spline(atol=1e-6):
    spline = CubicSpline()
    spline.nodes = np.array([5.0, 3.0, 2.0, 1.0])
    spline.coefficients = np.array([0.00906157, -0.22241211, 1.1993566, -6.094036])

    r_test = np.linspace(0.5, 4, 11).reshape(-1, 1)
    r_ref = np.array(
        [
            0.63661957,
            0.24074936,
            -0.18581057,
            -0.19665897,
            -0.02487797,
            0.09462216,
            0.11103271,
            0.07803873,
            0.04451948,
            0.02229485,
            0.00906157,
        ]
    ).reshape(-1, 1)

    np.testing.assert_allclose(spline(r_test), r_ref, atol=atol)
