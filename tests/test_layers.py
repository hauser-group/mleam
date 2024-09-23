import pytest
import numpy as np
from mleam.layers import CubicSpline, NaturalCubicSpline, ClampedCubicSpline
from scipy.interpolate import CubicSpline as ScipyCubicSpline


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


@pytest.mark.parametrize(
    "layer, kwargs, bc_type",
    [
        (NaturalCubicSpline, {}, ((2, 0.0), (2, 0.0))),
        (ClampedCubicSpline, {"dy": (0.0, 0.0)}, ((1, 0.0), (1, 0.0))),
        (ClampedCubicSpline, {"dy": (-0.1, 0.2)}, ((1, -0.1), (1, 0.2))),
    ],
)
def test_cubic_spline_vs_scipy(layer, kwargs, bc_type, atol=1e-6):
    x_fit = np.array([0, 0.8, 2.1, 3.0, 3.5, 4.0])

    def target_fun(x):
        return np.sinc(x)

    scipy_spline = ScipyCubicSpline(x_fit, target_fun(x_fit), bc_type=bc_type)

    x_test = np.linspace(-0.1, 4.1, 101)

    test_spline = layer(x_fit, target_fun(x_fit), **kwargs)

    np.testing.assert_allclose(
        test_spline(x_test.reshape(-1, 1)).numpy().flatten(),
        scipy_spline(x_test),
        atol=atol,
    )
