import pytest
import numpy as np
from mleam.layers import (
    CubicSpline,
    NaturalCubicSpline,
    ClampedCubicSpline,
    ClampedCubicHermiteSpline,
    QuinticHermiteSpline,
)
from scipy.interpolate import CubicSpline as ScipyCubicSpline
from scipy.interpolate import make_interp_spline


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
        (ClampedCubicHermiteSpline, {"dy": (0.0, 0.0)}, ((1, 0.0), (1, 0.0))),
        (ClampedCubicHermiteSpline, {"dy": (-0.1, 0.2)}, ((1, -0.1), (1, 0.2))),
    ],
)
def test_cubic_spline_vs_scipy(layer, kwargs, bc_type, atol=1e-6):
    x_fit = np.array([0, 0.8, 2.0, 2.1, 3.0, 3.5, 4.0])

    def target_fun(x):
        return np.sinc(x)

    scipy_spline = ScipyCubicSpline(x_fit, target_fun(x_fit), bc_type=bc_type)

    x_test = np.linspace(-0.1, 4.1, 101)

    test_spline = layer(x_fit, y=target_fun(x_fit), **kwargs)

    np.testing.assert_allclose(
        test_spline(x_test.reshape(-1, 1)).numpy().flatten(),
        scipy_spline(x_test),
        atol=atol,
    )


def test_quintic_spline_vs_scipy(atol=1e-5):
    x_fit = np.array([0, 0.8, 2.0, 2.1, 3.0, 3.5, 4.0])

    def target_fun(x):
        return np.sinc(x)

    scipy_spline = make_interp_spline(x_fit, target_fun(x_fit), k=5)

    x_test = np.linspace(-0.1, 4.1, 101)

    test_spline = QuinticHermiteSpline(
        x_fit,
        y=scipy_spline(x_fit),
        dy=scipy_spline.derivative(1)(x_fit),
        ddy=scipy_spline.derivative(2)(x_fit),
    )

    np.testing.assert_allclose(
        test_spline(x_test.reshape(-1, 1)).numpy().flatten(),
        scipy_spline(x_test),
        atol=atol,
    )


@pytest.mark.parametrize(
    "x, y, dy, ddy, ref",
    [
        (
            np.array([0.5, 1.0, 1.6]),
            np.array([1.0, 2.0, 3.2]),
            np.array([2.0, 2.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            0.0,
        ),
        (  # Following tests are evaluated with WolframAlpha
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            120 / 7,
        ),
        (  # Test invariance to x spacing
            np.array([0.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            120 / 7,
        ),
        (  # Test scaling with respect to y
            np.array([0.0, 1.0]),
            np.array([0.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            4 * 120 / 7,
        ),
        (  # Has to be symmetric
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            120 / 7,
        ),
        (  # Now for given derivatives
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            192 / 35,
        ),
        (  # Again, should be symmetric
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            192 / 35,
        ),
        (  # Finally, given second derivative
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 1.0]),
            3 / 35,
        ),
        (  # And again, test for symmetry
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            3 / 35,
        ),
    ],
)
def test_quintic_hermite_spline_regularization(x, y, dy, ddy, ref, atol=1e-5):
    layer = QuinticHermiteSpline(x=x, y=y, dy=dy, ddy=ddy)

    np.testing.assert_allclose(layer.losses[0], ref, atol=atol)
