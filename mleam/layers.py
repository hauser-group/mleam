import numpy as np
import tensorflow as tf
from mleam.constants import InputNormType


class PolynomialCutoffFunction(tf.keras.layers.Layer):
    def __init__(self, pair_type, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1),
            name="cut_a_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(a),
        )
        self.b = self.add_weight(
            shape=(1),
            name="cut_b_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(b),
        )

    def call(self, r):
        r_scaled = (r - self.a) / (self.b - self.a)
        result = tf.where(
            tf.less(r, self.b),
            1.0
            + ((15 - 6 * r_scaled) * r_scaled - 10) * r_scaled * r_scaled * r_scaled,
            tf.zeros_like(r),
        )
        return tf.where(tf.less_equal(r, self.a), tf.ones_like(r), result)


class PolynomialCutoffFunctionMask(tf.keras.layers.Layer):
    def __init__(self, pair_type, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1),
            name="cut_a_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(a),
        )
        self.b = self.add_weight(
            shape=(1),
            name="cut_b_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(b),
        )

    def call(self, r):
        result = tf.where(tf.less_equal(r, self.a), tf.ones_like(r), tf.zeros_like(r))
        cond = tf.logical_and(tf.greater(r, self.a), tf.less(r, self.b))
        idx = tf.where(cond)
        r_scaled = tf.boolean_mask((r - self.a) / (self.b - self.a), cond)
        updates = 1.0 - 10.0 * r_scaled**3 + 15.0 * r_scaled**4 - 6.0 * r_scaled**5
        result = tf.tensor_scatter_nd_update(result, idx, updates)
        return result


class SmoothCutoffFunction(tf.keras.layers.Layer):
    def __init__(self, pair_type, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1),
            name="cut_a_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(a),
        )
        self.b = self.add_weight(
            shape=(1),
            name="cut_b_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(b),
        )

    def call(self, r):
        condition = tf.logical_and(tf.greater(r, self.a), tf.less(r, self.b))
        # In order to avoid evaluation of the exponential function outside
        # of the scope of the cutoff_function an additional tf.where is used
        r_save = tf.where(condition, r, 0.5 * (self.a + self.b) * tf.ones_like(r))
        result = tf.where(
            condition,
            1.0
            / (
                1.0
                + tf.exp(
                    ((self.a - self.b) * (2 * r_save - self.a - self.b))
                    / ((r_save - self.a) * (r_save - self.b))
                )
            ),
            tf.zeros_like(r),
        )
        return tf.where(tf.less_equal(r, self.a), tf.ones_like(r), result)


class OffsetLayer(tf.keras.layers.Layer):
    def __init__(self, type, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.offset = self.add_weight(
            shape=(1),
            name="%s_offset" % type,
            trainable=trainable,
            initializer=tf.zeros_initializer(),
        )

    def call(self, inp):
        return self.offset * tf.ones_like(inp)


class InputNormalizationAndShift(tf.keras.layers.Layer):
    """Computes (r/r0 - 1). This is done in a separate layer in order
    to share the r0 weight.
    """

    def __init__(self, pair_type, r0=2.7, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.r0 = self.add_weight(
            shape=(1),
            name="r0_" + pair_type,
            trainable=trainable,
            initializer=tf.constant_initializer(r0),
        )

    def call(self, r):
        return r / self.r0 - 1.0


class PairInteraction(tf.keras.layers.Layer):
    """Calls a pair interaction and multiplies it be the cutoff function"""

    def __init__(self, pair_interaction, cutoff_function, **kwargs):
        super().__init__(**kwargs)
        self.pair_interaction = pair_interaction
        self.cutoff_function = cutoff_function

    def call(self, r):
        return self.pair_interaction(r) * self.cutoff_function(r)


class NormalizedPairInteraction(PairInteraction):
    """Normalizes the input and feeds it to a pair potential"""

    def __init__(self, input_normalization, **kwargs):
        super().__init__(**kwargs)
        self.input_normalization = input_normalization

    def call(self, r):
        return self.pair_interaction(
            self.input_normalization(r)
        ) * self.cutoff_function(r)


class CubicSpline(tf.keras.layers.Layer):
    """Base class that only defines the call functionality"""

    def call(self, r):
        delta_r = self.nodes[tf.newaxis, :] - r
        return tf.reduce_sum(
            tf.where(
                delta_r > 0.0,
                self.coefficients[tf.newaxis, :] * delta_r**3,
                tf.zeros_like(delta_r),
            ),
            axis=-1,
            keepdims=True,
        )


class BaseCubicSpline(tf.keras.layers.Layer):
    def __init__(self, x, y=None, x_name="x", y_name="y", **kwargs):
        """
        Initialize with x (data points) and y (TensorFlow Variables).
        x: 1D numpy array of shape (n,) for the interpolation points
        y: optional, 1D numpy array of shape (n,) for the values at the interpolation points
        """
        super().__init__(**kwargs)
        assert (x[1:] - x[:-1] > 0.0).all()
        if y is not None:
            assert len(x) == len(y)
        # NOTE: To make the x trainable we would need to also enforce their ordering somehow
        self.x = self.add_weight(
            shape=x.shape,
            name=x_name,
            trainable=False,
            initializer=tf.constant_initializer(x),
        )
        self.y = self.add_weight(
            shape=x.shape,
            name=y_name,
            trainable=True,
            initializer=tf.zeros_initializer()
            if y is None
            else tf.constant_initializer(y),
        )

        # Compute the distances between the x points (h_i = x_{i+1} - x_i)
        # NOTE: no equivalent for np.diff in tensorflow
        self.h = self.x[1:] - self.x[:-1]

    def compute_coefficients(self):
        """
        Compute the coefficients of the cubic spline.
        """
        # Compute the matrix system to solve for the second derivatives
        A = self._construct_tridiagonal_matrix()
        b = self._construct_rhs()

        # Solve for the second derivatives (M)
        M = tf.linalg.tridiagonal_solve(A, b, diagonals_format="compact")

        return M

    def call(self, x_new):
        """
        Evaluate the spline values at r points using the coefficients.
        """
        x_new = tf.squeeze(x_new)
        M = self.compute_coefficients()

        # Find the interval where each x_new belongs
        idx = tf.searchsorted(self.x, x_new) - 1
        # To enable extrapolation use the edge polynomials whenever
        # the index is < 0 or > n - 1.
        idx = tf.clip_by_value(idx, 0, tf.shape(self.x)[0] - 2)

        # Get corresponding h values for each x_new
        h = tf.gather(self.h, idx)

        # Compute spline components
        a = (tf.gather(self.x, idx + 1) - x_new) / h
        b = (x_new - tf.gather(self.x, idx)) / h

        # Compute spline value using the natural cubic spline formula
        spline_value = (
            a * tf.gather(self.y, idx)
            + b * tf.gather(self.y, idx + 1)
            + ((a**3 - a) * tf.gather(M, idx) + (b**3 - b) * tf.gather(M, idx + 1))
            * (h**2)
            / 6
        )

        return tf.reshape(spline_value, (-1, 1))


class CubicHermiteSpline(tf.keras.layers.Layer):
    def __init__(
        self,
        x,
        y=None,
        x_name="x",
        y_name="y",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (x[1:] - x[:-1] > 0.0).all()
        if y is not None:
            assert len(x) == len(y)
        # NOTE: To make the x trainable we would need to also enforce their ordering somehow
        self.x = self.add_weight(
            shape=x.shape,
            name=x_name,
            trainable=False,
            initializer=tf.constant_initializer(x),
        )
        self.y = self.add_weight(
            shape=x.shape,
            name=y_name,
            trainable=True,
            initializer=tf.zeros_initializer()
            if y is None
            else tf.constant_initializer(y),
        )

    def call(self, x_new):
        """
        Evaluate the spline values at x_new using the coefficients.
        """
        x_new = tf.squeeze(x_new)
        y, dy = self.compute_coefficients()

        # Find the interval where each x_new belongs
        idx = tf.searchsorted(self.x, x_new) - 1
        # To enable extrapolation use the edge polynomials whenever
        # the index is < 0 or > n - 1.
        idx = tf.clip_by_value(idx, 0, tf.shape(self.x)[0] - 2)

        # Get corresponding h values for each x_new
        h = tf.gather(self.x[1:] - self.x[:-1], idx)

        # Compute spline components
        t = (x_new - tf.gather(self.x, idx)) / h

        y_left = tf.gather(y, idx)
        y_right = tf.gather(y, idx + 1)
        dy_left = tf.gather(dy, idx)
        dy_right = tf.gather(dy, idx + 1)

        # Compute spline value using the cubic spline formula
        spline_value = (
            (2 * (y_left - y_right) + h * (dy_left + dy_right)) * t**3
            + (3 * (-y_left + y_right) + h * (-2 * dy_left - dy_right)) * t**2
            + h * dy_left * t
            + y_left
        )

        return tf.reshape(spline_value, (-1, 1))


class ClampedCubicHermiteSpline(CubicHermiteSpline):
    def __init__(
        self,
        *args,
        dy=[0, 0],
        dy_trainable=True,
        dy_name="dy",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dy = self.add_weight(
            shape=(2,),
            name=dy_name,
            trainable=dy_trainable,
            initializer=tf.zeros_initializer()
            if dy is None
            else tf.constant_initializer(dy),
        )

    def _construct_tridiagonal_matrix(self):
        # Last element of superdiag is ignored
        h = self.x[1:] - self.x[:-1]
        superdiag = tf.concat([tf.zeros((1,)), h[:-1], tf.zeros((1,))], axis=0)
        maindiag = tf.concat(
            [tf.ones((1,)), 2 * (h[:-1] + h[1:]), tf.ones((1,))],
            axis=0,
        )
        # First element of subdiag is ignored
        subdiag = tf.concat([tf.zeros((1,)), h[1:], tf.zeros((1,))], axis=0)
        return tf.stack([superdiag, maindiag, subdiag], axis=0)

    def _construct_rhs(self):
        """
        Construct the right-hand side vector b for the clamped cubic spline system.
        """
        y = self.y
        h = self.x[1:] - self.x[:-1]
        b = tf.concat(
            [
                self.dy[:1],
                3
                * (
                    h[1:] * (y[1:-1] - y[:-2]) / h[:-1]
                    + h[:-1] * (y[2:] - y[1:-1]) / h[1:]
                ),
                self.dy[-1:],
            ],
            axis=0,
        )

        return b

    def compute_coefficients(self):
        """
        Compute the coefficients of the cubic spline.
        """
        # Compute the matrix system to solve for the second derivatives
        A = self._construct_tridiagonal_matrix()
        b = self._construct_rhs()

        return self.y, tf.linalg.tridiagonal_solve(A, b, diagonals_format="compact")


class NaturalQuinticSpline(tf.keras.layers.Layer):
    def __init__(
        self, x, y=None, dy=None, x_name="x", y_name="y", dy_name="dy", **kwargs
    ):
        super().__init__(**kwargs)
        assert (x[1:] - x[:-1] > 0.0).all()
        if y is not None:
            assert len(x) == len(y)
        if dy is not None:
            assert len(x) == len(dy)

        self.x = self.add_weight(
            shape=x.shape,
            name=x_name,
            trainable=False,
            initializer=tf.constant_initializer(x),
        )
        self.y = self.add_weight(
            shape=x.shape,
            name=y_name,
            trainable=True,
            initializer=tf.zeros_initializer()
            if y is None
            else tf.constant_initializer(y),
        )
        self.dy = self.add_weight(
            shape=x.shape,
            name=dy_name,
            trainable=True,
            initializer=tf.zeros_initializer()
            if dy is None
            else tf.constant_initializer(dy),
        )
        self.h = self.x[1:] - self.x[:-1]

    def _construct_tridiagonal_matrix(self):
        # Last element of superdiag is ignored
        superdiag = tf.concat([tf.zeros((1,)), self.h[:-1], tf.zeros((1,))], axis=0)
        maindiag = tf.concat(
            [-3 * tf.ones((1,)), -3 * (self.h[:-1] + self.h[1:]), -3 * tf.ones((1,))],
            axis=0,
        )
        # First element of subdiag is ignored
        subdiag = tf.concat([tf.zeros((1,)), self.h[1:], tf.zeros((1,))], axis=0)
        return tf.stack([superdiag, maindiag, subdiag], axis=0)

    def _construct_rhs(self):
        """
        Construct the right-hand side vector b for the clamped cubic spline system.
        """
        y = self.y
        dy = self.dy
        h = self.h
        b = 20 * tf.concat(
            [
                -(y[1:2] - y[:1]) / h[:1] ** 2
                + (3 * dy[:1] + 2 * dy[1:2]) / (5 * h[:1]),
                h[:-1]
                * h[1:]
                * (
                    (y[1:-1] - y[:-2]) / h[:-1] ** 3
                    - (y[2:] - y[1:-1]) / h[1:] ** 3
                    - (2 * dy[:-2] + 3 * dy[1:-1]) / (5 * h[:-1] ** 2)
                    + (3 * dy[1:-1] + 2 * dy[2:]) / (5 * h[1:] ** 2)
                ),
                (y[-1:] - y[-2:-1]) / h[-1:] ** 2
                - (3 * dy[-1:] + 2 * dy[-2:-1]) / (5 * h[-1:]),
            ],
            axis=0,
        )

        return b

    def compute_coefficients(self):
        """
        Compute the coefficients of the cubic spline.
        """
        # Compute the matrix system to solve for the second derivatives
        A = self._construct_tridiagonal_matrix()
        b = self._construct_rhs()

        return tf.linalg.tridiagonal_solve(A, b, diagonals_format="compact")

    def call(self, x_new):
        """
        Evaluate the spline values at x_new using the coefficients.
        """
        x_new = tf.squeeze(x_new)
        ddy = self.compute_coefficients()

        # Find the interval where each x_new belongs
        idx = tf.searchsorted(self.x, x_new) - 1
        # To enable extrapolation use the edge polynomials whenever
        # the index is < 0 or > n - 1.
        idx = tf.clip_by_value(idx, 0, tf.shape(self.x)[0] - 2)

        # Get corresponding h values for each x_new
        h = tf.gather(self.h, idx)

        # Compute spline components
        t = (x_new - tf.gather(self.x, idx)) / h

        y_left = tf.gather(self.y, idx)
        y_right = tf.gather(self.y, idx + 1)
        dy_left = tf.gather(self.dy, idx)
        dy_right = tf.gather(self.dy, idx + 1)
        ddy_left = tf.gather(ddy, idx)
        ddy_right = tf.gather(ddy, idx + 1)

        # Compute spline value using the quintic spline formula
        spline_value = (
            (
                6 * (y_right - y_left)
                - 3 * h * (dy_left + dy_right)
                + 0.5 * h**2 * (ddy_right - ddy_left)
            )
            * t**5
            + (
                -15 * (y_right - y_left)
                + h * (8 * dy_left + 7 * dy_right)
                - 0.5 * h**2 * (2 * ddy_right - 3 * ddy_left)
            )
            * t**4
            + (
                10 * (y_right - y_left)
                - 2 * h * (3 * dy_left + 2 * dy_right)
                + 0.5 * h**2 * (ddy_right - 3 * ddy_left)
            )
            * t**3
            + 0.5 * h**2 * ddy_left * t**2
            + h * dy_left * t
            + y_left
        )

        return tf.reshape(spline_value, (-1, 1))


class QuinticHermiteSpline(tf.keras.layers.Layer):
    def __init__(
        self,
        x,
        y=None,
        dy=None,
        ddy=None,
        x_name="x",
        y_name="y",
        dy_name="dy",
        ddy_name="ddy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (x[1:] - x[:-1] > 0.0).all()
        if y is not None:
            assert len(x) == len(y)
        if dy is not None:
            assert len(x) == len(dy)

        self.x = self.add_weight(
            shape=x.shape,
            name=x_name,
            trainable=False,
            initializer=tf.constant_initializer(x),
        )

        y = np.zeros(len(x)) if y is None else y
        dy = np.zeros(len(x)) if dy is None else dy
        ddy = np.zeros(len(x)) if ddy is None else ddy

        init_function_values = np.stack([y, dy, ddy], axis=0)
        self.function_values = self.add_weight(
            shape=(3, len(x)),
            name=y_name,
            trainable=True,
            initializer=tf.constant_initializer(init_function_values),
            regularizer=self._curvature_regularizer,
        )

    def _get_coefficients(self):
        return self.function_values

    def _curvature_regularizer(self, x):
        h = self.x[1:] - self.x[:-1]
        y, dy, ddy = x

        reg = tf.reduce_sum(
            h**4 / 35 * (3 * ddy[:-1] ** 2 + ddy[:-1] * ddy[1:] + 3 * ddy[1:] ** 2)
            + 2
            * h**3
            / 35
            * (
                11 * ddy[:-1] * dy[:-1]
                + 4 * ddy[:-1] * dy[1:]
                - 4 * ddy[1:] * dy[:-1]
                - 11 * ddy[1:] * dy[1:]
            )
            + 6
            * h**2
            / 35
            * (
                5
                * (
                    ddy[:-1] * y[:-1]
                    - ddy[:-1] * y[1:]
                    - ddy[1:] * y[:-1]
                    + ddy[1:] * y[1:]
                )
                + 32 * dy[:-1] ** 2
                + 36 * dy[:-1] * dy[1:]
                + 32 * dy[1:] ** 2
            )
            + 120 * h / 7 * (dy[:-1] + dy[1:]) * (y[:-1] - y[1:])
            + 120 / 7 * (y[:-1] - y[1:]) ** 2
        )
        return reg

    def call(self, x_new):
        """
        Evaluate the spline values at x_new using the coefficients.
        """
        x_new = tf.squeeze(x_new)
        y, dy, ddy = self._get_coefficients()

        # Find the interval where each x_new belongs
        idx = tf.searchsorted(self.x, x_new) - 1
        # To enable extrapolation use the edge polynomials whenever
        # the index is < 0 or > n - 1.
        idx = tf.clip_by_value(idx, 0, tf.shape(self.x)[0] - 2)

        # Get corresponding h values for each x_new
        h = tf.gather(self.x[1:] - self.x[:-1], idx)

        # Compute spline components
        t = (x_new - tf.gather(self.x, idx)) / h

        y_left = tf.gather(y, idx)
        y_right = tf.gather(y, idx + 1)
        dy_left = tf.gather(dy, idx)
        dy_right = tf.gather(dy, idx + 1)
        ddy_left = tf.gather(ddy, idx)
        ddy_right = tf.gather(ddy, idx + 1)

        # Compute spline value using the quintic spline formula
        spline_value = (
            (
                6 * (y_right - y_left)
                - 3 * h * (dy_left + dy_right)
                + 0.5 * h**2 * (ddy_right - ddy_left)
            )
            * t**5
            + (
                -15 * (y_right - y_left)
                + h * (8 * dy_left + 7 * dy_right)
                - 0.5 * h**2 * (2 * ddy_right - 3 * ddy_left)
            )
            * t**4
            + (
                10 * (y_right - y_left)
                - 2 * h * (3 * dy_left + 2 * dy_right)
                + 0.5 * h**2 * (ddy_right - 3 * ddy_left)
            )
            * t**3
            + 0.5 * h**2 * ddy_left * t**2
            + h * dy_left * t
            + y_left
        )

        return tf.reshape(spline_value, (-1, 1))


class NaturalCubicSpline(BaseCubicSpline):
    def _construct_tridiagonal_matrix(self):
        """
        Construct the tridiagonal matrix for the natural cubic spline system.
        """
        # Last element of superdiag is ignored
        superdiag = tf.concat([tf.zeros((1,)), self.h[1:], tf.zeros((1,))], axis=0)
        maindiag = tf.concat(
            [tf.ones((1,)), 2 * (self.h[:-1] + self.h[1:]), tf.ones((1,))],
            axis=0,
        )
        # First element of subdiag is ignored
        subdiag = tf.concat([tf.zeros((1,)), self.h[:-1], tf.zeros((1,))], axis=0)
        return tf.stack([superdiag, maindiag, subdiag], axis=0)

    def _construct_rhs(self):
        """
        Construct the right-hand side vector b for the natural cubic spline system.
        """
        # The zeros at the start and end correspond to the natural spline boundary
        # conditions
        return tf.concat(
            [
                tf.zeros((1,)),
                6
                * (
                    (self.y[2:] - self.y[1:-1]) / self.h[1:]
                    - (self.y[1:-1] - self.y[:-2]) / self.h[:-1]
                ),
                tf.zeros((1,)),
            ],
            axis=0,
        )


class ClampedCubicSpline(BaseCubicSpline):
    def __init__(self, x, y=None, dy=[0, 0], dy_trainable=True, dy_name="dy", **kwargs):
        super().__init__(x, y=y, **kwargs)
        self.dy = self.add_weight(
            shape=(2,),
            name=dy_name,
            trainable=dy_trainable,
            initializer=tf.zeros_initializer()
            if dy is None
            else tf.constant_initializer(dy),
        )

    def _construct_tridiagonal_matrix(self):
        """
        Construct the tridiagonal matrix for the clamped cubic spline system.
        """
        # Last element of superdiag is ignored
        superdiag = tf.concat([self.h, tf.zeros((1,))], axis=0)
        maindiag = tf.concat(
            [2 * self.h[:1], 2 * (self.h[:-1] + self.h[1:]), 2 * self.h[-1:]],
            axis=0,
        )
        # First element of subdiag is ignored
        subdiag = tf.concat([tf.zeros((1,)), self.h], axis=0)
        return tf.stack([superdiag, maindiag, subdiag], axis=0)

    def _construct_rhs(self):
        """
        Construct the right-hand side vector b for the clamped cubic spline system.
        """
        y = self.y
        h = self.h
        b = tf.concat(
            [
                6 * ((y[1] - y[0]) / h[0] - self.dy[:1]),
                6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1]),
                6 * (self.dy[-1:] - (y[-1] - y[-2]) / h[-1]),
            ],
            axis=0,
        )

        return b


class PairPhi(tf.keras.layers.Layer):
    input_norm = InputNormType.NONE

    def __init__(self, pair_type: str, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type


class PairPhiScaledShiftedInput(PairPhi):
    input_norm = InputNormType.SCALED_SHIFTED


class PairRho(tf.keras.layers.Layer):
    input_norm = InputNormType.NONE

    def __init__(self, pair_type: str, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type


class PairRhoScaledShiftedInput(PairRho):
    input_norm = InputNormType.SCALED_SHIFTED


class BornMayer(PairPhiScaledShiftedInput):
    def __init__(self, pair_type, A=0.2, p=9.2, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.A = self.add_weight(
            shape=(1), name="A_" + pair_type, initializer=tf.constant_initializer(A)
        )
        self.p = self.add_weight(
            shape=(1), name="p_" + pair_type, initializer=tf.constant_initializer(p)
        )
        self._supports_ragged_inputs = True

    def call(self, r_normalized):
        return 2 * self.A * tf.exp(-self.p * r_normalized)


class SuttonChenPhi(PairPhi):
    def __init__(
        self, pair_type: str, c: float = 3.0, n: int = 6, n_trainable=False, **kwargs
    ):
        super().__init__(pair_type, **kwargs)
        self.c = self.add_weight(
            shape=(1), name="c_" + pair_type, initializer=tf.constant_initializer(c)
        )
        self.n = self.add_weight(
            shape=(1),
            name="n_" + pair_type,
            initializer=tf.constant_initializer(n),
            trainable=n_trainable,
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return (self.c / r) ** self.n


class DoubleSuttonChenPhi(PairPhi):
    def __init__(
        self,
        pair_type: str,
        c_1: float = 3.0,
        n_1: int = 6,
        c_2: float = 3.0,
        n_2: int = 8,
        **kwargs,
    ):
        super().__init__(pair_type, **kwargs)
        self.c_1 = self.add_weight(
            shape=(1), name=f"c_1_{pair_type}", initializer=tf.constant_initializer(c_1)
        )
        self.c_2 = self.add_weight(
            shape=(1), name=f"c_2_{pair_type}", initializer=tf.constant_initializer(c_2)
        )
        self.n_1 = self.add_weight(
            shape=(1),
            name=f"n_1_{pair_type}",
            initializer=tf.constant_initializer(n_1),
            trainable=False,
        )
        self.n_2 = self.add_weight(
            shape=(1),
            name=f"n_2_{pair_type}",
            initializer=tf.constant_initializer(n_2),
            trainable=False,
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return (self.c_1 / r) ** self.n_1 + (self.c_2 / r) ** self.n_2


class FinnisSinclairPhi(PairPhi):
    def __init__(
        self,
        pair_type: str,
        c: float = 5.0,
        c0: float = 1.0,
        c1: float = 0.0,
        c2: float = 0.0,
        **kwargs,
    ):
        super().__init__(pair_type, **kwargs)
        self.c = self.add_weight(
            shape=(1), name="c_" + pair_type, initializer=tf.constant_initializer(c)
        )
        self.c0 = self.add_weight(
            shape=(1), name="c0_" + pair_type, initializer=tf.constant_initializer(c0)
        )
        self.c1 = self.add_weight(
            shape=(1), name="c1_" + pair_type, initializer=tf.constant_initializer(c1)
        )
        self.c2 = self.add_weight(
            shape=(1), name="c2_" + pair_type, initializer=tf.constant_initializer(c2)
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return tf.where(
            r <= self.c,
            (r - self.c) ** 2 * (self.c0 + self.c1 * r + self.c2 * r**2),
            tf.zeros_like(r),
        )


class CubicSplinePhi(PairPhi, CubicSpline):
    def __init__(self, pair_type, r_k, a_k, **kwargs):
        super().__init__(pair_type, **kwargs)
        assert len(r_k) == len(a_k)
        self.nodes = self.add_weight(
            shape=(len(r_k)),
            name=f"r_k_{pair_type}",
            initializer=tf.constant_initializer(r_k),
            trainable=False,
        )
        self.coefficients = self.add_weight(
            shape=(len(a_k)),
            name=f"a_k_{pair_type}",
            initializer=tf.constant_initializer(a_k),
        )


class MorsePhi(PairPhiScaledShiftedInput):
    def __init__(self, pair_type: str, D: float = 1, a: float = 1, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.D = self.add_weight(
            shape=(1), name=f"D_{pair_type}", initializer=tf.constant_initializer(D)
        )
        self.a = self.add_weight(
            shape=(1), name=f"a_{pair_type}", initializer=tf.constant_initializer(a)
        )

    def call(self, r_normalized):
        return self.D * (
            tf.exp(-2 * self.a * r_normalized) - 2 * tf.exp(-self.a * r_normalized)
        )


class DoubleExpPhi(PairPhiScaledShiftedInput):
    def __init__(self, pair_type, A_1=1.6, p_1=3.5, A_2=0.8, p_2=1.0, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.A_1 = self.add_weight(
            shape=(1), name=f"A_1_{pair_type}", initializer=tf.constant_initializer(A_1)
        )
        self.p_1 = self.add_weight(
            shape=(1), name=f"p_1_{pair_type}", initializer=tf.constant_initializer(p_1)
        )
        self.A_2 = self.add_weight(
            shape=(1), name=f"A_2_{pair_type}", initializer=tf.constant_initializer(A_2)
        )
        self.p_2 = self.add_weight(
            shape=(1), name=f"p_2_{pair_type}", initializer=tf.constant_initializer(p_2)
        )
        self._supports_ragged_inputs = True

    def call(self, r_normalized):
        return self.A_1 * tf.exp(-self.p_1 * r_normalized) + self.A_2 * tf.exp(
            -self.p_2 * r_normalized
        )


class NaturalCubicSplinePhi(PairPhi, NaturalCubicSpline):
    def __init__(self, pair_type, r_k, a_k, **kwargs):
        self.pair_type = pair_type
        NaturalCubicSpline.__init__(
            self,
            x=r_k,
            y=a_k,
            x_name=f"r_k_{pair_type}",
            y_name=f"a_k_{pair_type}",
            **kwargs,
        )


class NaturalQuinticSplinePhi(PairPhi, NaturalQuinticSpline):
    def __init__(self, pair_type, r_k, a_k=None, da_k=None, **kwargs):
        self.pair_type = pair_type
        NaturalQuinticSpline.__init__(
            self,
            x=r_k,
            y=a_k,
            dy=da_k,
            x_name=f"r_k_{pair_type}",
            y_name=f"a_k_{pair_type}",
            dy_name=f"da_k_{pair_type}",
            **kwargs,
        )


class FittedQuinticSplinePhi(PairPhi, QuinticHermiteSpline):
    def __init__(self, pair_type, r_k, a_k=None, da_k=None, dda_k=None, **kwargs):
        self.pair_type = pair_type
        QuinticHermiteSpline.__init__(
            self,
            x=r_k,
            y=a_k,
            dy=da_k,
            ddy=dda_k,
            x_name=f"r_k_{pair_type}",
            y_name=f"a_k_{pair_type}",
            dy_name=f"da_k_{pair_type}",
            ddy_name=f"dda_k_{pair_type}",
            **kwargs,
        )


class ClampedCubicSplinePhi(PairPhi, ClampedCubicSpline):
    def __init__(self, pair_type, r_k, a_k=None, da_k=None, **kwargs):
        self.pair_type = pair_type
        ClampedCubicSpline.__init__(
            self,
            x=r_k,
            y=a_k,
            dy=da_k,
            x_name=f"r_k_{pair_type}",
            y_name=f"a_k_{pair_type}",
            dy_name=f"da_k_{pair_type}",
            **kwargs,
        )


class ExpRho(PairRhoScaledShiftedInput):
    def __init__(self, pair_type, xi=1.6, q=3.5, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.xi = self.add_weight(
            shape=(1), name="xi_" + pair_type, initializer=tf.constant_initializer(xi)
        )
        self.q = self.add_weight(
            shape=(1), name="q_" + pair_type, initializer=tf.constant_initializer(q)
        )
        self._supports_ragged_inputs = True

    def call(self, r_normalized):
        return self.xi**2 * tf.exp(-2 * self.q * r_normalized)


class SuttonChenRho(PairRho):
    def __init__(
        self, pair_type: str, a: float = 3.0, m: int = 6, m_trainable=False, **kwargs
    ):
        super().__init__(pair_type, **kwargs)
        self.a = self.add_weight(
            shape=(1), name="a_" + pair_type, initializer=tf.constant_initializer(a)
        )
        self.m = self.add_weight(
            shape=(1),
            name="m_" + pair_type,
            initializer=tf.constant_initializer(m),
            trainable=m_trainable,
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return (self.a / r) ** self.m


class DoubleSuttonChenRho(PairRho):
    def __init__(
        self,
        pair_type: str,
        a_1: float = 3.0,
        m_1: int = 6,
        a_2: float = 3.0,
        m_2: int = 8,
        **kwargs,
    ):
        super().__init__(pair_type, **kwargs)
        self.a_1 = self.add_weight(
            shape=(1), name=f"a_1_{pair_type}", initializer=tf.constant_initializer(a_1)
        )
        self.a_2 = self.add_weight(
            shape=(1), name=f"a_2_{pair_type}", initializer=tf.constant_initializer(a_2)
        )
        self.m_1 = self.add_weight(
            shape=(1),
            name=f"m_1_{pair_type}",
            initializer=tf.constant_initializer(m_1),
            trainable=False,
        )
        self.m_2 = self.add_weight(
            shape=(1),
            name=f"m_2_{pair_type}",
            initializer=tf.constant_initializer(m_2),
            trainable=False,
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return (self.a_1 / r) ** self.m_1 + (self.a_2 / r) ** self.m_2


class FinnisSinclairRho(PairRho):
    def __init__(
        self,
        pair_type: str,
        A: float = 1.0,
        d: float = 5.0,
        beta: float = 0.0,
        beta_trainable: bool = True,
        **kwargs,
    ):
        super().__init__(pair_type, **kwargs)
        self.A = self.add_weight(
            shape=(1), name="A_" + pair_type, initializer=tf.constant_initializer(A)
        )
        self.d = self.add_weight(
            shape=(1), name="d_" + pair_type, initializer=tf.constant_initializer(d)
        )
        self.beta = self.add_weight(
            shape=(1),
            name="beta_" + pair_type,
            initializer=tf.constant_initializer(beta),
            trainable=beta_trainable,
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return tf.where(
            r <= self.d,
            self.A * ((r - self.d) ** 2 + self.beta * (r - self.d) ** 3 / self.d),
            tf.zeros_like(r),
        )


class CubicSplineRho(PairRho, CubicSpline):
    def __init__(self, pair_type, R_k, A_k, **kwargs):
        super().__init__(pair_type, **kwargs)
        assert len(R_k) == len(A_k)
        self.nodes = self.add_weight(
            shape=(len(R_k)),
            name=f"R_k_{pair_type}",
            initializer=tf.constant_initializer(R_k),
            trainable=False,
        )
        self.coefficients = self.add_weight(
            shape=(len(A_k)),
            name=f"A_k_{pair_type}",
            initializer=tf.constant_initializer(A_k),
        )


class NaturalCubicSplineRho(PairRho, NaturalCubicSpline):
    def __init__(self, pair_type, R_k, A_k=None, **kwargs):
        self.pair_type = pair_type
        NaturalCubicSpline.__init__(
            self,
            x=R_k,
            y=A_k,
            x_name=f"R_k_{pair_type}",
            y_name=f"A_k_{pair_type}",
            **kwargs,
        )


class NaturalQuinticSplineRho(PairRho, NaturalQuinticSpline):
    def __init__(self, pair_type, R_k, A_k=None, dA_k=None, **kwargs):
        self.pair_type = pair_type
        NaturalQuinticSpline.__init__(
            self,
            x=R_k,
            y=A_k,
            dy=dA_k,
            x_name=f"R_k_{pair_type}",
            y_name=f"A_k_{pair_type}",
            dy_name=f"dA_k_{pair_type}",
            **kwargs,
        )


class FittedQuinticSplineRho(PairRho, QuinticHermiteSpline):
    def __init__(self, pair_type, R_k, A_k=None, dA_k=None, ddA_k=None, **kwargs):
        self.pair_type = pair_type
        QuinticHermiteSpline.__init__(
            self,
            x=R_k,
            y=A_k,
            dy=dA_k,
            ddy=ddA_k,
            x_name=f"R_k_{pair_type}",
            y_name=f"A_k_{pair_type}",
            dy_name=f"dA_k_{pair_type}",
            ddy_name=f"ddA_k_{pair_type}",
            **kwargs,
        )


class ClampedCubicSplineRho(PairRho, ClampedCubicSpline):
    def __init__(self, pair_type, R_k, A_k=None, dA_k=None, **kwargs):
        self.pair_type = pair_type
        ClampedCubicSpline.__init__(
            self,
            x=R_k,
            y=A_k,
            dy=dA_k,
            x_name=f"R_k_{pair_type}",
            y_name=f"A_k_{pair_type}",
            dy_name=f"dA_k_{pair_type}",
            **kwargs,
        )


class DoubleExpRho(PairRhoScaledShiftedInput):
    def __init__(self, pair_type, xi_1=1.6, q_1=3.5, xi_2=0.8, q_2=1.0, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.xi_1 = self.add_weight(
            shape=(1),
            name="xi_1_" + pair_type,
            initializer=tf.constant_initializer(xi_1),
        )
        self.q_1 = self.add_weight(
            shape=(1), name="q_1_" + pair_type, initializer=tf.constant_initializer(q_1)
        )
        self.xi_2 = self.add_weight(
            shape=(1),
            name="xi_2_" + pair_type,
            initializer=tf.constant_initializer(xi_2),
        )
        self.q_2 = self.add_weight(
            shape=(1), name="q_2_" + pair_type, initializer=tf.constant_initializer(q_2)
        )
        self._supports_ragged_inputs = True

    def call(self, r_normalized):
        return self.xi_1**2 * tf.exp(
            -2 * self.q_1 * r_normalized
        ) + self.xi_2**2 * tf.exp(-2 * self.q_2 * r_normalized)


class VoterRho(PairRho):
    def __init__(self, pair_type: str, xi: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.xi = self.add_weight(
            shape=(1),
            name=f"xi_{pair_type}",
            initializer=tf.constant_initializer(xi),
        )
        self.beta = self.add_weight(
            shape=(1),
            name=f"beta_{pair_type}",
            initializer=tf.constant_initializer(beta),
        )
        self._supports_ragged_inputs = True

    def call(self, r):
        return (
            self.xi**2
            * r**6
            * (tf.exp(-self.beta * r) + 2**9 * tf.exp(-2 * self.beta * r))
        )


class NNRho(PairRhoScaledShiftedInput):
    def __init__(self, pair_type, layers=[20, 20], regularization=None, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.dense_layers = []
        if regularization:
            regularization = tf.keras.regularizers.L2(l2=regularization)
        for n in layers:
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    n, activation="tanh", kernel_regularizer=regularization
                )
            )
        # Last layer is linear
        self.dense_layers.append(tf.keras.layers.Dense(1))

    def call(self, r_normalized):
        nn_results = self.dense_layers[0](tf.expand_dims(r_normalized, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return tf.squeeze(nn_results, axis=-1)


class NNRhoExp(PairRhoScaledShiftedInput):
    def __init__(self, pair_type, layers=[20, 20], regularization=None, **kwargs):
        super().__init__(pair_type, **kwargs)
        self.dense_layers = []
        if regularization:
            regularization = tf.keras.regularizers.L2(l2=regularization)
        for n in layers:
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    n, activation="tanh", kernel_regularizer=regularization
                )
            )
        # Last layer is linear
        self.dense_layers.append(tf.keras.layers.Dense(1))

    def call(self, r_normalized):
        nn_results = self.dense_layers[0](tf.expand_dims(r_normalized, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return tf.exp(tf.squeeze(nn_results, axis=-1))


class SqrtEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, rho):
        return -tf.math.sqrt(rho)


class JohnsonEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        atom_type: str,
        F0: float = 0.5,
        eta: float = 0.5,
        F1: float = 0.5,
        zeta: float = 0.5,
        power_law_trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.F0 = self.add_weight(
            shape=(1), name=f"F0_{atom_type}", initializer=tf.constant_initializer(F0)
        )
        self.eta = self.add_weight(
            shape=(1), name=f"eta_{atom_type}", initializer=tf.constant_initializer(eta)
        )
        self.F1 = self.add_weight(
            shape=(1),
            name=f"F1_{atom_type}",
            initializer=tf.constant_initializer(F1),
            trainable=power_law_trainable,
        )
        self.zeta = self.add_weight(
            shape=(1),
            name=f"zeta_{atom_type}",
            initializer=tf.constant_initializer(zeta),
            trainable=power_law_trainable,
        )

    def call(self, rho):
        return (
            -self.F0 * (1 - self.eta * tf.math.log(rho)) * rho**self.eta
            - self.F1 * rho**self.zeta
        )


class ExtendedEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c0 = self.add_weight(
            shape=(1), name="c0", initializer=tf.constant_initializer(1.0)
        )
        self.c1 = self.add_weight(
            shape=(1), name="c1", initializer=tf.constant_initializer(0.001)
        )

    def call(self, rho):
        return -tf.math.sqrt(rho) * (self.c0 + self.c1 * rho)


class ExtendedEmbeddingV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c0 = self.add_weight(
            shape=(1), name="c0", initializer=tf.constant_initializer(0.7)
        )
        self.c1 = self.add_weight(
            shape=(1), name="c1", initializer=tf.constant_initializer(0.05)
        )

    def call(self, rho):
        return -self.c0 * tf.math.sqrt(rho) - self.c1 * rho


class ExtendedEmbeddingV3(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c0 = self.add_weight(
            shape=(1), name="c0", initializer=tf.constant_initializer(0.5)
        )
        self.c1 = self.add_weight(
            shape=(1), name="c1", initializer=tf.constant_initializer(0.5)
        )
        self.c2 = self.add_weight(
            shape=(1), name="c2", initializer=tf.constant_initializer(0.05)
        )

    def call(self, rho):
        return -tf.math.sqrt(rho) * (self.c0 + self.c1 * tf.tanh(self.c2 * rho))


class ExtendedEmbeddingV4(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c0 = self.add_weight(
            shape=(1), name="c0", initializer=tf.constant_initializer(0.5)
        )
        self.c1 = self.add_weight(
            shape=(1), name="c1", initializer=tf.constant_initializer(0.05)
        )

    def call(self, rho):
        return -tf.math.sqrt(rho) * (self.c0 + (1 - self.c0) * tf.tanh(self.c1 * rho))


class NNSqrtEmbedding(tf.keras.layers.Layer):
    def __init__(self, layers=[20, 20], regularization=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = []
        if regularization:
            regularization = tf.keras.regularizers.L2(l2=regularization)
        for n in layers:
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    n, activation="tanh", kernel_regularizer=regularization
                )
            )
        # Last layer is linear and has a bias value of one
        self.dense_layers.append(
            tf.keras.layers.Dense(
                1,
                bias_initializer="ones",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            )
        )

    def call(self, rho):
        nn_results = self.dense_layers[0](rho)
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return -tf.math.sqrt(rho) * nn_results


class AtomicNeuralNetwork(tf.keras.layers.Layer):
    def __init__(
        self, layers=[20, 20], regularization=None, offset_trainable=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.offset_trainable = offset_trainable
        self.dense_layers = []
        if regularization:
            regularization = tf.keras.regularizers.L2(l2=regularization)
        for n in layers:
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    n, activation="tanh", kernel_regularizer=regularization
                )
            )
        # Last layer is linear
        if offset_trainable:
            self.dense_layers.append(tf.keras.layers.Dense(1))
        else:
            self.dense_layers.append(tf.keras.layers.Dense(1, use_bias=False))

    def call(self, Gs):
        # Gs.shape = (None, num_Gs)
        nn_results = self.dense_layers[0](Gs)
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        if self.offset_trainable:
            return nn_results
        offset = self.dense_layers[0](-tf.ones([1, Gs.shape[1]]))
        for layer in self.dense_layers[1:]:
            offset = layer(offset)
        return nn_results - offset


class MinMaxDescriptorNorm(tf.keras.layers.Layer):
    def __init__(self, Gs_min, Gs_max, **kwargs):
        super().__init__(**kwargs)
        self.Gs_min = Gs_min
        self.Gs_max = Gs_max

    def call(self, Gs):
        # Gs.shape = (None, num_Gs)
        return (Gs - self.Gs_min) / (self.Gs_max - self.Gs_min) - 1
