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

    @tf.function
    def call(self, r):
        r_scaled = (r - self.a) / (self.b - self.a)
        result = tf.where(
            tf.less(r, self.b),
            1.0 - 10.0 * r_scaled**3 + 15.0 * r_scaled**4 - 6.0 * r_scaled**5,
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

    @tf.function
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

    @tf.function
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

    @tf.function
    def call(self, inp):
        return self.offset * tf.ones_like(inp)


class InputNormalization(tf.keras.layers.Layer):
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

    @tf.function
    def call(self, r):
        return r / self.r0 - 1.0


class PairInteraction(tf.keras.layers.Layer):
    """Calls a pair interaction and multiplies it be the cutoff function"""

    def __init__(self, pair_interaction, cutoff_function, **kwargs):
        super().__init__(**kwargs)
        self.pair_interaction = pair_interaction
        self.cutoff_function = cutoff_function

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        return self.pair_interaction(r) * self.cutoff_function(r)


class NormalizedPairInteraction(PairInteraction):
    """Normalizes the input and feeds it to a pair potential"""

    def __init__(self, input_normalization, **kwargs):
        super().__init__(**kwargs)
        self.input_normalization = input_normalization

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        return self.pair_interaction(
            self.input_normalization(r)
        ) * self.cutoff_function(r)


class CubicSpline(tf.keras.layers.Layer):
    """Base class that only defines the call functionality"""

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        # r = (None, 1)
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


class PairPhi(tf.keras.layers.Layer):
    input_norm = InputNormType.NONE


class PairPhiScaledInput(PairPhi):
    input_norm = InputNormType.SCALED


class PairRho(tf.keras.layers.Layer):
    input_norm = InputNormType.NONE


class PairRhoScaledInput(PairRho):
    input_norm = InputNormType.SCALED


class BornMayer(PairPhiScaledInput):
    def __init__(self, pair_type, A=0.2, p=9.2, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.A = self.add_weight(
            shape=(1), name="A_" + pair_type, initializer=tf.constant_initializer(A)
        )
        self.p = self.add_weight(
            shape=(1), name="p_" + pair_type, initializer=tf.constant_initializer(p)
        )
        self._supports_ragged_inputs = True

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        return 2 * self.A * tf.exp(-self.p * r_normalized)


class SuttonChenPhi(PairPhi):
    def __init__(self, pair_type: str, c: float = 3.0, n: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.c = self.add_weight(
            shape=(1), name="c_" + pair_type, initializer=tf.constant_initializer(c)
        )
        self.n = self.add_weight(
            shape=(1),
            name="n_" + pair_type,
            initializer=tf.constant_initializer(n),
            trainable=False,
        )
        self._supports_ragged_inputs = True

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        # r = (None, 1)
        return (self.c / r) ** self.n


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
        super().__init__(**kwargs)
        self.pair_type = pair_type
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        # r = (None, 1)
        return tf.where(
            r <= self.c,
            (r - self.c) ** 2 * (self.c0 + self.c1 * r + self.c2 * r**2),
            tf.zeros_like(r),
        )


class CubicSplinePhi(PairPhi, CubicSpline):
    def __init__(self, pair_type, r_k, a_k, **kwargs):
        super().__init__(**kwargs)
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


class MorsePhi(PairPhiScaledInput):
    def __init__(self, pair_type: str, D: float = 1, a: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.D = self.add_weight(
            shape=(1), name=f"D_{pair_type}", initializer=tf.constant_initializer(D)
        )
        self.a = self.add_weight(
            shape=(1), name=f"a_{pair_type}", initializer=tf.constant_initializer(a)
        )

    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        return self.D * (
            tf.exp(-2 * self.a * r_normalized) - 2 * tf.exp(-self.a * r_normalized)
        )


class PhiDoubleExp(PairPhiScaledInput):
    def __init__(self, pair_type, A_1=1.6, p_1=3.5, A_2=0.8, p_2=1.0, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        return self.A * tf.exp(-self.p_1 * r_normalized) + self.A_2 * tf.exp(
            -self.p_2 * r_normalized
        )


class RhoExp(PairRhoScaledInput):
    def __init__(self, pair_type, xi=1.6, q=3.5, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.xi = self.add_weight(
            shape=(1), name="xi_" + pair_type, initializer=tf.constant_initializer(xi)
        )
        self.q = self.add_weight(
            shape=(1), name="q_" + pair_type, initializer=tf.constant_initializer(q)
        )
        self._supports_ragged_inputs = True

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        return self.xi**2 * tf.exp(-2 * self.q * r_normalized)


class SuttonChenRho(PairRho):
    def __init__(self, pair_type: str, a: float = 3.0, m: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1), name="a_" + pair_type, initializer=tf.constant_initializer(a)
        )
        self.m = self.add_weight(
            shape=(1),
            name="m_" + pair_type,
            initializer=tf.constant_initializer(m),
            trainable=False,
        )
        self._supports_ragged_inputs = True

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        # r = (None, 1)
        return (self.a / r) ** self.m


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
        super().__init__(**kwargs)
        self.pair_type = pair_type
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        # r = (None, 1)
        return tf.where(
            r <= self.d,
            self.A * ((r - self.d) ** 2 + self.beta * (r - self.d) ** 3 / self.d),
            tf.zeros_like(r),
        )


class CubicSplineRho(PairRho, CubicSpline):
    def __init__(self, pair_type, R_k, A_k, **kwargs):
        super().__init__(**kwargs)
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


class RhoDoubleExp(PairRhoScaledInput):
    def __init__(self, pair_type, xi_1=1.6, q_1=3.5, xi_2=0.8, q_2=1.0, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        return self.xi_1**2 * tf.exp(
            -2 * self.q_1 * r_normalized
        ) + self.xi_2**2 * tf.exp(-2 * self.q_2 * r_normalized)


def VoterRho(PairRho):
    def __init__(self, pair_type: str, xi: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r):
        # r.shape = (None, 1)
        return (
            self.xi**2
            * r**6
            * (tf.exp(-self.beta * r) + 2**9 * tf.exp(-2 * self.beta * r))
        )


class NNRho(PairRhoScaledInput):
    def __init__(self, pair_type, layers=[20, 20], regularization=None, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        nn_results = self.dense_layers[0](tf.expand_dims(r_normalized, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return tf.squeeze(nn_results, axis=-1)


class NNRhoExp(PairRhoScaledInput):
    def __init__(self, pair_type, layers=[20, 20], regularization=None, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, r_normalized):
        # r_normalized.shape = (None, 1)
        nn_results = self.dense_layers[0](tf.expand_dims(r_normalized, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return tf.exp(tf.squeeze(nn_results, axis=-1))


class SqrtEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.keras.backend.floatx()),
        )
    )
    def call(self, rho):
        # rho.shape = (None, 1)
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

    @tf.function
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

    @tf.function
    def call(self, Gs):
        # Gs.shape = (None, num_Gs)
        return (Gs - self.Gs_min) / (self.Gs_max - self.Gs_min) - 1
