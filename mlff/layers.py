import tensorflow as tf


class PolynomialCutoffFunction(tf.keras.layers.Layer):

    def __init__(self, pair_type, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1), name='cut_a_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(a))
        self.b = self.add_weight(
            shape=(1), name='cut_b_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(b))

    @tf.function
    def call(self, r):
        r_scaled = (r - self.a)/(self.b - self.a)
        result = tf.where(
            tf.less(r, self.b),
            1. - 10. * r_scaled**3 + 15. * r_scaled**4 - 6. * r_scaled**5,
            tf.zeros_like(r))
        return tf.where(tf.less_equal(r, self.a), tf.ones_like(r), result)


class PolynomialCutoffFunctionMask(tf.keras.layers.Layer):

    def __init__(self, pair_type, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1), name='cut_a_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(a))
        self.b = self.add_weight(
            shape=(1), name='cut_b_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(b))

    @tf.function
    def call(self, r):
        result = tf.where(
            tf.less_equal(r, self.a), tf.ones_like(r), tf.zeros_like(r))
        cond = tf.logical_and(tf.greater(r, self.a), tf.less(r, self.b))
        idx = tf.where(cond)
        r_scaled = tf.boolean_mask((r - self.a)/(self.b - self.a), cond)
        updates = 1. - 10. * r_scaled**3 + 15. * r_scaled**4 - 6. * r_scaled**5
        result = tf.tensor_scatter_nd_update(result, idx, updates)
        return result


class SmoothCutoffFunction(tf.keras.layers.Layer):

    def __init__(self, pair_type, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.a = self.add_weight(
            shape=(1), name='cut_a_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(a))
        self.b = self.add_weight(
            shape=(1), name='cut_b_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(b))

    @tf.function
    def call(self, r):
        condition = tf.logical_and(tf.greater(r, self.a), tf.less(r, self.b))
        # In order to avoid evaluation of the exponential function outside
        # of the scope of the cutoff_function an additional tf.where is used
        r_save = tf.where(
            condition, r, 0.5*(self.a + self.b)*tf.ones_like(r))
        result = tf.where(
            condition,
            1./(1. + tf.exp(((self.a - self.b)*(2*r_save - self.a - self.b)) /
                            ((r_save - self.a)*(r_save - self.b)))),
            tf.zeros_like(r))
        return tf.where(tf.less_equal(r, self.a), tf.ones_like(r), result)


class OffsetLayer(tf.keras.layers.Layer):

    def __init__(self, type, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.offset = self.add_weight(
            shape=(1), name='%s_offset' % type, trainable=trainable,
            initializer=tf.zeros_initializer())

    @tf.function
    def call(self, inp):
        return self.offset*tf.ones_like(inp)


class InputNormalization(tf.keras.layers.Layer):
    """ Computes (r/r0 - 1). This is done in a separate layer in order
    to share the r0 weight.
    """
    def __init__(self, pair_type, r0=2.7, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.r0 = self.add_weight(
            shape=(1), name='r0_' + pair_type, trainable=trainable,
            initializer=tf.constant_initializer(r0))

    @tf.function
    def call(self, r):
        return r/self.r0 - 1.0


class BornMayer(tf.keras.layers.Layer):

    def __init__(self, pair_type, A=0.2, p=9.2, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.A = self.add_weight(
            shape=(1), name='A_' + pair_type,
            initializer=tf.constant_initializer(A))
        self.p = self.add_weight(
            shape=(1), name='p_' + pair_type,
            initializer=tf.constant_initializer(p))
        self._supports_ragged_inputs = True

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, r_normalized):
        """r_normalized.shape = (None,)"""
        return self.A*tf.exp(-self.p*r_normalized)


class RhoExp(tf.keras.layers.Layer):

    def __init__(self, pair_type, xi=1.6, q=3.5, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.xi = self.add_weight(
            shape=(1), name='xi_' + pair_type,
            initializer=tf.constant_initializer(xi))
        self.q = self.add_weight(
            shape=(1), name='q_' + pair_type,
            initializer=tf.constant_initializer(q))
        self._supports_ragged_inputs = True

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, r_normalized):
        """r_normalized.shape = (None,)"""
        return self.xi*tf.exp(-self.q*r_normalized)


class RhoTwoExp(tf.keras.layers.Layer):

    def __init__(self, pair_type, xi_1=1.6, q_1=3.5,
                 xi_2=0.8, q_2=1.0, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.xi_1 = self.add_weight(
            shape=(1), name='xi_1_' + pair_type,
            initializer=tf.constant_initializer(xi_1))
        self.q_1 = self.add_weight(
            shape=(1), name='q_1_' + pair_type,
            initializer=tf.constant_initializer(q_1))
        self.xi_2 = self.add_weight(
            shape=(1), name='xi_2_' + pair_type,
            initializer=tf.constant_initializer(xi_2))
        self.q_2 = self.add_weight(
            shape=(1), name='q_2_' + pair_type,
            initializer=tf.constant_initializer(q_2))
        self._supports_ragged_inputs = True

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, r_normalized):
        """r_normalized.shape = (None,)

        The output of this function will be squared. Therefore to avoid a
        binomial mixing term the sqrt of the sum of squares is returned"""
        return tf.sqrt(self.xi_1**2*tf.exp(-2*self.q_1*r_normalized)
                       + self.xi_2**2*tf.exp(-2*self.q_2*r_normalized))


class NNRho(tf.keras.layers.Layer):

    def __init__(self, pair_type, layers=[20, 20], reg=None, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.dense_layers = []
        if reg:
            reg = tf.keras.regularizers.L2(l2=reg)
        for n in layers:
            self.dense_layers.append(tf.keras.layers.Dense(
                n, activation='tanh', kernel_regularizer=reg))
        # Last layer is linear
        self.dense_layers.append(tf.keras.layers.Dense(1))

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, r_normalized):
        """r_normalized.shape = (None,)"""
        nn_results = self.dense_layers[0](
            tf.expand_dims(r_normalized, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return tf.squeeze(nn_results, axis=-1)


class NNRhoExp(tf.keras.layers.Layer):

    def __init__(self, pair_type, layers=[20, 20], reg=None, **kwargs):
        super().__init__(**kwargs)
        self.pair_type = pair_type
        self.dense_layers = []
        if reg:
            reg = tf.keras.regularizers.L2(l2=reg)
        for n in layers:
            self.dense_layers.append(tf.keras.layers.Dense(
                n, activation='tanh', kernel_regularizer=reg))
        # Last layer is linear
        self.dense_layers.append(tf.keras.layers.Dense(1))

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, r_normalized):
        """r_normalized.shape = (None,)"""
        nn_results = self.dense_layers[0](
            tf.expand_dims(r_normalized, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return tf.exp(tf.squeeze(nn_results, axis=-1))


class PairInteraction(tf.keras.layers.Layer):
    """Normalizes the input and feeds it to a pair potential"""

    def __init__(self, input_normalization,
                 pair_interaction, cutoff_function, **kwargs):
        super().__init__(**kwargs)
        self.input_normalization = input_normalization
        self.pair_interaction = pair_interaction
        self.cutoff_function = cutoff_function

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, r):
        return (self.pair_interaction(self.input_normalization(r))
                * self.cutoff_function(r))


class SqrtEmbedding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, rho):
        """rho.shape = (None,)"""
        return -tf.math.sqrt(rho)


class ExtendedEmbedding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c0 = self.add_weight(
            shape=(1), name='c0',
            initializer=tf.constant_initializer(1.))
        self.c1 = self.add_weight(
            shape=(1), name='c1',
            initializer=tf.constant_initializer(0.001))

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, rho):
        """rho.shape = (None,)"""
        return -tf.math.sqrt(rho)*(self.c0 + self.c1*rho)


class NNSqrtEmbedding(tf.keras.layers.Layer):

    def __init__(self, layers=[20, 20], reg=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = []
        if reg:
            reg = tf.keras.regularizers.L2(l2=reg)
        for n in layers:
            self.dense_layers.append(tf.keras.layers.Dense(
                n, activation='tanh', kernel_regularizer=reg))
        # Last layer is linear and has a bias value of one
        self.dense_layers.append(tf.keras.layers.Dense(
            1, bias_initializer='ones',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3))
        )

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.keras.backend.floatx()),))
    def call(self, rho):
        """rho.shape = (None,)"""
        nn_results = self.dense_layers[0](tf.expand_dims(rho, axis=-1))
        for layer in self.dense_layers[1:]:
            nn_results = layer(nn_results)
        return -tf.math.sqrt(rho)*tf.squeeze(nn_results, axis=-1)


class AtomicNeuralNetwork(tf.keras.layers.Layer):

    def __init__(self, layers=[20, 20], reg=None,
                 offset_trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.offset_trainable = offset_trainable
        self.dense_layers = []
        if reg:
            reg = tf.keras.regularizers.L2(l2=reg)
        for n in layers:
            self.dense_layers.append(tf.keras.layers.Dense(
                n, activation='tanh', kernel_regularizer=reg))
        # Last layer is linear
        if offset_trainable:
            self.dense_layers.append(tf.keras.layers.Dense(1))
        else:
            self.dense_layers.append(tf.keras.layers.Dense(1, use_bias=False))

    @tf.function
    def call(self, Gs):
        """Gs.shape = (None, num_Gs)"""
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
        """Gs.shape = (None, num_Gs)"""
        return (Gs - self.Gs_min)/(self.Gs_max - self.Gs_min) - 1
