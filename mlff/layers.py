import tensorflow as tf

class PolynomialCutoffFunction(tf.keras.layers.Layer):
    
    def __init__(self, types, a=5.0, b=7.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.a = self.add_weight(
            shape=(1), name='cut_a_' + types, trainable=trainable,
            initializer=tf.constant_initializer(a))
        self.b = self.add_weight(
            shape=(1), name='cut_b_' + types, trainable=trainable,
            initializer=tf.constant_initializer(b))    
    
    @tf.function
    def call(self, r):
        r_scaled = (r - self.a)/(self.b - self.a)
        result = tf.where(
            tf.logical_and(tf.greater(r, self.a), tf.less(r, self.b)), 
            1.0 - 10.0 * r_scaled**3 + 15 * r_scaled**4 - 6 * r_scaled**5, tf.zeros_like(r))
        return tf.where(tf.less_equal(r, self.a), tf.ones_like(r), result)
        

class InputNormalization(tf.keras.layers.Layer):
    """ Computes (r/r0 - 1). This is done in a separate layer in order
    to share the r0 weight.
    """
    def __init__(self, types, r0=2.7, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.r0 = self.add_weight(
            shape=(1), name='r0_' + types, trainable=trainable,
            initializer=tf.constant_initializer(r0))
    
    @tf.function
    def call(self, r):
        return r/self.r0 - 1

class BornMayer(tf.keras.layers.Layer):
    
    def __init__(self, types, A=0.2, p=9.2, **kwargs):
        super().__init__(**kwargs)
        self.A = self.add_weight(
            shape=(1), name='A_' + types,
            initializer=tf.constant_initializer(A))
        self.p = self.add_weight(
            shape=(1), name='p_' + types,
            initializer=tf.constant_initializer(p))
        self._supports_ragged_inputs = True
        
    @tf.function
    def call(self, r_normalized):
        return self.A*tf.exp(-self.p*r_normalized)
    
class RhoExp(tf.keras.layers.Layer):
    
    def __init__(self, types, xi=1.6, q=3.5, **kwargs):
        super().__init__(**kwargs)
        self.xi = self.add_weight(
            shape=(1), name='xi_' + types,
            initializer=tf.constant_initializer(xi))
        self.q = self.add_weight(
            shape=(1), name='q_' + types,
            initializer=tf.constant_initializer(q))
        self._supports_ragged_inputs = True
        
    @tf.function
    def call(self, r_normalized):
        return self.xi**2*tf.exp(-2*self.q*r_normalized)
    
class PairInteraction(tf.keras.layers.Layer):
    """Multiplies a pair-wise interaction with the cutoff function"""
    
    def __init__(self, input_normalization, 
                 pair_interaction, cutoff_function, **kwargs):
        super().__init__(**kwargs)
        self.input_normalization = input_normalization
        self.pair_interaction = pair_interaction
        self.cutoff_function = cutoff_function
       
    @tf.function
    def call(self, r):
        return (self.pair_interaction(self.input_normalization(r))
                * self.cutoff_function(r))
    
class SqrtEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @tf.function
    def call(self, rho):
        return -tf.math.sqrt(rho)
