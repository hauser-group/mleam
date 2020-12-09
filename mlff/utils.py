import numpy as np
import tensorflow as tf


@tf.function
def distances_and_pair_types(xyz, types, n_types, cutoff=10.0):
    n = tf.shape(xyz, out_type=tf.int64)[0]
    r_vec = tf.expand_dims(xyz, 0) - tf.expand_dims(xyz, 1)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1, keepdims=True))
    mask = tf.logical_and(tf.logical_not(tf.eye(n, dtype=tf.bool)),
                          tf.less_equal(distances[:, :, 0], cutoff))

    min_ij = tf.math.minimum(types[:, tf.newaxis], types[tf.newaxis, :])
    pair_types = (
        n_types*min_ij - (min_ij*(min_ij-1))//2
        + tf.abs(tf.subtract(types[:, tf.newaxis], types[tf.newaxis, :])))

    # Should be possible in a single step:
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n))], axis=2)
    gradient = tf.scatter_nd(indices, -r_vec/distances, (n, n, n, 3))
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n))], axis=2)
    gradient = tf.tensor_scatter_nd_add(gradient, indices, r_vec/distances)

    distances = tf.ragged.boolean_mask(distances, mask)
    pair_types = tf.ragged.boolean_mask(pair_types, mask)
    gradient = tf.ragged.boolean_mask(gradient, mask)
    # convert into ragged_rank=2
    row_lens = gradient.row_lengths()
    gradient = tf.RaggedTensor.from_nested_row_lengths(
        tf.reshape(gradient.flat_values, (-1, 3)),
        (row_lens, n*tf.ones(tf.reduce_sum(row_lens), dtype=tf.int64)))
    return distances, pair_types, gradient


@tf.function
def get_pair_type(types, n):
    """input: int tensor of shape (None, 1)
              int of the total number of elements"""

    min_ij = tf.math.minimum(types[:, tf.newaxis], types[tf.newaxis, :])
    mask = tf.logical_not(tf.eye(tf.shape(types)[0], dtype=tf.bool))
    return tf.ragged.boolean_mask(
        n*min_ij - (min_ij*(min_ij-1))//2
        + tf.abs(tf.subtract(types[:, tf.newaxis], types[tf.newaxis, :])),
        mask)


@tf.function
def distances_with_gradient(xyz):
    n = tf.shape(xyz, out_type=tf.int64)[0]
    mask = tf.logical_not(tf.eye(n, dtype=tf.bool))
    r_vec = tf.expand_dims(xyz, 0) - tf.expand_dims(xyz, 1)
    distances = tf.sqrt(tf.reduce_sum(r_vec**2, axis=-1, keepdims=True))

    # Should be possible in a single step:
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n))], axis=2)
    gradient = tf.scatter_nd(indices, -r_vec/distances, (n, n, n, 3))
    indices = tf.stack(
        [tf.broadcast_to(tf.expand_dims(tf.range(n), 1), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n)),
         tf.broadcast_to(tf.expand_dims(tf.range(n), 0), (n, n))], axis=2)
    gradient = tf.tensor_scatter_nd_add(gradient, indices, r_vec/distances)

    # Previously used numpy version

    # gradient = np.zeros((n, n, n, 3))
    # gradient[np.expand_dims(np.arange(n), 1),
    #          np.expand_dims(np.arange(n), 0),
    #          np.expand_dims(np.arange(n), 1), :] = -r_vec/distances
    # gradient[np.expand_dims(np.arange(n), 1),
    #          np.expand_dims(np.arange(n), 0),
    #          np.expand_dims(np.arange(n), 0), :] = r_vec/distances

    # Slicing into the masked tensor would be significanlty more difficult
    # but could be achieved similar to this n=4 example
    # gradient[np.expand_dims(np.arange(4), 1),
    #          np.expand_dims(np.arange(3), 0),
    #         [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], :])

    # gradient = tf.RaggedTensor.from_nested_row_lengths(
    #    gradient[mask2], ((n-1)*tf.ones(n, dtype=tf.int64),
    #                       n*tf.ones(n*(n-1), dtype=tf.int64)))
    gradient = tf.RaggedTensor.from_nested_row_lengths(
        tf.reshape(gradient[mask], (-1, 3)),
        ((n-1)*tf.ones(n, dtype=tf.int64), n*tf.ones(n*(n-1), dtype=tf.int64)))
    return tf.ragged.boolean_mask(distances, mask), gradient


class RaggedMeanSquaredError(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.squared_difference(y_pred, y_true))


class ConstantExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, constant_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.constant_steps = constant_steps

    @tf.function
    def __call__(self, step):
        return tf.where(step <= self.constant_steps,
                        self.initial_learning_rate,
                        super().__call__(step - self.constant_steps))

    def get_config(self):
        config = super().get_config()
        config.update({'constant_steps': self.constant_steps})
        return config


def opt_fun_factory(model, loss, train_x, train_y, val_x=None, val_y=None,
                    save_path=None):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable
            parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t.
        # parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model, train_x, train_y)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(
            loss_value, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        grads = tf.dynamic_stitch(idx, grads)

        if f.val:
            val_loss = loss(model, val_x, val_y)
            if val_loss < f.best_val:
                tf.py_function(
                    lambda epoch, val_loss:
                        model.save_weights(save_path.format(epoch=epoch,
                                                            val_loss=val_loss)
                                           ),
                    inp=[f.iter, val_loss], Tout=[])
                f.best_val.assign(val_loss)
        else:
            val_loss = 0.0

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "train_loss:", loss_value,
                 'val_loss:', val_loss)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[(loss_value, val_loss)], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []
    f.val = val_x is not None and val_y is not None
    f.best_val = tf.Variable(1e10)

    return f


def pretrain_rho(model, params, max_iter=50000, tol=1e-5,
                 r_vec=np.reshape(np.linspace(0, 6.2, 101, dtype=np.float32),
                                  (-1, 1)),
                 optimizer=tf.keras.optimizers.Adam()):
    from mlff.layers import (InputNormalization, PolynomialCutoffFunction,
                             RhoExp, PairInteraction)

    exp_rhos = {}
    nn_rhos = {}
    pair_types = ['NiNi', 'NiPt', 'PtPt']
    for pair_type in pair_types:
        normalized_input = InputNormalization(
            pair_type, r0=params[('r0', pair_type)], trainable=False)
        cutoff_function = PolynomialCutoffFunction(
            pair_type, a=params[('cut_a', pair_type)],
            b=params[('cut_b', pair_type)])
        pair_potential = RhoExp(
            pair_type, xi=params[('xi', pair_type)],
            q=params[('q', pair_type)])

        exp_rhos[pair_type] = PairInteraction(
            normalized_input, pair_potential, cutoff_function)
        nn_rhos[pair_type] = model.layers[
            [l.name for l in model.layers].index('%s-rho' % pair_type)]

    @tf.function
    def rho_loss():
        return tf.reduce_sum(tf.reduce_mean(
            [tf.math.squared_difference(
                 nn_rhos[pair_type](r_vec), exp_rhos[pair_type](r_vec))
             for pair_type in pair_types], axis=1))

    for i in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            loss = rho_loss()
            if loss < tol:
                return True
        grads = tape.gradient(
            loss, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if i % 500 == 0:
            print(i, loss.numpy())
    return False
