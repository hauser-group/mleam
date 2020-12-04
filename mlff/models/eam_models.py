import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_where_op import where as ragged_where
from itertools import combinations_with_replacement
from mlff.layers import (PairInteraction, PolynomialCutoffFunction,
                         OffsetLayer, InputNormalization, BornMayer,
                         RhoExp, RhoTwoExp, NNRho, NNRhoExp,
                         SqrtEmbedding, ExtendedEmbedding, ExtendedEmbeddingV2,
                         ExtendedEmbeddingV3, NNSqrtEmbedding)
from mlff.utils import distances_and_pair_types


class EAMPotential(tf.keras.Model):
    """Base class for all EAM based potentials"""

    def __init__(self, atom_types, build_forces=False,
                 preprocessed_input=False, cutoff=10.,
                 method='partition_stitch', force_method='old'):
        """
        atom_types: list of str used to construct all the model layers. Note
                                the order is important as it determines the
                                corresponding integer representation that is
                                fed to the model.
        build_forces: boolean determines if the graph for evaluating forces
                              should be constructed.
        preprocessed_input: boolean switches between input of atomic positions
                                    and interatomic distances. Preprocessed
                                    is currently faster as the calculation of
                                    interatomic distances in the model can not
                                    be parallelized.
        cutoff: float if preprocessed_input=False this determines the global
                      cutoff until which interatomic distances are calculated.
        method: str switch for different implementations of the main body
                    options are:
                    - 'partition_stitch' default and probably savest choice
                    - 'where' uses branchless programming can be significantly
                              faster as it is easier to parallelize
                    - 'gather_scatter' old implementation, no longer maintained
        force_method: str switch for debugging, options are:
                    - 'old' default
                    - 'new'
        """
        super().__init__()

        self.atom_types = atom_types
        self.type_dict = {}
        for i, t in enumerate(atom_types):
            self.type_dict[t] = i
        self.atom_pair_types = []
        self.pair_type_dict = {}
        for i, (t1, t2) in enumerate(
                combinations_with_replacement(self.atom_types, 2)):
            t = ''.join([t1, t2])
            self.atom_pair_types.append(t)
            self.pair_type_dict[t] = i
        self.build_forces = build_forces
        self.preprocessed_input = preprocessed_input
        self.method = method
        self.force_method = force_method

        inputs = {'types': tf.keras.Input(shape=(None, 1), ragged=True,
                                          dtype=tf.int32)}
        if self.preprocessed_input:
            inputs['pair_types'] = tf.keras.Input(shape=(None, None, 1),
                                                  ragged=True,
                                                  dtype=inputs['types'].dtype)
            inputs['distances'] = tf.keras.Input(shape=(None, None, 1),
                                                 ragged=True)
            if self.build_forces:
                # [batchsize] x N x (N-1) x N x 3
                inputs['dr_dx'] = tf.keras.Input(shape=(None, None, None, 3),
                                                 ragged=True)
        else:
            self.cutoff = cutoff
            inputs['positions'] = tf.keras.Input(shape=(None, 3), ragged=True)

        self._set_inputs(inputs)

        (self.pair_potentials, self.pair_rho,
         self.embedding_functions, self.offsets) = self.build_functions()

    def build_functions(self):
        pass

    def call(self, inputs):
        if self.preprocessed_input:
            return self.shallow_call(inputs)
        return self.deep_call(inputs)

    def deep_call(self, inputs):
        types = inputs['types']
        positions = inputs['positions']
        distances, pair_types, dr_dx = tf.map_fn(
            lambda x: distances_and_pair_types(
                x[0], x[1], len(self.atom_types), self.cutoff),
            (positions, types),
            fn_output_signature=(
                tf.RaggedTensorSpec(shape=[None, None, 1], ragged_rank=1,
                                    dtype=positions.dtype),
                tf.RaggedTensorSpec(shape=[None, None, 1], ragged_rank=1,
                                    dtype=types.dtype),
                tf.RaggedTensorSpec(shape=[None, None, None, 3], ragged_rank=2,
                                    dtype=positions.dtype))
            )

        if self.build_forces:
            return self.main_body_with_forces(types, distances,
                                              pair_types, dr_dx)
        return self.main_body_no_forces(types, distances, pair_types)

    @tf.function
    def shallow_call(self, inputs):
        types = inputs['types']
        distances = inputs['distances']
        pair_types = inputs['pair_types']

        if self.build_forces:
            dr_dx = inputs['dr_dx']
            return self.main_body_with_forces(types, distances,
                                              pair_types, dr_dx)
        return self.main_body_no_forces(types, distances, pair_types)

    @tf.function(input_signature=(
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, 1]), tf.int32, 1, tf.int64),
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, None, 1]), tf.keras.backend.floatx(),
            2, tf.int64),
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, None, 1]), tf.int32, 2, tf.int64)))
    def main_body_no_forces(self, types, distances, pair_types):
        """Calculates the energy per atom by calling the main body
        """
        if self.method == 'partition_stitch':
            energy = self.body_partition_stitch(types, distances, pair_types)
        elif self.method == 'gather_scatter':
            energy = self.body_gather_scatter(types, distances, pair_types)
        elif self.method == 'where':
            energy = self.body_where(types, distances, pair_types)
        else:
            raise NotImplementedError('Unknown method %s' % self.method)
        number_of_atoms = tf.cast(types.row_lengths(), energy.dtype,
                                  name='number_of_atoms')
        energy_per_atom = tf.divide(
            energy, tf.expand_dims(number_of_atoms, axis=-1),
            name='energy_per_atom')

        return {'energy_per_atom': energy_per_atom}

    @tf.function(input_signature=(
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, 1]), tf.int32, 1, tf.int64),
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, None, 1]), tf.keras.backend.floatx(),
            2, tf.int64),
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, None, 1]), tf.int32, 2, tf.int64),
        tf.RaggedTensorSpec(
            tf.TensorShape([None, None, None, None, 3]),
            tf.keras.backend.floatx(), 3, tf.int64)))
    def main_body_with_forces(self, types, distances, pair_types, dr_dx):
        """Calculates the energy per atom and the derivative of the total
           energy with respect to the distances
        """
        with tf.GradientTape() as tape:
            tape.watch(distances.flat_values)
            if self.method == 'partition_stitch':
                energy = self.body_partition_stitch(types, distances,
                                                    pair_types)
            elif self.method == 'gather_scatter':
                energy = self.body_gather_scatter(types, distances, pair_types)
            elif self.method == 'where':
                energy = self.body_where(types, distances, pair_types)
            else:
                raise NotImplementedError('Unknown method %s' % self.method)
        number_of_atoms = tf.cast(types.row_lengths(), energy.dtype,
                                  name='number_of_atoms')
        energy_per_atom = tf.divide(
            energy, tf.expand_dims(number_of_atoms, axis=-1),
            name='energy_per_atom')

        if self.force_method == 'old':
            # Probably not should not be reshaped to RaggedTensor only to sum
            # over these dimensions in the next step.
            dE_dr = tf.RaggedTensor.from_nested_row_splits(
                tape.gradient(energy, distances.flat_values),
                distances.nested_row_splits, name='dE_dr')
            # dr_dx.shape = (batch_size, None, None, None, 3)
            # dE_dr.shape = (batch_size, None, None, 1)
            # Sum over atom indices i and j. Force is the negative gradient.
            forces = -tf.reduce_sum(dr_dx * tf.expand_dims(dE_dr, -1),
                                    axis=(-3, -4), name='dE_dr_times_dr_dx')
        elif self.force_method == 'new':
            dE_dr = tf.reshape(
                tape.gradient(energy, distances.flat_values),
                tf.TensorShape([types.shape[0], None, 1, 1]), name='dE_dr')
            # dr_dx.shape = (batch_size, None, None, None, 3)
            # dE_dr.shape = (batch_size, None, 1, 1)
            forces = -tf.reduce_sum(
                dr_dx.merge_dims(1, 2) * dE_dr,
                axis=-3, name='dE_dr_times_dr_dx')
        else:
            raise NotImplementedError(
                'Unknown force method %s' % self.force_method)
        return {'energy_per_atom': energy_per_atom, 'forces': forces}

    @tf.function
    def body_where(self, types, distances, pair_types):
        """"""
        rho = distances
        phi = distances
        for t in self.atom_pair_types:
            cond = tf.equal(pair_types, self.pair_type_dict[t])
            rho = ragged_where(
                cond,
                tf.ragged.map_flat_values(self.pair_rho[t], rho), rho)
            phi = ragged_where(
                cond,
                tf.ragged.map_flat_values(self.pair_potentials[t], phi), phi)

        # Sum over atoms j
        sum_rho = tf.reduce_sum(rho**2, axis=-2, name='sum_rho')
        sum_phi = tf.reduce_sum(phi, axis=-2, name='sum_phi')

        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        embedding_energies = tf.math.maximum(sum_rho, 1e-30)
        for t in self.atom_types:
            cond = tf.equal(types, self.type_dict[t])
            # tf.abs(x) necessary here since most embedding functions do not
            # support negative inputs but embedding energies are typically
            # negative and tf.where applies the function to every vector entry
            embedding_energies = ragged_where(
                cond,
                tf.ragged.map_flat_values(
                    lambda x: (self.embedding_functions[t](tf.abs(x))
                               + self.offsets[t](x)), embedding_energies),
                embedding_energies)
        atomic_energies = sum_phi + embedding_energies

        # Sum over atoms i
        return tf.reduce_sum(atomic_energies, axis=-2, name='energy')

    @tf.function
    def body_partition_stitch(self, types, distances, pair_types):
        """main body using dynamic_partition and dynamic_stitch methods"""
        pair_type_indices = tf.dynamic_partition(
                tf.expand_dims(tf.range(tf.size(distances)), -1),
                pair_types.flat_values, len(self.atom_pair_types),
                name='pair_type_indices')
        # Partition distances according to the pair_type
        partitioned_r = tf.dynamic_partition(
            distances.flat_values, pair_types.flat_values,
            len(self.atom_pair_types), name='partitioned_r')
        rho = [self.pair_rho[t](tf.expand_dims(part, -1))
               for t, part in zip(self.atom_pair_types, partitioned_r)]
        phi = [self.pair_potentials[t](tf.expand_dims(part, -1))
               for t, part in zip(self.atom_pair_types, partitioned_r)]

        rho = tf.dynamic_stitch(pair_type_indices, rho)
        phi = tf.dynamic_stitch(pair_type_indices, phi)

        # Reshape to ragged tensors
        phi = tf.RaggedTensor.from_nested_row_splits(
            phi, distances.nested_row_splits)
        rho = tf.RaggedTensor.from_nested_row_splits(
            rho, distances.nested_row_splits)

        # Sum over atoms j
        sum_rho = tf.reduce_sum(rho**2, axis=-2, name='sum_rho')
        sum_phi = tf.reduce_sum(phi, axis=-2, name='sum_phi')

        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        sum_rho = tf.math.maximum(sum_rho, 1e-30)

        # Embedding energy
        partitioned_sum_rho = tf.dynamic_partition(
            sum_rho, types, len(self.atom_types),
            name='partitioned_sum_rho')
        type_indices = tf.dynamic_partition(
            tf.expand_dims(tf.range(tf.size(sum_rho)), -1),
            types.flat_values, len(self.atom_types), name='type_indices')
        # energy offset is added here
        embedding_energies = [
            self.embedding_functions[t](tf.expand_dims(rho_t, -1))
            + self.offsets[t](tf.expand_dims(rho_t, -1))
            for t, rho_t in zip(self.atom_types, partitioned_sum_rho)]
        embedding_energies = tf.dynamic_stitch(
            type_indices, embedding_energies, name='embedding_energies')

        atomic_energies = sum_phi.flat_values + embedding_energies
        # Reshape to ragged
        atomic_energies = tf.RaggedTensor.from_row_splits(
            atomic_energies, types.row_splits, name='atomic_energies')
        # Sum over atoms i
        return tf.reduce_sum(atomic_energies, axis=-2, name='energy')

    @tf.function
    def body_gather_scatter(self, types, distances, pair_types):
        """main body using gather and scatter methods"""
        phi = tf.zeros_like(distances.flat_values)
        rho = tf.zeros_like(distances.flat_values)

        for ij, type_ij in enumerate(self.atom_pair_types):
            # Flat values necessary until where properly supports
            # ragged tensors
            indices = tf.where(tf.equal(pair_types, ij).flat_values)[:, 0]
            masked_distances = tf.gather(distances.flat_values, indices)
            phi = tf.tensor_scatter_nd_update(
                phi, tf.expand_dims(indices, -1),
                self.pair_potentials[type_ij](masked_distances))
            rho = tf.tensor_scatter_nd_update(
                rho, tf.expand_dims(indices, -1),
                self.pair_rho[type_ij](masked_distances))
        # Reshape back to ragged tensors
        phi = tf.RaggedTensor.from_nested_row_splits(
            phi, distances.nested_row_splits)
        rho = tf.RaggedTensor.from_nested_row_splits(
            rho, distances.nested_row_splits)
        # Sum over atoms j and flatten again
        atomic_energies = tf.reduce_sum(phi, axis=-2).flat_values
        sum_rho = tf.reduce_sum(rho**2, axis=-2).flat_values
        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        sum_rho = tf.math.maximum(sum_rho, 1e-30)
        for i, t in enumerate(self.atom_types):
            indices = tf.where(tf.equal(types, i).flat_values)[:, 0]
            atomic_energies = tf.tensor_scatter_nd_add(
                atomic_energies, tf.expand_dims(indices, -1),
                self.embedding_functions[t](tf.gather(sum_rho, indices)))
        # Reshape to ragged
        atomic_energies = tf.RaggedTensor.from_row_splits(
            atomic_energies, types.row_splits)
        # Sum over atoms i
        return tf.reduce_sum(atomic_energies, axis=-2, name='energy')


class SMATB(EAMPotential):

    def __init__(self, atom_types, params={}, r0_trainable=False,
                 offset_trainable=True, reg=None, **kwargs):
        """TODO: __init__ can not be called with params={}: raises
        tensorflow error"""
        # Determine the maximum cutoff value to pass to DeepEAMPotential.
        # Defaults to 7.5 if 'cut_b' if missing for one or all pair_types.
        # The 'or' in the max function is used as fallback in case the list
        # comprehension returns an empty list
        cutoff = max([params.get(key, 7.5)
                      for key in params if key[0] == 'cut_b'] or [7.5])
        self.params = params
        self.r0_trainable = r0_trainable
        self.offset_trainable = offset_trainable
        self.reg = reg
        super().__init__(atom_types, cutoff=cutoff, **kwargs)

    def build_functions(self):
        # Todo should probably move to parent class
        pair_potentials = {}
        pair_rho = {}
        for (t1, t2) in combinations_with_replacement(self.atom_types, 2):
            pair_type = ''.join([t1, t2])
            normalized_input = InputNormalization(
                pair_type, r0=self.params.get(('r0', pair_type), 2.7),
                trainable=self.r0_trainable)
            cutoff_function = PolynomialCutoffFunction(
                pair_type, a=self.params.get(('cut_a', pair_type), 5.0),
                b=self.params.get(('cut_b', pair_type), 7.5))
            pair_potential = self.get_pair_potential(pair_type)
            rho = self.get_rho(pair_type)
            pair_potentials[pair_type] = PairInteraction(
                normalized_input, pair_potential, cutoff_function,
                name='%s-phi' % pair_type)
            pair_rho[pair_type] = PairInteraction(
                normalized_input, rho, cutoff_function,
                name='%s-rho' % pair_type)
        embedding_functions = {t: self.get_embedding(t)
                               for t in self.atom_types}
        offsets = {t: OffsetLayer(t, self.offset_trainable,
                                  name='%s-offset' % t)
                   for t in self.atom_types}
        return pair_potentials, pair_rho, embedding_functions, offsets

    def get_pair_potential(self, pair_type):
        return BornMayer(pair_type, A=self.params.get(('A', pair_type), 0.2),
                         p=self.params.get(('p', pair_type), 9.2),
                         name='Phi-%s' % pair_type)

    def get_rho(self, pair_type):
        return RhoExp(pair_type, xi=self.params.get(('xi', pair_type), 1.6),
                      q=self.params.get(('q', pair_type), 3.5),
                      name='Rho-%s' % pair_type)

    def get_embedding(self, type):
        return SqrtEmbedding(name='%s-Embedding' % type)


class ExtendedEmbeddingModel(SMATB):

    def get_embedding(self, type):
        return ExtendedEmbedding(name='%s-Embedding' % type)


class ExtendedEmbeddingV2Model(SMATB):

    def get_embedding(self, type):
        return ExtendedEmbeddingV2(name='%s-Embedding' % type)


class ExtendedEmbeddingV3Model(SMATB):

    def get_embedding(self, type):
        return ExtendedEmbeddingV3(name='%s-Embedding' % type)


class NNEmbeddingModel(SMATB):

    def get_embedding(self, type):
        return NNSqrtEmbedding(
            layers=self.params.get(('F_layers', type), [20, 20]),
            reg=self.reg, name='%s-Embedding' % type)


class RhoTwoExpModel(SMATB):

    def get_rho(self, pair_type):
        return RhoTwoExp(pair_type,
                         xi_1=self.params.get(('xi_1', pair_type), 1.6),
                         q_1=self.params.get(('q_1', pair_type), 3.5),
                         xi_2=self.params.get(('xi_2', pair_type), 0.8),
                         q_2=self.params.get(('q_2', pair_type), 1.0),
                         name='Rho-%s' % pair_type)


class NNRhoModel(SMATB):

    def get_rho(self, pair_type):
        return NNRho(
            pair_type,
            layers=self.params.get(('rho_layers', pair_type), [20, 20]),
            reg=self.reg, name='Rho-%s' % pair_type)


class NNRhoExpModel(SMATB):

    def get_rho(self, pair_type):
        return NNRhoExp(
            pair_type,
            layers=self.params.get(('rho_layers', pair_type), [20, 20]),
            reg=self.reg, name='Rho-%s' % pair_type)


class ExtendedEmbeddingRhoTwoExpModel(ExtendedEmbeddingModel, RhoTwoExpModel):
    """Combination of ExtendedEmbeddingModel and RhoTwoExpModel"""


class ExtendedEmbeddingV3RhoTwoExpModel(ExtendedEmbeddingV3Model,
                                        RhoTwoExpModel):
    """Combination of ExtendedEmbeddingV3Model and RhoTwoExpModel"""


class NNEmbeddingNNRhoModel(NNEmbeddingModel, NNRhoModel):
    """Combination of NNEmbeddingModel and NNRhoModel"""


class NNEmbeddingNNRhoExpModel(NNEmbeddingModel, NNRhoExpModel):
    """Combination of NNEmbeddingModel and NNRhoExpModel"""
