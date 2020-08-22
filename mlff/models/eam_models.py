import tensorflow as tf
from itertools import combinations_with_replacement
from mlff.layers import (PairInteraction, PolynomialCutoffFunction,
                         InputNormalization, BornMayer, RhoExp, RhoNN,
                         SqrtEmbedding, NNSqrtEmbedding)
from mlff.utils import distances_and_pair_types


class EAMPotential(tf.keras.Model):
    """Base class for all EAM based potentials"""

    def __init__(self, atom_types, build_forces=False):
        super().__init__()

        self.atom_types = sorted(atom_types)
        self.atom_pair_types = []
        for (t1, t2) in combinations_with_replacement(self.atom_types, 2):
            self.atom_pair_types.append(''.join([t1, t2]))
        self.build_forces = build_forces
        (self.pair_potentials, self.pair_rho,
         self.embedding_functions) = self.build_functions()

    def build_functions(self):
        pass

    @tf.function
    def main_body_no_forces(self, types, distances, pair_types):
        """Calculates the energy per atom by calling body_partition_stitch()
        """
        energy = self.body_partition_stitch(types, distances, pair_types)
        # energy = self.body_gather_scatter(types, distances, pair_types)
        number_of_atoms = tf.cast(types.row_lengths(), energy.dtype,
                                  name='number_of_atoms')
        energy_per_atom = tf.divide(
            energy, tf.expand_dims(number_of_atoms, axis=-1),
            name='energy_per_atom')

        return energy_per_atom

    @tf.function
    def main_body_with_forces(self, types, distances, pair_types, dr_dx):
        """Calculates the energy per atom and the derivative of the total
           energy with respect to the distances
        """
        with tf.GradientTape() as tape:
            tape.watch(distances.flat_values)
            energy = self.body_partition_stitch(types, distances, pair_types)
            # energy = self.body_gather_scatter(types, distances, pair_types)
        number_of_atoms = tf.cast(types.row_lengths(), energy.dtype,
                                  name='number_of_atoms')
        energy_per_atom = tf.divide(
            energy, tf.expand_dims(number_of_atoms, axis=-1),
            name='energy_per_atom')

        dE_dr = tf.RaggedTensor.from_nested_row_splits(
            tape.gradient(energy, distances.flat_values),
            distances.nested_row_splits, name='dE_dr')
        # dr_dx.shape = (batch_size, None, None, None, 3)
        # dE_dr.shape = (batch_size, None, None, 1)
        # Sum over atom indices i and j. Force is the negative gradient.
        forces = -tf.reduce_sum(dr_dx * tf.expand_dims(dE_dr, -1),
                                axis=(-3, -4), name='dE_dr_times_dr_dx')
        return energy_per_atom, forces

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
        rho = [self.pair_rho[t](part)
               for t, part in zip(self.atom_pair_types, partitioned_r)]
        phi = [self.pair_potentials[t](part)
               for t, part in zip(self.atom_pair_types, partitioned_r)]

        rho = tf.expand_dims(tf.dynamic_stitch(pair_type_indices, rho), -1)
        phi = tf.expand_dims(tf.dynamic_stitch(pair_type_indices, phi), -1)

        # Reshape to ragged tensors
        phi = tf.RaggedTensor.from_nested_row_splits(
            phi, distances.nested_row_splits)
        rho = tf.RaggedTensor.from_nested_row_splits(
            rho, distances.nested_row_splits)

        # Sum over atoms j
        sum_rho = tf.reduce_sum(rho, axis=-2, name='SumRho')
        sum_phi = tf.reduce_sum(phi, axis=-2, name='SumPhi')

        # Embedding energy
        partitioned_sum_rho = tf.dynamic_partition(
            sum_rho, types, len(self.atom_types),
            name='partitioned_sum_rho')
        type_indices = tf.dynamic_partition(
            tf.expand_dims(tf.range(tf.size(sum_rho)), -1),
            types.flat_values, len(self.atom_types), name='type_indices')
        embedding_energies = [
            self.embedding_functions[t](rho_t)
            for t, rho_t in zip(self.atom_types, partitioned_sum_rho)]
        atomic_energies = (
            sum_phi.flat_values
            + tf.expand_dims(
                tf.dynamic_stitch(type_indices, embedding_energies,
                                  name='embedding_energies'),
                -1))
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
        sum_rho = tf.reduce_sum(rho, axis=-2).flat_values
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


class DeepEAMPotential(EAMPotential):

    def __init__(self, atom_types, cutoff=10.0, **kwargs):
        super().__init__(atom_types, **kwargs)

        self.cutoff = cutoff
        types = tf.keras.Input(shape=(None, 1), ragged=True, dtype=tf.int32)
        positions = tf.keras.Input(shape=(None, 3), ragged=True)
        inputs = {'types': types, 'positions': positions}

        self._set_inputs(inputs)

    # using tf.function raises:
    # LookupError: No gradient defined for operation
    # 'map/RaggedFromVariant_2/RaggedTensorFromVariant'
    # (op type: RaggedTensorFromVariant)
    # @tf.function
    def call(self, inputs):
        types = inputs['types']
        positions = inputs['positions']
        distances, pair_types, dr_dx = tf.map_fn(
            lambda x: distances_and_pair_types(
                x[0], x[1], len(self.atom_types), self.cutoff),
            (positions, types),
            fn_output_signature=(tf.RaggedTensorSpec(
                                    shape=[None, None, 1], ragged_rank=1,
                                    dtype=positions.dtype),
                                 tf.RaggedTensorSpec(
                                    shape=[None, None, 1], ragged_rank=1,
                                    dtype=types.dtype),
                                 tf.RaggedTensorSpec(
                                    shape=[None, None, None, 3],
                                    ragged_rank=2, dtype=positions.dtype))
            )

        if self.build_forces:
            return self.main_body_with_forces(types, distances,
                                              pair_types, dr_dx)
        return self.main_body_no_forces(types, distances, pair_types)


class ShallowEAMPotential(EAMPotential):

    def __init__(self, atom_types, cutoff=None, **kwargs):
        """The cutoff argument is only for compatibility"""
        super().__init__(atom_types, **kwargs)
        types = tf.keras.Input(shape=(None, 1), ragged=True, dtype=tf.int32)
        distances = tf.keras.Input(shape=(None, None, 1), ragged=True)
        pair_types = tf.keras.Input(shape=(None, None, 1), ragged=True,
                                    dtype=types.dtype)
        inputs = {'types': types, 'pair_types': pair_types,
                  'distances': distances}
        if self.build_forces:
            # [batchsize] x N x (N-1) x N x 3
            inputs['dr_dx'] = tf.keras.Input(shape=(None, None, None, 3),
                                             ragged=True)

        self._set_inputs(inputs)

    @tf.function
    def call(self, inputs):
        types = inputs['types']
        distances = inputs['distances']
        pair_types = inputs['pair_types']

        if self.build_forces:
            dr_dx = inputs['dr_dx']
            return self.main_body_with_forces(types, distances,
                                              pair_types, dr_dx)
        return self.main_body_no_forces(types, distances, pair_types)


class SMATB(DeepEAMPotential):

    def __init__(self, atom_types, params={},
                 r0_trainable=False, **kwargs):
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
        super().__init__(atom_types, cutoff=cutoff, **kwargs)

    def build_functions(self):
        # Todo should probably move to parent class
        pair_potentials = {}
        pair_rho = {}
        for (t1, t2) in combinations_with_replacement(self.atom_types, 2):
            type_i = ''.join([t1, t2])
            normalized_input = InputNormalization(
                type_i, r0=self.params.get(('r0', type_i), 2.7),
                trainable=self.r0_trainable)
            cutoff_function = PolynomialCutoffFunction(
                type_i, a=self.params.get(('cut_a', type_i), 5.0),
                b=self.params.get(('cut_b', type_i), 7.5))
            pair_potential = self.get_pair_potential(type_i)
            rho = self.get_rho(type_i)
            pair_potentials[type_i] = PairInteraction(
                normalized_input, pair_potential, cutoff_function,
                name='%s-phi' % type_i)
            pair_rho[type_i] = PairInteraction(
                normalized_input, rho, cutoff_function, name='%s-rho' % type_i)
        embedding_functions = {}
        for t in self.atom_types:
            embedding_functions[t] = self.get_embedding(t)
        return pair_potentials, pair_rho, embedding_functions

    def get_pair_potential(self, pair_type):
        return BornMayer(pair_type, A=self.params.get(('A', pair_type), 0.2),
                         p=self.params.get(('p', pair_type), 9.2))

    def get_rho(self, pair_type):
        return RhoExp(pair_type, xi=self.params.get(('xi', pair_type), 1.6),
                      q=self.params.get(('q', pair_type), 3.5))

    def get_embedding(self, type):
        return SqrtEmbedding(name='%s-Embedding' % type)


class NNEmbeddingModel(SMATB):

    def get_embedding(self, type):
        return NNSqrtEmbedding(
            layers=self.params.get(('F_layers', type), [20, 20]),
            name='%s-Embedding' % type)


class NNRhoModel(SMATB):

    def get_rho(self, pair_type):
        return RhoNN(
            pair_type,
            layers=self.params.get(('rho_layers', pair_type), [20, 20]))


class NNEmbeddingNNRhoModel(NNEmbeddingModel, NNRhoModel):
    """Combination of NNEmbeddingModel and NNRhoModel"""
