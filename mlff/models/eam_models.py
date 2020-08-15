import tensorflow as tf
from itertools import combinations_with_replacement
from mlff.layers import (PairInteraction, PolynomialCutoffFunction,
                         InputNormalization, BornMayer, RhoExp,
                         SqrtEmbedding)
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
    def body_partition_stitch(self, types, distances, pair_types):
        """main body using dynamic_partition and dynamic_stitch methods"""
        pair_type_indices = tf.dynamic_partition(
                tf.expand_dims(tf.range(tf.size(distances)), -1),
                pair_types.flat_values, len(self.atom_pair_types),
                name='pair_type_indices')
        # Partition distances according to the pair_type
        partitioned_r = tf.dynamic_partition(
            distances.flat_values * 1, pair_types.flat_values,
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
        sum_rho = tf.reduce_sum(rho, axis=-2)
        sum_phi = tf.reduce_sum(phi, axis=-2)

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
                tf.dynamic_stitch(type_indices, embedding_energies), -1))
        # Reshape to ragged
        atomic_energies = tf.RaggedTensor.from_row_splits(
            atomic_energies, types.row_splits)
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
                                    shape=[None, None, 1], ragged_rank=1),
                                 tf.RaggedTensorSpec(
                                    shape=[None, None, 1], ragged_rank=1,
                                    dtype=tf.int32),
                                 tf.RaggedTensorSpec(
                                    shape=[None, None, None, 3],
                                    ragged_rank=2))
            )

        with tf.GradientTape() as tape:
            tape.watch(distances.flat_values)
            energy = self.body_partition_stitch(types, distances, pair_types)
            # energy = self.body_gather_scatter(types, distances, pair_types)
        energy_per_atom = energy/tf.expand_dims(tf.cast(
            types.row_lengths(), tf.float32, name='number_of_atoms'), axis=-1)

        if self.build_forces:
            dE_dr = tf.RaggedTensor.from_nested_row_splits(
                tape.gradient(energy, distances.flat_values),
                distances.nested_row_splits)
            gradients = tf.reduce_sum(dr_dx * tf.expand_dims(dE_dr, -1),
                                      axis=(-3, -4))
            return energy_per_atom, gradients
        return energy_per_atom


class ShallowEAMPotential(EAMPotential):

    def __init__(self, atom_types, **kwargs):
        super().__init__(atom_types, **kwargs)
        types = tf.keras.Input(shape=(None, 1), ragged=True, dtype=tf.int32)
        distances = tf.keras.Input(shape=(None, None, 1), ragged=True)
        pair_types = tf.keras.Input(shape=(None, None, 1), ragged=True,
                                    dtype=tf.int32)
        inputs = {'types': types, 'pair_types': pair_types,
                  'distances': distances}
        if self.build_forces:
            # [batchsize] x N x (N-1) x N x 3
            dr_dx = tf.keras.Input(shape=(None, None, None, 3), ragged=True)
            inputs['dr_dx'] = dr_dx

        self._set_inputs(inputs)

    # using tf.function raises:
    # LookupError: No gradient defined for operation
    # 'map/RaggedFromVariant_2/RaggedTensorFromVariant'
    # (op type: RaggedTensorFromVariant)
    # @tf.function
    def call(self, inputs):
        types = inputs['types']
        distances = inputs['distances']
        pair_types = inputs['pair_types']

        with tf.GradientTape() as tape:
            tape.watch(distances.flat_values)
            energy = self.body_partition_stitch(types, distances, pair_types)
            # energy = self.body_gather_scatter(types, distances, pair_types)
        energy_per_atom = energy/tf.expand_dims(tf.cast(
            types.row_lengths(), tf.float32, name='number_of_atoms'), axis=-1)

        if self.build_forces:
            dr_dx = inputs['dr_dx']
            dE_dr = tf.RaggedTensor.from_nested_row_splits(
                tape.gradient(energy, distances.flat_values),
                distances.nested_row_splits)
            gradients = tf.reduce_sum(dr_dx * tf.expand_dims(dE_dr, -1),
                                      axis=(-3, -4))
            return energy_per_atom, gradients
        return energy_per_atom


class SMATB(DeepEAMPotential):

    def __init__(self, atom_types, initial_params={},
                 r0_trainable=False, **kwargs):
        # Determine the maximum cutoff value to pass to DeepEAMPotential.
        # Defaults to 7.5 if 'cut_b' if missing for one or all pair_types.
        cutoff = max([initial_params.get(key, 7.5)
                      for key in initial_params if key[0] == 'cut_b'] or [7.5])
        self.initial_params = initial_params
        self.r0_trainable = r0_trainable
        super().__init__(atom_types, cutoff=cutoff, **kwargs)

    def build_functions(self):
        pair_potentials = {}
        pair_rho = {}
        for (t1, t2) in combinations_with_replacement(self.atom_types, 2):
            type_i = ''.join([t1, t2])
            normalized_input = InputNormalization(
                type_i, r0=self.initial_params.get(('r0', type_i), 2.7),
                trainable=self.r0_trainable)
            cutoff_function = PolynomialCutoffFunction(
                type_i, a=self.initial_params.get(('cut_a', type_i), 5.0),
                b=self.initial_params.get(('cut_b', type_i), 7.5))
            pair_potential = BornMayer(
                type_i, A=self.initial_params.get(('A', type_i), 0.2),
                p=self.initial_params.get(('p', type_i), 9.2))
            rho = RhoExp(type_i,
                         xi=self.initial_params.get(('xi', type_i), 1.6),
                         q=self.initial_params.get(('q', type_i), 3.5))
            pair_potentials[type_i] = PairInteraction(
                normalized_input, pair_potential, cutoff_function,
                name='%s-phi' % type_i)
            pair_rho[type_i] = PairInteraction(
                normalized_input, rho, cutoff_function, name='%s-rho' % type_i)
        embedding_functions = {}
        for t in self.atom_types:
            embedding_functions[t] = SqrtEmbedding(name='%s-Embedding' % t)
        return pair_potentials, pair_rho, embedding_functions
