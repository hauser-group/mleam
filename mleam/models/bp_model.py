import tensorflow as tf
from mleam.layers import AtomicNeuralNetwork


class BehlerParrinello(tf.keras.Model):
    def __init__(
        self,
        atom_types,
        num_Gs,
        layers=None,
        reg=None,
        offset_trainable=True,
        build_forces=False,
    ):
        super().__init__()
        self.atom_types = atom_types
        self.num_Gs = num_Gs
        self.build_forces = build_forces
        layers = layers or {t: [20, 20] for t in atom_types}

        self.atomic_neural_nets = {
            t: AtomicNeuralNetwork(
                layers[t], offset_trainable=offset_trainable, reg=reg, name="%s-ANN" % t
            )
            for t in atom_types
        }
        # TODO: Fix: this does somehow not create the weight variables...
        for t in atom_types:
            self.atomic_neural_nets[t].build(input_shape=(None, num_Gs[t]))

        types = tf.keras.Input(shape=(None, 1), ragged=True, dtype=tf.int32)
        Gs = tf.keras.Input(shape=(None, None), ragged=True)
        inputs = {"types": types, "Gs": Gs}

        if self.build_forces:
            inputs["dGs"] = tf.keras.Input(shape=(None, None, None, 3), ragged=True)

        self._set_inputs(inputs)

    @tf.function
    def call(self, inputs):
        types = inputs["types"]
        Gs = inputs["Gs"]

        if self.build_forces:
            dGs = inputs["dGs"]
            return self.main_body_with_forces(types, Gs, dGs)
        return self.main_body_no_forces(types, Gs)

    @tf.function
    def main_body_no_forces(self, types, Gs):
        """Calculates the energy per atom by calling body_partition_stitch()"""
        energy = self.body_partition_stitch(types, Gs)
        number_of_atoms = tf.cast(
            types.row_lengths(), energy.dtype, name="number_of_atoms"
        )
        energy_per_atom = tf.divide(
            energy, tf.expand_dims(number_of_atoms, axis=-1), name="energy_per_atom"
        )

        return {"energy_per_atom": energy_per_atom}

    @tf.function
    def main_body_with_forces(self, types, Gs, dGs):
        """Calculates the energy per atom and the derivative of the total
        energy with respect to the distances
        """
        with tf.GradientTape() as tape:
            tape.watch(Gs.flat_values)
            energy = self.body_partition_stitch(types, Gs)
        number_of_atoms = tf.cast(
            types.row_lengths(), energy.dtype, name="number_of_atoms"
        )
        energy_per_atom = tf.divide(
            energy, tf.expand_dims(number_of_atoms, axis=-1), name="energy_per_atom"
        )

        dE_dG = tf.expand_dims(
            tf.RaggedTensor.from_nested_row_splits(
                tape.gradient(energy, Gs.flat_values),
                Gs.nested_row_splits,
                name="dE_dG",
            ),
            -1,
        )
        # dGs.shape = (batch_size, None, None, None, 3)
        # dE_dr.shape = (batch_size, None, None, 1)
        # Sum over atom indices i and j. Force is the negative gradient.
        forces = -tf.reduce_sum(
            dGs * tf.expand_dims(dE_dG, -1), axis=(-3, -4), name="dE_dG_times_dG_dx"
        )
        return {"energy_per_atom": energy_per_atom, "forces": forces}

    @tf.function
    def body_partition_stitch(self, types, Gs):
        """Returns the total energy"""
        # Partition Gs according to their atom type.
        # types needs to be broadcasted to the same shape as Gs by ones_like.
        # Gives a list of len(atom_types) of 1D tensors that has to be reshaped
        # before it goes through the neural network
        partitioned_Gs = tf.dynamic_partition(
            Gs, types * tf.ones_like(Gs, dtype=tf.int32), len(self.atom_types)
        )
        # reshape and apply AtomicNeuralNetwork functions
        partitioned_energies = [
            self.atomic_neural_nets[t](tf.reshape(Gi, (-1, self.num_Gs[t])))
            for t, Gi in zip(self.atom_types, partitioned_Gs)
        ]

        # Find indices of the atomic energies in the original tensor
        type_indices = tf.dynamic_partition(
            tf.expand_dims(tf.range(tf.size(types)), -1),
            types.flat_values,
            2,
            name="type_indices",
        )
        # Stitch the partitioned energies according to the indices
        # and reshape back to a RaggedTensor
        atomic_energies = tf.RaggedTensor.from_row_splits(
            tf.dynamic_stitch(type_indices, partitioned_energies),
            types.row_splits,
            name="atomic_energies",
        )
        # Summation over the atoms gives a (batch_size, 1) Tensor
        return tf.reduce_sum(atomic_energies, axis=-2, name="energy")
