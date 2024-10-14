import tensorflow as tf
import numpy as np
from typing import List
from tensorflow.python.ops.ragged.ragged_where_op import where as ragged_where
from itertools import combinations_with_replacement
from mleam.layers import (
    PairInteraction,
    NormalizedPairInteraction,
    PolynomialCutoffFunction,
    OffsetLayer,
    InputNormalizationAndShift,
    BornMayer,
    ExpRho,
    SuttonChenPhi,
    SuttonChenRho,
    DoubleSuttonChenPhi,
    DoubleSuttonChenRho,
    FinnisSinclairPhi,
    FinnisSinclairRho,
    CubicSplinePhi,
    CubicSplineRho,
    DoubleExpPhi,
    DoubleExpRho,
    FittedQuinticSplinePhi,
    FittedQuinticSplineRho,
    NNRho,
    NNRhoExp,
    SqrtEmbedding,
    JohnsonEmbedding,
    ExtendedEmbedding,
    ExtendedEmbeddingV2,
    ExtendedEmbeddingV3,
    ExtendedEmbeddingV4,
    NNSqrtEmbedding,
)
from mleam.preprocessing import (
    distances_and_pair_types,
    distances_and_pair_types_no_grad,
)
from mleam.constants import InputNormType
import warnings


class EAMPotential(tf.keras.Model):
    """Base class for all EAM based potentials"""

    hyperparams = {"offset_trainable": False}

    def __init__(
        self,
        atom_types: List[str],
        params={},
        hyperparams={},
        build_forces=False,
        preprocessed_input=False,
        cutoff=None,
        method="partition_stitch",
        force_method="old",
        **kwargs,
    ):
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
                    - 'partition_stitch' default and probably safest choice
                    - 'where' uses branchless programming can be significantly
                              faster as it is easier to parallelize
                    - 'gather_scatter' old implementation, no longer maintained
        """
        super().__init__(**kwargs)

        self.atom_types = atom_types
        self.params = params
        self.hyperparams.update(hyperparams)

        self.type_dict = {}
        for i, t in enumerate(atom_types):
            self.type_dict[t] = i
        self.atom_pair_types = []
        self.pair_type_dict = {}
        for i, (t1, t2) in enumerate(combinations_with_replacement(self.atom_types, 2)):
            t = "".join([t1, t2])
            self.atom_pair_types.append(t)
            self.pair_type_dict[t] = i

        self.build_forces = build_forces
        self.preprocessed_input = preprocessed_input
        self.method = method

        inputs = {"types": tf.keras.Input(shape=(None, 1), ragged=True, dtype=tf.int32)}

        if self.preprocessed_input:
            inputs["pair_types"] = tf.keras.Input(
                shape=(None, None, 1), ragged=True, dtype=inputs["types"].dtype
            )
            inputs["distances"] = tf.keras.Input(shape=(None, None, 1), ragged=True)
            if self.build_forces:
                # [batchsize] x N x N x 3, see preprocessing.py for further details
                inputs["dr_dx"] = tf.keras.Input(shape=(None, None, 3), ragged=True)
        else:
            self.cutoff = cutoff
            if self.cutoff is None:
                # Determine the maximum cutoff value to pass to DeepEAMPotential.
                # Defaults to 7.5 if 'cut_b' if missing for one or all pair_types.
                # The 'or' in the max function is used as fallback in case the list
                # comprehension returns an empty list
                self.cutoff = max(
                    [params.get(key, 7.5) for key in params if key[0] == "cut_b"]
                    or [7.5]
                )
            inputs["positions"] = tf.keras.Input(shape=(None, 3), ragged=True)

        self._set_inputs(inputs)

        (
            self.pair_potentials,
            self.pair_rho,
            self.embedding_functions,
            self.offsets,
        ) = self.build_functions()

    def build_functions(self):
        pair_potentials = {}
        pair_rho = {}
        for t1, t2 in combinations_with_replacement(self.atom_types, 2):
            pair_type = "".join([t1, t2])

            cutoff_function = PolynomialCutoffFunction(
                pair_type,
                a=self.params.get(("cut_a", pair_type), 5.0),
                b=self.params.get(("cut_b", pair_type), 7.5),
            )
            pair_potential = self.get_pair_potential(pair_type)
            rho = self.get_rho(pair_type)

            if (
                pair_potential.input_norm == InputNormType.SCALED_SHIFTED
                or rho.input_norm == InputNormType.SCALED_SHIFTED
            ):
                # A input normalization layer is needed for potential
                # sharing of the parameter r0 between phi and rho layers
                scaled_input = InputNormalizationAndShift(
                    pair_type,
                    r0=self.params.get(("r0", pair_type), 2.7),
                    trainable=self.hyperparams.get("r0_trainable", False),
                )

            if pair_potential.input_norm == InputNormType.SCALED_SHIFTED:
                pair_potentials[pair_type] = NormalizedPairInteraction(
                    input_normalization=scaled_input,
                    pair_interaction=pair_potential,
                    cutoff_function=cutoff_function,
                    name=f"{pair_type}-phi",
                )
            else:
                pair_potentials[pair_type] = PairInteraction(
                    pair_interaction=pair_potential,
                    cutoff_function=cutoff_function,
                    name=f"{pair_type}-phi",
                )

            if rho.input_norm == InputNormType.SCALED_SHIFTED:
                pair_rho[pair_type] = NormalizedPairInteraction(
                    input_normalization=scaled_input,
                    pair_interaction=rho,
                    cutoff_function=cutoff_function,
                    name=f"{pair_type}-rho",
                )
            else:
                pair_rho[pair_type] = PairInteraction(
                    pair_interaction=rho,
                    cutoff_function=cutoff_function,
                    name=f"{pair_type}-rho",
                )
        embedding_functions = {t: self.get_embedding(t) for t in self.atom_types}
        offsets = {
            t: OffsetLayer(t, self.hyperparams["offset_trainable"], name=f"{t}-offset")
            for t in self.atom_types
        }
        return pair_potentials, pair_rho, embedding_functions, offsets

    @tf.function
    def call(self, inputs):
        if self.preprocessed_input:
            return self.shallow_call(inputs)
        return self.deep_call(inputs)

    @tf.function
    def deep_call(self, inputs):
        types = inputs["types"]
        positions = inputs["positions"]

        if self.build_forces:
            distances, pair_types, dr_dx = distances_and_pair_types(
                positions, types, len(self.atom_types), diagonal=self.cutoff
            )
            return self.main_body_with_forces(types, distances, pair_types, dr_dx)
        (
            distances,
            pair_types,
        ) = distances_and_pair_types_no_grad(
            positions, types, len(self.atom_types), diagonal=self.cutoff
        )
        return self.main_body_no_forces(types, distances, pair_types)

    @tf.function
    def shallow_call(self, inputs):
        types = inputs["types"]
        distances = inputs["distances"]
        pair_types = inputs["pair_types"]

        if self.build_forces:
            dr_dx = inputs["dr_dx"]
            return self.main_body_with_forces(types, distances, pair_types, dr_dx)
        return self.main_body_no_forces(types, distances, pair_types)

    @tf.function
    def main_body_no_forces(self, types, distances, pair_types):
        """Calculates the energy per atom by calling the main body"""
        if self.method == "partition_stitch":
            energy = self.body_partition_stitch(types, distances, pair_types)
        elif self.method == "gather_scatter":
            energy = self.body_gather_scatter(types, distances, pair_types)
        elif self.method == "where":
            energy = self.body_where(types, distances, pair_types)
        elif self.method == "dense_where":
            energy = self.body_dense_where(types, distances, pair_types)
        else:
            raise NotImplementedError("Unknown method %s" % self.method)

        if isinstance(types, tf.RaggedTensor):
            number_of_atoms = tf.expand_dims(
                tf.cast(types.row_lengths(), energy.dtype, name="number_of_atoms"),
                axis=-1,
            )
        else:
            number_of_atoms = tf.cast(
                tf.math.count_nonzero(types >= 0, axis=-2),
                energy.dtype,
                name="number_of_atoms",
            )

        energy_per_atom = tf.divide(energy, number_of_atoms, name="energy_per_atom")

        return {"energy": energy, "energy_per_atom": energy_per_atom}

    @tf.function
    def main_body_with_forces(self, types, distances, pair_types, dr_dx):
        """Calculates the energy per atom and the derivative of the total
        energy with respect to the distances
        """
        with tf.GradientTape() as tape:
            tape.watch(distances)
            results = self.main_body_no_forces(types, distances, pair_types)

        dE_dr = tape.gradient(results["energy"], distances)
        results["forces"] = -(
            tf.reduce_sum(dE_dr * dr_dx, axis=-3, name="dE_dr_times_dr_dx_sum_i")
            - tf.reduce_sum(dE_dr * dr_dx, axis=-2, name="dE_dr_times_dr_dx_sum_j")
        )

        return results

    @tf.function
    def body_where(self, types, distances, pair_types):
        """"""
        rho = tf.zeros_like(distances)
        phi = tf.zeros_like(distances)
        for t in self.atom_pair_types:
            cond = tf.equal(pair_types, self.pair_type_dict[t])
            rho = ragged_where(
                cond, tf.ragged.map_flat_values(self.pair_rho[t], distances), rho
            )
            phi = ragged_where(
                cond, tf.ragged.map_flat_values(self.pair_potentials[t], distances), phi
            )

        # Sum over atoms j
        sum_rho = tf.reduce_sum(rho, axis=-2, name="sum_rho")
        sum_phi = tf.reduce_sum(phi, axis=-2, name="sum_phi")

        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        embedding_energies = 1e-30 * tf.ones_like(sum_rho)
        for t in self.atom_types:
            cond = tf.equal(types, self.type_dict[t])
            # tf.abs(x) necessary here since most embedding functions do not
            # support negative inputs but embedding energies are typically
            # negative and tf.where applies the function to every vector entry
            embedding_energies = ragged_where(
                cond,
                tf.ragged.map_flat_values(
                    lambda x: (self.embedding_functions[t](x) + self.offsets[t](x)),
                    sum_rho,
                ),
                embedding_energies,
            )
        atomic_energies = 0.5 * sum_phi + embedding_energies

        # Sum over atoms i
        return tf.reduce_sum(atomic_energies, axis=-2, name="energy")

    @tf.function
    def body_dense_where(self, types, distances, pair_types):
        """"""
        rho = tf.zeros_like(distances)
        phi = tf.zeros_like(distances)
        for t in self.atom_pair_types:
            cond = tf.equal(pair_types, self.pair_type_dict[t])
            rho = tf.where(cond, self.pair_rho[t](distances), rho)
            phi = tf.where(cond, self.pair_potentials[t](distances), phi)

        # Sum over atoms j
        sum_rho = tf.reduce_sum(rho, axis=-2, name="sum_rho")
        sum_phi = tf.reduce_sum(phi, axis=-2, name="sum_phi")

        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        embedding_energies = 1e-30 * tf.ones_like(sum_rho)
        for t in self.atom_types:
            cond = tf.equal(types, self.type_dict[t])
            embedding_energies = tf.where(
                cond,
                self.embedding_functions[t](sum_rho) + self.offsets[t](sum_rho),
                embedding_energies,
            )
        atomic_energies = 0.5 * sum_phi + embedding_energies

        # Sum over atoms i
        return tf.reduce_sum(atomic_energies, axis=-2, name="energy")

    def _custom_ragged_sum(self, a, name=None):
        return tf.RaggedTensor.from_row_splits(
            tf.math.segment_sum(a.flat_values, a.nested_value_rowids()[1], name=name),
            a.row_splits,
        )

    @tf.function
    def body_partition_stitch(self, types, distances, pair_types):
        """main body using dynamic_partition and dynamic_stitch methods"""
        pair_type_indices = tf.dynamic_partition(
            tf.expand_dims(
                tf.range(tf.size(distances)), -1, name="range_for_pair_indices"
            ),
            pair_types.flat_values,
            len(self.atom_pair_types),
            name="pair_type_indices",
        )
        # Partition distances according to the pair_type
        partitioned_r = tf.dynamic_partition(
            distances.flat_values,
            pair_types.flat_values,
            len(self.atom_pair_types),
            name="partitioned_r",
        )
        rho = [
            self.pair_rho[t](tf.expand_dims(part, -1))
            for t, part in zip(self.atom_pair_types, partitioned_r)
        ]
        phi = [
            self.pair_potentials[t](tf.expand_dims(part, -1))
            for t, part in zip(self.atom_pair_types, partitioned_r)
        ]

        rho = tf.dynamic_stitch(pair_type_indices, rho, name="stitch_rho")
        phi = tf.dynamic_stitch(pair_type_indices, phi, name="stitch_phi")

        # Reshape to ragged tensors
        rho = tf.RaggedTensor.from_nested_row_splits(
            rho,
            distances.nested_row_splits,
            name="reshape_rho_to_ragged",
            validate=False,
        )
        phi = tf.RaggedTensor.from_nested_row_splits(
            phi,
            distances.nested_row_splits,
            name="reshape_phi_to_ragged",
            validate=False,
        )

        # Sum over atoms j
        sum_rho = tf.reduce_sum(rho, axis=-2, name="sum_rho")
        sum_phi = tf.reduce_sum(phi, axis=-2, name="sum_phi")

        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        sum_rho = tf.math.maximum(sum_rho, 1e-30)

        # Embedding energy
        partitioned_sum_rho = tf.dynamic_partition(
            sum_rho.flat_values,
            types.flat_values,
            len(self.atom_types),
            name="partitioned_sum_rho",
        )
        type_indices = tf.dynamic_partition(
            tf.expand_dims(tf.range(tf.size(sum_rho)), -1),
            types.flat_values,
            len(self.atom_types),
            name="type_indices",
        )
        # energy offset is added here
        embedding_energies = [
            self.embedding_functions[t](tf.expand_dims(rho_t, -1))
            + self.offsets[t](tf.expand_dims(rho_t, -1))
            for t, rho_t in zip(self.atom_types, partitioned_sum_rho)
        ]
        embedding_energies = tf.dynamic_stitch(
            type_indices, embedding_energies, name="embedding_energies"
        )

        atomic_energies = 0.5 * sum_phi.flat_values + embedding_energies
        # Reshape to ragged: (N_structure, N_atoms, 1)
        atomic_energies = tf.RaggedTensor.from_row_splits(
            atomic_energies,
            types.row_splits,
            name="atomic_energies",
            validate=False,
        )
        # Sum over atoms i.
        return tf.reduce_sum(atomic_energies, axis=-2, name="energy")

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
                phi,
                tf.expand_dims(indices, -1),
                self.pair_potentials[type_ij](masked_distances),
            )
            rho = tf.tensor_scatter_nd_update(
                rho,
                tf.expand_dims(indices, -1),
                self.pair_rho[type_ij](masked_distances),
            )
        # Reshape back to ragged tensors
        phi = tf.RaggedTensor.from_nested_row_splits(phi, distances.nested_row_splits)
        rho = tf.RaggedTensor.from_nested_row_splits(rho, distances.nested_row_splits)
        # Sum over atoms j and flatten again
        atomic_energies = 0.5 * tf.reduce_sum(phi, axis=-2).flat_values
        sum_rho = tf.reduce_sum(rho, axis=-2).flat_values
        # Make sure that sum_rho is never exactly zero since this leads to
        # problems in the gradient of the square root embedding function
        sum_rho = tf.math.maximum(sum_rho, 1e-30)
        for i, t in enumerate(self.atom_types):
            indices = tf.where(tf.equal(types, i).flat_values)[:, 0]
            atomic_energies = tf.tensor_scatter_nd_add(
                atomic_energies,
                tf.expand_dims(indices, -1),
                self.embedding_functions[t](tf.gather(sum_rho, indices)),
            )
        # Reshape to ragged
        atomic_energies = tf.RaggedTensor.from_row_splits(
            atomic_energies, types.row_splits
        )
        # Sum over atoms i
        return tf.reduce_sum(atomic_energies, axis=-2, name="energy")

    def tabulate(
        self,
        filename,
        atomic_numbers,
        atomic_masses,
        lattice_constants=dict(),
        lattice_types=dict(),
        cutoff_rho=120.0,
        nrho=10000,
        cutoff=6.2,
        nr=10000,
    ):
        """
        Uses the atsim module to tabulate the potential for use in LAMMPS or
        similar programs.

        filename: str path of the output file. Note that the extension
                     .eam.fs is added
        atomic_numbers: dict containing the atomic_numbers of all
                            self.atom_types
        atomic_masses: dict containing the atomic mass of all self.atom_types

        """
        from atsim.potentials import EAMPotential, Potential
        from atsim.potentials.eam_tabulation import SetFL_FS_EAMTabulation

        warnings.warn(
            "Long range behavior of the tabulated potentials differs"
            " from the tensorflow implementation!"
        )

        def wrapper(fun):
            def wrapped_fun(x):
                return fun(tf.reshape(x, (1, 1)))

            return wrapped_fun

        pair_potentials = []
        pair_densities = {t: {} for t in self.atom_types}
        for t1, t2 in combinations_with_replacement(self.atom_types, 2):
            pair_type = "".join([t1, t2])

            pair_potentials.append(
                Potential(
                    t1,
                    t2,
                    wrapper(self.pair_potentials[pair_type]),
                )
            )
            pair_densities[t1][t2] = wrapper(self.pair_rho[pair_type])
            pair_densities[t2][t1] = wrapper(self.pair_rho[pair_type])

        eam_potentials = []
        for t in self.atom_types:
            eam_potentials.append(
                EAMPotential(
                    t,
                    atomic_numbers[t],
                    atomic_masses[t],
                    wrapper(self.embedding_functions[t]),
                    pair_densities[t],
                    latticeConstant=lattice_constants.get(t, 0.0),
                    latticeType=lattice_types.get(t, "fcc"),
                )
            )

        tabulation = SetFL_FS_EAMTabulation(
            pair_potentials, eam_potentials, cutoff, nr, cutoff_rho, nrho
        )

        with open("".join([filename, ".eam.fs"]), "w") as outfile:
            tabulation.write(outfile)


class SMATB(EAMPotential):
    def get_pair_potential(self, pair_type):
        return BornMayer(
            pair_type,
            A=self.params.get(("A", pair_type), 0.2),
            p=self.params.get(("p", pair_type), 9.2),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type):
        return ExpRho(
            pair_type,
            xi=self.params.get(("xi", pair_type), 1.6),
            q=self.params.get(("q", pair_type), 3.5),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class FinnisSinclair(EAMPotential):
    def get_pair_potential(self, pair_type):
        return FinnisSinclairPhi(
            pair_type,
            c=self.params.get(("c", pair_type), 5.0),
            c0=self.params.get(("c0", pair_type), 1.0),
            c1=self.params.get(("c1", pair_type), 1.0),
            c2=self.params.get(("c2", pair_type), 1.0),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type):
        return FinnisSinclairRho(
            pair_type,
            A=self.params.get(("A", pair_type), 1.0),
            d=self.params.get(("d", pair_type), 5.0),
            beta=self.params.get(("beta", pair_type), 0.0),
            beta_trainable=self.hyperparams.get("beta_trainable", True),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class SuttonChen(EAMPotential):
    def get_pair_potential(self, pair_type):
        return SuttonChenPhi(
            pair_type,
            c=self.params.get(("c", pair_type), 3.0),
            n=self.params.get(("n", pair_type), 6),
            n_trainable=self.hyperparams.get("n_trainable", False),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type):
        return SuttonChenRho(
            pair_type,
            a=self.params.get(("a", pair_type), 3.0),
            m=self.params.get(("m", pair_type), 6),
            m_trainable=self.hyperparams.get("m_trainable", False),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class DoubleSuttonChen(EAMPotential):
    def get_pair_potential(self, pair_type):
        return DoubleSuttonChenPhi(
            pair_type,
            c_1=self.params.get(("c_1", pair_type), 3.0),
            n_1=self.params.get(("n_1", pair_type), 6),
            c_2=self.params.get(("c_2", pair_type), 3.0),
            n_2=self.params.get(("n_2", pair_type), 8),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type):
        return DoubleSuttonChenRho(
            pair_type,
            a_1=self.params.get(("a_1", pair_type), 3.0),
            m_1=self.params.get(("m_1", pair_type), 6),
            a_2=self.params.get(("a_2", pair_type), 3.0),
            m_2=self.params.get(("m_2", pair_type), 8),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class Johnson(EAMPotential):
    def get_pair_potential(self, pair_type: str):
        return BornMayer(
            pair_type,
            A=self.params.get(("A", pair_type), 0.2),
            p=self.params.get(("p", pair_type), 9.2),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type: str):
        return ExpRho(
            pair_type,
            xi=self.params.get(("xi", pair_type), 1.6),
            q=self.params.get(("q", pair_type), 3.5),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type: str):
        return JohnsonEmbedding(
            atom_type,
            F0=self.params.get(("F0", atom_type), 0.5),
            eta=self.params.get(("eta", atom_type), 0.5),
            F1=self.params.get(("F1", atom_type), 0.5),
            zeta=self.params.get(("zeta", atom_type), 0.5),
            power_law_trainable=self.hyperparams.get("power_law_trainable", True),
            name=f"{atom_type}-Embedding",
        )


class Ackland(EAMPotential):
    def get_pair_potential(self, pair_type: str):
        r_k = self.params.get(("r_k", pair_type), np.linspace(2, 5, 6))
        return CubicSplinePhi(
            pair_type,
            r_k=r_k,
            a_k=self.params.get(("a_k", pair_type), np.zeros_like(r_k)),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type: str):
        R_k = self.params.get(("R_k", pair_type), np.array([2.0, 3.0]))
        return CubicSplineRho(
            pair_type,
            R_k=R_k,
            A_k=self.params.get(("A_k", pair_type), np.zeros_like(R_k)),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type: str):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class DoubleExp(EAMPotential):
    def get_pair_potential(self, pair_type):
        return DoubleExpPhi(
            pair_type,
            A_1=self.params.get(("A_1", pair_type), 0.2),
            p_1=self.params.get(("p_1", pair_type), 9.2),
            A_2=self.params.get(("A_2", pair_type), 0.05),
            p_2=self.params.get(("p_2", pair_type), 0.1),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type):
        return DoubleExpRho(
            pair_type,
            xi_1=self.params.get(("xi_1", pair_type), 1.6),
            q_1=self.params.get(("q_1", pair_type), 3.5),
            xi_2=self.params.get(("xi_2", pair_type), 0.1),
            q_2=self.params.get(("q_2", pair_type), 0.1),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type: str):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class FittedQuinticSpline(EAMPotential):
    def get_pair_potential(self, pair_type: str):
        return FittedQuinticSplinePhi(
            pair_type,
            r_k=self.params.get(("r_k", pair_type), np.array([2.0, 3.0])),
            a_k=self.params.get(("a_k", pair_type)),
            da_k=self.params.get(("da_k", pair_type)),
            dda_k=self.params.get(("dda_k", pair_type)),
            name="Phi-%s" % pair_type,
        )

    def get_rho(self, pair_type: str):
        return FittedQuinticSplineRho(
            pair_type,
            R_k=self.params.get(("R_k", pair_type), np.array([2.0, 3.0])),
            A_k=self.params.get(("A_k", pair_type)),
            dA_k=self.params.get(("dA_k", pair_type)),
            ddA_k=self.params.get(("ddA_k", pair_type)),
            name="Rho-%s" % pair_type,
        )

    def get_embedding(self, atom_type: str):
        return SqrtEmbedding(name=f"{atom_type}-Embedding")


class CommonEmbeddingSMATB(SMATB):
    def build_functions(self):
        pair_potentials = {}
        pair_rho = {}
        for t1, t2 in combinations_with_replacement(self.atom_types, 2):
            pair_type = "".join([t1, t2])
            normalized_input = InputNormalizationAndShift(
                pair_type,
                r0=self.params.get(("r0", pair_type), 2.7),
                trainable=self.hyperparams.get("r0_trainable", False),
            )
            cutoff_function = PolynomialCutoffFunction(
                pair_type,
                a=self.params.get(("cut_a", pair_type), 5.0),
                b=self.params.get(("cut_b", pair_type), 7.5),
            )
            pair_potential = self.get_pair_potential(pair_type)
            rho = self.get_rho(pair_type)
            pair_potentials[pair_type] = NormalizedPairInteraction(
                input_normalization=normalized_input,
                pair_interaction=pair_potential,
                cutoff_function=cutoff_function,
                name="%s-phi" % pair_type,
            )
            pair_rho[pair_type] = NormalizedPairInteraction(
                input_normalization=normalized_input,
                pair_interaction=rho,
                cutoff_function=cutoff_function,
                name="%s-rho" % pair_type,
            )
        embedding_function = self.get_embedding()
        embedding_functions = {t: embedding_function for t in self.atom_types}
        offsets = {
            t: OffsetLayer(t, self.hyperparams["offset_trainable"], name=f"{t}-offset")
            for t in self.atom_types
        }
        return pair_potentials, pair_rho, embedding_functions, offsets


class ExtendedEmbeddingModel(SMATB):
    def get_embedding(self, type):
        return ExtendedEmbedding(name="%s-Embedding" % type)


class ExtendedEmbeddingV2Model(SMATB):
    def get_embedding(self, type):
        return ExtendedEmbeddingV2(name="%s-Embedding" % type)


class ExtendedEmbeddingV3Model(SMATB):
    def get_embedding(self, type):
        return ExtendedEmbeddingV3(name="%s-Embedding" % type)


class ExtendedEmbeddingV4Model(SMATB):
    def get_embedding(self, type):
        return ExtendedEmbeddingV4(name="%s-Embedding" % type)


class CommonExtendedEmbeddingV4Model(CommonEmbeddingSMATB):
    def get_embedding(self):
        return ExtendedEmbeddingV4(name="Common-Embedding")


class NNEmbeddingModel(SMATB):
    def get_embedding(self, atom_type):
        return NNSqrtEmbedding(
            layers=self.params.get(("F_layers", atom_type), [20, 20]),
            regularization=self.hyperparams.get("regularization", 1e-5),
            name="%s-Embedding" % atom_type,
        )


class CommonNNEmbeddingModel(CommonEmbeddingSMATB):
    def get_embedding(self):
        # Tensorflow requires keys to have a consistent type, therefore a
        # tuple is used.
        return NNSqrtEmbedding(
            layers=self.params.get(("F_layers",), [20, 20]),
            regularization=self.hyperparams.get("regularization", 1e-5),
            name="Common-Embedding",
        )


class RhoTwoExpModel(SMATB):
    def get_rho(self, pair_type):
        return DoubleExpRho(
            pair_type,
            xi_1=self.params.get(("xi_1", pair_type), 1.6),
            q_1=self.params.get(("q_1", pair_type), 3.5),
            xi_2=self.params.get(("xi_2", pair_type), 0.8),
            q_2=self.params.get(("q_2", pair_type), 1.0),
            name="Rho-%s" % pair_type,
        )


class NNRhoModel(SMATB):
    def get_rho(self, pair_type):
        return NNRho(
            pair_type,
            layers=self.params.get(("rho_layers", pair_type), [20, 20]),
            regularization=self.hyperparams.get("regularization", 1e-5),
            name="Rho-%s" % pair_type,
        )


class NNRhoExpModel(SMATB):
    def get_rho(self, pair_type):
        return NNRhoExp(
            pair_type,
            layers=self.params.get(("rho_layers", pair_type), [20, 20]),
            regularization=self.hyperparams.get("regularization", 1e-5),
            name="Rho-%s" % pair_type,
        )


class ExtendedEmbeddingRhoTwoExpModel(ExtendedEmbeddingModel, RhoTwoExpModel):
    """Combination of ExtendedEmbeddingModel and RhoTwoExpModel"""


class ExtendedEmbeddingV3RhoTwoExpModel(ExtendedEmbeddingV3Model, RhoTwoExpModel):
    """Combination of ExtendedEmbeddingV3Model and RhoTwoExpModel"""


class ExtendedEmbeddingV4RhoTwoExpModel(ExtendedEmbeddingV4Model, RhoTwoExpModel):
    """Combination of ExtendedEmbeddingV4Model and RhoTwoExpModel"""


class NNEmbeddingNNRhoModel(NNEmbeddingModel, NNRhoModel):
    """Combination of NNEmbeddingModel and NNRhoModel"""


class NNEmbeddingNNRhoExpModel(NNEmbeddingModel, NNRhoExpModel):
    """Combination of NNEmbeddingModel and NNRhoExpModel"""


class CommonNNEmbeddingNNRhoModel(CommonNNEmbeddingModel):
    def get_rho(self, pair_type):
        return NNRho(
            pair_type,
            layers=self.params.get(("rho_layers", pair_type), [20, 20]),
            regularization=self.hyperparams.get("regularization", 1e-5),
            name="Rho-%s" % pair_type,
        )


class CommonExtendedEmbeddingV4RhoTwoExpModel(CommonExtendedEmbeddingV4Model):
    def get_rho(self, pair_type):
        return DoubleExpRho(
            pair_type,
            xi_1=self.params.get(("xi_1", pair_type), 1.6),
            q_1=self.params.get(("q_1", pair_type), 3.5),
            xi_2=self.params.get(("xi_2", pair_type), 0.8),
            q_2=self.params.get(("q_2", pair_type), 1.0),
            name="Rho-%s" % pair_type,
        )
