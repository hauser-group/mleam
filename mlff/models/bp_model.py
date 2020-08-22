import tensorflow as tf
from mlff.utils import distances_and_pair_types


class BehlerParrinelloModel():

    def __init__(self):
        pass


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
