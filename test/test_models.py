import unittest
import numpy as np
import tensorflow as tf
from mlff.models import SMATB
from utils import derive_scalar_wrt_array


class ModelTest():
    class ModelTest(unittest.TestCase):

        def test_save_and_load(self):
            """Only testing save_weights as standard save does not seem to
            work with ragged output tensors."""
            # Random input
            xyzs = tf.RaggedTensor.from_tensor(
                tf.random.normal((1, 4, 3)), lengths=[4])
            types = tf.ragged.constant([[[0], [1], [0], [1]]], ragged_rank=1)

            model = self.get_random_model(atom_types=['Ni', 'Pt'])
            ref_e, ref_forces = model({'positions': xyzs, 'types': types})

            model.save_weights('./tmp_model.h5')

            # Build fresh model
            model = self.get_model(['Ni', 'Pt'])
            # Model needs to be called once before loading in order to
            # determine all weight sizes...
            model({'positions': xyzs, 'types': types})
            model.load_weights('./tmp_model.h5')
            new_e, new_forces = model({'positions': xyzs, 'types': types})

            np.testing.assert_allclose(new_e.numpy(), ref_e.numpy())
            np.testing.assert_allclose(new_forces.to_tensor().numpy(),
                                       ref_forces.to_tensor().numpy())

        def test_derivative(self, atol=1e-3):
            """ Problem when testing the derivative is the low numerical
                accuracy of the float32 used in the models. The fix for now
                is to use small values of dx and atol"""
            # Number of atoms
            N = 4
            # Random input
            xyzs = tf.RaggedTensor.from_tensor(
                tf.random.normal((1, N, 3)), lengths=[N])
            types = tf.ragged.constant([[[0], [1], [0], [1]]], ragged_rank=1)

            model = self.get_random_model(atom_types=['Ni', 'Pt'])
            _, forces = model({'positions': xyzs, 'types': types})

            def fun(x):
                """ Model puts out energy_per_atom, which has to be multiplied
                    by N to get the total energy"""
                xyzs = tf.RaggedTensor.from_tensor(x, lengths=[N])
                return N*model({'positions': xyzs, 'types': types})[0]
            num_forces = derive_scalar_wrt_array(fun, xyzs.to_tensor().numpy(),
                                                 dx=1e-2)

            np.testing.assert_allclose(forces.to_tensor().numpy(),
                                       num_forces, atol=atol)


class SMATBTest(ModelTest.ModelTest):

    def get_model(self, atom_types=['Ni', 'Pt']):
        # {'foo': 0} is a workaround for a bug in __init__ that should be
        # fixed ASAP
        return SMATB(atom_types, initial_params={'foo': 0}, build_forces=True)

    def get_random_model(self, atom_types=['Ni', 'Pt']):
        # Generate 12 random positive numbers for the SMATB parameters
        p = np.abs(np.random.randn(12))
        initial_params = {
            ('A', 'PtPt'): p[0], ('A', 'NiPt'): p[1], ('A', 'NiNi'): p[2],
            ('xi', 'PtPt'): p[3], ('xi', 'NiPt'): p[4], ('xi', 'NiNi'): p[5],
            ('p', 'PtPt'): p[6], ('p', 'NiPt'): p[7], ('p', 'NiNi'): p[8],
            ('q', 'PtPt'): p[9], ('q', 'NiPt'): p[10], ('q', 'NiNi'): p[11],
            ('r0', 'PtPt'): 2.77, ('r0', 'NiPt'): 2.63, ('r0', 'NiNi'): 2.49,
            ('cut_a', 'PtPt'): 4.087, ('cut_b', 'PtPt'): 5.006,
            ('cut_a', 'NiPt'): 4.087, ('cut_b', 'NiPt'): 4.434,
            ('cut_a', 'NiNi'): 3.620, ('cut_b', 'NiNi'): 4.434}
        model = SMATB(atom_types, initial_params=initial_params,
                      build_forces=True)

        return model


if __name__ == '__main__':
    unittest.main()
