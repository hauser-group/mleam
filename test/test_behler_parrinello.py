import unittest
import json
import tensorflow as tf
import numpy as np
from mlff.models import BehlerParrinello


class BehlerParrinelloTest(unittest.TestCase):

    def test_saving_and_loading(self):
        with open('behler_parrinello_reference/NiPt13.json', 'r') as fin:
            data = json.load(fin)

        type_dict = {'Ni': 0, 'Pt': 1}
        types = tf.ragged.constant(
            [[[type_dict[t]] for t in data['types']]], ragged_rank=1)
        Gs = tf.ragged.constant([data['Gs']], ragged_rank=2)
        dGs = tf.ragged.constant([data['dGs']], ragged_rank=3)

        model = BehlerParrinello(['Ni', 'Pt'], num_Gs={'Ni': 50, 'Pt': 50},
                                 build_forces=True)

        prediction = model({'types': types, 'Gs': Gs, 'dGs': dGs})
        energy, forces = prediction['energy_per_atom'], prediction['forces']
        model.save_weights('./tmp_model.h5')

        model2 = BehlerParrinello(['Ni', 'Pt'], num_Gs={'Ni': 50, 'Pt': 50},
                                  build_forces=True)

        # Model needs to be called once before loading in order to
        # determine all weight sizes...
        model2({'types': types, 'Gs': Gs, 'dGs': dGs})

        model2.load_weights('./tmp_model.h5')
        prediction2 = model2({'types': types, 'Gs': Gs, 'dGs': dGs})
        energy2, forces2 = (prediction2['energy_per_atom'],
                            prediction2['forces'])

        np.testing.assert_allclose(energy2.numpy(), energy.numpy())
        np.testing.assert_allclose(forces2.numpy(), forces.numpy())

    def test_versus_mlpot(self):
        tf.keras.backend.clear_session()
        with open(
                'behler_parrinello_reference/mlpot_results.json', 'r') as fin:
            data = json.load(fin)
        energy_ref = np.array(data['energy'])
        forces_ref = np.array(data['forces'])

        with open('behler_parrinello_reference/NiPt13.json', 'r') as fin:
            data = json.load(fin)

        type_dict = {'Ni': 0, 'Pt': 1}
        types = tf.ragged.constant(
            [[[type_dict[t]] for t in data['types']]], ragged_rank=1)
        Gs = tf.ragged.constant([data['Gs']], ragged_rank=2)
        dGs = tf.ragged.constant([data['dGs']], ragged_rank=3)

        model = BehlerParrinello(['Ni', 'Pt'], num_Gs={'Ni': 50, 'Pt': 50},
                                 layers={'Ni': [20], 'Pt': [20]},
                                 build_forces=True)
        # Model needs to be called once before loading in order to
        # determine all weight sizes...
        model({'types': types, 'Gs': Gs, 'dGs': dGs})
        model.load_weights('behler_parrinello_reference/saved_model.h5')

        prediction = model({'types': types, 'Gs': Gs, 'dGs': dGs})
        energy, forces = prediction['energy_per_atom'], prediction['forces']
        energy = energy[0]*13

        # Save weights again for debugging
        model.save_weights('./tmp_model.h5')
        np.testing.assert_allclose(energy.numpy(), energy_ref, atol=1e-6)
        np.testing.assert_allclose(forces.numpy(), forces_ref, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
