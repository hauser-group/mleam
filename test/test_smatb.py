import unittest
import json
import numpy as np
import tensorflow as tf
from mleam.models import SMATB


class SMATBTest(unittest.TestCase):

    params = {
        ('A', 'PtPt'): 0.1602, ('A', 'NiPt'): 0.1346,
        ('A', 'NiNi'): 0.0845, ('xi', 'PtPt'): 2.1855,
        ('xi', 'NiPt'): 2.3338, ('xi', 'NiNi'): 1.405,
        ('p', 'PtPt'): 13.00, ('p', 'NiPt'): 14.838, ('p', 'NiNi'): 11.73,
        ('q', 'PtPt'): 3.13, ('q', 'NiPt'): 3.036, ('q', 'NiNi'): 1.93,
        ('r0', 'PtPt'): 2.77, ('r0', 'NiPt'): 2.63, ('r0', 'NiNi'): 2.49,
        ('cut_a', 'PtPt'): 4.08707719,
        ('cut_b', 'PtPt'): 5.0056268338740553,
        ('cut_a', 'NiPt'): 4.08707719,
        ('cut_b', 'NiPt'): 4.4340500673763259,
        ('cut_a', 'NiNi'): 3.62038672,
        ('cut_b', 'NiNi'): 4.4340500673763259}

    def test_versus_lammps(self):
        # Read reference geometry
        with open('LAMMPS_SMATB_reference/data.NiPt', 'r') as fin:
            flag = False
            # Convert to tensorflow types
            type_dict = {'1': 0, '2': 1}
            types = []
            positions = []
            for line in fin:
                if line.startswith('Atoms # atomic'):
                    # Skip one line:
                    line = next(fin)
                    flag = True
                elif flag:
                    sp = line.split()
                    types.append(type_dict[sp[1]])
                    positions.append([float(s) for s in sp[2:5]])
        N = len(types)
        # Read LAMMPS energy
        with open('LAMMPS_SMATB_reference/log.lammps', 'r') as fin:
            for line in fin:
                if line.startswith('Step Temp E_pair E_mol TotEng Press'):
                    sp = next(fin).split()
                    e_ref = float(sp[4])
        # Read LAMMPS forces:
        with open('LAMMPS_SMATB_reference/dump.force', 'r') as fin:
            flag = False
            forces_ref = []
            for line in fin:
                if line.startswith('ITEM: ATOMS id type fx fy fz'):
                    flag = True
                elif flag:
                    sp = line.split()
                    forces_ref.append([float(s) for s in sp[2:5]])

        model = SMATB(['Ni', 'Pt'], params=self.params, build_forces=True)

        types = tf.expand_dims(tf.ragged.constant([types]), axis=2)
        positions = tf.ragged.constant([positions], ragged_rank=1)
        prediction = model({'types': types, 'positions': positions})
        energy, forces = prediction['energy_per_atom'], prediction['forces']

        np.testing.assert_allclose(energy.numpy()[0]*N, e_ref, rtol=1e-6)
        np.testing.assert_allclose(forces.numpy()[0], forces_ref, atol=1e-5)

    def test_versus_ferrando_code(self):

        with open('../data/test_data.json', 'r') as fin:
            data = json.load(fin)

        type_dict = {'Ni': 0, 'Pt': 1}
        e_ref = data['e_smatb']

        types = tf.ragged.constant(
            [[[type_dict[ti]] for ti in type_vec]
             for type_vec in data['symbols']], ragged_rank=1)
        positions = tf.ragged.constant(data['positions'], ragged_rank=1)
        Ns = tf.cast(types.row_lengths(), dtype=tf.float32)

        model = SMATB(['Ni', 'Pt'], params=self.params, build_forces=True)

        e_model = model({'types': types,
                         'positions': positions})['energy_per_atom']
        e_model = tf.squeeze(e_model)*Ns

        # High tolerance since we know that the Ferrando code uses a different
        # cutoff style
        np.testing.assert_allclose(e_model.numpy(), e_ref, rtol=1e-3)

    def test_body_methods(self, atol=1e-6):
        tf.keras.backend.clear_session()
        # Random input
        xyzs = tf.RaggedTensor.from_tensor(
            tf.random.normal((1, 4, 3)), lengths=[4])
        types = tf.ragged.constant([[[0], [1], [0], [1]]], ragged_rank=1)

        # Generate 12 random positive numbers for the SMATB parameters
        p = np.abs(np.random.randn(12))
        random_params = {
            ('A', 'PtPt'): p[0], ('A', 'NiPt'): p[1], ('A', 'NiNi'): p[2],
            ('xi', 'PtPt'): p[3], ('xi', 'NiPt'): p[4], ('xi', 'NiNi'): p[5],
            ('p', 'PtPt'): p[6], ('p', 'NiPt'): p[7], ('p', 'NiNi'): p[8],
            ('q', 'PtPt'): p[9], ('q', 'NiPt'): p[10], ('q', 'NiNi'): p[11],
            ('r0', 'PtPt'): 2.77, ('r0', 'NiPt'): 2.63, ('r0', 'NiNi'): 2.49,
            ('cut_a', 'PtPt'): 4.087, ('cut_b', 'PtPt'): 5.006,
            ('cut_a', 'NiPt'): 4.087, ('cut_b', 'NiPt'): 4.434,
            ('cut_a', 'NiNi'): 3.620, ('cut_b', 'NiNi'): 4.434}

        model = SMATB(['Ni', 'Pt'], params=random_params, build_forces=True,
                      method='partition_stitch')
        prediction_1 = model({'positions': xyzs, 'types': types})
        e_1, forces_1 = (prediction_1['energy_per_atom'],
                         prediction_1['forces'])
        model.save_weights('./tmp_model.h5')

        model_2 = SMATB(['Ni', 'Pt'], params=random_params, build_forces=True,
                        method='where')
        _ = model_2({'positions': xyzs, 'types': types})
        model_2.load_weights('./tmp_model.h5')
        prediction_2 = model_2({'positions': xyzs, 'types': types})
        e_2, forces_2 = (prediction_2['energy_per_atom'],
                         prediction_2['forces'])

        np.testing.assert_allclose(e_1.numpy(), e_2.numpy(),
                                   equal_nan=False, atol=1e-6)
        np.testing.assert_allclose(forces_1.to_tensor().numpy(),
                                   forces_2.to_tensor().numpy(),
                                   equal_nan=False, atol=1e-6)

        model_3 = SMATB(['Ni', 'Pt'], params=random_params, build_forces=True,
                        method='gather_scatter')
        _ = model_3({'positions': xyzs, 'types': types})
        model_3.load_weights('./tmp_model.h5')
        prediction_3 = model_3({'positions': xyzs, 'types': types})
        e_3, forces_3 = (prediction_3['energy_per_atom'],
                         prediction_3['forces'])

        np.testing.assert_allclose(e_1.numpy(), e_3.numpy(),
                                   equal_nan=False, atol=1e-6)
        np.testing.assert_allclose(forces_1.to_tensor().numpy(),
                                   forces_3.to_tensor().numpy(),
                                   equal_nan=False, atol=1e-6)

    def test_tabulation(self, atol=5e-2):
        try:
            from atsim.potentials import EAMPotential, Potential
        except ImportError:
            print('Skipping tabulation test')
            return
        model = SMATB(['Ni', 'Pt'], params=self.params)
        model.tabulate('tmp', atomic_numbers=dict(Ni=28, Pt=78),
                       atomic_masses=dict(Ni=58.6934, Pt=195.084),
                       cutoff_rho=100.0, nrho=10000, cutoff=6.0, nr=10000)

        def compare_tabs(start_ind):
            ref = np.loadtxt('LAMMPS_SMATB_reference/NiPt.eam.fs',
                             skiprows=start_ind, max_rows=10000)
            test = np.loadtxt('tmp.eam.fs', skiprows=start_ind, max_rows=10000)
            np.testing.assert_allclose(test, ref, atol=atol)

        # F Ni:
        compare_tabs(6)
        # F Pt:
        compare_tabs(6+30000+1)
        # rho NiNi:
        compare_tabs(6+10000)
        # rho NiPt:
        compare_tabs(6+20000)
        # rho PtNi:
        compare_tabs(6+30000+1+10000)
        # rho PtPt:
        compare_tabs(6+30000+1+20000)
        # phi NiNi:
        compare_tabs(6+30000+1+30000)
        # phi NiPt:
        compare_tabs(6+30000+1+30000+10000)
        # phi PtPt:
        compare_tabs(6+30000+1+30000+20000)


if __name__ == '__main__':
    unittest.main()
