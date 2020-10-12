import unittest
import json
import numpy as np
import tensorflow as tf
from mlff.models import SMATB


class SMATBTest(unittest.TestCase):

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

        model = SMATB(['Ni', 'Pt'], params=params, build_forces=True)

        types = tf.expand_dims(tf.ragged.constant([types]), axis=2)
        positions = tf.ragged.constant([positions], ragged_rank=1)
        energy, forces = model({'types': types, 'positions': positions})

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

        model = SMATB(['Ni', 'Pt'], params=params, build_forces=True)

        e_model, _ = model({'types': types, 'positions': positions})
        e_model = tf.squeeze(e_model)*Ns

        # High tolerance since we know that the Ferrando code uses a different
        # cutoff style
        np.testing.assert_allclose(e_model.numpy(), e_ref, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
