import json
import h5py
import tensorflow as tf
import numpy as np
from mlpot.nnpotentials import BPpotential

atom_types = ['Ni', 'Pt']
model = BPpotential(atom_types, input_dims=[50, 50], build_forces=True)

with open('NiPt13.json', 'r') as fin:
    data = json.load(fin)

types = data['types']
Gs, dGs = data['Gs'], data['dGs']
Gs_by_type = {t: [] for t in atom_types}
dGs_by_type = {t: [] for t in atom_types}
for Gi, dGi, ti in zip(Gs, dGs, types):
    Gs_by_type[ti].append(Gi)
    dGs_by_type[ti].append(dGi)

feed_dict = {model.target: np.zeros(1),
             model.target_forces: np.zeros((1, len(types), 3))}
for t in atom_types:
    feed_dict[model.atom_indices[t]] = np.zeros((len(Gs_by_type[t]), 1))
    feed_dict[model.atomic_contributions[t].input] = Gs_by_type[t]
    feed_dict[
        model.atomic_contributions[t].derivatives_input
        ] = dGs_by_type[t]

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.variables_initializer(model.variables))
    energy, forces = sess.run([model.E_predict, model.F_predict], feed_dict)
    variables = sess.run(model.variables)
    print(model.variables)
    print(energy, forces)
    print([var.shape for var in variables])

with open('mlpot_results.json', 'w') as f:
    json.dump({'energy': energy.tolist(), 'forces': forces.tolist()}, f)

with h5py.File('saved_model.h5', 'w') as f:
    f.attrs['backend'] = b'tensorflow'
    f.attrs['layer_names'] = np.array([b'atomic_neural_network',
                                       b'atomic_neural_network_1'])
    f.create_dataset('atomic_neural_network/dense/kernel:0',
                     data=variables[0], dtype='f4')
    f.create_dataset('atomic_neural_network/dense/bias:0',
                     data=variables[1], dtype='f4')
    f.create_dataset('atomic_neural_network/dense_1/kernel:0',
                     data=variables[2], dtype='f4')
    f.create_dataset('atomic_neural_network/dense_1/bias:0',
                     data=variables[3], dtype='f4')
    f['atomic_neural_network'].attrs['weight_names'] = np.array(
        [b'dense/kernel:0', b'dense/bias:0',
         b'dense_1/kernel:0', b'dense_1/bias:0']
    )

    f.create_dataset('atomic_neural_network_1/dense_2/kernel:0',
                     data=variables[4], dtype='f4')
    f.create_dataset('atomic_neural_network_1/dense_2/bias:0',
                     data=variables[5], dtype='f4')
    f.create_dataset('atomic_neural_network_1/dense_3/kernel:0',
                     data=variables[6], dtype='f4')
    f.create_dataset('atomic_neural_network_1/dense_3/bias:0',
                     data=variables[7], dtype='f4')
    f['atomic_neural_network_1'].attrs['weight_names'] = np.array(
        [b'dense_2/kernel:0', b'dense_2/bias:0',
         b'dense_3/kernel:0', b'dense_3/bias:0']
    )
