import tensorflow as tf
import numpy as np
import json
from mlff.data_prep import dataset_from_json, preprocessed_dataset_from_json
from mlff.models import SMATB
import os

type_dict = {'Ni': 0, 'Pt': 1}

# Use smaller cutoff for preprocessed dataset!
dataset_train = preprocessed_dataset_from_json(
    '../data/train_data.json', type_dict, cutoff=5.006, batch_size=50)
dataset_val = preprocessed_dataset_from_json(
    '../data/val_data.json', type_dict, cutoff=5.006)
#dataset_train = dataset_from_json('../data/train_data.json', type_dict, batch_size=50)
#dataset_val = dataset_from_json('../data/val_data.json', type_dict)

params = {
    ('A', 'PtPt'): 0.1602, ('A', 'NiPt'): 0.1346, ('A', 'NiNi'): 0.0845,
    ('xi', 'PtPt'): 2.1855, ('xi', 'NiPt'): 2.3338, ('xi', 'NiNi'): 1.405,
    ('p', 'PtPt'): 13.00, ('p', 'NiPt'): 14.838, ('p', 'NiNi'): 11.73,
    ('q', 'PtPt'): 3.13, ('q', 'NiPt'): 3.036, ('q', 'NiNi'): 1.93,
    ('r0', 'PtPt'): 2.77, ('r0', 'NiPt'): 2.63, ('r0', 'NiNi'): 2.49,
    ('cut_a', 'PtPt'): 4.08707719, ('cut_b', 'PtPt'): 5.0056268338740553,
    ('cut_a', 'NiPt'): 4.08707719, ('cut_b', 'NiPt'): 4.4340500673763259,
    ('cut_a', 'NiNi'): 3.62038672, ('cut_b', 'NiNi'): 4.4340500673763259}
model = SMATB(['Ni', 'Pt'], params=params, build_forces=True)

class RaggedMeanSquaredError(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.squared_difference(y_pred, y_true))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=[RaggedMeanSquaredError(),
                    RaggedMeanSquaredError()])

run_name = 'SMATB/batch_size_50'
if not os.path.exists('./saved_models/%s' % run_name):
    os.makedirs('./saved_models/%s' % run_name)
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./saved_models/%s/model.{epoch:03d}-{val_loss:.4f}.h5' % run_name,
        save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/%s/' % run_name, profile_batch='10,20')
]

model.fit(x=dataset_train, validation_data=dataset_val, epochs=400,
          callbacks=my_callbacks, initial_epoch=0, verbose=2)