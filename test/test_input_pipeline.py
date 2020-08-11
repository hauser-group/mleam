import unittest
import numpy as np
import tensorflow as tf
from mlff.data_prep import distances_and_pair_types
from utils import rotation_matrix


class InputTest(unittest.TestCase):
    def test_distances_and_pair_types(self, atol=1e-6):
        # Distance matrix:
        # 0.0 1.5 3.0 2.5
        # 1.5 0.0 3.4 3.7
        # 3.0 3.4 0.0 2.5
        # 2.5 3.7 2.5 0.0
        xyzs = tf.constant([[0, 0, 0],
                            [1.5, 0, 0],
                            [0, 3.0, 0],
                            [0, 1.5, 2.0]])
        types = tf.expand_dims(tf.constant([0, 0, 1, 1]), axis=-1)

        r, pair_types, dr_dx = distances_and_pair_types(xyzs, types, 2,
                                                        cutoff=2.6)

        # This is awful, however with the current version of tensorflow
        # assertEqual(r.shape, tf.TensorShape([3, None, 1])) because the
        # __eq__ of None Dimensions returns None
        self.assertEqual(r.shape.as_list(), [4, None, 1])
        self.assertEqual(pair_types.shape.as_list(), [4, None, 1])
        self.assertEqual(dr_dx.shape.as_list(), [4, None, None, 3])

        np.testing.assert_allclose(r[0].numpy(), np.array([[1.5],
                                                           [2.5]]), atol=atol)
        np.testing.assert_allclose(r[1].numpy(), np.array([[1.5]]), atol=atol)
        np.testing.assert_allclose(r[2].numpy(), np.array([[2.5]]), atol=atol)
        np.testing.assert_allclose(r[3].numpy(), np.array([[2.5],
                                                           [2.5]]), atol=atol)

        # Test correct pair types
        # (0, 0) -> 0
        # (0, 1), (1, 0) -> 1
        # (1, 1) -> 2
        # Pair_type matrix:
        # 0 0 1 1
        # 0 0 1 1
        # 1 1 2 2
        # 1 1 2 2
        np.testing.assert_array_equal(pair_types[0, :, 0].numpy(), [0, 1])
        np.testing.assert_array_equal(pair_types[1, :, 0].numpy(), [0])
        np.testing.assert_array_equal(pair_types[2, :, 0].numpy(), [2])
        np.testing.assert_array_equal(pair_types[3, :, 0].numpy(), [1, 2])

        # Test invariance with respect to an arbitray rotation and translation
        # Construct random rotation matrix
        R = tf.constant(
            rotation_matrix(np.random.randn(3), np.random.randn(1)[0]),
            dtype=tf.float32)
        # Apply random translation before rotation
        xyzs2 = tf.matmul(xyzs + np.random.randn(3), R)

        r2, pair_types, dr_dx2 = distances_and_pair_types(xyzs2, types, 2,
                                                          cutoff=2.6)

        np.testing.assert_allclose(r.to_tensor().numpy(),
                                   r2.to_tensor().numpy(), atol=atol)

        # Test versus numerical derivative


if __name__ == '__main__':
    unittest.main()
