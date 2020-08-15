import unittest
import numpy as np
from utils import rotation_matrix


class RotationMatrixTest(unittest.TestCase):

    def test_orthogonality(self, atol=1e-10):
        R = rotation_matrix(np.random.randn(3), np.random.randn(1)[0])
        np.testing.assert_allclose(R.dot(R.T), R.T.dot(R), atol=atol)
        np.testing.assert_allclose(R.dot(R.T), np.eye(3), atol=atol)

    def test_determinant(self):
        R = rotation_matrix(np.random.randn(3), np.random.randn(1)[0])
        np.testing.assert_allclose(np.linalg.det(R), 1)


class NumericalDerivativeTest(unittest.TestCase):

    def test_quadratic_function(self):
        pass
