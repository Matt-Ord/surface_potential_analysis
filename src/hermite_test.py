import unittest

import numpy as np

from hermite import hermite_coefficient, hermite_value


class TestHermite(unittest.TestCase):
    def test_hermite_coefficient(self):
        cases = [
            (1, 0, 0),
            (0, 1, 0),
            (2, 1, 1),
            (-30240, 10, 0),
            (13440, 8, 4),
            (32, 5, 5),
            (-9216, 9, 7),
            (0, 8, 7),
            (1383782400, 15, 7),
        ]
        for (result, n, m) in cases:
            self.assertEqual(hermite_coefficient(n, m), result)

    def test_hermite_value_vectorization(self):
        points = np.linspace(0, 1, 10)
        out = hermite_value(4, points)
        for (x, result) in zip(points, out):
            self.assertEqual(hermite_value(4, x), result)
