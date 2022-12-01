import random
import unittest

import numpy as np
from scipy.constants import hbar

from energy_data import (
    EnergyData,
    add_back_symmetry_points,
    extend_z_data,
    repeat_original_data,
)


class TestSurfaceHamiltonian(unittest.TestCase):
    def test_repeat_original_data_shape(self) -> None:

        data: EnergyData = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "x_points": [0, 2 * np.pi * hbar],
            "y_points": [0, 2 * np.pi * hbar],
            "z_points": [0, 1],
        }

        extended = repeat_original_data(data)
        print(np.array(extended["points"]).shape)
        self.assertTrue(np.array_equal(np.array(extended["points"]).shape, (6, 6, 2)))

    def test_repeat_original_data_spacing(self) -> None:
        delta_xy = 2 * np.pi * hbar
        n_xy = random.randrange(1, 10) * 2 + 1
        xy_points = np.linspace(0, delta_xy, n_xy)

        data: EnergyData = {
            "points": np.zeros(shape=(n_xy - 1, n_xy - 1, 2)).tolist(),
            "x_points": xy_points[:-1].tolist(),
            "y_points": xy_points[:-1].tolist(),
            "z_points": [0, 1],
        }

        extended = repeat_original_data(data)
        expected_xy_points = np.linspace(-delta_xy, 2 * delta_xy, 3 * (n_xy - 1) + 1)

        self.assertTrue(np.array_equal(expected_xy_points[:-1], extended["x_points"]))
        self.assertTrue(np.array_equal(expected_xy_points[:-1], extended["y_points"]))

    def test_add_back_symmetry_points(self) -> None:
        delta_xy = 2 * np.pi * hbar
        n_xy = 3
        xy_points = np.linspace(0, delta_xy, n_xy)

        points_xy_plane = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        points = np.swapaxes(np.tile(points_xy_plane, (2, 1, 1)), 0, -1)

        data: EnergyData = {
            "points": points[:-1, :-1].tolist(),
            "x_points": xy_points[:-1].tolist(),
            "y_points": xy_points[:-1].tolist(),
            "z_points": [0, 1],
        }

        extended = add_back_symmetry_points(data)

        self.assertTrue(np.array_equal(xy_points, extended["x_points"]))
        self.assertTrue(np.array_equal(xy_points, extended["y_points"]))
        self.assertTrue(np.array_equal(extended["points"], points))

    def test_extend_z_data(self) -> None:
        points = [[[1.0, 2.0, 3.0]]]

        data: EnergyData = {
            "points": points,
            "x_points": [0],
            "y_points": [0],
            "z_points": [0, 1, 4],
        }

        extended = extend_z_data(data)

        self.assertTrue(np.array_equal(extended["points"], [[[1, 1, 1, 2, 3, 3, 3]]]))
        self.assertTrue(
            np.array_equal(extended["z_points"], [-2.0, -1.0, 0.0, 1.0, 4.0, 7.0, 10.0])
        )
