import random
import unittest

import numpy as np
from scipy.constants import hbar

from energy_data import EnergyData
from hamiltonian import SurfaceHamiltonian


class TestSurfaceHamiltonian(unittest.TestCase):
    def test_diagonal_energies(self) -> None:
        data: EnergyData = {
            "mass": 1,
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "sho_omega": 1 / hbar,
            "x_points": [0, 2 * np.pi * hbar],
            "y_points": [0, 2 * np.pi * hbar],
            "z_points": [0, 1],
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy()
        self.assertTrue(np.array_equal(diagonal_energy, expected))

    def test_get_all_coordinates(self) -> None:
        nkx = random.randrange(1, 20)
        nky = random.randrange(1, 20)
        nz = random.randrange(1, 100)

        xt, yt, zt = np.meshgrid(
            range(nkx),
            range(nky),
            range(nz),
            indexing="ij",
        )

        expected = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T

        data: EnergyData = {
            "mass": 1,
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "sho_omega": 1 / hbar,
            "x_points": [0, 2 * np.pi * hbar],
            "y_points": [0, 2 * np.pi * hbar],
            "z_points": [0, 1],
        }
        hamiltonian = SurfaceHamiltonian((nkx, nky, nz), data)
        coords = hamiltonian._get_all_coordinates()
        for (e, a) in zip(expected, coords):
            self.assertEqual(e[0], a[0])
            self.assertEqual(e[1], a[1])
            self.assertEqual(e[2], a[2])
            self.assertEqual(3, len(a))

    def test_get_index(self) -> None:
        nkx = random.randrange(1, 20)
        nky = random.randrange(1, 20)
        nz = random.randrange(1, 100)

        data: EnergyData = {
            "mass": 1,
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "sho_omega": 1 / hbar,
            "x_points": [0, 2 * np.pi * hbar],
            "y_points": [0, 2 * np.pi * hbar],
            "z_points": [0, 1],
        }
        hamiltonian = SurfaceHamiltonian((nkx, nky, nz), data)
        coords = hamiltonian._get_all_coordinates()
        for (i, c) in enumerate(coords):
            self.assertEqual(i, hamiltonian.get_index(*c))

    def test_get_sho_potential(self) -> None:
        data: EnergyData = {
            "mass": 1,
            "points": np.zeros(shape=(2, 2, 5)).tolist(),
            "sho_omega": 1,
            "x_points": [0, 2 * np.pi * hbar],
            "y_points": [0, 2 * np.pi * hbar],
            "z_points": [-2, -1, 0, 1, 2],
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        self.assertTrue(np.array_equal(expected, hamiltonian.get_sho_potential()))

    def test_get_sho_subtracted_points(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)
        x_points = np.linspace(0, 2 * np.pi * hbar, nx).tolist()
        y_points = np.linspace(0, 2 * np.pi * hbar, ny).tolist()
        z_points = np.linspace(-20, 200, nz).tolist()

        data: EnergyData = {
            "mass": 1,
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "sho_omega": 1,
            "x_points": x_points,
            "y_points": y_points,
            "z_points": z_points,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data)

        data2: EnergyData = {
            "mass": 1,
            "points": np.tile(hamiltonian.get_sho_potential(), (nx, ny, 1)).tolist(),
            "sho_omega": 1,
            "x_points": x_points,
            "y_points": y_points,
            "z_points": z_points,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data2)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(nx, ny, nz))

        self.assertTrue(np.allclose(expected, actual))

    def test_get_off_diagonal_energies(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)
        x_points = np.linspace(0, 2 * np.pi * hbar, nx).tolist()
        y_points = np.linspace(0, 2 * np.pi * hbar, ny).tolist()
        z_points = np.linspace(-20, 200, nz).tolist()

        resolution = (2, 2, 2)

        data: EnergyData = {
            "mass": 1,
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "sho_omega": 1,
            "x_points": x_points,
            "y_points": y_points,
            "z_points": z_points,
        }
        hamiltonian = SurfaceHamiltonian(resolution, data)

        data2: EnergyData = {
            "mass": 1,
            "points": np.tile(hamiltonian.get_sho_potential(), (nx, ny, 1)).tolist(),
            "sho_omega": 1,
            "x_points": x_points,
            "y_points": y_points,
            "z_points": z_points,
        }
        hamiltonian = SurfaceHamiltonian(resolution, data2)
        actual = hamiltonian.get_off_diagonal_energies()
        expected_shape = (np.prod(resolution), np.prod(resolution))

        self.assertTrue(np.array_equal(actual, np.zeros(shape=expected_shape)))
