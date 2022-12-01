import random
import unittest

import numpy as np
from scipy.constants import hbar

from energy_data import EnergyInterpolation
from hamiltonian import SurfaceHamiltonian, SurfaceHamiltonianConfig


def generate_random_potential(width=5):
    random_array = np.random.rand(width + 1, width + 1)

    out = np.zeros_like(random_array, dtype=float)
    out += random_array[::+1, ::+1]
    out += random_array[::-1, ::+1]
    out += random_array[::+1, ::-1]
    out += random_array[::-1, ::-1]
    out += random_array[::+1, ::+1].T
    out += random_array[::-1, ::+1].T
    out += random_array[::+1, ::-1].T
    out += random_array[::-1, ::-1].T
    return out[:width, :width]


def generate_symmetrical_points(height, width=5):
    return np.swapaxes([generate_random_potential(width) for _ in range(height)], 0, -1)


class TestSurfaceHamiltonian(unittest.TestCase):
    def test_diagonal_energies(self) -> None:
        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy()

        self.assertTrue(np.array_equal(diagonal_energy, expected))

    def test_get_all_coordinates(self) -> None:
        nkx = random.randrange(1, 20)
        nky = random.randrange(1, 20)
        nz = random.randrange(1, 100)

        xt, yt, zt = np.meshgrid(range(nkx), range(nky), range(nz), indexing="ij")
        expected = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T
        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((nkx, nky, nz), data, config)
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

        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((nkx, nky, nz), data, config)

        coords = hamiltonian._get_all_coordinates()
        for (i, c) in enumerate(coords):
            self.assertEqual(i, hamiltonian.get_index(*c))

    def test_get_sho_potential(self) -> None:
        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(2, 2, 5)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        self.assertTrue(np.array_equal(expected, hamiltonian.get_sho_potential()))

    def test_get_sho_subtracted_points(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)

        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -20,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        data2: EnergyInterpolation = {
            "points": np.tile(hamiltonian.get_sho_potential(), (nx, ny, 1)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data2, config)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(nx, ny, nz))

        self.assertTrue(np.allclose(expected, actual))

    def test_delta_x(self):
        nx = random.randrange(2, 10) * 2
        ny = random.randrange(2, 10) * 2
        x_points = np.linspace(0, 2 * np.pi * hbar, num=nx)
        y_points = np.linspace(0, 2 * np.pi * hbar, num=ny)
        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx - 1, ny - 1, 5)).tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        self.assertAlmostEqual(x_points[-1], hamiltonian.delta_x)
        self.assertAlmostEqual(y_points[-1], hamiltonian.delta_y)

    def test_fft(self) -> None:
        x_points = np.linspace(0, 2 * np.pi * hbar, 4, endpoint=False).tolist()
        y_points = np.linspace(0, 2 * np.pi * hbar, 4, endpoint=False).tolist()

        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(4, 4, 5)).tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        points = np.tile(hamiltonian.get_sho_potential(), (4, 4, 1))
        points[0, 0, 0] = 1
        # points[4, 4, 0] = 1
        # points[4, 0, 0] = 1
        # points[0, 4, 0] = 1

        # points = generate_random_points()

        data2: EnergyInterpolation = {
            "points": points.tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data2, config)
        print(hamiltonian.points)
        print(hamiltonian.get_sho_subtracted_points())
        print(np.imag(hamiltonian.get_ft_potential()[:, :, 0]))
        print(np.real(hamiltonian.get_ft_potential()[:, :, 0]))

        self.assertTrue(np.all(np.isreal(hamiltonian.get_ft_potential())))

    def test_get_fft_is_real(self) -> None:
        nx = random.randrange(1, 10) * 2
        ny = nx
        nz = random.randrange(2, 100)
        x_points = np.linspace(0, 2 * np.pi * hbar, nx).tolist()
        y_points = np.linspace(0, 2 * np.pi * hbar, ny).tolist()

        points = generate_symmetrical_points(nz, nx)
        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        self.assertTrue(np.all(np.isreal(hamiltonian.get_ft_potential())))

    def test_get_off_diagonal_energies_zero(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)
        x_points = np.linspace(0, 2 * np.pi * hbar, nx).tolist()
        y_points = np.linspace(0, 2 * np.pi * hbar, ny).tolist()

        resolution = (2, 2, 2)
        config: SurfaceHamiltonianConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -20,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian(resolution, data, config)

        data2: EnergyInterpolation = {
            "points": np.tile(hamiltonian.get_sho_potential(), (nx, ny, 1)).tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian(resolution, data2, config)
        actual = hamiltonian.get_off_diagonal_energies()
        expected_shape = (np.prod(resolution), np.prod(resolution))

        self.assertTrue(np.array_equal(actual, np.zeros(shape=expected_shape)))
