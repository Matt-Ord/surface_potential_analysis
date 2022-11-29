import math
from functools import cached_property
from typing import Any, Tuple

import numpy as np
import numpy.fft as fft
import scipy.special
from numpy.typing import NDArray
from scipy.constants import hbar

from energy_data import EnergyData


def transform_potential(potential: NDArray[Any]):
    return fft.fft2(potential, axes=(0, 1))


def get_off_diagonal_energy(potential: NDArray[Any]):
    pass


def calculate_sho_wavefunction(z_points, sho_omega, mass, n):
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_z = z_points * norm

    prefactor = np.sqrt(norm / 2**n * math.factorial(n))
    hermite = scipy.special.eval_hermite(n, normalized_z)
    exponential = np.exp(-np.square(normalized_z) / 2)
    return prefactor * hermite * exponential


def check_hermite_normalization(z_points, sho_omega, mass, n):
    pass


def is_valid_data(data: EnergyData):
    shape = np.array(data["points"]).shape
    x_points = len(data["x_points"])
    y_points = len(data["y_points"])
    z_points = len(data["z_points"])
    return (
        len(shape) == 3
        and shape[0] == x_points
        and shape[1] == y_points
        and shape[2] == z_points
    )


class SurfaceHamiltonian:
    """Mass in Kg"""

    data: EnergyData

    _resolution: Tuple[int, int, int]

    _hamiltonian: NDArray[Any]

    def __init__(self, resolution: Tuple[int, int, int], data: EnergyData) -> None:
        if not is_valid_data(data):
            raise AssertionError("Data has incorrect dimensions")
        self.data = data
        self._resolution = resolution
        self._hamiltonian = np.diag(self._calculate_diagonal_energy())

    @property
    def points(self):
        return np.array(self.data["points"])

    @property
    def x_points(self):
        return np.array(self.data["x_points"])

    @property
    def y_points(self):
        return np.array(self.data["y_points"])

    @property
    def z_points(self):
        return np.array(self.data["z_points"])

    @property
    def mass(self):
        return self.data["mass"]

    @property
    def sho_omega(self):
        return self.data["sho_omega"]

    @cached_property
    def dkx(self):
        return 2 * np.pi / (self.x_points[-1] - self.x_points[0])

    @cached_property
    def dky(self):
        return 2 * np.pi / (self.y_points[-1] - self.y_points[0])

    @cached_property
    def dz(self):
        delta_z = self.z_points[1] - self.z_points[0]
        return delta_z / self.z_points.shape[0]

    def _get_all_coordinates(self) -> NDArray:
        return np.array([x for x in np.ndindex(*self._resolution)])

    def _calculate_diagonal_energy(self) -> NDArray[Any]:

        coords = self._get_all_coordinates()
        x_coords, y_coords, z_coords = coords.T

        x_energy = (hbar * self.dkx * x_coords) ** 2 / (2 * self.mass)
        y_energy = (hbar * self.dky * y_coords) ** 2 / (2 * self.mass)
        z_energy = (hbar * self.sho_omega) * (z_coords + 0.5)
        return x_energy + y_energy + z_energy

    def get_index(self, nkx: int, nky: int, nz: int) -> int:
        ikx = nkx * self._resolution[1] * self._resolution[2]
        iky = nky * self._resolution[2]
        return ikx + iky + nz

    def get_sho_potential(self) -> NDArray:
        return 0.5 * self.mass * self.sho_omega**2 * np.square(self.z_points)

    def get_sho_subtracted_points(self) -> NDArray:
        return np.subtract(self.points, self.get_sho_potential())

    def get_ft_potential(self) -> NDArray:
        subtracted_potential = self.get_sho_subtracted_points()
        return np.fft.ifft2(subtracted_potential, axes=(0, 1))

    def get_off_diagonal_energies(self) -> NDArray:
        energies = np.zeros(
            shape=(
                self._resolution[0],
                self._resolution[1],
                self._resolution[2],
                self._resolution[2],
            ),
            dtype=complex,
        )
        ft_pot = self.get_ft_potential()

        for [nkx, nky, nz1] in self._get_all_coordinates():
            for nz2 in range(self._resolution[2]):
                hermite1 = calculate_sho_wavefunction(
                    self.z_points, self.sho_omega, self.mass, nz1
                )
                hermite2 = calculate_sho_wavefunction(
                    self.z_points, self.sho_omega, self.mass, nz2
                )
                ft_pot_points = ft_pot[nkx, nky]
                fourier_transform = np.sum(hermite1 * hermite2 * ft_pot_points)

                norm = self.dz
                energies[nkx, nky, nz1, nz2] = norm * fourier_transform

        n_coordinates = len(self._get_all_coordinates())
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates), dtype=complex)
        for [nkx1, nky1, nz1] in self._get_all_coordinates():
            for [nkx2, nky2, nz2] in self._get_all_coordinates():
                index1 = self.get_index(nkx1, nky1, nz1)
                index2 = self.get_index(nkx2, nky2, nz2)
                dkx = nkx2 - nkx1
                dky = nky2 - nky1
                hamiltonian[index1, index2] = energies[dkx, dky, nz1, nz2]

        return hamiltonian


if __name__ == "__main__":
    data: EnergyData = {
        "mass": 1,
        "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        "sho_omega": 1 / hbar,
        "x_points": [0, 2 * np.pi * hbar],
        "y_points": [0, 2 * np.pi * hbar],
        "z_points": [0, 1],
    }
    hamiltonian = SurfaceHamiltonian((2, 2, 2), data)
