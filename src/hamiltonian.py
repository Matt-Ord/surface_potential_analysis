import math
from functools import cached_property
from typing import Any, Tuple, TypedDict

import numpy as np
import scipy.special
from numpy.typing import NDArray
from scipy.constants import hbar

from energy_data import EnergyInterpolation


def calculate_sho_wavefunction(z_points, sho_omega, mass, n):
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_z = z_points * norm

    prefactor = np.sqrt(norm / 2**n * math.factorial(n))
    hermite = scipy.special.eval_hermite(n, normalized_z)
    exponential = np.exp(-np.square(normalized_z) / 2)
    return prefactor * hermite * exponential


def check_hermite_normalization(z_points, sho_omega, mass, n):
    pass


class SurfaceHamiltonianConfig(TypedDict):
    sho_omega: float
    """Angular frequency (in rad s-1) of the sho we will fit using"""
    mass: float
    """Mass in Kg"""
    z_offset: float
    """z position of the nz=0 position in the sho well"""


class SurfaceHamiltonian:

    _potential: EnergyInterpolation

    _config: SurfaceHamiltonianConfig

    _resolution: Tuple[int, int, int]

    _hamiltonian: NDArray[Any]

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        potential: EnergyInterpolation,
        config: SurfaceHamiltonianConfig,
    ) -> None:

        self._potential = potential
        self._config = config
        self._resolution = resolution
        self._hamiltonian = np.diag(self._calculate_diagonal_energy())

    @property
    def points(self):
        return np.array(self._potential["points"])

    @property
    def x_points(self):
        """
        Calculate the lattice coordinates in the x direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_x, self.points.shape[0], endpoint=False)

    @property
    def y_points(self):
        """
        Calculate the lattice coordinates in the y direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_y, self.points.shape[1], endpoint=False)

    @property
    def z_points(self):
        nz = self.points.shape[2]
        z_start = self._config["z_offset"]
        z_end = self._config["z_offset"] + (nz - 1) * self._potential["dz"]
        return np.linspace(z_start, z_end, nz)

    @property
    def mass(self):
        return self._config["mass"]

    @property
    def sho_omega(self):
        return self._config["sho_omega"]

    @cached_property
    def delta_x(self):
        return self._potential["delta_x"]

    @cached_property
    def dkx(self):
        return 2 * np.pi / self.delta_x

    @cached_property
    def delta_y(self):
        return self._potential["delta_y"]

    @cached_property
    def dky(self):
        return 2 * np.pi / (self.delta_y)

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
        fft_potential = np.fft.ifft2(subtracted_potential, axes=(0, 1))
        if not np.all(np.isreal(np.real_if_close(fft_potential))):
            raise AssertionError("FFT was not real!")
        return np.real_if_close(fft_potential)

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
    data: EnergyInterpolation = {
        "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        "delta_x": 2 * np.pi * hbar,
        "delta_y": 2 * np.pi * hbar,
        "dz": 1,
    }
    config: SurfaceHamiltonianConfig = {"mass": 1, "sho_omega": 1 / hbar, "z_offset": 0}
    hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)
