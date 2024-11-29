from __future__ import annotations

import unittest
from typing import Literal

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.legacy import (
    ExplicitBasis3d,
    FundamentalPositionBasis,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.sho_basis import (
    SHOBasisConfig,
    calculate_sho_wavefunction,
    infinate_sho_basis_3d_from_config,
    sho_basis_3d_from_config,
)

rng = np.random.default_rng()


def _normalize_sho_basis(basis: ExplicitBasis3d[int, int]) -> ExplicitBasis3d[int, int]:
    vectors = basis.vectors
    turning_point = vectors[
        np.arange(vectors.shape[0]),
        np.argmax(np.abs(vectors[:, : vectors.shape[1] // 2]), axis=1),
    ]

    normalized = np.exp(-1j * np.angle(turning_point))[:, np.newaxis] * vectors
    return ExplicitBasis3d(basis.delta_x, normalized)


class SHOBasisTest(unittest.TestCase):
    def test_sho_normalization(self) -> None:
        mass = hbar**2
        sho_omega = 1 / hbar
        x_points = np.linspace(-10, 10, 1001, dtype=np.complex128)

        for iz1 in range(12):
            for iz2 in range(12):
                sho_1 = calculate_sho_wavefunction(x_points, mass, sho_omega, iz1)
                sho_2 = calculate_sho_wavefunction(x_points, mass, sho_omega, iz2)
                sho_norm = (x_points[1] - x_points[0]) * np.sum(
                    sho_1 * sho_2, dtype=float
                )

                if iz1 == iz2:
                    self.assertAlmostEqual(sho_norm, 1.0)
                else:
                    self.assertAlmostEqual(sho_norm, 0.0)

    def test_calculate_sho_wavefunction(self) -> None:
        mass = hbar**2
        sho_omega = 1 / hbar
        z_points = np.linspace(
            -10,
            10,
            rng.integers(low=0, high=1000),
            dtype=np.complex128,  # type: ignore bad libary types
        )

        norm = np.sqrt(mass * sho_omega / hbar)

        phi_0_norm = np.sqrt(norm / np.sqrt(np.pi))
        phi_0_expected = phi_0_norm * np.exp(-((z_points * norm) ** 2) / 2)
        phi_0_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 0)

        np.testing.assert_allclose(phi_0_expected, phi_0_actual)

        phi_1_norm = np.sqrt(2 * norm / np.sqrt(np.pi))
        phi_1_expected = phi_1_norm * z_points * np.exp(-((z_points * norm) ** 2) / 2)
        phi_1_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 1)

        np.testing.assert_allclose(phi_1_expected, phi_1_actual)

        phi_2_norm = np.sqrt(norm / (2 * np.sqrt(np.pi)))
        phi_2_poly = (2 * z_points**2 - 1) * np.exp(-((z_points * norm) ** 2) / 2)
        phi_2_expected = phi_2_norm * phi_2_poly
        phi_2_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 2)

        np.testing.assert_allclose(phi_2_expected, phi_2_actual)

        phi_3_norm = np.sqrt(norm / (3 * np.sqrt(np.pi)))
        phi_3_poly = (2 * z_points**3 - 3 * z_points) * np.exp(
            -((z_points * norm) ** 2) / 2
        )
        phi_3_expected = phi_3_norm * phi_3_poly
        phi_3_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 3)

        np.testing.assert_allclose(phi_3_expected, phi_3_actual)

    def test_get_sho_rust(self) -> None:
        mass = hbar**2 * rng.random()
        sho_omega = rng.random() / hbar
        z_points = np.linspace(
            -20 * rng.random(), 20 * rng.random(), 1000, dtype=np.float64
        )

        for n in range(14):
            actual = hamiltonian_generator.get_sho_wavefunction(
                z_points.tolist(), sho_omega, mass, n
            )
            expected = calculate_sho_wavefunction(z_points, sho_omega, mass, n)

            np.testing.assert_allclose(actual, expected)

    def test_infinate_sho_basis_from_config_normalization(self) -> None:
        nz = 12
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }
        parent: FundamentalPositionBasis3d[Literal[1001]] = FundamentalPositionBasis(
            np.array([0, 0, 20]), 1001
        )
        basis = BasisUtil(infinate_sho_basis_3d_from_config(parent, config, 12))
        np.testing.assert_almost_equal(
            np.ones((nz,)), np.sum(basis.vectors * np.conj(basis.vectors), axis=1)
        )

    def test_sho_config_basis_normalization(self) -> None:
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -5 * np.pi]),
        }
        parent = FundamentalPositionBasis(np.array([0, 0, 10 * np.pi]), 1001)

        axis = BasisUtil(sho_basis_3d_from_config(parent, config, 12))

        norm = np.linalg.norm(axis.vectors, axis=1)
        np.testing.assert_array_almost_equal(norm, np.ones_like(norm))

        norm = np.linalg.norm(axis.vectors, axis=1)
        np.testing.assert_array_almost_equal(norm, np.ones_like(norm))

    def test_infinate_sho_normal_sho_config(self) -> None:
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -5 * np.pi]),
        }
        parent = FundamentalPositionBasis(np.array([0, 0, 10 * np.pi]), 1001)

        basis1 = _normalize_sho_basis(
            infinate_sho_basis_3d_from_config(parent, config, 16)
        )
        basis2 = _normalize_sho_basis(sho_basis_3d_from_config(parent, config, 16))
        np.testing.assert_array_almost_equal(
            BasisUtil(basis1).vectors,
            BasisUtil(basis2).vectors,
        )
