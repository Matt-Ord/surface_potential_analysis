from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, Literal

import hamiltonian_generator
import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import hbar

from surface_potential_analysis.basis.legacy import (
    FundamentalPositionBasis,
    FundamentalPositionBasis1d,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    StackedBasis,
    TransformedPositionBasis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.hamiltonian_builder import (
    momentum_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.hamiltonian_builder.sho_subtracted_basis import (
    _SurfaceHamiltonianUtil,  # type: ignore this is test file
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.build import (
    position_basis_3d_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.stacked_basis.sho_basis import (
    SHOBasisConfig,
    calculate_x_distances,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.util.interpolation import interpolate_points_rfftn

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import (
        Potential,
    )


rng = np.random.default_rng()


def _generate_random_potential(
    width: int = 5,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    random_array = rng.random((width + 1, width + 1))

    out = np.zeros_like(random_array, dtype=float)
    out += random_array[::+1, ::+1]
    out += random_array[::-1, ::+1]
    out += random_array[::+1, ::-1]
    out += random_array[::-1, ::-1]
    out += random_array[::+1, ::+1].T
    out += random_array[::-1, ::+1].T
    out += random_array[::+1, ::-1].T
    out += random_array[::-1, ::-1].T
    return out[:width, :width]  # type: ignore[no-any-return]


def _generate_symmetrical_points(
    height: int, width: int = 5
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    return np.swapaxes(
        [_generate_random_potential(width) for _ in range(height)], 0, -1
    ).ravel()  # type: ignore[no-any-return]


class HamiltonianBuilderTest(unittest.TestCase):
    def test_hamiltonian_from_potential_momentum(self) -> None:
        potential: Potential[
            StackedBasis[FundamentalTransformedPositionBasis1d[Literal[100]]]
        ] = {
            "basis": StackedBasis(
                FundamentalTransformedPositionBasis(np.array([1]), 100)
            ),
            "data": np.array(rng.random(100), dtype=np.complex128),
        }
        actual = momentum_basis.hamiltonian_from_potential(potential)

        converted = convert_potential_to_basis(
            potential, tuple_basis_as_fundamental(potential["basis"])
        )
        expected = convert_operator_to_basis(
            {
                "basis": StackedBasis(converted["basis"], converted["basis"]),
                "data": np.diag(converted["data"]),
            },
            StackedBasis(potential["basis"], potential["basis"]),
        )
        np.testing.assert_array_almost_equal(expected["data"], actual["data"])

    def test_diagonal_energies(self) -> None:
        resolution = (2, 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -1]),
        }
        potential: Potential[Any] = {
            "data": np.zeros(4 * 4 * 3, dtype=np.complex128),
            "basis": position_basis_3d_from_shape(
                (4, 4, 3),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, 3]),
                    ]
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy(  # type: ignore this is testing file # noqa: SLF001
            np.array([0, 0, 0])
        )

        np.testing.assert_array_almost_equal(diagonal_energy, expected)

    def test_get_sho_potential(self) -> None:
        resolution = (2, 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -2]),
        }
        potential: Potential[Any] = {
            "data": np.zeros((4, 4, 5), dtype=np.complex128).ravel(),
            "basis": position_basis_3d_from_shape(
                (4, 4, 5),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, 5]),
                    ]
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        np.testing.assert_equal(expected, hamiltonian.get_sho_potential())

    def test_get_sho_subtracted_points(self) -> None:
        nx = rng.integers(2, 20)  # type: ignore bad libary types
        ny = rng.integers(2, 20)  # type: ignore bad libary types
        nz = rng.integers(2, 100)  # type: ignore bad libary types

        resolution = (nx, ny, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -20]),
        }
        potential: Potential[Any] = {
            "data": np.zeros((2 * nx, 2 * ny, nz), dtype=np.complex128).ravel(),
            "basis": position_basis_3d_from_shape(
                (2 * nx, 2 * ny, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        potential2: Potential[Any] = {
            "data": np.tile(
                hamiltonian.get_sho_potential(), (2 * nx, 2 * ny, 1)
            ).ravel(),
            "basis": position_basis_3d_from_shape(
                (2 * nx, 2 * ny, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential2, config, resolution)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(2 * nx, 2 * ny, nz))

        np.testing.assert_allclose(expected, actual)

    def test_get_fft_is_real(self) -> None:
        width = rng.integers(1, 10) * 2  # type: ignore bad libary types
        nz = rng.integers(2, 100)  # type: ignore bad libary types

        points = _generate_symmetrical_points(nz, width)
        resolution = (width // 2, width // 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -20]),
        }
        potential: Potential[Any] = {
            "data": points,
            "basis": position_basis_3d_from_shape(
                (width, width, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)
        ft_potential = hamiltonian.get_ft_potential()
        np.testing.assert_almost_equal(
            np.imag(ft_potential), np.zeros_like(ft_potential)
        )
        np.testing.assert_almost_equal(np.real(ft_potential), ft_potential)

    def test_is_almost_hermitian(self) -> None:
        width = rng.integers(1, 10) * 2  # type: ignore bad libary types
        nz = rng.integers(2, 100)  # type: ignore bad libary types

        points = _generate_symmetrical_points(nz, width)
        np.testing.assert_allclose(
            points.reshape(width, width, nz)[1:, 1:],
            points.reshape(width, width, nz)[1:, 1:][::-1, ::-1],
        )
        resolution = (width // 2, width // 2, 10)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -2]),
        }
        potential: Potential[Any] = {
            "data": points,
            "basis": position_basis_3d_from_shape(
                (width, width, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        np.testing.assert_allclose(
            hamiltonian.hamiltonian(np.array([0, 0, 0]))["data"],
            hamiltonian.hamiltonian(np.array([0, 0, 0]))["data"].conjugate().T,
        )

    def test_get_hermite_val_rust(self) -> None:
        n = rng.integers(1, 10)  # type: ignore bad libary types
        x = (rng.random() * 10) - 5
        self.assertAlmostEqual(
            hamiltonian_generator.get_hermite_val(x, n),
            scipy.special.eval_hermite(n, [x]).item(0),  # type: ignore bad libary types
            places=6,
        )

    def test_calculate_off_diagonal_energies_rust(self) -> None:
        nx = rng.integers(2, 20)  # type: ignore bad libary types
        ny = rng.integers(2, 20)  # type: ignore bad libary types
        nz = 100

        resolution = (nx // 2, ny // 2, 14)
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, 0]),
        }
        potential: Potential[Any] = {
            "data": np.zeros(shape=(nx, ny, nz), dtype=np.complex128).ravel(),
            "basis": position_basis_3d_from_shape(
                (nx, ny, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        np.testing.assert_allclose(
            hamiltonian._calculate_off_diagonal_energies_fast(),  # noqa: SLF001 # type: ignore this is test file
            hamiltonian._calculate_off_diagonal_energies(),  # noqa: SLF001 # type: ignore this is test file
        )

    def test_total_surface_hamiltonian_simple(self) -> None:
        shape = np.array([3, 3, 200])  # np.random.randint(1, 2, size=3, dtype=int)
        nz = 6

        expected_basis = StackedBasis(
            FundamentalPositionBasis(np.array([2 * np.pi, 0, 0]), shape.item(0)),
            FundamentalPositionBasis(np.array([0, 2 * np.pi, 0]), shape.item(1)),
            FundamentalPositionBasis(np.array([0, 0, 5 * np.pi]), shape.item(2)),
        )
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -2.5 * np.pi]),
        }
        bloch_fraction = np.array([0, 0, 0])

        points = np.tile(
            (
                0.5
                * config["mass"]
                * config["sho_omega"] ** 2
                * np.square(
                    calculate_x_distances(expected_basis[2], config["x_origin"])
                )
            ),
            (shape.item(0), shape.item(1), 1),
        )

        potential: Potential[Any] = {
            "basis": expected_basis,
            "data": points.astype(np.complex128).ravel(),
        }

        momentum_builder_result = momentum_basis.total_surface_hamiltonian(
            potential, config["mass"], bloch_fraction
        )

        interpolated_points = interpolate_points_rfftn(
            points, s=(2 * points.shape[0], 2 * points.shape[1]), axes=(0, 1)
        )

        interpolated_potential: Potential[Any] = {
            "basis": StackedBasis(
                FundamentalPositionBasis(
                    potential["basis"][0].delta_x, interpolated_points.shape[0]
                ),
                FundamentalPositionBasis(
                    potential["basis"][1].delta_x, interpolated_points.shape[1]
                ),
                FundamentalPositionBasis(
                    potential["basis"][2].delta_x, interpolated_points.shape[2]
                ),
            ),
            "data": interpolated_points,  # type: ignore[typeddict-item]
        }

        actual = sho_subtracted_basis.total_surface_hamiltonian(
            interpolated_potential,
            config,
            bloch_fraction,
            (points.shape[0], points.shape[1], nz),
        )

        expected = convert_operator_to_basis(
            momentum_builder_result,
            actual["basis"],
        )

        np.testing.assert_array_almost_equal(actual["data"], expected["data"])

    def test_total_surface_hamiltonian(self) -> None:
        shape = np.array([3, 3, 200])
        nz = 6

        expected_basis = StackedBasis(
            FundamentalPositionBasis(np.array([2 * np.pi, 0, 0]), shape.item(0)),
            FundamentalPositionBasis(np.array([0, 2 * np.pi, 0]), shape.item(1)),
            FundamentalPositionBasis(np.array([0, 0, 5 * np.pi]), shape.item(2)),
        )
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -2.5 * np.pi]),
        }
        bloch_fraction = np.array([0, 0, 0])

        points = rng.random(shape)

        potential: Potential[Any] = {
            "basis": expected_basis,
            "data": points.astype(np.complex128).ravel(),
        }

        momentum_builder_result = momentum_basis.total_surface_hamiltonian(
            potential, config["mass"], bloch_fraction
        )

        interpolated_points = interpolate_points_rfftn(
            points, s=(2 * points.shape[0], 2 * points.shape[1]), axes=(0, 1)
        )

        interpolated_potential: Potential[Any] = {
            "basis": StackedBasis(
                FundamentalPositionBasis(
                    potential["basis"][0].delta_x, interpolated_points.shape[0]
                ),
                FundamentalPositionBasis(
                    potential["basis"][1].delta_x, interpolated_points.shape[1]
                ),
                FundamentalPositionBasis(
                    potential["basis"][2].delta_x, interpolated_points.shape[2]
                ),
            ),
            "points": interpolated_points,  # type: ignore[typeddict-item]
        }

        actual = sho_subtracted_basis.total_surface_hamiltonian(
            interpolated_potential,
            config,
            bloch_fraction,
            (points.shape[0], points.shape[1], nz),
        )

        expected = convert_operator_to_basis(momentum_builder_result, actual["basis"])

        np.testing.assert_array_almost_equal(actual["data"], expected["data"])

    def test_momentum_builder_sho_hamiltonian(self) -> None:
        mass = hbar**2
        omega = 1 / hbar
        basis = StackedBasis(FundamentalPositionBasis(np.array([30]), 1000))
        util = BasisUtil(basis)
        potential: Potential[StackedBasis[FundamentalPositionBasis1d[int]]] = {
            "basis": basis,
            "data": 0.5
            * mass
            * omega**2
            * np.linalg.norm(util.x_points_stacked - 15, axis=0) ** 2,
        }
        hamiltonian = momentum_basis.total_surface_hamiltonian(
            potential, mass, np.array([0])
        )
        eigenstates = calculate_eigenvectors_hermitian(
            hamiltonian, subset_by_index=(0, 50)
        )
        expected = hbar * omega * (util.stacked_nk_points[0].astype(np.float64) + 0.5)
        np.testing.assert_almost_equal(expected[:50], eigenstates["eigenvalue"][:50])

        in_basis = convert_potential_to_basis(
            potential, stacked_basis_as_fundamental_momentum_basis(potential["basis"])
        )
        hamiltonian2 = momentum_basis.total_surface_hamiltonian(
            in_basis, mass, np.array([0])
        )
        eigenstates2 = calculate_eigenvectors_hermitian(
            hamiltonian2, subset_by_index=(0, 50)
        )
        np.testing.assert_almost_equal(expected[:50], eigenstates2["eigenvalue"][:50])

        extended: Potential[StackedBasis[TransformedPositionBasis[int, int, int]]] = {
            "basis": StackedBasis(TransformedPositionBasis(np.array([30]), 1000, 2000)),
            "data": in_basis["data"] * np.sqrt(2000 / 1000),
        }
        converted = convert_potential_to_basis(
            extended, stacked_basis_as_fundamental_momentum_basis(extended["basis"])
        )
        hamiltonian3 = momentum_basis.total_surface_hamiltonian(
            converted, mass, np.array([0])
        )
        eigenstates3 = calculate_eigenvectors_hermitian(
            hamiltonian3, subset_by_index=(0, 50)
        )
        np.testing.assert_almost_equal(expected[:50], eigenstates3["eigenvalue"][:50])
