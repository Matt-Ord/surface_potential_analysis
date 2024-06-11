from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    TupleBasisLike,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_expectation,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator

rng = np.random.default_rng()


class HamiltonianEigenstates(unittest.TestCase):
    def test_calculate_energy_diagonal(self) -> None:
        basis = StackedBasis[Any, Any, Any](
            FundamentalTransformedPositionBasis(
                np.array([1, 0, 0]),
                rng.integers(1, 10),  # type: ignore bad libary types
            ),
            FundamentalTransformedPositionBasis(
                np.array([0, 1, 0]),
                rng.integers(1, 10),  # type: ignore bad libary types
            ),
            FundamentalTransformedPositionBasis(
                np.array([0, 0, 1]),
                rng.integers(1, 10),  # type: ignore bad libary types
            ),
        )
        energies = rng.random((basis).n)
        hamiltonian: SingleBasisOperator[TupleBasisLike[Any, Any, Any]] = {
            "basis": StackedBasis(basis, basis),
            "data": np.diag(energies),
        }
        actual: list[complex] = []
        for i in range(basis.n):
            vector = np.zeros(shape=(basis.n), dtype=complex)
            vector[i] = np.exp(1j * 2 * np.pi * rng.random())

            actual.append(
                calculate_expectation(
                    hamiltonian,
                    {"basis": basis, "data": vector},
                )
            )

        np.testing.assert_allclose(energies, actual)
