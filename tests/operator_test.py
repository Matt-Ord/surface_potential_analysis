from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedBasis,
)
from surface_potential_analysis.basis.explicit_basis import ExplicitBasis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator_list import (
    DiagonalOperatorList,
    OperatorList,
    as_operator_list,
    operator_list_from_iter,
    operator_list_into_iter,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import Operator

rng = np.random.default_rng()


def _random_orthonormal(n: int) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    random_matrix = rng.random((n, n))
    q, _ = np.linalg.qr(random_matrix)
    np.testing.assert_array_almost_equal(np.linalg.inv(q), np.conj(np.transpose(q)))
    return q


class OperatorTest(unittest.TestCase):
    def test_as_operator_list(self) -> None:
        n = rng.integers(3, 10)
        m = rng.integers(3, 10)
        diagonal = rng.integers(0, 1000, (n, m)).astype(np.complex128)
        expected = np.zeros((n, m, m), dtype=np.complex128)
        for i in range(n):
            for j in range(m):
                expected[i, j, j] = diagonal[i, j]

        diagonal_list: DiagonalOperatorList[
            FundamentalBasis[int], FundamentalBasis[int], FundamentalBasis[int]
        ] = {
            "basis": TupleBasis(
                FundamentalBasis(n),
                TupleBasis(FundamentalBasis(m), FundamentalBasis(m)),
            ),
            "data": diagonal.reshape(-1),
        }
        actual = as_operator_list(diagonal_list)

        np.testing.assert_array_almost_equal(expected, actual["data"].reshape(n, m, m))

    def test_convert_operator_list_to_basis(self) -> None:
        n = rng.integers(3, 10)
        m = rng.integers(3, 10)
        data = rng.random((n, m, m)).astype(np.complex128)

        op_list: OperatorList[
            FundamentalBasis[int], FundamentalBasis[int], FundamentalBasis[int]
        ] = {
            "basis": TupleBasis(
                FundamentalBasis(n),
                TupleBasis(FundamentalBasis(m), FundamentalBasis(m)),
            ),
            "data": data.reshape(-1),
        }

        new_basis = ExplicitBasis[Any, Any].from_state_vectors(
            {
                "basis": TupleBasis(FundamentalBasis(m), op_list["basis"][1][0]),
                "data": _random_orthonormal(m).ravel(),
            }
        )
        basis = TupleBasis(new_basis, new_basis)

        actual = convert_operator_list_to_basis(op_list, basis)
        expected = operator_list_from_iter(
            [
                convert_operator_to_basis(op, basis)
                for op in operator_list_into_iter(op_list)
            ]
        )
        np.testing.assert_array_almost_equal(actual["data"], expected["data"])

        converted_back = convert_operator_list_to_basis(actual, op_list["basis"][1])

        np.testing.assert_array_almost_equal(converted_back["data"], op_list["data"])

    def test_convert_operator_to_basis_momentum(self) -> None:
        m = rng.integers(3, 10)
        data = rng.random((m, m)).astype(np.complex128)

        operator: Operator[FundamentalBasis[int], FundamentalBasis[int]] = {
            "basis": TupleBasis(FundamentalBasis(m), FundamentalBasis(m)),
            "data": data.reshape(-1),
        }

        new_basis = FundamentalTransformedBasis(m)
        basis = TupleBasis(new_basis, new_basis)

        actual = convert_operator_to_basis(operator, basis)

        converted_back = convert_operator_to_basis(actual, operator["basis"])

        np.testing.assert_array_almost_equal(converted_back["data"], operator["data"])

    def test_convert_operator_to_basis_explicit(self) -> None:
        m = rng.integers(3, 10)
        data = rng.random((m, m)).astype(np.complex128)

        operator: Operator[FundamentalBasis[int], FundamentalBasis[int]] = {
            "basis": TupleBasis(FundamentalBasis(m), FundamentalBasis(m)),
            "data": data.reshape(-1),
        }

        new_basis = ExplicitBasis[FundamentalBasis[int], Any].from_basis(
            FundamentalTransformedBasis(m)
        )
        basis = TupleBasis(new_basis, new_basis)

        actual = convert_operator_to_basis(operator, basis)
        converted_back = convert_operator_to_basis(actual, operator["basis"])
        np.testing.assert_array_almost_equal(converted_back["data"], operator["data"])

        new_basis_1 = ExplicitBasis[Any, Any].from_state_vectors(
            {
                "basis": TupleBasis(
                    FundamentalBasis(m), FundamentalTransformedBasis(m)
                ),
                "data": np.eye(m, dtype=np.complex128).ravel(),
            }
        )
        basis_1 = TupleBasis(new_basis_1, new_basis_1)

        actual_1 = convert_operator_to_basis(operator, basis_1)
        np.testing.assert_array_almost_equal(actual["data"], actual_1["data"])
        converted_back_1 = convert_operator_to_basis(actual_1, operator["basis"])
        np.testing.assert_array_almost_equal(converted_back_1["data"], operator["data"])

        new_basis_2 = ExplicitBasis[Any, Any].from_state_vectors(
            {
                "basis": TupleBasis(
                    FundamentalBasis(m), FundamentalTransformedBasis(m)
                ),
                "data": _random_orthonormal(m).ravel(),
            }
        )
        basis_2 = TupleBasis(new_basis_2, new_basis_2)

        actual_2 = convert_operator_to_basis(operator, basis_2)
        converted_back_2 = convert_operator_to_basis(actual_2, operator["basis"])

        np.testing.assert_almost_equal(
            np.linalg.norm(converted_back_2["data"]), np.linalg.norm(operator["data"])
        )
        np.testing.assert_array_almost_equal(converted_back_2["data"], operator["data"])
