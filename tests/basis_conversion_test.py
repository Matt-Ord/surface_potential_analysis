from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.legacy import (
    FundamentalTransformedPositionBasis,
    convert_vector,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from tests.utils import get_random_explicit_basis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import BasisWithLengthLike

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _NDInv = TypeVar("_NDInv", bound=int)

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)
    _NF1Inv = TypeVar("_NF1Inv", bound=int)

rng = np.random.default_rng()


def get_basis_conversion_matrix(
    basis_0: BasisWithLengthLike[_N0Inv, _NF0Inv, _NDInv],
    basis_1: BasisWithLengthLike[_N1Inv, _NF1Inv, _NDInv],
) -> np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex128]]:
    """
    Get the matrix to convert one set of axis axes into another.

    Parameters
    ----------
    basis_0 : BasisLike[_N0Inv, _NF0Inv]
    basis_1 : BasisLike[_N1Inv, _NF1Inv]

    Returns
    -------
    np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]
    """
    vectors_0 = BasisUtil(basis_0).vectors
    vectors_1 = BasisUtil(basis_1).vectors
    return np.dot(vectors_0, np.conj(vectors_1).T)  # type: ignore[no-any-return]


def convert_vector_simple(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex128]],
    initial_basis: BasisWithLengthLike,
    final_basis: BasisWithLengthLike,
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    matrix = get_basis_conversion_matrix(initial_basis, final_basis)
    return np.moveaxis(np.tensordot(vector, matrix, axes=([axis], [0])), -1, axis)  # type: ignore[no-any-return]


class BasisConversionTest(unittest.TestCase):
    def test_as_explicit_position_basis_momentum_normalization(self) -> None:
        fundamental_n = rng.integers(2, 5)  # type: ignore bad library types
        n = rng.integers(1, fundamental_n)  # type: ignore bad library types

        basis = get_random_explicit_basis(1, fundamental_n=fundamental_n, n=n)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(BasisUtil(basis).vectors, axis=1), np.ones(n)
        )

        actual = convert_vector(
            BasisUtil(basis).vectors, basis_as_fundamental_basisbasis), basis
        )
        expected = np.eye(n)
        np.testing.assert_almost_equal(actual, expected)
        actual = convert_vector(
            BasisUtil(basis).vectors, basis_as_fundamental_basisbasis), basis
        )
        expected = convert_vector_simple(
            BasisUtil(basis).vectors, basis_as_fundamental_basisbasis), basis
        )
        np.testing.assert_almost_equal(actual, expected)

        actual = convert_vector(
            np.eye(fundamental_n),
            basis_as_fundamental_basisbasis),
            basis,
        )
        expected = BasisUtil(basis).vectors.T
        np.testing.assert_almost_equal(actual, expected)

        actual = convert_vector(
            np.eye(fundamental_n),
            basis_as_fundamental_basisbasis),
            basis,
        )
        expected = convert_vector_simple(
            np.eye(fundamental_n).astype(np.complex128),
            basis_as_fundamental_basisbasis),
            basis,
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_as_explicit_position_basis_momentum(self) -> None:
        n = rng.integers(5, 10)  # type: ignore bad libary types

        basis = FundamentalTransformedPositionBasis(np.array([1]), n)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(BasisUtil(basis).vectors, axis=1), np.ones(n)
        )
        np.testing.assert_array_almost_equal(
            BasisUtil(basis).vectors,
            np.exp(
                (1j * 2 * np.pi)
                * np.arange(n)[:, np.newaxis]
                * np.linspace(0, 1, n, endpoint=False)[np.newaxis, :]
            )
            / np.sqrt(n),
        )

        np.testing.assert_array_almost_equal(
            convert_vector(
                np.eye(n), basis, basis_as_fundamental_basisbasis)
            ),
            np.exp(
                (1j * 2 * np.pi)
                * np.arange(n)[:, np.newaxis]
                * np.linspace(0, 1, n, endpoint=False)[np.newaxis, :]
            )
            / np.sqrt(n),
        )

        np.testing.assert_array_almost_equal(
            convert_vector(
                np.eye(n), basis, basis_as_fundamental_basisbasis)
            ),
            convert_vector_simple(
                np.eye(n).astype(np.complex128),
                basis,
                basis_as_fundamental_basisbasis),
            ),
        )

    def test_get_basis_conversion_matrix_diagonal(self) -> None:
        fundamental_n = rng.integers(2, 5)  # type: ignore bad libary types
        n = rng.integers(1, fundamental_n)  # type: ignore bad libary types

        basis_0 = BasisUtil(
            get_random_explicit_basis(1, fundamental_n=fundamental_n, n=n)
        )
        np.testing.assert_array_equal(basis_0.vectors.shape, (n, fundamental_n))

        matrix = get_basis_conversion_matrix(basis_0, basis_0)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_basis(1, fundamental_n=fundamental_n, n=n)

        matrix = get_basis_conversion_matrix(basis_1, basis_1)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

    def test_basis_conversion_matrix_position(self) -> None:
        fundamental_n = rng.integers(2, 5)  # type: ignore bad libary types
        n = rng.integers(1, fundamental_n)  # type: ignore bad libary types
        basis_0 = get_random_explicit_basis(1, fundamental_n=fundamental_n, n=n)

        matrix = get_basis_conversion_matrix(basis_0, basis_0)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_basis(1, fundamental_n=fundamental_n, n=n)

        matrix = get_basis_conversion_matrix(basis_1, basis_1)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))
