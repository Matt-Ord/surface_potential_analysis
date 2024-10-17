from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import special_ortho_group

from surface_potential_analysis.basis.legacy import (
    ExplicitBasisWithLength,
    FundamentalPositionBasis,
    StackedBasis,
    TransformedPositionBasis,
    convert_matrix,
    convert_vector,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.interpolation import (
    interpolate_points_fftn,
    pad_ft_points,
)
from tests.utils import get_random_explicit_basis

rng = np.random.default_rng()


class BasisConfigConversionTest(unittest.TestCase):
    def test_explicit_basis_vectors(self) -> None:
        fundamental_n = rng.integers(2, 5)  # type: ignore bad libary types

        axis = get_random_explicit_basis(1, fundamental_n=fundamental_n)

        expected = axis.vectors
        actual = BasisUtil(axis).vectors
        np.testing.assert_array_equal(expected, actual)

        a = axis.__from_fundamental__(actual)
        np.testing.assert_array_almost_equal(a, np.eye(axis.n))

    def test_convert_vector_normalization(self) -> None:
        fundamental_shape = (rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5))  # type: ignore bad libary types

        _basis_0 = StackedBasis(
            get_random_explicit_basis(3, fundamental_n=fundamental_shape[0]),
            get_random_explicit_basis(3, fundamental_n=fundamental_shape[1]),
            get_random_explicit_basis(3, fundamental_n=fundamental_shape[2]),
        )
        # Note this only holds if the space spanned by _basis_1 contains the space of _basis_0
        _basis_1 = StackedBasis(
            get_random_explicit_basis(
                3, fundamental_n=fundamental_shape[0], n=fundamental_shape[0]
            ),
            get_random_explicit_basis(
                3, fundamental_n=fundamental_shape[1], n=fundamental_shape[1]
            ),
            get_random_explicit_basis(
                3, fundamental_n=fundamental_shape[2], n=fundamental_shape[2]
            ),
        )

        vector = special_ortho_group.rvs(_basis_0.n)[0]
        converted = convert_vector(vector, _basis_0, _basis_1)

        np.testing.assert_equal(converted.size, _basis_1.n)

        np.testing.assert_array_almost_equal(np.linalg.norm(converted), 1)

        actual_reversed = convert_vector(converted, _basis_1, _basis_0)
        np.testing.assert_array_almost_equal(actual_reversed, vector)

    def test_convert_vector_equivalent(self) -> None:
        fundamental_shape = (rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5))  # type: ignore bad libary types

        _basis_0 = StackedBasis(
            get_random_explicit_basis(3, fundamental_n=fundamental_shape[0]),
            get_random_explicit_basis(3, fundamental_n=fundamental_shape[1]),
            get_random_explicit_basis(3, fundamental_n=fundamental_shape[2]),
        )
        _basis_1 = _basis_0

        vector = special_ortho_group.rvs(_basis_0.n)[0]
        converted = convert_vector(vector, _basis_0, _basis_1)

        np.testing.assert_array_almost_equal(np.linalg.norm(converted), 1)
        np.testing.assert_array_almost_equal(converted, vector)

    def test_convert_vector_truncated_momentum(self) -> None:
        fundamental_n = rng.integers(3, 5)  # type: ignore bad libary types
        n = rng.integers(2, fundamental_n)  # type: ignore bad libary types

        basis_0 = TransformedPositionBasis(np.array([1]), n, fundamental_n)

        actual = convert_vector(
            np.identity(fundamental_n),
            basis_as_transformed_basis(basis_0),
            basis_0,
        )
        expected = pad_ft_points(np.eye(fundamental_n), s=(n,), axes=(1,))

        np.testing.assert_array_almost_equal(actual, expected)

    def test_convert_vector_truncated(self) -> None:
        fundamental_n = rng.integers(3, 10)  # type: ignore bad libary types
        n = rng.integers(2, fundamental_n)  # type: ignore bad libary types

        basis_0 = FundamentalPositionBasis(np.array([1]), n)
        momentum = basis_as_transformed_basis(basis_0)

        initial_points = rng.random(n).astype(np.complex128)
        converted = convert_vector(initial_points, basis_0, momentum)
        truncated_basis = TransformedPositionBasis(np.array([1]), n, fundamental_n)
        scaled = converted * np.sqrt(fundamental_n / n)
        actual = convert_vector(
            scaled,
            truncated_basis,
            basis_as_transformed_basis(truncated_basis),
        )
        expected = interpolate_points_fftn(
            initial_points, s=(fundamental_n,), axes=(0,)
        )
        np.testing.assert_array_almost_equal(actual, expected)

        actual_simple = convert_vector(
            scaled,
            truncated_basis,
            basis_as_transformed_basis(truncated_basis),
        )
        np.testing.assert_array_almost_equal(actual, actual_simple)

    def test_convert_matrix_truncated(self) -> None:
        fundamental_n = rng.integers(3, 10)  # type: ignore bad libary types
        n = rng.integers(2, fundamental_n)  # type: ignore bad libary types

        truncated_basis = TransformedPositionBasis(np.array([1]), n, fundamental_n)
        final_basis = TransformedPositionBasis(
            np.array([1]), fundamental_n, fundamental_n
        )
        matrix = rng.random((n, n))

        actual = convert_matrix(
            matrix, truncated_basis, final_basis, truncated_basis, final_basis
        )
        expected = pad_ft_points(matrix, s=(fundamental_n, fundamental_n), axes=(0, 1))
        np.testing.assert_array_almost_equal(actual, expected)

    def test_convert_matrix_round_trip(self) -> None:
        fundamental_n = rng.integers(3, 10)  # type: ignore bad libary types
        n = rng.integers(2, fundamental_n)  # type: ignore bad libary types

        small_basis = TransformedPositionBasis(np.array([1]), n, n)
        truncated_large_basis = TransformedPositionBasis(
            np.array([1]), n, fundamental_n
        )
        large_basis = TransformedPositionBasis(
            np.array([1]), fundamental_n, fundamental_n
        )

        matrix = rng.random((n, n))
        converted = convert_matrix(
            matrix,
            basis_as_fundamental_basissmall_basis),
            small_basis,
            basis_as_fundamental_basissmall_basis),
            small_basis,
        )
        converted = converted * n / fundamental_n
        converted = convert_matrix(
            converted,
            truncated_large_basis,
            basis_as_fundamental_basistruncated_large_basis),
            truncated_large_basis,
            basis_as_fundamental_basistruncated_large_basis),
        )

        actual = convert_matrix(
            converted,
            basis_as_fundamental_basistruncated_large_basis),
            truncated_large_basis,
            basis_as_fundamental_basistruncated_large_basis),
            truncated_large_basis,
        )
        actual = actual * fundamental_n / n
        actual = convert_matrix(
            actual,
            small_basis,
            basis_as_fundamental_basissmall_basis),
            small_basis,
            basis_as_fundamental_basissmall_basis),
        )

        np.testing.assert_array_almost_equal(actual, matrix)

        actual = convert_matrix(
            converted,
            basis_as_fundamental_basislarge_basis),
            large_basis,
            basis_as_fundamental_basislarge_basis),
            large_basis,
        )
        actual = convert_matrix(
            actual,
            large_basis,
            truncated_large_basis,
            large_basis,
            truncated_large_basis,
        )
        actual = actual * fundamental_n / n
        actual = convert_matrix(
            actual,
            small_basis,
            basis_as_fundamental_basissmall_basis),
            small_basis,
            basis_as_fundamental_basissmall_basis),
        )
        np.testing.assert_array_almost_equal(actual, matrix)

    def test_convert_vector_explicit(self) -> None:
        fundamental_n = rng.integers(3, 10)  # type: ignore bad libary types
        n = rng.integers(2, fundamental_n)  # type: ignore bad libary types

        vectors = special_ortho_group.rvs(fundamental_n)[:n]
        axis = ExplicitBasisWithLength(np.array([0]), vectors.astype(np.complex128))
        actual = convert_vector(
            np.eye(n),
            axis,
            basis_as_fundamental_basisaxis),
        )
        expected = axis.vectors
        np.testing.assert_array_almost_equal(actual, expected)
