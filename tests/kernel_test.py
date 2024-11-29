from __future__ import annotations

import unittest

import numpy as np

from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
    StackedBasis,
)
from surface_potential_analysis.kernel.kernel import (
    DiagonalNoiseKernel,
    NoiseKernel,
    as_noise_kernel,
    get_diagonal_noise_kernel,
    get_noise_kernel,
    get_single_factorized_noise_operators,
    get_single_factorized_noise_operators_diagonal,
)
from surface_potential_analysis.operator.operator_list import (
    as_operator_list,
)

rng = np.random.default_rng()


class KernelTest(unittest.TestCase):
    def test_single_factorized_kernel(self) -> None:
        n = rng.integers(3, 10)

        # Kernel such that G_ij,kl = \del_ij \del_kl G_i k
        data = rng.random(size=(n**2, n**2)) + 1j * rng.random(size=(n**2, n**2))
        data += np.transpose(np.conj(data))
        data = data.reshape(n, n, n, n).swapaxes(0, 1).reshape(-1)

        basis = FundamentalBasis(n)
        kernel: NoiseKernel[
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
        ] = {
            "basis": StackedBasis(
                StackedBasis(basis, basis), StackedBasis(basis, basis)
            ),
            "data": data.reshape(-1),
        }

        actual = get_single_factorized_noise_operators(kernel)
        converted = get_noise_kernel(actual)
        np.testing.assert_array_almost_equal(
            converted["data"].reshape(n, n, n, n), kernel["data"].reshape(n, n, n, n)
        )

    def test_single_factorized_kernel_block_diagonal(self) -> None:
        n = 2  # rng.integers(3, 10)

        # Kernel such that G_ij,kl = \del_ij \del_kl G_i k
        operator = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
        operator += np.transpose(np.conj(operator))
        data = np.zeros((n, n, n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                data[i, i, j, j] = operator[i, j]

        basis = FundamentalBasis(n)
        kernel: NoiseKernel[
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
        ] = {
            "basis": StackedBasis(
                StackedBasis(basis, basis), StackedBasis(basis, basis)
            ),
            "data": data.reshape(-1),
        }

        actual = get_single_factorized_noise_operators(kernel)
        expected = np.linalg.eig(operator)
        # What happened to the second eigenvalue...
        np.testing.assert_array_almost_equal(
            actual["eigenvalue"][np.argsort(np.abs(actual["eigenvalue"]))[2:]],
            expected.eigenvalues,
        )

        converted = get_noise_kernel(actual)
        np.testing.assert_array_almost_equal(converted["data"], kernel["data"])

    def test_single_factorized_diagonal_kernel(self) -> None:
        n = rng.integers(3, 10)

        # Kernel such that G_ij,kl = \del_ij \del_kl G_i k
        data = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
        data += np.transpose(np.conj(data))

        basis = FundamentalBasis(n)
        kernel: DiagonalNoiseKernel[
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
        ] = {
            "basis": StackedBasis(
                StackedBasis(basis, basis), StackedBasis(basis, basis)
            ),
            "data": data.reshape(-1),
        }

        actual = get_single_factorized_noise_operators_diagonal(kernel)

        converted = get_diagonal_noise_kernel(actual)

        np.testing.assert_array_almost_equal(converted["data"], kernel["data"])

        truncated = get_diagonal_noise_kernel(
            {
                "eigenvalue": actual["eigenvalue"][1:],
                "basis": StackedBasis(FundamentalBasis(n - 1), actual["basis"][1]),
                "data": data.reshape(n, -1)[1:].reshape(-1),
            }
        )
        truncated_actual = get_diagonal_noise_kernel(
            get_single_factorized_noise_operators_diagonal(truncated)
        )

        np.testing.assert_array_almost_equal(
            truncated["data"], truncated_actual["data"]
        )

    def test_diagonal_kernel_as_kernel(self) -> None:
        n = rng.integers(3, 10)

        # Kernel such that G_ij,kl = \del_ij \del_kl G_i k
        data = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
        data += np.transpose(np.conj(data))

        basis = FundamentalBasis(n)
        kernel: DiagonalNoiseKernel[
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
        ] = {
            "basis": StackedBasis(
                StackedBasis(basis, basis), StackedBasis(basis, basis)
            ),
            "data": data.reshape(-1),
        }

        factorized = get_single_factorized_noise_operators_diagonal(kernel)
        factorized_full = as_operator_list(factorized)
        factorized_full["eigenvalue"] = factorized["eigenvalue"]

        full_from_factorized = get_noise_kernel(factorized_full)
        full_from_original = as_noise_kernel(kernel)

        np.testing.assert_array_almost_equal(
            full_from_original["data"], full_from_factorized["data"]
        )

    def test_as_noise_kernel(self) -> None:
        n = rng.integers(3, 10)
        m = rng.integers(3, 10)
        diagonal = rng.integers(0, 1000, (n, m)).astype(np.complex128)
        expected = np.zeros((n, n, m, m), dtype=np.complex128)
        for i in range(n):
            for j in range(m):
                expected[i, i, j, j] = diagonal[i, j]

        diagonal_kernel: DiagonalNoiseKernel[
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
            FundamentalBasis[BasisMetadata],
        ] = {
            "basis": StackedBasis(
                StackedBasis(FundamentalBasis(n), FundamentalBasis(n)),
                StackedBasis(FundamentalBasis(m), FundamentalBasis(m)),
            ),
            "data": diagonal.reshape(-1),
        }
        actual = as_noise_kernel(diagonal_kernel)

        np.testing.assert_array_almost_equal(
            expected, actual["data"].reshape(n, n, m, m)
        )
