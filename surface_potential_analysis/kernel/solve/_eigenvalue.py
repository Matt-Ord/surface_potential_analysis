from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
)
from surface_potential_analysis.kernel.kernel import (
    DiagonalNoiseKernel,
    DiagonalNoiseOperatorList,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        DiagonalNoiseOperatorList,
        NoiseKernel,
        NoiseOperatorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[int, int])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])


def get_noise_operators_eigenvalue(
    kernel: NoiseKernel[_B0, _B1, _B0, _B1],
) -> NoiseOperatorList[FundamentalBasis[int], _B0, _B1]:
    r"""
    Given a noise kernel, find the noise operator which diagonalizes the kernel.

    Eigenvalue method.

    Note these are the operators `L`

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B0, _B0, _B0]
    fit_method: Literal["explicit", "poly fit", "eigenvalue"],
                method used to generate noise operators.
        _description_

    Returns
    -------
    NoiseOperatorList[FundamentalBasis[int], _B0, _B0]
        _description_
    """
    data = (
        kernel["data"]
        .reshape(*kernel["basis"][0].shape, *kernel["basis"][1].shape)
        .swapaxes(0, 1)
        .reshape(kernel["basis"][0].n, kernel["basis"][1].n)
    )
    # Find the n^2 operators which are independent
    # I think this is always true
    np.testing.assert_array_almost_equal(data, np.conj(np.transpose(data)))

    res = np.linalg.eigh(data)
    np.testing.assert_array_almost_equal(
        data,
        np.einsum(  # type: ignore unknown
            "ak,k,kb->ab",
            res.eigenvectors,
            res.eigenvalues,
            np.conj(np.transpose(res.eigenvectors)),
        ),
    )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}
    return {
        "basis": TupleBasis(FundamentalBasis(kernel["basis"][0].n), kernel["basis"][0]),
        "data": np.conj(np.transpose(res.eigenvectors)).reshape(-1),
        "eigenvalue": res.eigenvalues,
    }


def get_noise_operators_diagonal_eigenvalue(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
) -> DiagonalNoiseOperatorList[FundamentalBasis[int], _B0, _B1]:
    r"""
    For a diagonal kernel it is possible to find N independent noise sources, each of which is diagonal.

    Each of these will be represented by a particular noise operator
    ```latex
    Z_i \ket{i}\bra{i}
    ```
    Note we return a list of noise operators, rather than a single noise operator,
    as it is not currently possible to represent a sparse StackedBasis (unless it can
    be represented as a StackedBasis of individual sparse Basis)

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]
        _description_

    Returns
    -------
    DiagonalNoiseOperator[BasisLike[Any, Any], BasisLike[Any, Any]]
        _description_
    """
    data = kernel["data"].reshape(kernel["basis"][0][0].n, -1)
    # Find the n^2 operators which are independent

    # This should be true if our operators are hermitian - a requirement
    # for our finite temperature correction.
    # For isotropic noise, it is always possible to force this to be true
    # As long as we have evenly spaced k (we have to take symmetric and antisymmetric combinations)
    np.testing.assert_allclose(
        data, np.conj(np.transpose(data)), err_msg="kernel non hermitian"
    )
    res = np.linalg.eigh(data)

    np.testing.assert_allclose(
        data,
        np.einsum(  # type: ignore einsum
            "k,ak,kb->ab",
            res.eigenvalues,
            res.eigenvectors,
            np.conj(np.transpose(res.eigenvectors)),
        ),
        rtol=1e-4,
    )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}
    return {
        "basis": TupleBasis(
            FundamentalBasis(kernel["basis"][0][0].n), kernel["basis"][0]
        ),
        "data": np.conj(np.transpose(res.eigenvectors)).reshape(-1),
        "eigenvalue": res.eigenvalues,
    }
