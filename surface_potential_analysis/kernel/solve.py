from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.kernel.fit import get_cos_series_expansion
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    NoiseKernel,
    NoiseOperatorList,
    SingleBasisDiagonalNoiseOperatorList,
    get_noise_operators_isotropic_stacked,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points
from surface_potential_analysis.util.util import slice_along_axis

_TBL0 = TypeVar("_B0", bound=TupleBasisWithLengthLike[Any, Any])

_B0 = TypeVar("_B0", bound=BasisLike[int, int])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

_TB0 = TypeVar("_TB0", bound=TupleBasisLike[*tuple[Any, ...]])


def get_noise_operators_real_isotropic_stacked(
    kernel: IsotropicNoiseKernel[_TB0],
    *,
    shape: tuple[int, ...] | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _TB0
]:
    """
    Get the noise operators, expanding the kernel about each axis individually.

    FFT method.

    Returns
    -------
    SingleBasisDiagonalNoiseOperatorList[
        TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _TB0
    ]
    """
    np.testing.assert_allclose(np.imag(kernel["data"]), 0)
    shape = tuple(n // 2 for n in kernel["basis"].shape) if shape is None else shape

    standard_operators = get_noise_operators_isotropic_stacked(
        kernel, shape=tuple(2 * n + 1 for n in shape)
    )

    data = standard_operators["data"].reshape(*standard_operators["basis"][0].shape, -1)

    np.testing.assert_allclose(
        standard_operators["eigenvalue"][1::],
        standard_operators["eigenvalue"][1::][::-1],
        rtol=1e-8,
    )

    for axis, n in enumerate(shape):
        cloned = data.copy()
        # Build (e^(ikx) +- e^(-ikx)) operators
        cos_slice = slice_along_axis(slice(1, 1 + n), axis)
        sin_slice = slice_along_axis(slice(1 + n, None), axis)
        reverse_slice = slice_along_axis(slice(None, None, -1), axis)

        data[cos_slice] = (
            cloned[cos_slice] + cloned[sin_slice][reverse_slice]
        ) / np.sqrt(2)
        data[sin_slice] = (
            cloned[cos_slice][reverse_slice] - cloned[sin_slice]
        ) / np.sqrt(2)

    return {
        "basis": standard_operators["basis"],
        "data": data.ravel(),
        "eigenvalue": standard_operators["eigenvalue"],
    }


def get_noise_operators_taylor_expansion(
    kernel: IsotropicNoiseKernel[_TBL0],
    *,
    n: int = 1,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    """Calculate the noise operators for a general isotropic noise kernel.

    Polynomial fitting to get Taylor expansion.

    Parameters
    ----------
    kernel: IsotropicNoiseKernel[TupleBasisWithLengthLike[Any, Any]]
    n: int, by default 1

    Returns
    -------
    The noise operators formed using the 2n+1 lowest fourier terms, and the corresponding coefficients.

    """
    basis_x = stacked_basis_as_fundamental_position_basis(kernel["basis"])
    delta_x = np.linalg.norm(BasisUtil(basis_x).delta_x_stacked[0])
    k = 2 * np.pi / basis_x.shape[0]
    delta_k = (2 * np.pi / delta_x).item()
    nx_points = BasisUtil(basis_x).fundamental_stacked_nx_points[0]
    displacements = (
        (BasisUtil(basis_x).fundamental_stacked_nk_points[0])
        * (BasisUtil(basis_x).dx_stacked[0])
    )
    kernel_data = kernel["data"]
    # weight is chosen such that the 2n+1 points around the origin are selected for fitting
    weight = pad_ft_points(np.ones(2 * n + 1), (basis_x.n,), (0,))
    noise_polynomial = np.polynomial.Polynomial.fit(
        x=displacements,
        y=kernel_data,
        deg=np.arange(0, 2 * n + 1, 2),
        w=weight,
        domain=[-np.min(displacements), np.min(displacements)],
    )
    noise_coefficients = noise_polynomial.convert().coef[::2]
    operator_coefficients = get_cos_series_expansion(
        true_noise_coeff=noise_coefficients,
        d_k=delta_k,
        n_polynomials=n,
    )

    sines = [
        np.sin(i * k * nx_points).astype(np.complex128) for i in np.arange(1, n + 1)
    ]
    coses = [
        np.cos(i * k * nx_points).astype(np.complex128) for i in np.arange(1, n + 1)
    ]
    data = np.append(np.ones_like(nx_points).astype(np.complex128), [sines, coses])

    return {
        "basis": TupleBasis(FundamentalBasis(2 * n + 1), TupleBasis(basis_x, basis_x)),
        "data": data.astype(np.complex128),
        "eigenvalue": (
            np.concatenate([operator_coefficients, operator_coefficients[1:]])
        ).astype(np.complex128),
    }


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
    # np.testing.assert_array_almost_equal(
    #     data,
    #     np.einsum(
    #         "ak,k,kb->ab",
    #         res.eigenvectors,
    #         res.eigenvalues,
    #         np.conj(np.transpose(res.eigenvectors)),
    #     ),
    # )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}
    return {
        "basis": TupleBasis(FundamentalBasis(kernel["basis"][0].n), kernel["basis"][0]),
        "data": np.conj(np.transpose(res.eigenvectors)).reshape(-1),
        "eigenvalue": res.eigenvalues,
    }
