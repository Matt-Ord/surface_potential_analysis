from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

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
from surface_potential_analysis.kernel.fit import (
    get_trig_series_coefficients,
    get_trig_series_data,
)
from surface_potential_analysis.kernel.kernel import (
    DiagonalNoiseKernel,
    DiagonalNoiseOperatorList,
    get_diagonal_noise_kernel,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        DiagonalNoiseOperatorList,
        IsotropicNoiseKernel,
        NoiseKernel,
        NoiseOperatorList,
        SingleBasisDiagonalNoiseOperatorList,
    )

_TBL0 = TypeVar("_TBL0", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])

_B0 = TypeVar("_B0", bound=BasisLike[int, int])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

_TB0 = TypeVar("_TB0", bound=TupleBasisLike[*tuple[Any, ...]])


def get_noise_operators_isotropic_stacked(
    kernel: IsotropicNoiseKernel[_TB0],
    *,
    shape: tuple[int, ...] | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalNoiseOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _TB0
]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    beta(x - x') = 1 / N \sum_k |f(k)|^2 e^(ikx) for some f(k)
    |f(k)|^2 = \sum_x beta(x) e^(-ik.x)

    The independent noise operators are then given by

    L(k) = 1 / N \sum_x e^(ikx) S(x)

    The inddependent operators can therefore be found directly using a FFT
    of the noise beta(x).

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]
        _description_

    Returns
    -------
    DiagonalNoiseOperator[BasisLike[Any, Any], BasisLike[Any, Any]]
        _description_
    """
    shape = kernel["basis"].shape if shape is None else shape
    if assert_periodic:
        ratio = tuple(n % s for n, s in zip(kernel["basis"].shape, shape))
        # Is 2 * np.pi * N / s equal to A * 2 * np.pi for some integer A
        np.testing.assert_array_almost_equal(
            ratio, 0, err_msg="Operators requested are not periodic"
        )
    shape_basis = fundamental_stacked_basis_from_shape(shape)

    coefficients = np.fft.ifftn(
        pad_ft_points(
            kernel["data"].reshape(kernel["basis"].shape),
            shape,
            tuple(range(len(shape))),
        ),
        norm="forward",
    )

    coefficients *= kernel["basis"].n / coefficients.size

    # Operators e^(ik_n0,n1,.. x_m0,m1,..) / sqrt(prod(Mi))
    # with k_n0,n1 = 2 * np.pi * (n0,n1,...) / prod(Ni), ni = 0...Ni
    # and x_m0,m1 = (m0,m1,...), mi = 0...Mi
    k = tuple(2 * np.pi / n for n in shape)
    nk_points = BasisUtil(kernel["basis"]).stacked_nk_points
    i_points = BasisUtil(shape_basis).stacked_nk_points

    operators = np.array(
        [
            np.exp(1j * np.einsum("i,i,ij->j", k, i, nk_points))
            / np.sqrt(kernel["basis"].n)
            for i in zip(*i_points)
        ]
    )

    return {
        "basis": TupleBasis(
            shape_basis,
            TupleBasis(kernel["basis"], kernel["basis"]),
        ),
        "data": operators.ravel(),
        "eigenvalue": coefficients.ravel(),
    }


def get_noise_operators_real_isotropic_stacked_fft(
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


def get_noise_operators_stacked_taylor_expansion(
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
    operator_coefficients = get_trig_series_coefficients(
        polynomial_coefficients=noise_coefficients,
        d_k=delta_k,
        n_coses=n,
    )

    data = get_trig_series_data(k, nx_points, n=n)

    return {
        "basis": TupleBasis(FundamentalBasis(2 * n + 1), TupleBasis(basis_x, basis_x)),
        "data": data.astype(np.complex128),
        "eigenvalue": operator_coefficients.astype(np.complex128),
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
        np.einsum(
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


def truncate_diagonal_noise_kernel(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1], *, n: int | slice
) -> DiagonalNoiseKernel[_B0, _B1, _B0, _B1]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    n : int

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators = get_noise_operators_diagonal_eigenvalue(kernel)

    arg_sort = np.argsort(np.abs(operators["eigenvalue"]))
    args = arg_sort[-n::] if isinstance(n, int) else arg_sort[::-1][n]
    return get_diagonal_noise_kernel(
        {
            "basis": TupleBasis(FundamentalBasis(args.size), operators["basis"][1]),
            "data": operators["data"]
            .reshape(operators["basis"][0].n, -1)[args]
            .ravel(),
            "eigenvalue": operators["eigenvalue"][args],
        }
    )
