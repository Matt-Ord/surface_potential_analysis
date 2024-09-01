from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisLike,
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.util.interpolation import pad_ft_points
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        IsotropicNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator_list import (
        SingleBasisDiagonalOperatorList,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList

    _TBL0 = TypeVar("_TBL0", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])

    _B0 = TypeVar("_B0", bound=BasisLike[int, int])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

    _TB0 = TypeVar("_TB0", bound=TupleBasisLike[*tuple[Any, ...]])
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[Any, Any, Any])


def _assert_periodic_sample(
    basis_shape: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    ratio = tuple(n % s for n, s in zip(basis_shape, shape, strict=True))
    # Is 2 * np.pi * N / s equal to A * 2 * np.pi for some integer A
    message = (
        "Operators requested for a sample which does not evenly divide the basis shape\n"
        "This would result in noise operators which are not periodic"
    )
    try:
        np.testing.assert_array_almost_equal(ratio, 0, err_msg=message)
    except AssertionError:
        raise AssertionError(message) from None


def _get_operators_for_isotropic_noise(
    basis: _B0,
    *,
    n: int | None = None,
    fundamental_n: int | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalOperatorList[FundamentalBasis[int], _B0]:
    fundamental_n = basis.n if fundamental_n is None else fundamental_n
    if assert_periodic:
        _assert_periodic_sample((basis.n,), (fundamental_n,))
    n = fundamental_n if n is None else n
    # Operators e^(ik_n x_m) / sqrt(M)
    # with k_n = 2 * np.pi * n / N, n = 0...N
    # and x_m = m, m = 0...M
    k = 2 * np.pi / fundamental_n
    nk_points = BasisUtil(basis).nk_points[np.newaxis, :]
    i_points = np.arange(0, n)[:, np.newaxis]

    operators = np.exp(1j * i_points * k * nk_points) / np.sqrt(basis.n)
    return {
        "basis": TupleBasis(
            FundamentalBasis(fundamental_n),
            TupleBasis(basis, basis),
        ),
        "data": operators.astype(np.complex128).ravel(),
    }


def _get_noise_eigenvalues_isotropic_fft(
    kernel: IsotropicNoiseKernel[_B0], *, fundamental_n: int | None = None
) -> ValueList[FundamentalBasis[int]]:
    fundamental_n = kernel["basis"].n if fundamental_n is None else fundamental_n
    coefficients = np.fft.ifft(
        pad_ft_points(kernel["data"], (fundamental_n,), (0,)),
        norm="forward",
    )
    coefficients *= kernel["basis"].n / fundamental_n
    return {
        "basis": FundamentalBasis(fundamental_n),
        "data": coefficients,
    }


def get_noise_operators_isotropic_fft(
    kernel: IsotropicNoiseKernel[_B0],
    *,
    fundamental_n: int | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalNoiseOperatorList[FundamentalBasis[int], _B0]:
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
    fundamental_n = kernel["basis"].n if fundamental_n is None else fundamental_n

    operators = _get_operators_for_isotropic_noise(
        kernel["basis"], fundamental_n=fundamental_n, assert_periodic=assert_periodic
    )
    eigenvalues = _get_noise_eigenvalues_isotropic_fft(
        kernel, fundamental_n=fundamental_n
    )

    return {
        "basis": operators["basis"],
        "eigenvalue": eigenvalues["data"],
        "data": operators["data"],
    }


def get_operators_for_real_isotropic_noise(
    basis: _B0,
    *,
    n: int | None = None,
    fundamental_n: int | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalOperatorList[FundamentalBasis[int], _B0]:
    """Get operators used for real isotropic noise.

    returns the 2n - 1 smallest operators (n frequencies)

    Parameters
    ----------
    basis : _B0
        _description_
    n : int | None, optional
        _description_, by default None

    Returns
    -------
    SingleBasisDiagonalOperatorList[FundamentalBasis[int], _B0]
        _description_
    """
    fundamental_n = basis.n if fundamental_n is None else fundamental_n
    if assert_periodic:
        _assert_periodic_sample((basis.n,), (fundamental_n,))
    n = (fundamental_n // 2) if n is None else n

    k = 2 * np.pi / fundamental_n
    nk_points = BasisUtil(basis).nk_points[np.newaxis, :]

    sines = np.sin(np.arange(n - 1, 0, -1)[:, np.newaxis] * nk_points * k)
    coses = np.cos(np.arange(0, n)[:, np.newaxis] * nk_points * k)
    data = np.concatenate([coses, sines]).astype(np.complex128) / np.sqrt(basis.n)

    # Equivalent to
    # ! data = standard_operators["data"].reshape(kernel["basis"].n, -1)
    # ! end = fundamental_n // 2 + 1
    # Build (e^(ikx) +- e^(-ikx)) operators
    # ! data[1:end] = np.sqrt(2) * np.real(data[1:end])
    # ! data[end:] = np.sqrt(2) * np.imag(np.conj(data[end:]))
    return {
        "basis": TupleBasis(FundamentalBasis(data.shape[0]), TupleBasis(basis, basis)),
        "data": data,
    }


def get_noise_operators_real_isotropic_fft(
    kernel: IsotropicNoiseKernel[_B0],
    *,
    n: int | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalNoiseOperatorList[FundamentalBasis[int], _B0]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    beta(x - x') = 1 / N \sum_k |f(k)|^2 e^(ikx) for some f(k)
    |f(k)|^2 = \sum_x beta(x) e^(-ik.x)

    The independent noise operators are then given by

    L(k) = 1 / N \sum_x e^(ikx) S(x)

    The inddependent operators can therefore be found directly using a FFT
    of the noise beta(x).

    For a real kernel, the coefficients of  e^(+-ikx) are the same
    we can therefore equivalently use cos(x) and sin(x) as the basis
    for the kernel.

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]

    Returns
    -------
    SingleBasisDiagonalNoiseOperatorList[FundamentalBasis[int], FundamentalBasis[int]]
    """
    np.testing.assert_allclose(np.imag(kernel["data"]), 0)

    fundamental_n = kernel["basis"].n if n is None else 2 * n + 1

    operators = get_operators_for_real_isotropic_noise(
        kernel["basis"], fundamental_n=fundamental_n, assert_periodic=assert_periodic
    )

    eigenvalues = _get_noise_eigenvalues_isotropic_fft(
        kernel, fundamental_n=fundamental_n
    )

    np.testing.assert_allclose(
        eigenvalues["data"][1::],
        eigenvalues["data"][1::][::-1],
        rtol=1e-8,
    )

    return {
        "basis": operators["basis"],
        "data": operators["data"],
        "eigenvalue": eigenvalues["data"],
    }


def _get_operators_for_isotropic_stacked_noise(
    basis: _SB0,
    *,
    shape: tuple[int, ...] | None = None,
    fundamental_shape: tuple[int, ...] | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _SB0
]:
    fundamental_shape = basis.shape if fundamental_shape is None else fundamental_shape
    if assert_periodic:
        _assert_periodic_sample(basis.shape, fundamental_shape)
    shape = fundamental_shape if shape is None else shape
    # Operators e^(ik_n0,n1,.. x_m0,m1,..) / sqrt(prod(Mi))
    # with k_n0,n1 = 2 * np.pi * (n0,n1,...) / prod(Ni), ni = 0...Ni
    # and x_m0,m1 = (m0,m1,...), mi = 0...Mi
    k = tuple(2 * np.pi / n for n in fundamental_shape)
    shape_basis = fundamental_stacked_basis_from_shape(fundamental_shape)

    nk_points = BasisUtil(basis).fundamental_stacked_nk_points
    i_points = BasisUtil(shape_basis).stacked_nk_points

    operators = np.array(
        [
            np.exp(1j * np.einsum("i,i,ij->j", k, i, nk_points))  # type: ignore einsum
            / np.sqrt(basis.n)
            for i in zip(*i_points)
        ]
    )
    return {
        "basis": TupleBasis(
            shape_basis,
            TupleBasis(basis, basis),
        ),
        "data": operators.astype(np.complex128).ravel(),
    }


def _get_noise_eigenvalues_isotropic_stacked_fft(
    kernel: IsotropicNoiseKernel[_TB0],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
) -> ValueList[TupleBasisLike[*tuple[FundamentalBasis[int], ...]]]:
    fundamental_shape = (
        kernel["basis"].shape if fundamental_shape is None else fundamental_shape
    )

    coefficients = np.fft.ifftn(
        pad_ft_points(
            kernel["data"].reshape(kernel["basis"].shape),
            fundamental_shape,
            tuple(range(len(fundamental_shape))),
        ),
        norm="forward",
    )

    coefficients *= kernel["basis"].n / coefficients.size
    return {
        "basis": fundamental_stacked_basis_from_shape(fundamental_shape),
        "data": coefficients.ravel(),
    }


def get_noise_operators_isotropic_stacked_fft(
    kernel: IsotropicNoiseKernel[_TB0],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
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
    fundamental_shape = (
        kernel["basis"].shape if fundamental_shape is None else fundamental_shape
    )

    operators = _get_operators_for_isotropic_stacked_noise(
        kernel["basis"],
        fundamental_shape=fundamental_shape,
        assert_periodic=assert_periodic,
    )
    eigenvalues = _get_noise_eigenvalues_isotropic_stacked_fft(
        kernel, fundamental_shape=fundamental_shape
    )

    return {
        "basis": operators["basis"],
        "data": operators["data"],
        "eigenvalue": eigenvalues["data"],
    }


def get_operators_for_real_isotropic_stacked_noise(
    basis: _SB0,
    *,
    shape: tuple[int, ...] | None = None,
    fundamental_shape: tuple[int, ...] | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _SB0
]:
    """Get operators used for real isotropic noise.

    Parameters
    ----------
    basis : _B0
        _description_
    n_terms : int | None, optional
        _description_, by default None

    Returns
    -------
    SingleBasisDiagonalOperatorList[FundamentalBasis[int], _B0]
        _description_
    """
    fundamental_shape = basis.shape if fundamental_shape is None else fundamental_shape
    complex_operators = _get_operators_for_isotropic_stacked_noise(
        basis,
        shape=shape,
        fundamental_shape=fundamental_shape,
        assert_periodic=assert_periodic,
    )
    data = (
        complex_operators["data"]
        .reshape(*complex_operators["basis"][0].shape, -1)
        .copy()
    )

    for axis, (n, basis_n) in enumerate(zip(fundamental_shape, basis.shape)):
        cloned = data.copy()
        # Build (e^(ikx) +- e^(-ikx)) operators
        # Index of highest frequency positive
        max_cos_idx = n // 2

        # Note we ignore the N / 2 frequency
        # ie the last operator if we have even
        cos_end_idx = min(max_cos_idx, (basis_n + 1) // 2 - 1)
        sin_start_idx = max(max_cos_idx + 1, n - (basis_n + 1) // 2 + 1)
        # If n != basis_n, we must have an odd number of points
        assert n == basis_n or n % 2 == 1

        cos_slice = slice_along_axis(slice(1, cos_end_idx + 1), axis)
        conj_cos_slice = slice_along_axis(slice(None, sin_start_idx - 1, -1), axis)
        data[cos_slice] = (cloned[cos_slice] + cloned[conj_cos_slice]) / np.sqrt(2)

        sin_slice = slice_along_axis(slice(sin_start_idx, None), axis)
        conj_sin_slice = slice_along_axis(slice(cos_end_idx, 0, -1), axis)
        data[sin_slice] = (cloned[sin_slice] - cloned[conj_sin_slice]) * 1j / np.sqrt(2)

    return {
        "basis": complex_operators["basis"],
        "data": data,
    }


def get_noise_operators_real_isotropic_stacked_fft(
    kernel: IsotropicNoiseKernel[_TB0],
    *,
    shape: tuple[int, ...] | None = None,
    assert_periodic: bool = True,
) -> SingleBasisDiagonalNoiseOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _TB0
]:
    """
    Get the noise operators, expanding the kernel about each axis individually.

    Returns
    -------
    SingleBasisDiagonalNoiseOperatorList[
        TupleBasisLike[*tuple[FundamentalBasis[int], ...]], _TB0
    ]
    """
    np.testing.assert_allclose(np.imag(kernel["data"]), 0)

    fundamental_shape = (
        kernel["basis"].shape if shape is None else tuple(2 * n + 1 for n in shape)
    )

    operators = get_operators_for_real_isotropic_stacked_noise(
        kernel["basis"],
        fundamental_shape=fundamental_shape,
        assert_periodic=assert_periodic,
    )
    # TODO(matt): assert has correct symmetry  # noqa: FIX002
    eigenvalues = _get_noise_eigenvalues_isotropic_stacked_fft(
        kernel, fundamental_shape=fundamental_shape
    )

    return {
        "basis": operators["basis"],
        "data": operators["data"],
        "eigenvalue": eigenvalues["data"],
    }
