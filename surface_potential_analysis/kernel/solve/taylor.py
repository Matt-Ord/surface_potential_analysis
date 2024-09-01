from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from scipy.special import factorial  # type:ignore bad stb file

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.kernel.build import (
    get_fundamental_axis_kernels_from_isotropic,
)
from surface_potential_analysis.kernel.kernel import (
    get_full_noise_operators_from_axis_operators,
)
from surface_potential_analysis.kernel.solve._fft import (
    get_operators_for_real_isotropic_noise,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasis,
    )
    from surface_potential_analysis.kernel.kernel import (
        IsotropicNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
    )

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])

    _B0 = TypeVar("_B0", bound=BasisLike[int, int])
    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def _get_cos_coefficients_for_taylor_series(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    n_terms = polynomial_coefficients.size if n_terms is None else n_terms

    atol = 1e-8 * np.max(polynomial_coefficients).item()
    is_nonzero = np.isclose(polynomial_coefficients, 0, atol=atol)
    first_nonzero = np.argmax(is_nonzero)
    if first_nonzero == 0 and is_nonzero.item(0) is False:
        first_nonzero = is_nonzero.size
    n_nonzero_terms = min(first_nonzero, n_terms)
    polynomial_coefficients = polynomial_coefficients[:n_nonzero_terms]

    i = np.arange(0, n_nonzero_terms).reshape(1, -1)
    m = np.arange(0, n_nonzero_terms).reshape(-1, 1)
    coefficients_prefactor = ((-1) ** m) / (factorial(2 * m))
    coefficients_matrix = coefficients_prefactor * (i ** (2 * m))
    cos_series_coefficients = np.linalg.solve(
        coefficients_matrix, polynomial_coefficients
    )
    out = np.zeros(n_terms, np.float64)
    out[:n_nonzero_terms] = cos_series_coefficients.T
    return out


def _get_coefficients_for_taylor_series(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    cos_series_coefficients = _get_cos_coefficients_for_taylor_series(
        polynomial_coefficients, n_terms=n_terms
    )
    sin_series_coefficients = cos_series_coefficients[:0:-1]
    return np.concatenate([cos_series_coefficients, sin_series_coefficients])


def get_noise_operators_explicit_taylor_expansion(
    basis: _B0,
    polynomial_coefficients: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    _B0,
]:
    """Note polynomial_coefficients should be properly normalized."""
    n_terms = (basis.n // 2) if n_terms is None else n_terms

    data = get_operators_for_real_isotropic_noise(basis, n=n_terms)

    # coefficients for the Taylor expansion of the trig terms
    coefficients = _get_coefficients_for_taylor_series(
        polynomial_coefficients=polynomial_coefficients,
    )
    coefficients *= basis.n

    return {
        "basis": data["basis"],
        "data": data["data"].astype(np.complex128),
        "eigenvalue": coefficients.astype(np.complex128),
    }


def get_noise_operators_real_isotropic_taylor_expansion(
    kernel: IsotropicNoiseKernel[_BL0],
    *,
    n: int | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    FundamentalPositionBasis[Any, Any],
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
    basis_x = basis_as_fundamental_position_basis(kernel["basis"])

    n_states: int = basis_x.n
    n = (n_states + 1) // 2 if n is None else n

    # weight is chosen such that the 2n+1 points around the origin are selected for fitting
    weight = pad_ft_points(np.ones(2 * n + 1), (n_states,), (0,))
    points = np.cos(np.arange(n_states) * (2 * np.pi / n_states))

    # use T_n(cos(x)) = cos(nx) to find the coefficients
    noise_polynomial = cast(
        np.polynomial.Polynomial,
        np.polynomial.Chebyshev.fit(  # type: ignore unknown
            x=points,
            y=kernel["data"],
            deg=n,
            w=weight,
            domain=(-1, 1),
        ),
    )

    operator_coefficients = np.concatenate(
        [noise_polynomial.coef, noise_polynomial.coef[:0:-1]]
    ).astype(np.complex128)
    operator_coefficients *= n_states

    operators = get_operators_for_real_isotropic_noise(basis_x, n=n + 1)

    return {
        "basis": operators["basis"],
        "data": operators["data"],
        "eigenvalue": operator_coefficients,
    }


def get_stacked_noise_operators_real_isotropic_taylor_expansion(
    kernel: IsotropicNoiseKernel[_SBV0],
    *,
    shape: tuple[int | None, ...] | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    TupleBasis[*tuple[FundamentalBasis[int], ...]],
    TupleBasis[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    axis_kernels = get_fundamental_axis_kernels_from_isotropic(kernel)
    shape = tuple(None for _ in axis_kernels) if shape is None else shape

    operators_list = tuple(
        get_noise_operators_real_isotropic_taylor_expansion(
            kernel,
            n=n,
        )
        for (kernel, n) in zip(axis_kernels, shape)
    )
    return get_full_noise_operators_from_axis_operators(operators_list)
