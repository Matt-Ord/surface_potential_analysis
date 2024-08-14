# measure runtime of different parts of fit functions here, then use this in plot.py in reduced caldeira
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from scipy.special import factorial

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.kernel.solve import (
    get_operators_for_real_isotropic_noise,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        IsotropicNoiseKernel,
    )

_TBL0 = TypeVar("_TBL0", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])
_B0 = TypeVar("_B0", bound=BasisLike[int, int])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])


# poly fit
def _get_time_poly_fit(
    n_states: int, kernel: IsotropicNoiseKernel[_TBL0], n: int
) -> float:
    x = np.cos(
        pad_ft_points(np.arange(n_states), (2 * n + 1,), (0,)) * (2 * np.pi / n_states)
    )
    y = pad_ft_points(kernel["data"], (2 * n + 1,), (0,))
    deg = np.arange(0, n + 1)
    ts = datetime.datetime.now(tz=datetime.UTC)
    _noise_polynomial = cast(
        np.polynomial.Polynomial,
        np.polynomial.Chebyshev.fit(
            x=x,
            y=y,
            deg=deg,
        ),
    )
    te = datetime.datetime.now(tz=datetime.UTC)
    return (te - ts).total_seconds()


def _get_time_for_get_op_for_real_isotropic_noise(
    basis: _B0,
    n: int,
) -> float:
    # this is used in both poly fit and explicit
    ts = datetime.datetime.now(tz=datetime.UTC)
    _operators = get_operators_for_real_isotropic_noise(basis, n_terms=n + 1)
    te = datetime.datetime.now(tz=datetime.UTC)
    return (te - ts).total_seconds()


def get_time_stacked_taylor_expansion(
    kernel: IsotropicNoiseKernel[_TBL0],
    *,
    n: int = 1,
) -> list[float]:
    basis_x = stacked_basis_as_fundamental_position_basis(kernel["basis"])
    n_states: int = basis_x.n
    poly_fit_time = _get_time_poly_fit(n_states, kernel, n)
    time_for_get_op_for_real_isotropic_noise = (
        _get_time_for_get_op_for_real_isotropic_noise(basis_x, n + 1)
    )

    return [poly_fit_time, time_for_get_op_for_real_isotropic_noise]


# fft
def _get_time_ifft(
    kernel: IsotropicNoiseKernel[_B0],
    n: int | None = None,
) -> float:
    ts = datetime.datetime.now(tz=datetime.UTC)
    _coefficients = np.fft.ifft(
        pad_ft_points(kernel["data"], (n,), (0,)),
        norm="forward",
    )
    te = datetime.datetime.now(tz=datetime.UTC)
    return (te - ts).total_seconds()


def get_time_real_isotropic_fft(
    kernel: IsotropicNoiseKernel[_B0],
) -> float:
    return _get_time_ifft(kernel, n=kernel["basis"].n)


# eigenvalue
def get_time_eigh(kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1]) -> float:
    data = kernel["data"].reshape(kernel["basis"][0][0].n, -1)
    ts = datetime.datetime.now(tz=datetime.UTC)
    _res = np.linalg.eigh(data)
    te = datetime.datetime.now(tz=datetime.UTC)
    return (te - ts).total_seconds()


# explicit
def get_time_explicit(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    n_terms: int | None,
    a: float,
    lambda_: float,
) -> list[float]:
    j = np.arange(0, n_terms)
    polynomial_coefficients = (a**2 / factorial(j)) * ((-1 / (2 * lambda_**2)) ** j)
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
    ts = datetime.datetime.now(tz=datetime.UTC)
    _cos_series_coefficients = np.linalg.solve(
        coefficients_matrix, polynomial_coefficients
    )
    te = datetime.datetime.now(tz=datetime.UTC)
    time_for_get_cos_series_coeff = (te - ts).total_seconds()

    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    time_for_get_op_for_real_isotropic_noise = (
        _get_time_for_get_op_for_real_isotropic_noise(basis_x, n_terms)
    )
    return [time_for_get_cos_series_coeff, time_for_get_op_for_real_isotropic_noise]
