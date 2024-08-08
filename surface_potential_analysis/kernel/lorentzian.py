from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVarTuple

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore no stub

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.kernel.build import build_isotropic_kernel_from_function
from surface_potential_analysis.kernel.solve import (
    get_noise_operators_explicit_taylor_expansion,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        IsotropicNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
    )

_B0s = TypeVarTuple("_B0s")


def get_effective_lorentzian_parameter(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> tuple[float, float]:
    """
    Generate a set of lorentzian parameters A, Lambda for a friction coefficient eta.

    Parameters
    ----------
    basis : TupleBasisLike[
        _description_
    eta : float
    temperature : float
    lambda_factor : float, optional
        lambda_factor, by default 2*np.sqrt(2)

    Returns
    -------
    tuple[float, float]
        (A, lambda_)
    """
    util = BasisUtil(basis)
    smallest_max_displacement = np.min(np.linalg.norm(util.delta_x_stacked, axis=1)) / 2
    lambda_ = smallest_max_displacement / lambda_factor
    # mu = A / lambda
    mu = np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_
    return (a, lambda_)


def get_lorentzian_isotropic_noise_kernel(
    basis: TupleBasisWithLengthLike[*_B0s],
    a: float,
    lambda_: float,
) -> IsotropicNoiseKernel[
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    """Get an isotropic noise kernel for a lorentzian correllation.

    beta(x,x') = a**2 * lambda_**2 / ((x-x')**2 + lambda_**2)

    Parameters
    ----------
    basis : TupleBasisWithLengthLike[*_B0s]
    a : float
    lambda_ : float

    Returns
    -------
    IsotropicNoiseKernel[
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    ]
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return a**2 * lambda_**2 / (displacements**2 + lambda_**2).astype(np.complex128)

    return build_isotropic_kernel_from_function(basis, fn)


def _get_explicit_taylor_coefficients_lorentzian(
    a: float,
    lambda_: float,
    *,
    n_terms: int = 1,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    i = np.arange(0, n_terms + 1)
    return a**2 * ((-1 / (lambda_**2)) ** i)


def get_lorentzian_operators_explicit_taylor(
    a: float,
    lambda_: float,
    basis: TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]],
    *,
    n_terms: int | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    """Calculate the noise operators for an isotropic lorentzian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of lorentzian
    noise lambda_/(x^2 + lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Parameters
    ----------
    lambda_: float, the HWHM
    basis: TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]
    n: int, by default 1

    Return in the order of [const term, first n sine terms, first n cos terms]
    and also their corresponding coefficients.
    """
    # currently only support 1D
    assert basis.ndim == 1
    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    n_terms = (basis_x[0].n // 2) if n_terms is None else n_terms

    # expand gaussian and define array containing coefficients for each term in the polynomial
    # coefficients for the explicit Taylor expansion of the gaussian noise
    # Normalize lambda
    delta_x = np.linalg.norm(BasisUtil(basis).delta_x_stacked[0])
    normalized_lambda = 2 * np.pi * lambda_ / delta_x
    polynomial_coefficients = _get_explicit_taylor_coefficients_lorentzian(
        a, normalized_lambda.item(), n_terms=n_terms
    )

    return get_noise_operators_explicit_taylor_expansion(
        basis_x, polynomial_coefficients, n_terms=n_terms
    )
