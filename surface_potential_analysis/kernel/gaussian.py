from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar  # type:ignore bad stub file
from scipy.special import factorial  # type:ignore bad stub file

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_basis,
)
from surface_potential_analysis.basis.legacy import (
    StackedBasisWithVolumeLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.kernel.build import (
    build_axis_kernel_from_function_stacked,
    build_isotropic_kernel_from_function_stacked,
    get_temperature_corrected_diagonal_noise_operators,
    truncate_diagonal_noise_operator_list,
)
from surface_potential_analysis.kernel.conversion import (
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.kernel import (
    AxisKernel,
    DiagonalNoiseOperatorList,
    SingleBasisDiagonalNoiseOperatorList,
    as_diagonal_kernel_from_isotropic,
    get_diagonal_noise_operators_from_axis,
)
from surface_potential_analysis.kernel.solve._fft import (
    get_periodic_noise_operators_real_isotropic_stacked_fft,
)
from surface_potential_analysis.kernel.solve._taylor import (
    get_linear_noise_operators_explicit_taylor_expansion,
    get_periodic_noise_operators_explicit_taylor_expansion,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from slate.basis._basis import FundamentalBasis
    from slate.metadata._metadata import BasisMetadata

    from surface_potential_analysis.basis.legacy import (
        BasisWithLengthLike,
        FundamentalPositionBasis,
        StackedBasisWithVolumeLike,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        IsotropicNoiseKernel,
        SingleBasisDiagonalNoiseKernel,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)


def get_gaussian_isotropic_noise_kernel(
    basis: StackedBasisWithVolumeLike,
    a: float,
    lambda_: float,
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Get the noise kernel for a gaussian correllated surface.

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
        _description_
    eta : float
        _description_
    temperature : float
        _description_
    lambda_factor : float, optional
        _description_, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
        _description_
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return a**2 * np.exp(-(displacements**2) / (2 * lambda_**2)).astype(
            np.complex128,
        )

    return build_isotropic_kernel_from_function_stacked(basis, fn)


def get_gaussian_axis_noise_kernel(
    basis: StackedBasisWithVolumeLike,
    a: float,
    lambda_: float,
) -> AxisKernel[FundamentalPositionBasis]:
    """
    Get the noise kernel for a gaussian correllated surface.

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
        _description_
    eta : float
        _description_
    temperature : float
        _description_
    lambda_factor : float, optional
        _description_, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
        _description_
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return a**2 * np.exp(-(displacements**2) / (2 * lambda_**2)).astype(
            np.complex128,
        )

    return build_axis_kernel_from_function_stacked(basis, fn)


def get_gaussian_noise_kernel(
    basis: StackedBasisWithVolumeLike,
    a: float,
    lambda_: float,
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Get the noise kernel for a gaussian correllated surface.

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
        _description_
    eta : float
        _description_
    temperature : float
        _description_
    lambda_factor : float, optional
        _description_, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
        _description_
    """
    return as_diagonal_kernel_from_isotropic(
        get_gaussian_isotropic_noise_kernel(basis, a, lambda_)
    )


def get_effective_gaussian_parameters(
    basis: StackedBasisWithVolumeLike,
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> tuple[float, float]:
    """
    Generate a set of Gaussian parameters A, Lambda for a friction coefficient eta.

    This is done to match the quadratic coefficient (A^2/(2 lambda^2))
    beta(x,x') = A^2(1-(x-x')^2/(lambda^2))

    to the caldeira leggett noise

    beta(x,x') = 2 * eta * Boltzmann * temperature / hbar**2

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
    smallest_max_displacement = np.min(np.linalg.norm(util.delta_x_stacked, axis=1))
    lambda_ = smallest_max_displacement / lambda_factor
    # mu = A / lambda
    mu = np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_
    return (a, lambda_)


def get_effective_gaussian_noise_kernel(
    basis: StackedBasisWithVolumeLike,
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Get the noise kernel for a gaussian correllated surface, given the Caldeira leggett parameters.

    This chooses the largest possible wavelength, such that the smallest correllation between
    any two points is a**2 * np.exp(- lambda_factor ** 2 / 2), where a**2 is the max correllation

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
    eta : float
    temperature : float
    lambda_factor : float, optional
        lambda_factor, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
    """
    a, lambda_ = get_effective_gaussian_parameters(
        basis, eta, temperature, lambda_factor=lambda_factor
    )
    return get_gaussian_noise_kernel(basis, a, lambda_)


def get_effective_gaussian_isotropic_noise_kernel(
    basis: StackedBasisWithVolumeLike,
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Get the noise kernel for a gaussian correllated surface, given the Caldeira leggett parameters.

    This chooses the largest possible wavelength, such that the smallest correllation between
    any two points is a**2 * np.exp(- lambda_factor ** 2 / 2), where a**2 is the max correllation

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
    eta : float
    temperature : float
    lambda_factor : float, optional
        lambda_factor, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
    """
    a, lambda_ = get_effective_gaussian_parameters(
        basis, eta, temperature, lambda_factor=lambda_factor
    )
    return get_gaussian_isotropic_noise_kernel(basis, a, lambda_)


def get_gaussian_noise_operators_periodic(
    basis: StackedBasisWithVolumeLike,
    a: float,
    lambda_: float,
    *,
    truncation: Iterable[int] | None = None,
) -> DiagonalNoiseOperatorList[
    FundamentalBasis[BasisMetadata],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    kernel = get_gaussian_isotropic_noise_kernel(basis, a, lambda_)

    operators = get_periodic_noise_operators_real_isotropic_stacked_fft(kernel)
    truncation = range(operators["basis"][0].size) if truncation is None else truncation
    return truncate_diagonal_noise_operator_list(operators, truncation=truncation)


def get_effective_gaussian_noise_operators_periodic(
    basis: StackedBasisWithVolumeLike,
    eta: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[BasisMetadata],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    a, lambda_ = get_effective_gaussian_parameters(basis, eta, temperature)
    return get_gaussian_noise_operators_periodic(
        basis, a, lambda_, truncation=truncation
    )


def get_temperature_corrected_gaussian_noise_operators(
    hamiltonian: SingleBasisOperator[_SBV0],
    a: float,
    lambda_: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> SingleBasisNoiseOperatorList[FundamentalBasis[BasisMetadata], _SBV0]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    operators = get_gaussian_noise_operators_periodic(
        hamiltonian["basis"][0], a, lambda_, truncation=truncation
    )
    corrected = get_temperature_corrected_diagonal_noise_operators(
        hamiltonian, operators, temperature
    )
    return convert_noise_operator_list_to_basis(corrected, hamiltonian["basis"])


def get_temperature_corrected_effective_gaussian_noise_operators(
    hamiltonian: SingleBasisOperator[_SBV0],
    eta: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> SingleBasisNoiseOperatorList[FundamentalBasis[BasisMetadata], _SBV0]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian["basis"][0], eta, temperature
    )

    return get_temperature_corrected_gaussian_noise_operators(
        hamiltonian, a, lambda_, temperature, truncation=truncation
    )


def _get_explicit_taylor_coefficients_gaussian(
    a: float,
    lambda_: float,
    *,
    n_terms: int = 1,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    i = np.arange(0, n_terms)
    return (a**2 / factorial(i)) * ((-1 / (2 * lambda_**2)) ** i)


def get_periodic_gaussian_operators_explicit_taylor(
    basis: BasisWithLengthLike[int, Any, Any],
    a: float,
    lambda_: float,
    *,
    n_terms: int | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[BasisMetadata],
    FundamentalPositionBasis,
]:
    """Calculate the noise operators for an isotropic gaussian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of gaussian
    noise (a^2)*e^(-x^2/2*lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Return in the order of [const term, first n cos terms, first n sin terms]
    and also their corresponding coefficients.
    """
    basis_x = basis_as_fundamental_basis(basis)
    n_terms = (basis_x.n // 2) if n_terms is None else n_terms

    # expand gaussian and define array containing coefficients for each term in the polynomial
    # coefficients for the explicit Taylor expansion of the gaussian noise
    # Normalize lambda
    delta_x = np.linalg.norm(BasisUtil(basis).delta_x)
    normalized_lambda = 2 * np.pi * lambda_ / delta_x
    polynomial_coefficients = _get_explicit_taylor_coefficients_gaussian(
        a, normalized_lambda.item(), n_terms=n_terms
    )

    return get_periodic_noise_operators_explicit_taylor_expansion(
        basis_x, polynomial_coefficients, n_terms=n_terms
    )


def get_linear_gaussian_noise_operators_explicit_taylor(
    basis: BasisWithLengthLike[int, int, int],
    a: float,
    lambda_: float,
    *,
    n_terms: int | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[BasisMetadata],
    FundamentalPositionBasis,
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    basis_x = basis_as_fundamental_basis(basis)
    n_terms = basis.n if n_terms is None else n_terms

    # expand gaussian and define array containing coefficients for each term in the polynomial
    # coefficients for the explicit Taylor expansion of the gaussian noise
    # Normalize lambda
    delta_x = np.linalg.norm(BasisUtil(basis).delta_x)
    normalized_lambda = 2 * np.pi * lambda_ / delta_x
    polynomial_coefficients = _get_explicit_taylor_coefficients_gaussian(
        a, normalized_lambda.item(), n_terms=n_terms
    )

    return get_linear_noise_operators_explicit_taylor_expansion(
        basis_x, polynomial_coefficients, n_terms=n_terms
    )


def get_periodic_gaussian_operators_explicit_taylor_stacked(
    basis: StackedBasisWithVolumeLike,
    a: float,
    lambda_: float,
    *,
    shape: tuple[int, ...] | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[BasisMetadata], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Calculate the noise operators for an isotropic gaussian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of gaussian
    noise (a^2)*e^(-x^2/2*lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Return in the order of [const term, first n cos terms, first n sin terms]
    and also their corresponding coefficients.
    """
    basis_x = tuple_basis_as_fundamental(basis)

    axis_operators = tuple(
        get_periodic_gaussian_operators_explicit_taylor(
            basis_x[i],
            a,
            lambda_,
            n_terms=None if shape is None else shape[i],
        )
        for i in range(basis.n_dim)
    )

    return get_diagonal_noise_operators_from_axis(axis_operators)


def get_linear_gaussian_operators_explicit_taylor_stacked(
    basis: StackedBasisWithVolumeLike,
    a: float,
    lambda_: float,
    *,
    shape: tuple[int, ...] | None = None,
) -> SingleBasisDiagonalNoiseOperatorList[
    TupleBasisLike[*tuple[FundamentalBasis[BasisMetadata], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Calculate the noise operators for an isotropic gaussian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of gaussian
    noise (a^2)*e^(-x^2/2*lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Return in the order of [const term, first n cos terms, first n sin terms]
    and also their corresponding coefficients.
    """
    basis_x = tuple_basis_as_fundamental(basis)

    axis_operators = tuple(
        get_linear_gaussian_noise_operators_explicit_taylor(
            basis_x[i],
            a,
            lambda_,
            n_terms=None if shape is None else shape[i],
        )
        for i in range(basis.n_dim)
    )

    return get_diagonal_noise_operators_from_axis(axis_operators)
