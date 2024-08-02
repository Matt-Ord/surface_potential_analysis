from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar

from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.kernel.build import (
    build_isotropic_kernel_from_function,
    get_temperature_corrected_diagonal_noise_operators,
)
from surface_potential_analysis.kernel.conversion import (
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.kernel import (
    DiagonalNoiseOperatorList,
    IsotropicNoiseKernel,
    SingleBasisDiagonalNoiseKernel,
    SingleBasisNoiseOperatorList,
    as_diagonal_kernel_from_isotropic,
    get_noise_operators_real_isotropic_stacked,
    truncate_diagonal_noise_operators,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def get_gaussian_isotropic_noise_kernel(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    a: float,
    lambda_: float,
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]] ]
        _description_
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return a**2 * np.exp(-(displacements**2) / (2 * lambda_**2)).astype(
            np.complex128,
        )

    return build_isotropic_kernel_from_function(basis, fn)


def get_gaussian_noise_kernel(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    a: float,
    lambda_: float,
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]] ]
        _description_
    """
    return as_diagonal_kernel_from_isotropic(
        get_gaussian_isotropic_noise_kernel(basis, a, lambda_)
    )


def get_effective_gaussian_parameters(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> tuple[float, float]:
    """
    Generate a set of Gaussian parameters A, Lambda for a friction coefficient eta.

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


def get_effective_gaussian_noise_kernel(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]] ]
    """
    a, lambda_ = get_effective_gaussian_parameters(
        basis, eta, temperature, lambda_factor=lambda_factor
    )
    return get_gaussian_noise_kernel(basis, a, lambda_)


def get_effective_gaussian_isotropic_noise_kernel(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]] ]
    """
    a, lambda_ = get_effective_gaussian_parameters(
        basis, eta, temperature, lambda_factor=lambda_factor
    )
    return get_gaussian_isotropic_noise_kernel(basis, a, lambda_)


def get_gaussian_noise_operators(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    a: float,
    lambda_: float,
    *,
    truncation: Iterable[int] | None = None,
) -> DiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
        FundamentalBasis[int],
        FundamentalPositionBasis[Any, Literal[1]],
    ]

    """
    kernel = get_gaussian_isotropic_noise_kernel(basis, a, lambda_)

    operators = get_noise_operators_real_isotropic_stacked(kernel)
    truncation = range(operators["basis"][0].n) if truncation is None else truncation
    return truncate_diagonal_noise_operators(operators, truncation=truncation)


def get_effective_gaussian_noise_operators(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    eta: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> DiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
        FundamentalBasis[int],
        FundamentalPositionBasis[Any, Literal[1]],
    ]

    """
    a, lambda_ = get_effective_gaussian_parameters(basis, eta, temperature)
    return get_gaussian_noise_operators(basis, a, lambda_, truncation=truncation)


def get_temperature_corrected_gaussian_noise_operators(
    hamiltonian: SingleBasisOperator[_SBV0],
    a: float,
    lambda_: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> SingleBasisNoiseOperatorList[FundamentalBasis[int], _SBV0]:
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
        FundamentalBasis[int],
        FundamentalPositionBasis[Any, Literal[1]],
    ]

    """
    operators = get_gaussian_noise_operators(
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
) -> SingleBasisNoiseOperatorList[FundamentalBasis[int], _SBV0]:
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
        FundamentalBasis[int],
        FundamentalPositionBasis[Any, Literal[1]],
    ]

    """
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian["basis"][0], eta, temperature
    )

    return get_temperature_corrected_gaussian_noise_operators(
        hamiltonian, a, lambda_, temperature, truncation=truncation
    )
