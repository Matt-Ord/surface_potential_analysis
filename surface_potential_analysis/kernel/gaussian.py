from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, TypeVarTuple, cast

import numpy as np
from scipy.constants import Boltzmann, hbar

from surface_potential_analysis.basis.basis_like import BasisLike, BasisWithLengthLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    SingleBasisDiagonalNoiseKernel,
    SingleBasisDiagonalNoiseOperatorList,
    SingleBasisNoiseOperatorList,
    get_noise_operators_diagonal,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator_list import (
    add_list_list,
    as_operator_list,
    get_commutator_operator_list,
    scale_operator_list,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.operator.operator_list import (
        SingleBasisOperatorList,
    )

_B1 = TypeVar(
    "_B1",
    bound=BasisLike[Any, Any],
)

_B0 = TypeVar(
    "_B0",
    bound=BasisLike[Any, Any],
)
_B0s = TypeVarTuple("_B0s")


def _get_displacements(
    basis: TupleBasisLike[*_B0s],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    util = BasisUtil(
        cast(TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]], basis)
    )
    step = tuple(
        (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n - (n // 2)
        for (n_x_points, n) in zip(
            util.fundamental_stacked_nx_points,
            util.fundamental_shape,
            strict=True,
        )
    )

    return np.linalg.norm(
        np.tensordot(step, util.dx_stacked, axes=(0, 0)),
        axis=2,
    )


def get_gaussian_noise_kernel(
    basis: TupleBasisWithLengthLike[*_B0s],
    a: float,
    lambda_: float,
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    displacements = _get_displacements(basis)
    correlation = a**2 * np.exp(-(displacements**2) / (2 * lambda_**2)).astype(
        np.complex128,
    )

    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    return {
        "basis": TupleBasis(
            TupleBasis(basis_x, basis_x),
            TupleBasis(basis_x, basis_x),
        ),
        "data": correlation.ravel(),
    }


def get_gaussian_isotropic_noise_kernel(
    basis: TupleBasisWithLengthLike[*_B0s],
    a: float,
    lambda_: float,
) -> IsotropicNoiseKernel[
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    displacements = _get_displacements(basis)[0]
    correlation = a**2 * np.exp(-(displacements**2) / (2 * lambda_**2)).astype(
        np.complex128,
    )

    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    return {
        "basis": basis_x,
        "data": correlation.ravel(),
    }


def get_effective_gaussian_parameters(
    basis: TupleBasisLike[*_B0s],
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
    util = BasisUtil(
        cast(TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]], basis)
    )
    smallest_max_displacement = np.min(np.linalg.norm(util.delta_x_stacked, axis=1)) / 2
    lambda_ = smallest_max_displacement / lambda_factor
    # mu = A / lambda
    mu = np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_
    return (a, lambda_)


def get_effective_gaussian_noise_kernel(
    basis: TupleBasisWithLengthLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> SingleBasisDiagonalNoiseKernel[
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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


def _get_temperature_corrected_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisOperatorList[
        _B0,
        _B1,
    ],
    temperature: float,
) -> SingleBasisOperatorList[
    _B0,
    _B1,
]:
    commutator = get_commutator_operator_list(hamiltonian, operators)
    correction = scale_operator_list(-1 / (4 * Boltzmann * temperature), commutator)
    return add_list_list(operators, correction)


def get_temperature_corrected_noise_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisNoiseOperatorList[
        _B0,
        _B1,
    ],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    _B0,
    _B1,
]:
    """
    Get the noise operators, applying the caldeira-legget like temperature correction.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B1]
    operators : SingleBasisDiagonalNoiseOperatorList[ _B0, _B1, ]
    temperature : float

    Returns
    -------
    SingleBasisNoiseOperatorList[ _B0, _B1]
    """
    corrected_operators = _get_temperature_corrected_operators(
        hamiltonian,
        operators,
        temperature,
    )

    return {
        "basis": corrected_operators["basis"],
        "data": corrected_operators["data"],
        "eigenvalue": operators["eigenvalue"],
    }


def get_temperature_corrected_diagonal_noise_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisDiagonalNoiseOperatorList[
        _B0,
        _B1,
    ],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    _B0,
    _B1,
]:
    """
    Get the noise operators, applying the caldeira-legget like temperature correction.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B1]
    operators : SingleBasisDiagonalNoiseOperatorList[ _B0, _B1, ]
    temperature : float

    Returns
    -------
    SingleBasisNoiseOperatorList[ _B0, _B1]
    """
    operators_full = as_operator_list(operators)
    return get_temperature_corrected_noise_operators(
        hamiltonian,
        {
            "basis": operators_full["basis"],
            "data": operators_full["data"],
            "eigenvalue": operators["eigenvalue"],
        },
        temperature,
    )


def get_temperature_corrected_gaussian_noise_operators(
    hamiltonian: SingleBasisOperator[TupleBasisWithLengthLike[*_B0s]],
    a: float,
    lambda_: float,
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    kernel = get_gaussian_noise_kernel(
        hamiltonian["basis"][0],
        a,
        lambda_,
    )
    operators = get_noise_operators_diagonal(kernel)
    hamiltonian_converted = convert_operator_to_basis(
        hamiltonian,
        operators["basis"][1],
    )

    return get_temperature_corrected_diagonal_noise_operators(
        hamiltonian_converted,
        operators,
        temperature,
    )


def get_temperature_corrected_effective_gaussian_noise_operators(
    hamiltonian: SingleBasisOperator[TupleBasisWithLengthLike[*_B0s]],
    eta: float,
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
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
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian["basis"][0], eta, temperature
    )

    return get_temperature_corrected_gaussian_noise_operators(
        hamiltonian,
        a,
        lambda_,
        temperature,
    )
