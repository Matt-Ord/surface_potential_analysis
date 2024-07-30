from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

from scipy.constants import Boltzmann

from surface_potential_analysis.basis.util import get_displacements_x
from surface_potential_analysis.kernel.kernel import (
    SingleBasisDiagonalNoiseOperatorList,
    SingleBasisNoiseOperatorList,
    as_noise_operators,
)
from surface_potential_analysis.operator.operations import (
    add_list_list,
    get_commutator_operator_list,
    scale_operator_list,
)
from surface_potential_analysis.operator.operator import SingleBasisOperator
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    import numpy as np

    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        IsotropicNoiseKernel,
        SingleBasisNoiseOperatorList,
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
    _B2 = TypeVar(
        "_B2",
        bound=BasisLike[Any, Any],
    )


def build_isotropic_kernel_from_function(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    """
    Get an Isotropic Kernel with a correllation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike[Any, Any, Any]
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ]
        beta(x-x'), the correllation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    ]
    """
    displacements = get_displacements_x(basis)[0]
    correlation = fn(displacements)

    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    return {"basis": basis_x, "data": correlation.ravel()}


def _get_temperature_corrected_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisOperatorList[
        _B0,
        _B2,
    ],
    temperature: float,
) -> SingleBasisOperatorList[
    _B0,
    _B1,
]:
    commutator = get_commutator_operator_list(hamiltonian, operators)
    correction = scale_operator_list(-1 / (4 * Boltzmann * temperature), commutator)
    return add_list_list(correction, operators)


def get_temperature_corrected_noise_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisNoiseOperatorList[
        _B0,
        _B2,
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
        _B2,
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
    operators_full = as_noise_operators(operators)
    return get_temperature_corrected_noise_operators(
        hamiltonian,
        operators_full,
        temperature,
    )
