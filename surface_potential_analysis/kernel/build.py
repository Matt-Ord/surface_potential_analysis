from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar

import numpy as np
from scipy.constants import Boltzmann  # type: ignore no stub

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import get_displacements_x
from surface_potential_analysis.kernel.kernel import (
    DiagonalNoiseKernel,
    SingleBasisNoiseOperatorList,
    get_diagonal_kernel_from_operators,
    get_full_kernel_from_operators,
)
from surface_potential_analysis.kernel.solve import (
    get_noise_operators_diagonal_eigenvalue,
    get_noise_operators_eigenvalue,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operations import (
    add_list_list,
    get_commutator_diagonal_operator_list,
    get_commutator_operator_list,
    scale_operator_list,
)
from surface_potential_analysis.operator.operator import SingleBasisOperator
from surface_potential_analysis.operator.operator_list import as_operator_list

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseOperatorList,
        IsotropicNoiseKernel,
        NoiseKernel,
        NoiseOperatorList,
        SingleBasisDiagonalNoiseOperatorList,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.operator.operator_list import (
        OperatorList,
        SingleBasisDiagonalOperatorList,
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
    displacements = get_displacements_x(basis)
    correlation = fn(displacements["data"].reshape(displacements["basis"].shape)[0])

    return {"basis": displacements["basis"][0], "data": correlation.ravel()}


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


def _get_temperature_corrected_diagonal_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisDiagonalOperatorList[
        _B0,
        _B2,
    ],
    temperature: float,
) -> OperatorList[_B0, _B2, _B2]:
    converted = convert_operator_to_basis(hamiltonian, operators["basis"][1])
    commutator = get_commutator_diagonal_operator_list(converted, operators)
    correction = scale_operator_list(-1 / (4 * Boltzmann * temperature), commutator)
    return add_list_list(correction, as_operator_list(operators))


def get_temperature_corrected_diagonal_noise_operators(
    hamiltonian: SingleBasisOperator[_B1],
    operators: SingleBasisDiagonalNoiseOperatorList[
        _B0,
        _B2,
    ],
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    _B0,
    _B2,
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
    corrected_operators = _get_temperature_corrected_diagonal_operators(
        hamiltonian,
        operators,
        temperature,
    )

    return {
        "basis": corrected_operators["basis"],
        "data": corrected_operators["data"],
        "eigenvalue": operators["eigenvalue"],
    }


def truncate_diagonal_noise_operator_list(
    operators: DiagonalNoiseOperatorList[_B2, _B0, _B1],
    truncation: Iterable[int],
) -> DiagonalNoiseOperatorList[FundamentalBasis[int], _B0, _B1]:
    """
    Get a truncated list of diagonal operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[FundamentalBasis[int], _B0, _B1]
    truncation : Iterable[int]

    Returns
    -------
    DiagonalNoiseOperatorList[FundamentalBasis[int], _B0, _B1]
    """
    args = np.argsort(operators["eigenvalue"])[::-1][np.array(list(truncation))]
    data = operators["data"].reshape(operators["basis"][0].n, -1)[args, :]
    return {
        "basis": TupleBasis(FundamentalBasis(data.shape[0]), operators["basis"][1]),
        "data": data,
        "eigenvalue": operators["eigenvalue"][args],
    }


def truncate_noise_operator_list(
    operators: NoiseOperatorList[_B2, _B0, _B1],
    truncation: Iterable[int],
) -> NoiseOperatorList[FundamentalBasis[int], _B0, _B1]:
    """
    Get a truncated list of operators.

    Parameters
    ----------
    operators : NoiseOperatorList[FundamentalBasis[int], _B0, _B1]
    truncation : Iterable[int]

    Returns
    -------
    NoiseOperatorList[FundamentalBasis[int], _B0, _B1]
    """
    args = np.argsort(operators["eigenvalue"])[::-1][np.array(list(truncation))]
    data = operators["data"].reshape(operators["basis"][0].n, -1)[args, :]
    return {
        "basis": TupleBasis(FundamentalBasis(data.shape[0]), operators["basis"][1]),
        "data": data,
        "eigenvalue": operators["eigenvalue"][args],
    }


def truncate_noise_kernel(
    kernel: NoiseKernel[_B0, _B1, _B0, _B1],
    truncation: Iterable[int],
) -> NoiseKernel[_B0, _B1, _B0, _B1]:
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
    operators = get_noise_operators_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return get_full_kernel_from_operators(truncated)


def truncate_diagonal_noise_kernel(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
    truncation: Iterable[int],
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

    truncated = truncate_diagonal_noise_operator_list(operators, truncation)
    return get_diagonal_kernel_from_operators(truncated)
