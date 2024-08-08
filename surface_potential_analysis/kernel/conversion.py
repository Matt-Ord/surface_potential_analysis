from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.basis_like import convert_vector
from surface_potential_analysis.kernel.kernel import get_full_kernel_from_operators
from surface_potential_analysis.kernel.solve import (
    get_noise_operators_diagonal_eigenvalue,
    get_noise_operators_eigenvalue,
)
from surface_potential_analysis.operator.conversion import (
    convert_diagonal_operator_list_to_basis,
    convert_operator_list_to_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        DiagonalNoiseOperatorList,
        IsotropicNoiseKernel,
        NoiseKernel,
        NoiseOperatorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _B3 = TypeVar("_B3", bound=BasisLike[Any, Any])
    _B4 = TypeVar("_B4", bound=BasisLike[Any, Any])


def convert_noise_operator_list_to_basis(
    operator: NoiseOperatorList[_B4, _B0, _B1],
    basis: TupleBasisLike[_B2, _B3],
) -> NoiseOperatorList[_B4, _B2, _B3]:
    """Given a noise operator, convert it to the given basis.

    Parameters
    ----------
    operator : NoiseOperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseOperatorList[_B4, _B2Inv, _B3Inv]
    """
    converted = convert_operator_list_to_basis(operator, basis)
    return {
        "basis": converted["basis"],
        "data": converted["data"],
        "eigenvalue": operator["eigenvalue"],
    }


def convert_diagonal_noise_operator_list_to_basis(
    operator: DiagonalNoiseOperatorList[_B4, _B0, _B1],
    basis: TupleBasisLike[_B2, _B3],
) -> NoiseOperatorList[_B4, _B2, _B3]:
    """Given a noise operator, convert it to the given basis.

    Parameters
    ----------
    operator : NoiseOperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseOperatorList[_B4, _B2Inv, _B3Inv]
    """
    converted = convert_diagonal_operator_list_to_basis(operator, basis)
    return {
        "basis": converted["basis"],
        "data": converted["data"],
        "eigenvalue": operator["eigenvalue"],
    }


def convert_kernel_to_basis(
    kernel: NoiseKernel[_B0, _B1, _B0, _B1],
    basis: TupleBasisLike[_B2, _B3],
) -> NoiseKernel[_B2, _B3, _B2, _B3]:
    """Convert the kernel to the given basis.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators = get_noise_operators_eigenvalue(kernel)
    converted = convert_noise_operator_list_to_basis(operators, basis)
    return get_full_kernel_from_operators(converted)


def convert_diagonal_kernel_to_basis(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
    basis: TupleBasisLike[_B2, _B3],
) -> NoiseKernel[_B2, _B3, _B2, _B3]:
    """Convert the kernel to the given basis.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators = get_noise_operators_diagonal_eigenvalue(kernel)
    converted = convert_diagonal_noise_operator_list_to_basis(operators, basis)
    return get_full_kernel_from_operators(converted)


def convert_isotropic_kernel_to_basis(
    kernel: IsotropicNoiseKernel[_B0],
    basis: _B2,
) -> IsotropicNoiseKernel[_B2]:
    """Convert the kernel to the given basis.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    data = convert_vector(kernel["data"], kernel["basis"], basis)
    return {"data": data, "basis": basis}
