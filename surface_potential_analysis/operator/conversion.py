from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from surface_potential_analysis.basis.legacy import (
    convert_matrix,
)
from surface_potential_analysis.operator.operator import DiagonalOperator, as_operator
from surface_potential_analysis.operator.operator_list import (
    diagonal_operator_list_as_full,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import BasisLike, TupleBasisLike
    from surface_potential_analysis.operator.operator import (
        Operator,
    )
    from surface_potential_analysis.operator.operator_list import (
        DiagonalOperatorList,
        OperatorList,
    )

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike)
    _B1Inv = TypeVar("_B1Inv", bound=BasisLike)
    _B2Inv = TypeVar("_B2Inv", bound=BasisLike)
    _B3Inv = TypeVar("_B3Inv", bound=BasisLike)
    _B4 = TypeVar("_B4", bound=BasisLike)


def convert_operator_to_basis(
    operator: Operator[_B0Inv, _B1Inv], basis: TupleBasisLike[_B2Inv, _B3Inv]
) -> Operator[_B2Inv, _B3Inv]:
    """
    Given an operator, convert it to the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    basis : _B3d1Inv

    Returns
    -------
    Eigenstate[_B3d1Inv]
    """
    converted = convert_matrix(
        operator["data"].reshape(operator["basis"].shape),
        operator["basis"][0],
        basis[0],
        operator["basis"][1],
        basis[1],
    )
    return {"basis": basis, "data": converted.reshape(-1)}


def convert_diagonal_operator_to_basis(
    operator: DiagonalOperator[_B0Inv, _B1Inv],
    basis: TupleBasisLike[_B2Inv, _B3Inv],
) -> Operator[_B2Inv, _B3Inv]:
    """Given an operator, convert it to the given basis.

    Parameters
    ----------
    operator : OperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    OperatorList[_B4, _B2Inv, _B3Inv]
    """
    full = as_operator(operator)
    return convert_operator_to_basis(full, basis)


def convert_operator_list_to_basis(
    operator: OperatorList[_B4, _B0Inv, _B1Inv],
    basis: TupleBasisLike[_B2Inv, _B3Inv],
) -> OperatorList[_B4, _B2Inv, _B3Inv]:
    """Given an operator, convert it to the given basis.

    Parameters
    ----------
    operator : OperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    OperatorList[_B4, _B2Inv, _B3Inv]
    """
    converted = convert_matrix(
        operator["data"].reshape(
            operator["basis"][0].size, *operator["basis"][1].shape
        ),
        operator["basis"][1][0],
        basis[0],
        operator["basis"][1][1],
        basis[1],
        axes=(1, 2),
    )
    return {
        "basis": VariadicTupleBasis((operator["basis"][0], basis), None),
        "data": converted.reshape(-1),
    }


def convert_diagonal_operator_list_to_basis(
    operator: DiagonalOperatorList[_B4, _B0Inv, _B1Inv],
    basis: TupleBasisLike[_B2Inv, _B3Inv],
) -> OperatorList[_B4, _B2Inv, _B3Inv]:
    """Given an operator, convert it to the given basis.

    Parameters
    ----------
    operator : OperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    OperatorList[_B4, _B2Inv, _B3Inv]
    """
    full = diagonal_operator_list_as_full(operator)
    return convert_operator_list_to_basis(full, basis)
