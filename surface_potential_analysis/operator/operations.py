from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisOperator,
    matmul_operator,
    subtract_operator,
)
from surface_potential_analysis.operator.operator_list import OperatorList

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.operator.operator_list import (
        OperatorList,
        SingleBasisOperatorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _B3 = TypeVar("_B3", bound=BasisLike[Any, Any])
    _B4 = TypeVar("_B4", bound=BasisLike[Any, Any])


def matmul_list_operator(
    lhs: OperatorList[_B3, _B0, _B1], rhs: Operator[_B4, _B2]
) -> OperatorList[_B3, _B0, _B2]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : OperatorList[_B3, _B0, _B1]
    rhs : Operator[_B1, _B2]

    Returns
    -------
    OperatorList[_B3, _B0, _B2]
    """
    converted = convert_operator_to_basis(
        rhs, TupleBasis(lhs["basis"][1][1], rhs["basis"][1])
    )

    data = np.tensordot(
        lhs["data"].reshape(-1, *lhs["basis"][1].shape),
        converted["data"].reshape(rhs["basis"].shape),
        axes=(2, 0),
    ).reshape(-1)
    return {
        "basis": TupleBasis(
            lhs["basis"][0], TupleBasis(lhs["basis"][1][0], rhs["basis"][1])
        ),
        "data": data,
    }


def matmul_operator_list(
    lhs: Operator[_B0, _B1], rhs: OperatorList[_B3, _B4, _B2]
) -> OperatorList[_B3, _B0, _B2]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : Operator[_B0, _B1]
    rhs : OperatorList[_B3, _B1, _B2]

    Returns
    -------
    OperatorList[_B3, _B0, _B2]
    """
    converted = convert_operator_list_to_basis(
        rhs, TupleBasis(lhs["basis"][1], rhs["basis"][1][1])
    )
    data = np.einsum(
        "ik,mkj->mij",
        lhs["data"].reshape(lhs["basis"].shape),
        converted["data"].reshape(-1, *converted["basis"][1].shape),
    ).reshape(-1)
    return {
        "basis": TupleBasis(
            converted["basis"][0], TupleBasis(lhs["basis"][0], converted["basis"][1][1])
        ),
        "data": data,
    }


def subtract_list_list(
    lhs: OperatorList[_B3, _B0, _B1], rhs: OperatorList[_B3, _B2, _B4]
) -> OperatorList[_B3, _B0, _B1]:
    """
    Subtract two operator list lhs-rhs.

    Parameters
    ----------
    lhs : OperatorList[_B3, _B0, _B1]
    rhs : OperatorList[_B3, _B0, _B1]

    Returns
    -------
    OperatorList[_B3, _B0, _B1]
    """
    converted = convert_operator_list_to_basis(rhs, lhs["basis"][1])
    return {
        "basis": lhs["basis"],
        "data": lhs["data"] - converted["data"],
    }


def get_commutator_operator_list(
    lhs: SingleBasisOperator[_B0], rhs: OperatorList[_B1, _B2, _B3]
) -> SingleBasisOperatorList[_B1, _B0]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to lhs rhs - rhs lhs

    Parameters
    ----------
    lhs : SingleBasisOperator[_B0]
    rhs : SingleBasisOperator[_B0]

    Returns
    -------
    SingleBasisOperator[_B0]
    """
    converted = convert_operator_list_to_basis(rhs, lhs["basis"])
    lhs_rhs = matmul_operator_list(lhs, converted)
    rhs_lhs = matmul_list_operator(converted, lhs)
    return subtract_list_list(lhs_rhs, rhs_lhs)


def add_list_list(
    lhs: OperatorList[_B3, _B0, _B1], rhs: OperatorList[_B3, _B2, _B4]
) -> OperatorList[_B3, _B0, _B1]:
    """
    Add two operator list lhs+rhs.

    Parameters
    ----------
    lhs : OperatorList[_B3, _B0, _B1]
    rhs : OperatorList[_B3, _B0, _B1]

    Returns
    -------
    OperatorList[_B3, _B0, _B1]
    """
    converted = convert_operator_list_to_basis(rhs, lhs["basis"][1])
    return {
        "basis": lhs["basis"],
        "data": lhs["data"] + converted["data"],
    }


def scale_operator_list(
    factor: complex, operator: OperatorList[_B3, _B0, _B1]
) -> OperatorList[_B3, _B0, _B1]:
    """
    Scale the operator list.

    Equivalent to multiplying each operator by factor

    Returns
    -------
    OperatorList[_B3, _B0, _B1]
    """
    return {
        "basis": operator["basis"],
        "data": operator["data"] * factor,
    }


def get_commutator(
    lhs: SingleBasisOperator[_B0], rhs: SingleBasisOperator[_B0]
) -> SingleBasisOperator[_B0]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to ths rhs - rhs lhs

    Parameters
    ----------
    lhs : SingleBasisOperator[_B0]
    rhs : SingleBasisOperator[_B0]

    Returns
    -------
    SingleBasisOperator[_B0]
    """
    lhs_rhs = matmul_operator(lhs, rhs)
    rhs_lhs = matmul_operator(rhs, lhs)
    return subtract_operator(lhs_rhs, rhs_lhs)
