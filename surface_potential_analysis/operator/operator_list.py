from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Iterable, TypedDict, TypeVar, Unpack

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import TupleBasis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.types import SingleFlatIndexLike

    from .operator import (
        DiagonalOperator,
        Operator,
    )
_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
_B3 = TypeVar("_B3", bound=BasisLike[Any, Any])

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)
_B2_co = TypeVar("_B2_co", bound=BasisLike[Any, Any], covariant=True)


class OperatorList(TypedDict, Generic[_B0_co, _B1_co, _B2_co]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: TupleBasisLike[_B0_co, TupleBasisLike[_B1_co, _B2_co]]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


SingleBasisOperatorList = OperatorList[_B0, _B1, _B1]


def select_operator(
    operator_list: OperatorList[_B0, _B1, _B2], idx: SingleFlatIndexLike
) -> Operator[_B1, _B2]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": operator_list["basis"][1],
        "data": operator_list["data"]
        .reshape(operator_list["basis"].shape)[idx]
        .reshape(-1),
    }


class DiagonalOperatorList(TypedDict, Generic[_B0_co, _B1_co, _B2_co]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: TupleBasisLike[_B0_co, TupleBasisLike[_B1_co, _B2_co]]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


SingleBasisDiagonalOperatorList = DiagonalOperatorList[_B0, _B1, _B1]


def select_operator_diagonal(
    operator_list: DiagonalOperatorList[_B0, _B1, _B2], idx: SingleFlatIndexLike
) -> DiagonalOperator[_B1, _B2]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": operator_list["basis"][1],
        "data": operator_list["data"]
        .reshape(operator_list["basis"][0].n, -1)[idx]
        .reshape(-1),
    }


def as_operator_list(
    diagonal_list: DiagonalOperatorList[_B0, _B1, _B2],
) -> OperatorList[_B0, _B1, _B2]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    n = diagonal_list["basis"][1].shape[0]
    data = np.einsum("ij,jk->ijk", diagonal_list["data"].reshape(-1, n), np.eye(n))

    return {
        "basis": diagonal_list["basis"],
        "data": data.ravel(),
    }


def operator_list_from_iter(
    iters: Iterable[Operator[_B1, _B2]],
) -> OperatorList[FundamentalBasis[int], _B1, _B2]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    operators = list(iters)
    basis = operators[0]["basis"]
    n = len(operators)
    data = np.array([x["data"] for x in operators])
    return {
        "basis": TupleBasis(FundamentalBasis(n), basis),
        "data": data.ravel(),
    }


def operator_list_into_iter(
    operators: OperatorList[Any, _B1, _B2],
) -> Iterable[Operator[_B1, _B2]]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    basis = operators["basis"][1]
    data = operators["data"].reshape(operators["basis"].shape)

    return [
        {
            "basis": basis,
            "data": d,
        }
        for d in data
    ]


def as_flat_operator(
    operator_list: SingleBasisDiagonalOperatorList[_B0, _B1],
) -> SingleBasisDiagonalOperator[TupleBasisLike[_B0, Unpack[tuple[Any, ...]]]]:
    """
    Given a diagonal operator list, re-interpret it as an operator.

    Parameters
    ----------
    operator_list : SingleBasisDiagonalOperatorList[_B0, _B1]

    Returns
    -------
    SingleBasisDiagonalOperator[TupleBasisLike[_B0, Unpack[tuple[Any, ...]]]]

    """
    basis = TupleBasis[_B0, Unpack[tuple[Any, ...]]](
        operator_list["basis"][0], *operator_list["basis"][1]
    )
    return {
        "basis": TupleBasis(basis, basis),
        "data": operator_list["data"],
    }


def as_diagonal_operator(
    operator_list: DiagonalOperatorList[_B0, _B1, _B2],
    idx: SingleFlatIndexLike,
) -> DiagonalOperator[_B1, _B2]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": operator_list["basis"][1],
        "data": operator_list["data"]
        .reshape(operator_list["basis"][0].n, -1)[idx]
        .reshape(-1),
    }


def sum_diagonal_operator_list_over_axes(
    states: DiagonalOperatorList[_B0, Any, Any], axes: tuple[int, ...]
) -> DiagonalOperatorList[_B0, Any, Any]:
    """
    given a diagonal operator list, sum the states over axes.

    Parameters
    ----------
    states : DiagonalOperatorList[Any, Any, _L0Inv]
    axes : tuple[int, ...]

    Returns
    -------
    DiagonalOperatorList[Any, Any, _L0Inv]
    """
    traced_basis = tuple(
        b for (i, b) in enumerate(states["basis"][1][0]) if i not in axes
    )
    # TODO: just wrong
    return {
        "basis": TupleBasis(states["basis"][0], TupleBasis(traced_basis, traced_basis)),
        "data": np.sum(
            states["data"].reshape(-1, *states["basis"][1][0].shape),
            axis=tuple(1 + np.array(axes, dtype=np.int_)),
        ).reshape(states["data"].shape[0], -1),
    }


def matmul_operator_list(
    lhs: Operator[_B0, _B1], rhs: OperatorList[_B3, _B1, _B2]
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
    data = np.einsum(
        "ik,mkj->mij",
        lhs["data"].reshape(lhs["basis"].shape),
        rhs["data"].reshape(-1, *rhs["basis"][1].shape),
    ).reshape(-1)
    return {
        "basis": TupleBasis(
            rhs["basis"][0], TupleBasis(lhs["basis"][0], rhs["basis"][1][1])
        ),
        "data": data,
    }


def matmul_list_operator(
    lhs: OperatorList[_B3, _B0, _B1], rhs: Operator[_B1, _B2]
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
    data = np.tensordot(
        lhs["data"].reshape(-1, *lhs["basis"][1].shape),
        rhs["data"].reshape(rhs["basis"].shape),
        axes=(2, 0),
    ).reshape(-1)
    return {
        "basis": TupleBasis(
            lhs["basis"][0], TupleBasis(lhs["basis"][1][0], rhs["basis"][1])
        ),
        "data": data,
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


def add_list_list(
    lhs: OperatorList[_B3, _B0, _B1], rhs: OperatorList[_B3, _B0, _B1]
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
    return {
        "basis": lhs["basis"],
        "data": lhs["data"] + rhs["data"],
    }


def subtract_list_list(
    lhs: OperatorList[_B3, _B0, _B1], rhs: OperatorList[_B3, _B0, _B1]
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
    return {
        "basis": lhs["basis"],
        "data": lhs["data"] - rhs["data"],
    }


def get_commutator_operator_list(
    lhs: SingleBasisOperator[_B0], rhs: SingleBasisOperatorList[_B1, _B0]
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
    lhs_rhs = matmul_operator_list(lhs, rhs)
    rhs_lhs = matmul_list_operator(rhs, lhs)
    return subtract_list_list(lhs_rhs, rhs_lhs)
