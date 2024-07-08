from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    BasisLike,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator_list import (
        SingleBasisDiagonalOperatorList,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        Eigenstate,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import SingleFlatIndexLike

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])


_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)

_SB0Inv = TypeVar("_SB0Inv", bound=TupleBasisLike[*tuple[Any, ...]])
_SB1Inv = TypeVar("_SB1Inv", bound=TupleBasisLike[*tuple[Any, ...]])


class Operator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: TupleBasisLike[_B0_co, _B1_co]
    # We need higher kinded types, and const generics to do this properly
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisOperator = Operator[_B0_co, _B0_co]
"""Represents an operator where both vector and dual vector uses the same basis"""


class DiagonalOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: TupleBasisLike[_B0_co, _B1_co]
    """Basis of the lhs (first index in array)"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class StatisticalDiagonalOperator(DiagonalOperator[_B0_co, _B1_co]):
    """Represents a statistical operator in the given basis."""

    standard_deviation: np.ndarray[tuple[int], np.dtype[np.float64]]


def as_operator(operator: DiagonalOperator[_B0, _B1]) -> Operator[_B0, _B1]:
    """
    Convert a diagonal operator into an operator.

    Parameters
    ----------
    operator : DiagonalOperator[_B0_co, _B1_co]

    Returns
    -------
    Operator[_B0_co, _B1_co]
    """
    return {"basis": operator["basis"], "data": np.diag(operator["data"])}


def as_diagonal_operator(operator: Operator[_B0, _B1]) -> DiagonalOperator[_B0, _B1]:
    """
    Convert an operator into a diagonal operator.

    Parameters
    ----------
    operator : DiagonalOperator[_B0_co, _B1_co]

    Returns
    -------
    Operator[_B0_co, _B1_co]
    """
    diagonal = np.diag(operator["data"].reshape(operator["basis"].shape))
    return {"basis": operator["basis"], "data": diagonal.reshape(-1)}


def sum_diagonal_operator_over_axes(
    operator: DiagonalOperator[_SB0Inv, _SB1Inv], axes: tuple[int, ...]
) -> DiagonalOperator[Any, Any]:
    """
    given a diagonal operator, sum the states over axes.

    Parameters
    ----------
    states : DiagonalOperator[Any, Any]
    axes : tuple[int, ...]

    Returns
    -------
    DiagonalOperator[Any, Any]
    """
    BasisUtil(operator["basis"])
    traced_basis = tuple(
        b for (i, b) in enumerate(operator["basis"][0]) if i not in axes
    )
    return {
        "basis": TupleBasis(TupleBasis(*traced_basis), TupleBasis(*traced_basis)),
        "data": np.sum(
            operator["data"].reshape(operator["basis"][0].shape), axis=axes
        ).reshape(-1),
    }


SingleBasisDiagonalOperator = DiagonalOperator[_B0, _B0]


def get_eigenvalue(
    eigenvalue_list: SingleBasisDiagonalOperator[BasisLike[Any, Any]],
    idx: SingleFlatIndexLike,
) -> np.complex128:
    """
    Get a single eigenvalue from the list.

    Parameters
    ----------
    eigenvalue_list : EigenvalueList[_L0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    np.complex_
    """
    return eigenvalue_list["data"][idx]


def average_eigenvalues(
    eigenvalues: SingleBasisDiagonalOperator[TupleBasisLike[*tuple[Any, ...]]],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> SingleBasisDiagonalOperator[TupleBasis[*tuple[Any, ...]]]:
    """
    Average eigenvalues over the given axis.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_B0Inv]
    axis : tuple[int, ...] | None, optional
        axis, by default None
    weights : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        weights, by default None

    Returns
    -------
    EigenvalueList[Any]
    """
    axis = tuple(range(eigenvalues["basis"].ndim)) if axis is None else axis
    basis = tuple(b for (i, b) in enumerate(eigenvalues["basis"][0]) if i not in axis)
    return {
        "basis": TupleBasis(TupleBasis(*basis), TupleBasis(*basis)),
        "data": np.average(
            eigenvalues["data"].reshape(*eigenvalues["basis"][0].shape),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }


def average_eigenvalues_list(
    eigenvalues: SingleBasisDiagonalOperatorList[_B0, _SB0Inv],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> SingleBasisDiagonalOperatorList[_B0, TupleBasis[*tuple[Any, ...]]]:
    """
    Average eigenvalues over the given axis.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_B0Inv]
    axis : tuple[int, ...] | None, optional
        axis, by default None
    weights : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        weights, by default None

    Returns
    -------
    EigenvalueList[Any]
    """
    axis = tuple(range(eigenvalues["basis"].ndim)) if axis is None else axis
    basis = tuple(
        b for (i, b) in enumerate(eigenvalues["basis"][1][0]) if i not in axis
    )
    return {
        "basis": TupleBasis(
            eigenvalues["basis"][0],
            TupleBasis(TupleBasis(*basis), TupleBasis(*basis)),
        ),
        "data": np.average(
            eigenvalues["data"].reshape(
                eigenvalues["basis"][0].n, *eigenvalues["basis"][1][0].shape
            ),
            axis=tuple(1 + ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }


def apply_function_to_operator(
    operator: SingleBasisOperator[_B0],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.complex128]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
) -> SingleBasisOperator[_B0]:
    res = np.linalg.eig(operator["data"].reshape(operator["basis"].shape))
    eigenvalues = fn(res.eigenvalues)
    data = np.einsum(
        "k,ak,kb->ab", eigenvalues, res.eigenvectors, np.linalg.inv(res.eigenvectors)
    )

    return {"basis": operator["basis"], "data": data}


def matmul_operator(
    lhs: Operator[_B0, _B1], rhs: Operator[_B1, _B2]
) -> Operator[_B0, _B2]:
    data = np.tensordot(
        lhs["data"].reshape(lhs["basis"].shape),
        rhs["data"].reshape(rhs["basis"].shape),
        axes=(1, 0),
    )
    return {"basis": TupleBasis(lhs["basis"][0], rhs["basis"][1]), "data": data}


def add_operator(a: Operator[_B0, _B1], b: Operator[_B0, _B1]) -> Operator[_B0, _B1]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Operator[_B0Inv]
    b : Operator[_B0Inv]

    Returns
    -------
    Operator[_B0Inv]
    """
    return {"basis": a["basis"], "data": a["data"] + b["data"]}


def subtract_operator(
    a: Operator[_B0, _B1], b: Operator[_B0, _B1]
) -> Operator[_B0, _B1]:
    """
    Subtract two operators (a-b).

    Parameters
    ----------
    a : Operator[_B0Inv]
    b : Operator[_B0Inv]

    Returns
    -------
    Operator[_B0Inv]
    """
    return {"basis": a["basis"], "data": a["data"] - b["data"]}


def apply_operator_to_state(
    lhs: Operator[_B0, _B1], state: StateVector[_B2]
) -> Eigenstate[_B0]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Operator[_B0Inv]
    b : Operator[_B0Inv]

    Returns
    -------
    Operator[_B0Inv]
    """
    converted = convert_state_vector_to_basis(state, lhs["basis"][1])
    data = np.einsum(
        "ik,k->i",
        lhs["data"].reshape(lhs["basis"].shape),
        converted["data"].reshape(converted["basis"].n),
    )
    norm = np.sqrt(np.sum(np.abs(np.square(data))))
    return {"basis": lhs["basis"][0], "data": data / norm, "eigenvalue": norm}
