from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Self,
    TypedDict,
    TypeVar,
    cast,
    override,
)

import numpy as np
from slate.array.array import SlateArray
from slate.basis._basis import Basis
from slate.basis.stacked import VariadicTupleBasis
from slate.metadata._metadata import BasisMetadata
from slate.metadata.stacked import StackedMetadata

from surface_potential_analysis.basis.legacy import BasisLike, TupleBasisLike
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator_list import (
        SingleBasisDiagonalOperatorList,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import (
        Eigenstate,
    )
    from surface_potential_analysis.state_vector.state_vector import LegacyStateVector
    from surface_potential_analysis.types import SingleFlatIndexLike

_B0 = TypeVar("_B0", bound=BasisLike)
_B1 = TypeVar("_B1", bound=BasisLike)
_B2 = TypeVar("_B2", bound=BasisLike)


_B0_co = TypeVar("_B0_co", bound=BasisLike, covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike, covariant=True)

_SB0Inv = TypeVar("_SB0Inv", bound=TupleBasisLike[*tuple[Any, ...]])
_SB1Inv = TypeVar("_SB1Inv", bound=TupleBasisLike[*tuple[Any, ...]])


class Operator[DT: np.generic, B: Basis[StackedMetadata[BasisMetadata, Any], Any]](
    SlateArray[DT, B]
):
    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self: Self, basis: B1
    ) -> Operator[DT, B1]:
        """Get the Operator with the basis set to basis."""
        return Operator(basis, self.basis.__convert_vector_into__(self.raw_data, basis))

    def __add__[_DT: np.number[Any], M: StackedMetadata[BasisMetadata, Any]](
        self: Operator[_DT, Basis[M, Any]], other: Operator[_DT, Basis[M, Any]]
    ) -> Operator[_DT, Basis[M, Any]]:
        res = self.raw_data + other.with_basis(self.basis).raw_data
        data = cast(np.ndarray[Any, np.dtype[_DT]], res)
        return Operator[_DT, Basis[M, Any]](self.basis, data)


class LegacyOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: VariadicTupleBasis[_B0_co, _B1_co, Any, np.complex128]
    # We need higher kinded types, and const generics to do this properly
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisOperator = LegacyOperator[_B0_co, _B0_co]
"""Represents an operator where both vector and dual vector uses the same basis"""


class LegacyDiagonalOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: VariadicTupleBasis[_B0_co, _B1_co, Any, np.complex128]
    """Basis of the lhs (first index in array)"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class StatisticalDiagonalOperator(LegacyDiagonalOperator[_B0_co, _B1_co]):
    """Represents a statistical operator in the given basis."""

    standard_deviation: np.ndarray[tuple[int], np.dtype[np.float64]]


def as_operator(operator: LegacyDiagonalOperator[_B0, _B1]) -> LegacyOperator[_B0, _B1]:
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


def as_diagonal_operator(
    operator: LegacyOperator[_B0, _B1],
) -> LegacyDiagonalOperator[_B0, _B1]:
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
    operator: LegacyDiagonalOperator[_SB0Inv, _SB1Inv], axes: tuple[int, ...]
) -> LegacyDiagonalOperator[Any, Any]:
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
    traced_basis = tuple(
        b for (i, b) in enumerate(operator["basis"][0]) if i not in axes
    )
    return {
        "basis": VariadicTupleBasis(
            (VariadicTupleBasis((traced_basis), None), None),
            VariadicTupleBasis((traced_basis), None),
        ),
        "data": np.sum(
            operator["data"].reshape(operator["basis"][0].shape), axis=axes
        ).reshape(-1),
    }


SingleBasisDiagonalOperator = LegacyDiagonalOperator[_B0, _B0]


def get_eigenvalue(
    eigenvalue_list: SingleBasisDiagonalOperator[BasisLike],
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
    axis = tuple(range(eigenvalues["basis"].n_dim)) if axis is None else axis
    basis = tuple(b for (i, b) in enumerate(eigenvalues["basis"][0]) if i not in axis)
    return {
        "basis": VariadicTupleBasis(
            (VariadicTupleBasis((basis), None), None), VariadicTupleBasis((basis), None)
        ),
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
    axis = tuple(range(eigenvalues["basis"].n_dim)) if axis is None else axis
    basis = tuple(
        b for (i, b) in enumerate(eigenvalues["basis"][1][0]) if i not in axis
    )
    return {
        "basis": TupleBasis(
            eigenvalues["basis"][0],
            VariadicTupleBasis(
                (VariadicTupleBasis((basis), None), None),
                VariadicTupleBasis((basis), None),
            ),
        ),
        "data": np.average(
            eigenvalues["data"].reshape(
                eigenvalues["basis"][0].size, *eigenvalues["basis"][1][0].shape
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
    data = np.einsum(  # type: ignore lib
        "k,ak,kb->ab", eigenvalues, res.eigenvectors, np.linalg.inv(res.eigenvectors)
    )

    return {"basis": operator["basis"], "data": data}


def matmul_operator(
    lhs: LegacyOperator[_B0, _B1], rhs: LegacyOperator[_B1, _B2]
) -> LegacyOperator[_B0, _B2]:
    data = np.tensordot(
        lhs["data"].reshape(lhs["basis"].shape),
        rhs["data"].reshape(rhs["basis"].shape),
        axes=(1, 0),
    )
    return {
        "basis": VariadicTupleBasis((lhs["basis"][0], rhs["basis"][1]), None),
        "data": data,
    }


def add_legacy_operator(
    a: LegacyOperator[_B0, _B1], b: LegacyOperator[_B0, _B1]
) -> LegacyOperator[_B0, _B1]:
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
    a: LegacyOperator[_B0, _B1], b: LegacyOperator[_B0, _B1]
) -> LegacyOperator[_B0, _B1]:
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
    lhs: LegacyOperator[_B0, _B1], state: LegacyStateVector[_B2]
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
    data = np.einsum(  # type: ignore lib
        "ik,k->i",
        lhs["data"].reshape(lhs["basis"].shape),
        converted["data"].reshape(converted["basis"].n),
    )
    norm = np.sqrt(np.sum(np.abs(np.square(data))))
    return {"basis": lhs["basis"][0], "data": data / norm, "eigenvalue": norm}
