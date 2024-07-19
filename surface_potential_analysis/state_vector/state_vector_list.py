from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
)
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.types import (
    SingleFlatIndexLike,
)
from surface_potential_analysis.util.util import get_data_in_axes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.operator.operator import Operator
    from surface_potential_analysis.state_vector.eigenstate_list import (
        EigenstateList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateDualVector,
        StateVector,
    )
    from surface_potential_analysis.types import (
        SingleFlatIndexLike,
        SingleStackedIndexLike,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _B3 = TypeVar("_B3", bound=BasisLike[Any, Any])
    _SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)


class StateVectorList(TypedDict, Generic[_B0_co, _B1_co]):
    """
    Represents a list of states.

    The first axis represents the basis of the list, and the second the basis of the states
    """

    basis: TupleBasisLike[_B0_co, _B1_co]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


def get_state_vector(
    state_list: StateVectorList[_B0, _B1], idx: SingleFlatIndexLike
) -> StateVector[_B1]:
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
        "basis": state_list["basis"][1],
        "data": state_list["data"].reshape(state_list["basis"].shape)[idx],
    }


def get_weighted_state_vector(
    state_list: StateVectorList[_B0, _B1], weights: StateVector[_B0]
) -> StateVector[_B1]:
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
    data = np.tensordot(
        weights["data"],
        state_list["data"].reshape(state_list["basis"][0].n, -1),
        axes=(0, 0),
    )
    return {"basis": state_list["basis"][1], "data": data}


def get_state_dual_vector(
    state_list: StateVectorList[_B0, _B1], idx: SingleFlatIndexLike
) -> StateDualVector[_B1]:
    """
    Get a single state dual vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": state_list["basis"][1],
        "data": np.conj(state_list["data"].reshape(state_list["basis"].shape)[idx]),
    }


def state_vector_list_into_iter(
    states: StateVectorList[_B0, _B1],
) -> Iterable[StateVector[_B1]]:
    """
    Select an eigenstate from an eigenstate collection.

    Parameters
    ----------
    collection : EigenstateColllection[_B0_co]
    bloch_idx : int
    band_idx : int

    Returns
    -------
    Eigenstate[_B0_co]
    """
    return [
        {
            "basis": states["basis"][1],
            "data": states["data"].reshape(states["basis"].shape)[idx],
        }
        for idx in range(states["basis"].shape[0])
    ]


def as_state_vector_list(
    states: Iterable[StateVector[_B1]],
) -> StateVectorList[FundamentalBasis[int], _B1]:
    states = list(states)
    return {
        "basis": TupleBasis(FundamentalBasis(len(states)), states[0]["basis"]),
        "data": np.array([w["data"] for w in states]).reshape(-1),
    }


def calculate_inner_products(
    state_0: StateVectorList[_B0, _B2],
    state_1: StateVectorList[_B1, _B3],
) -> Operator[_B0, _B1]:
    """
    Calculate the inner product of two states.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateDualVector[_B0Inv]

    Returns
    -------
    np.complex_
    """
    converted = convert_state_vector_list_to_basis(state_1, state_0["basis"][1])
    return {
        "basis": TupleBasis(state_0["basis"][0], state_1["basis"][0]),
        "data": np.einsum(
            "ik, jk -> ij",
            np.conj(state_0["data"]).reshape(state_0["basis"].shape),
            converted["data"].reshape(converted["basis"].shape),
        ).reshape(-1),
    }


def calculate_inner_products_elementwise(
    state_0: StateVectorList[_B0, _B2],
    state_1: StateVectorList[_B0, _B3],
) -> ValueList[_B0]:
    """
    Calculate the inner product of two states elementwise.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateDualVector[_B0Inv]

    Returns
    -------
    np.complex_
    """
    converted = convert_state_vector_list_to_basis(state_1, state_0["basis"][1])
    return {
        "basis": state_0["basis"][0],
        "data": np.einsum(
            "ik, ik -> i",
            np.conj(state_0["data"]).reshape(state_0["basis"].shape),
            converted["data"].reshape(converted["basis"].shape),
        ).reshape(-1),
    }


def calculate_inner_products_eigenvalues(
    state_0: EigenstateList[_B0, _B2],
    state_1: EigenstateList[_B1, _B2],
) -> Operator[_B0, _B1]:
    """
    Calculate the inner product of two states.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateDualVector[_B0Inv]

    Returns
    -------
    np.complex_
    """
    return {
        "basis": TupleBasis(state_0["basis"][0], state_1["basis"][0]),
        "data": np.einsum(
            "ik, jk, i, j -> ij",
            np.conj(state_0["data"]).reshape(state_0["basis"].shape),
            state_1["data"].reshape(state_1["basis"].shape),
            np.conj(state_0["eigenvalue"]),
            state_1["eigenvalue"],
        ).reshape(-1),
    }


def average_state_vector(
    probabilities: StateVectorList[_SB0, _B1],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> StateVectorList[Any, _B1]:
    """
    Average probabilities over several repeats.

    Parameters
    ----------
    probabilities : list[ProbabilityVectorList[_B0Inv, _L0Inv]]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    axis = tuple(range(probabilities["basis"][0].ndim)) if axis is None else axis
    basis = TupleBasis(
        *tuple(b for (i, b) in enumerate(probabilities["basis"][0]) if i not in axis)
    )
    return {
        "basis": TupleBasis(basis, probabilities["basis"][1]),
        "data": np.average(
            probabilities["data"].reshape(*probabilities["basis"][0].shape, -1),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }


def get_basis_states(
    basis: _B0,
) -> StateVectorList[FundamentalBasis[int], _B0]:
    """
    Get the eigenstates of a particular basis.

    Parameters
    ----------
    basis : _B0

    Returns
    -------
    StateVectorList[FundamentalBasis[int], _B0]

    """
    data = np.eye(basis.n, basis.n).astype(np.complex128)
    return {
        "basis": TupleBasis(FundamentalBasis(basis.n), basis),
        "data": data.reshape(-1),
    }


def get_state_along_axis(
    states: StateVectorList[_SB0, _B1],
    axes: tuple[int, ...] = (0,),
    idx: SingleStackedIndexLike | None = None,
) -> StateVectorList[Any, _B1]:
    """
    Get Probability from the list.

    Parameters
    ----------
    probabilities : ProbabilityVectorList[_B0Inv, _L0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    ProbabilityVector[_B0Inv]
    """
    ndim = states["basis"][0].ndim
    idx = tuple(0 for _ in range(ndim - len(axes))) if idx is None else idx
    final_basis = TupleBasis(
        *tuple(b for (i, b) in enumerate(states["basis"][0]) if i in axes)
    )

    vector = get_data_in_axes(
        states["data"].reshape(*states["basis"][0].shape, -1),
        (*axes, ndim),
        idx,
    ).reshape(-1)
    return {
        "basis": TupleBasis(final_basis, states["basis"][1]),
        "data": vector,
    }
