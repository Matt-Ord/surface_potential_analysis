from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, cast, overload

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.util import get_data_in_axes

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])


class ProbabilityVector(TypedDict, Generic[_B0]):
    """represents a state vector in a basis."""

    basis: _B0
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class ProbabilityVectorList(TypedDict, Generic[_B0, _B1]):
    """represents a list of probabilities in a basis."""

    basis: TupleBasisLike[_B0, _B1]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


def from_state_vector(state: StateVector[_B0]) -> ProbabilityVector[_B0]:
    """
    Get a probability vector for a given state vector.

    Parameters
    ----------
    state : StateVector[_B0Inv]

    Returns
    -------
    ProbabilityVector[_B0Inv]
    """
    return {"basis": state["basis"], "data": np.square(np.abs(state["data"]))}


def from_state_vector_list(
    states: StateVectorList[_B0, _B1],
) -> ProbabilityVectorList[_B0, _B1]:
    """
    Get a probability vector list for a given state vector list.

    Parameters
    ----------
    states : StateVectorList[_B0Inv, _L0Inv]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    return {"basis": states["basis"], "data": np.square(np.abs(states["data"]))}


def get_probability(
    probabilities: ProbabilityVectorList[_SB0, _B1],
    idx: SingleIndexLike | None = None,
) -> ProbabilityVector[_B1]:
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
    idx = 0 if idx is None else idx
    util = BasisUtil(probabilities["basis"][0])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    data = probabilities["data"].reshape(probabilities["basis"].shape)[idx]
    return {"basis": probabilities["basis"][1], "data": data}


def get_probability_along_axis(
    probabilities: ProbabilityVectorList[_SB0, _B1],
    axes: tuple[int, ...] = (0,),
    idx: SingleStackedIndexLike | None = None,
) -> ProbabilityVectorList[Any, _B1]:
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
    ndim = probabilities["basis"][0].ndim
    idx = tuple(0 for _ in range(ndim - 2)) if idx is None else idx
    final_basis = TupleBasis(
        *tuple(b for (i, b) in enumerate(probabilities["basis"][0]) if i in axes)
    )

    vector = get_data_in_axes(
        probabilities["data"].reshape(probabilities["basis"].shape),
        (*axes, ndim),
        idx,
    ).reshape(*final_basis.shape, -1)
    return {
        "basis": TupleBasis(final_basis, probabilities["basis"][1]),
        "data": vector,
    }


def sum_probability(
    probability: ProbabilityVector[_SB0], axis: tuple[int, ...] | None
) -> ProbabilityVector[Any]:
    """
    Sum the probabilities over the given axis.

    Parameters
    ----------
    probability : ProbabilityVector[_B0Inv]
    axis : tuple[int, ...] | None

    Returns
    -------
    ProbabilityVector[Any]
    """
    axis = tuple(range(probability["basis"].ndim)) if axis is None else axis
    basis = tuple(b for (i, b) in enumerate(probability["basis"]) if i not in axis)
    return {
        "basis": basis,
        "data": (
            np.sum(
                probability["data"].reshape(probability["basis"].shape), axis=axis
            ).reshape(-1)
        ),
    }


def sum_probabilities(
    probabilities: ProbabilityVectorList[_B0, _SB0],
    axis: tuple[int, ...] | None = None,
) -> ProbabilityVectorList[_B0, Any]:
    """
    Sum the probabilities over the given axis.

    Parameters
    ----------
    probabilities : ProbabilityVectorList[_B0Inv, _L0Inv]
    axis : tuple[int, ...] | None

    Returns
    -------
    ProbabilityVectorList[Any, _L0Inv]
    """
    axis = tuple(range(probabilities["basis"].ndim)) if axis is None else axis
    basis = TupleBasis(
        *tuple(b for (i, b) in enumerate(probabilities["basis"][1]) if i not in axis)
    )
    return {
        "basis": TupleBasis(probabilities["basis"][0], basis),
        "data": np.sum(
            probabilities["data"].reshape(-1, *probabilities["basis"][1].shape),
            axis=tuple(ax + 1 for ax in axis),
        ).reshape(-1),
    }


@overload
def average_probabilities(
    probabilities: ProbabilityVectorList[_SB0, _B1],
    axis: tuple[int, ...],
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> ProbabilityVectorList[Any, _B1]:
    ...


@overload
def average_probabilities(
    probabilities: ProbabilityVectorList[_B0, _B1],
    axis: None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> ProbabilityVector[_B1]:
    ...


def average_probabilities(
    probabilities: ProbabilityVectorList[_B0, _B1],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> ProbabilityVectorList[Any, _B1] | ProbabilityVector[_B1]:
    """
    Average probabilities over several repeats.

    Parameters
    ----------
    probabilities : list[ProbabilityVectorList[_B0Inv, _L0Inv]]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    if axis is None:
        if weights is not None:
            raise NotImplementedError
        return {
            "basis": probabilities["basis"][1],
            "data": np.average(
                probabilities["data"].reshape(*probabilities["basis"].shape),
                axis=0,
            ).reshape(-1),
        }

    old_basis = cast(TupleBasisLike[*tuple[Any, ...]], probabilities["basis"][0])
    basis = TupleBasis(*tuple(b for (i, b) in enumerate(old_basis) if i not in axis))
    return {
        "basis": TupleBasis(basis, probabilities["basis"][1]),
        "data": np.average(
            probabilities["data"].reshape(*old_basis.shape, -1),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }
