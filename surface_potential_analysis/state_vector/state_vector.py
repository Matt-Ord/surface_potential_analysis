from __future__ import annotations

from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike

_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)


class StateVector(TypedDict, Generic[_B0_co]):
    """represents a state vector in a basis."""

    basis: _B0_co
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class StateDualVector(TypedDict, Generic[_B0_co]):
    """represents a dual vector in a basis."""

    basis: _B0_co
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


def as_vector(vector: StateDualVector[_B0Inv]) -> StateVector[_B0Inv]:
    """
    Convert a state dual vector into a state vector.

    Parameters
    ----------
    vector : StateDualVector[_B0Inv]

    Returns
    -------
    StateVector[_B0Inv]
    """
    return {"basis": vector["basis"], "data": np.conj(vector["data"])}


def as_dual_vector(vector: StateVector[_B0Inv]) -> StateDualVector[_B0Inv]:
    """
    Convert a state vector into a state dual vector.

    Parameters
    ----------
    vector : StateVector[_B0Inv]

    Returns
    -------
    StateDualVector[_B0Inv]
    """
    return {"basis": vector["basis"], "data": np.conj(vector["data"])}


def calculate_normalization(
    state: StateVector[Any] | StateDualVector[Any],
) -> np.float64:
    """
    calculate the normalization of a state.

    This should always be 1

    Parameters
    ----------
    state: StateVector[Any] | StateDualVector[Any]

    Returns
    -------
    float
    """
    return np.sum(np.abs(state["data"]) ** 2).astype(np.float64)


def calculate_inner_product(
    state_0: StateVector[_B0Inv],
    state_1: StateDualVector[_B0Inv],
) -> complex:
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
    return np.tensordot(state_1["data"], state_0["data"], axes=(0, 0)).item(0)
