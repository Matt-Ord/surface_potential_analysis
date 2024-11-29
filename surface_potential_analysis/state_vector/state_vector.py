from __future__ import annotations

from typing import Any, Generic, Self, TypedDict, TypeVar, override

import numpy as np
from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis._basis import Basis
from slate.metadata._metadata import BasisMetadata

from surface_potential_analysis.basis.legacy import BasisLike

_B0Inv = TypeVar("_B0Inv", bound=BasisLike)

_B0_co = TypeVar("_B0_co", bound=BasisLike, covariant=True)


class StateVector[B: Basis[Any, np.complex128]](SlateArray[np.complex128, B]):
    """represents a state vector in a basis."""

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self: Self, basis: B1
    ) -> StateVector[B1]:
        """Get the Operator with the basis set to basis."""
        return StateVector(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __add__[_DT: np.number[Any], M: BasisMetadata](
        self: StateVector[Basis[M, np.complex128]],
        other: StateVector[Basis[M, np.complex128]],
    ) -> StateVector[Basis[M, np.complex128]]:
        data = self.raw_data + other.with_basis(self.basis).raw_data
        return StateVector(self.basis, data)


class LegacyStateVector(TypedDict, Generic[_B0_co]):
    """represents a state vector in a basis."""

    basis: _B0_co
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class LegacyStateDualVector(TypedDict, Generic[_B0_co]):
    """represents a dual vector in a basis."""

    basis: _B0_co
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


def as_legacy_vector(
    vector: LegacyStateDualVector[_B0Inv],
) -> LegacyStateVector[_B0Inv]:
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


def as_legacy_dual_vector(
    vector: LegacyStateVector[_B0Inv],
) -> LegacyStateDualVector[_B0Inv]:
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


def legacy_calculate_normalization(
    state: LegacyStateVector[Any] | LegacyStateDualVector[Any],
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


def legacy_calculate_inner_product(
    state_0: LegacyStateVector[_B0Inv],
    state_1: LegacyStateDualVector[_B0Inv],
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


def calculate_normalization[M: BasisMetadata](
    state: StateVector[Basis[M, np.complex128]],
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
    return np.sum(np.abs(state.raw_data) ** 2).astype(np.float64)


def calculate_inner_product[M: BasisMetadata](
    state_0: StateVector[Basis[M, np.complex128]],
    state_1: StateVector[Basis[M, np.complex128]],
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
    return np.tensordot(
        convert_array(state_1, state_0.basis.conjugate_basis()).raw_data,
        state_0.raw_data,
        axes=(0, 0),
    ).item(0)
