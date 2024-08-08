from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
)
from surface_potential_analysis.state_vector.state_vector import StateVector
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

if TYPE_CHECKING:
    import numpy as np

    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
    )

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)
_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


class ValueList(TypedDict, Generic[_B0_co]):
    """Represents some data listed over some basis."""

    basis: _B0_co
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class StatisticalValueList(ValueList[_B0_co]):
    """Represents some data listed over some basis."""

    standard_deviation: np.ndarray[tuple[int], np.dtype[np.float64]]


class Eigenstate(StateVector[_B0_co], TypedDict):
    """A State vector which is the eigenvector of some operator."""

    eigenvalue: complex | np.complex128


class EigenstateList(
    StateVectorList[_B0_co, _B1_co],
    TypedDict,
):
    """Represents a collection of eigenstates, each with the same basis."""

    eigenvalue: np.ndarray[tuple[int], np.dtype[np.complex128]]


def get_eigenvalues_list(
    states: EigenstateList[_B0, Any],
) -> SingleBasisDiagonalOperator[_B0]:
    """
    Extract eigenvalues from an eigenstate list.

    Parameters
    ----------
    states : EigenstateList[_B0, Any]

    Returns
    -------
    EigenvalueList[_B0]
    """
    return {
        "basis": TupleBasis(states["basis"][0], states["basis"][0]),
        "data": states["eigenvalue"],
    }
