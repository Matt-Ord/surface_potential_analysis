from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.block_fraction_basis import (
    BasisWithBlockFractionLike,
    ExplicitBlockFractionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.state_vector.state_vector import StateVector
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

from .eigenstate_calculation import calculate_eigenvectors_hermitian

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
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


_SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])
_BF0 = TypeVar("_BF0", bound=BasisWithBlockFractionLike[Any, Any])
EigenstateColllection = EigenstateList[_SB0, _B0]


def calculate_eigenstate_collection(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_L1], np.dtype[np.float64]]],
        SingleBasisOperator[_B0],
    ],
    bloch_fractions: np.ndarray[tuple[_L1, _L0], np.dtype[np.float64]],
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> EigenstateColllection[
    TupleBasisLike[ExplicitBlockFractionBasis[_L0], FundamentalBasis[int]], _B0
]:
    """
    Calculate an eigenstate collection with the given bloch phases.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_B3d0Inv]]
        Function used to generate the hamiltonian
    bloch_fractions : np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
        List of bloch phases
    subset_by_index : tuple[int, int] | None, optional
        subset_by_index, by default (0,0)

    Returns
    -------
    EigenstateColllection[_B3d0Inv]
    """
    subset_by_index = (0, 0) if subset_by_index is None else subset_by_index
    n_states = 1 + subset_by_index[1] - subset_by_index[0]

    basis = hamiltonian_generator(bloch_fractions[:, 0])["basis"][0]

    vectors = np.zeros(
        (bloch_fractions.shape[1], n_states * basis.n), dtype=np.complex128
    )
    eigenvalues = np.zeros((bloch_fractions.shape[1], n_states), dtype=np.complex128)

    for idx, bloch_fraction in enumerate(bloch_fractions.T):
        h = hamiltonian_generator(bloch_fraction)
        eigenstates = calculate_eigenvectors_hermitian(
            h, subset_by_index=subset_by_index
        )

        vectors[idx] = eigenstates["data"]
        eigenvalues[idx] = eigenstates["eigenvalue"]

    return {
        "basis": TupleBasis(
            TupleBasis(
                ExplicitBlockFractionBasis[_L0](bloch_fractions),
                FundamentalBasis(n_states),
            ),
            basis,
        ),
        "data": vectors.reshape(-1),
        "eigenvalue": eigenvalues.reshape(-1),
    }


def select_eigenstate(
    collection: EigenstateColllection[
        TupleBasisLike[_BF0, _B0],
        _B0_co,
    ],
    bloch_idx: int,
    band_idx: int,
) -> Eigenstate[_B0_co]:
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
    return {
        "basis": collection["basis"][1],
        "data": collection["data"].reshape(*collection["basis"][0].shape, -1)[
            bloch_idx, band_idx
        ],
        "eigenvalue": collection["eigenvalue"].reshape(
            *collection["basis"][0].shape, -1
        )[bloch_idx, band_idx],
    }


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
