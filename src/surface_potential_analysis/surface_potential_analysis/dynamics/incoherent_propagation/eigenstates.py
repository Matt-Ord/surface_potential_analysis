from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy

from .tunnelling_matrix import get_initial_pure_density_matrix_for_basis

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import DiagonalOperator
    from surface_potential_analysis.operator.operator_list import DiagonalOperatorList
    from surface_potential_analysis.state_vector.eigenstate_calculation import (
        EigenvectorList,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    from .tunnelling_basis import TunnellingSimulationBasis
    from .tunnelling_matrix import TunnellingMMatrix

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])


def calculate_tunnelling_eigenstates(
    matrix: TunnellingMMatrix[_B0Inv],
) -> EigenvectorList[_B0Inv, int]:
    """
    Given a tunnelling matrix, find the eigenstates.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_S0Inv]

    Returns
    -------
    TunnellingEigenstates[_S0Inv]
    """
    eigenvalues, vectors = scipy.linalg.eig(matrix["array"])
    return {
        "basis": matrix["basis"],
        "eigenvalues": eigenvalues - np.max(eigenvalues),
        "vectors": vectors.T,
    }


def get_operator_state_vector_decomposition(
    density_matrix: DiagonalOperator[_B0Inv, _B0Inv],
    eigenstates: StateVectorList[_B0Inv, _L0Inv],
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]:
    """
    Given a state and a set of TunnellingEigenstates decompose the state into the eigenstates.

    Parameters
    ----------
    state : TunnellingVector[_S0Inv]
        state to decompose
    eigenstates : TunnellingEigenstates[_S0Inv]
        set of eigenstates to decompose into

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
        A list of coefficients for each vector such that a[i] eigenstates["vectors"][i,:] = vector[:]
    """
    # eigenstates["vectors"] is the matrix such that the ith vector is
    # eigenstates["vectors"][i,:].
    # linalg.solve(a, b) = x where np.dot(a, x) == b, which is the sum
    # of the product over the last axis of x, so a[i] x[:, i] = b[:]
    # ie solved is the decomposition of b into the eigenvectors
    return scipy.linalg.solve(eigenstates["vectors"].T, density_matrix["vector"])  # type: ignore[no-any-return]


def get_equilibrium_state(
    eigenstates: EigenvectorList[_B0Inv, int],
) -> DiagonalOperator[_B0Inv, _B0Inv]:
    """
    Select the equilibrium tunnelling state from a list of eigenstates.

    Since all of the eigenstates have E < 0 except for the equilibrium
    this corresponds to the single "zero energy" state

    Parameters
    ----------
    eigenstates : TunnellingEigenstates[_S0Inv]

    Returns
    -------
    TunnellingVector[_S0Inv]
    """
    # We assume the surface is 'well connected', ie all initial states
    # end up in the equilibrium configuration
    initial = get_initial_pure_density_matrix_for_basis(eigenstates["basis"])
    coefficients = get_operator_state_vector_decomposition(initial, eigenstates)

    state_idx = np.argmax(eigenstates["eigenvalues"])
    # eigenstates["vectors"][state_idx] is not necessarily normalized,
    # and could contain negative or imaginary 'probabilities'
    vector = coefficients[state_idx] * eigenstates["vectors"][state_idx]
    return {
        "basis": eigenstates["basis"],
        "dual_basis": eigenstates["basis"],
        "vector": vector,
    }


def calculate_equilibrium_state(
    matrix: TunnellingMMatrix[_B0Inv],
) -> DiagonalOperator[_B0Inv, _B0Inv]:
    """
    Calculate the equilibrium tunnelling state for a given matrix.

    Since all of the eigenstates have E < 0 except for the equilibrium
    this corresponds to the single "zero energy" state

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]

    Returns
    -------
    TunnellingVector[_S0Inv]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    return get_equilibrium_state(eigenstates)


def get_tunnelling_simulation_state(
    eigenstates: EigenvectorList[_B0Inv, int],
    initial: DiagonalOperator[_B0Inv, _B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
) -> DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv]:
    """
    Get the StateVector given an initial state, and a list of eigenstates.

    Parameters
    ----------
    eigenstates : TunnellingEigenstates[_S0Inv]
        The eigenstates of the system
    initial_state : TunnellingVector[_S0Inv]
        The initial tunnelling state
    times : np.ndarray[tuple[_L1Inv], np.dtype[np.float_]]
        Times to calculate the occupation

    Returns
    -------
    StateVectorList[_B0Inv, _L1Inv]
    """
    coefficients = get_operator_state_vector_decomposition(initial, eigenstates)
    constants = coefficients[np.newaxis, :] * np.exp(
        eigenstates["eigenvalues"][np.newaxis, :] * times[:, np.newaxis]
    )
    vectors = np.tensordot(constants, eigenstates["vectors"], axes=(1, 0))
    return {
        "basis": eigenstates["basis"],
        "dual_basis": eigenstates["basis"],
        "vectors": vectors,
    }


def calculate_tunnelling_simulation_state(
    matrix: TunnellingMMatrix[_B0Inv],
    initial: DiagonalOperator[_B0Inv, _B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
) -> DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv]:
    """
    Get the StateVector given an initial state, and a tunnelling matrix.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
        _description_
    initial_state : StateVector[_B0Inv]
        _description_
    times : np.ndarray[tuple[_L1Inv], np.dtype[np.float_]]
        _description_

    Returns
    -------
    StateVectorList[_B0Inv, _L1Inv]
        _description_
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    return get_tunnelling_simulation_state(eigenstates, initial, times)