from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.time_basis_like import (
    ExplicitTimeBasis,
    FundamentalTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_tunnelling_eigenstates,
    calculate_tunnelling_simulation_state,
    get_equilibrium_state,
    get_operator_state_vector_decomposition,
    get_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    density_matrix_as_probability,
    density_matrix_list_as_probabilities,
    get_initial_pure_density_matrix_for_basis,
)
from surface_potential_analysis.dynamics.isf import calculate_isf_approximate_locations
from surface_potential_analysis.operator.operator import average_eigenvalues
from surface_potential_analysis.probability_vector.probability_vector import (
    ProbabilityVector,
    ProbabilityVectorList,
    average_probabilities,
    sum_probability,
)

if TYPE_CHECKING:
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
        TunnellingMMatrix,
    )
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import (
        DiagonalOperator,
        SingleBasisDiagonalOperator,
    )

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])


_L0Inv = TypeVar("_L0Inv", bound=int)


def calculate_isf_at_times(
    matrix: TunnellingMMatrix[_B0Inv],
    initial: DiagonalOperator[_B0Inv, _B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float64]],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
) -> SingleBasisDiagonalOperator[ExplicitTimeBasis[_L0Inv]]:
    """
    Calculate the ISF, assuming all states are approximately eigenstates of position.

    Parameters
    ----------
    initial_matrix : DiagonalOperator[_B0Inv, _B0Inv]
        Initial density matrix
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        direction along which to measure the ISF

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    final = calculate_tunnelling_simulation_state(matrix, initial, times)
    initial_occupation = density_matrix_as_probability(initial)
    final_occupation = density_matrix_list_as_probabilities(final)
    return calculate_isf_approximate_locations(initial_occupation, final_occupation, dk)


def calculate_equilibrium_state_averaged_isf(
    matrix: TunnellingMMatrix[_B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float64]],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
) -> SingleBasisDiagonalOperator[FundamentalTimeBasis[_L0Inv]]:
    """
    Calculate the ISF, averaging over the equilibrium occupation of each band.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    dk : np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    util = BasisUtil(matrix["basis"])
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    equilibrium = get_equilibrium_state(eigenstates)

    occupation_probabilities = sum_probability(
        density_matrix_as_probability(equilibrium), (0, 1)
    )
    eigenvalues = np.zeros((util.shape[2], times.size), dtype=np.complex128)
    for band in range(util.shape[2]):
        initial = get_initial_pure_density_matrix_for_basis(
            matrix["basis"][0], (0, 0, band)
        )
        initial_probability = density_matrix_as_probability(initial)
        final = get_tunnelling_simulation_state(eigenstates, initial, times)
        final_probabilities = density_matrix_list_as_probabilities(final)
        isf = calculate_isf_approximate_locations(
            initial_probability, final_probabilities, dk
        )
        eigenvalues[band] = isf
    isf_per_band: SingleBasisDiagonalOperator[
        TupleBasis[FundamentalBasis[int], ExplicitTimeBasis[_L0Inv]]
    ] = {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(util.shape[2]), ExplicitTimeBasis(times)),
            TupleBasis(FundamentalBasis(util.shape[2]), ExplicitTimeBasis(times)),
        ),
        "data": eigenvalues.reshape(-1),
    }
    averaged = average_eigenvalues(
        isf_per_band, (0,), weights=np.abs(occupation_probabilities["data"])
    )
    return {"basis": averaged["basis"][0][0], "data": averaged["data"]}


def calculate_equilibrium_initial_state_isf(
    matrix: TunnellingMMatrix[_B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float64]],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
) -> SingleBasisDiagonalOperator[FundamentalTimeBasis[_L0Inv]]:
    """
    Calculate the ISF, averaging over the equilibrium occupation of each band.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    dk : np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)

    vectors = np.zeros(
        (matrix["basis"].shape[2], times.size, matrix["basis"].n), dtype=np.complex128
    )
    for band in range(matrix["basis"].shape[2]):
        initial_state = get_initial_pure_density_matrix_for_basis(
            matrix["basis"][0], (0, 0, band)
        )
        final_state = get_tunnelling_simulation_state(eigenstates, initial_state, times)
        final_probabilities = density_matrix_list_as_probabilities(final_state)
        vectors[band] = final_probabilities["data"]
    probability_per_band: ProbabilityVectorList[
        TupleBasis[FundamentalBasis[int], ExplicitTimeBasis[_L0Inv]], _B0Inv
    ] = {
        "basis": TupleBasis(
            TupleBasis(
                FundamentalBasis(matrix["basis"].shape[2]), ExplicitTimeBasis(times)
            ),
            matrix["basis"][0],
        ),
        "data": vectors.reshape(-1),
    }

    equilibrium = get_equilibrium_state(eigenstates)
    occupation_probabilities = sum_probability(
        density_matrix_as_probability(equilibrium), (0, 1)
    )
    vector = np.zeros(matrix["basis"].shape, dtype=np.complex128)
    vector[0, 0, :] = occupation_probabilities["data"]
    initial: ProbabilityVector[_B0Inv] = {
        "basis": matrix["basis"][0],
        "data": vector.reshape(-1),
    }
    average_probability = average_probabilities(
        probability_per_band,
        weights=np.abs(occupation_probabilities["data"]),
        axis=(0,),
    )
    return calculate_isf_approximate_locations(initial, average_probability, dk)


@dataclass
class RateDecomposition(Generic[_L0Inv]):
    """Result of fitting a double exponential to an ISF."""

    eigenvalues: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]]
    coefficients: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]]


def get_rate_decomposition(
    matrix: TunnellingMMatrix[_B0Inv], initial: DiagonalOperator[_B0Inv, _B0Inv]
) -> RateDecomposition[int]:
    """
    Get the eigenvalues and relevant contribution of the rates in the simulation.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
    initial : DiagonalOperator[_B0Inv, _B0Inv]

    Returns
    -------
    RateDecomposition[int]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    coefficients = get_operator_state_vector_decomposition(initial, eigenstates)
    return RateDecomposition(eigenstates["data"], coefficients)
