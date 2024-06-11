from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import qutip
import qutip.ui
import scipy.sparse
from scipy.constants import hbar

from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
    _AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def get_state_vector_decomposition(
    initial_state: StateVector[_B0Inv],
    eigenstates: StateVectorList[_B1Inv, _B0Inv],
) -> SingleBasisDiagonalOperator[_B1Inv]:
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
        A list of coefficients for each vector such that a[i] eigenstates["data"][i,:] = vector[:]
    """
    return {
        "basis": TupleBasis(eigenstates["basis"][0], eigenstates["basis"][0]),
        "data": np.tensordot(
            np.conj(eigenstates["data"]).reshape(eigenstates["basis"].shape),
            initial_state["data"],
            axes=(1, 0),
        ).reshape(-1),
    }
    # eigenstates["data"] is the matrix such that the ith vector is
    # eigenstates["data"][i,:].
    # linalg.solve(a, b) = x where np.dot(a, x) == b, which is the sum
    # of the product over the last axis of x, so a[i] x[:, i] = b[:]
    # ie solved is the decomposition of b into the eigenvectors
    return scipy.linalg.solve(
        eigenstates["data"].reshape(eigenstates["basis"].shape).T, initial_state["data"]
    )  # type: ignore[no-any-return]


def solve_schrodinger_equation_decomposition(
    initial_state: StateVector[_B0Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B0Inv],
) -> StateVectorList[_AX0Inv, _B0Inv]:
    """
    Given an initial state, use the stochastic schrodinger equation to solve the dynamics of the system.

    Parameters
    ----------
    initial_state : StateVector[_B0Inv]
    times : np.ndarray[tuple[int], np.dtype[np.float_]]
    hamiltonian : SingleBasisOperator[_B0Inv]
    collapse_operators : list[SingleBasisOperator[_B0Inv]]

    Returns
    -------
    StateVectorList[_B0Inv, _L0Inv]
    """
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)
    np.testing.assert_array_almost_equal(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape),
        np.conj(hamiltonian["data"].reshape(hamiltonian["basis"].shape)).T,
    )
    coefficients = get_state_vector_decomposition(initial_state, eigenstates)
    np.testing.assert_array_almost_equal(
        np.tensordot(
            coefficients["data"],
            eigenstates["data"].reshape(eigenstates["basis"].shape),
            axes=(0, 0),
        ),
        initial_state["data"],
    )
    constants = coefficients["data"][np.newaxis, :] * np.exp(
        -1j
        * eigenstates["eigenvalue"][np.newaxis, :]
        * times.times[:, np.newaxis]
        / hbar
    )
    vectors = np.tensordot(
        constants, eigenstates["data"].reshape(eigenstates["basis"].shape), axes=(1, 0)
    )
    return {"basis": TupleBasis(times, eigenstates["basis"][1]), "data": vectors}


def solve_schrodinger_equation_diagonal(
    initial_state: StateVector[_B0Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisDiagonalOperator[_B0Inv],
) -> StateVectorList[_AX0Inv, _B0Inv]:
    """
    Given an initial state, use the schrodinger equation to solve the dynamics of the system.

    Parameters
    ----------
    initial_state : StateVector[_B0Inv]
    times : np.ndarray[tuple[int], np.dtype[np.float_]]
    hamiltonian : SingleBasisOperator[_B0Inv]
    collapse_operators : list[SingleBasisOperator[_B0Inv]]

    Returns
    -------
    StateVectorList[_B0Inv, _L0Inv]
    """
    data = initial_state["data"][np.newaxis, :] * np.exp(
        -1j * (hamiltonian["data"][np.newaxis, :]) * times.times[:, np.newaxis] / hbar
    )
    return {"basis": TupleBasis(times, hamiltonian["basis"][0]), "data": data}


def solve_schrodinger_equation(
    initial_state: StateVector[_B0Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B0Inv],
) -> StateVectorList[_AX0Inv, _B0Inv]:
    hamiltonian_qobj = qutip.Qobj(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape) / hbar,
    )
    initial_state_qobj = qutip.Qobj(initial_state["data"])
    result = qutip.sesolve(
        hamiltonian_qobj,
        initial_state_qobj,
        times.times,
        e_ops=[],
        options={
            "progress_bar": "enhanced",
            "store_states": True,
            "nsteps": times.step,
        },
    )
    return {
        "basis": TupleBasis(TupleBasis(times), hamiltonian["basis"][0]),
        "data": np.array(
            np.asarray([state.full().reshape(-1) for state in result.states]),
            dtype=np.complex128,
        ).reshape(-1),
    }
