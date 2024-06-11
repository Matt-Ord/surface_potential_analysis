from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
from sse_solver_py import solve_sse_euler_bra_ket

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.state_vector.state_vector_list import (
    get_basis_states,
    get_state_dual_vector,
    get_state_vector,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.explicit_basis import ExplicitBasis
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseOperator,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisDiagonalNoiseOperator[ExplicitBasis[Any, Any]]]
    | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]:
    ...


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisDiagonalNoiseOperator[ExplicitBasis[Any, Any]]]
    | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]:
    ...


def solve_stochastic_schrodinger_equation(  # type: ignore bad overload
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisDiagonalNoiseOperator[ExplicitBasis[Any, Any]]]
    | None = None,
    *,
    n_trajectories: int = 1,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Any], _AX0Inv], _B1Inv]:
    assert times.offset == 0

    data = np.zeros(
        (n_trajectories, times.n, initial_state["data"].size), dtype=np.complex128
    )
    collapse_operators = [] if collapse_operators is None else collapse_operators
    amplitudes = list[complex]()
    bra = list[complex]()
    ket = list[complex]()
    for operator in collapse_operators:
        # TODO: is it correct to pass bra as a dual_vector..
        states_bra = get_basis_states(operator["basis"][0])
        assert states_bra["basis"][0].n == 1
        state_bra = get_state_dual_vector(states_bra, 0)
        ket.extend(state_bra["data"])

        states_ket = get_basis_states(operator["basis"][0])
        assert states_ket["basis"][0].n == 1
        state_ket = get_state_vector(states_ket, 0)
        ket.extend(state_ket["data"])

        amplitudes.append(operator["data"].item())

    for i in range(n_trajectories):
        out = solve_sse_euler_bra_ket(
            list(initial_state["data"]),
            list(hamiltonian["data"]),
            amplitudes,
            bra,
            ket,
            times.n,
            times.step,
            times.fundamental_dt,
        )
        data[i] = np.array(out).reshape(times.n, -1)

    return {
        "data": data.reshape(-1),
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(n_trajectories), times),
            initial_state["basis"],
        ),
    }
