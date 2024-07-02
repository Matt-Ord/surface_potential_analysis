from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import qutip
import qutip.ui
import scipy.sparse
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.tunnelling_basis import (
    get_basis_from_shape,
)
from surface_potential_analysis.dynamics.util import build_hop_operator, get_hop_shift
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products,
    get_state_along_axis,
    get_state_vector,
    get_weighted_state_vector,
)

try:
    from sse_solver_py import SimulationConfig, SSEMethod, solve_sse, solve_sse_banded
except ImportError:
    # if sse_solver_py is not installed, create a dummy functions

    def solve_sse(  # noqa: D103
        initial_state: list[complex],  # noqa: ARG001
        hamiltonian: list[list[complex]],  # noqa: ARG001
        operators: list[list[list[complex]]],  # noqa: ARG001
        config: SimulationConfig,  # noqa: ARG001
    ) -> list[complex]:
        msg = "sse_solver_py not installed, add sse_solver_py feature"
        raise NotImplementedError(msg)

    def solve_sse_banded(  # noqa: PLR0913, PLR0917, D103
        initial_state: list[complex],  # noqa: ARG001
        hamiltonian_diagonal: list[list[complex]],  # noqa: ARG001
        hamiltonian_offset: list[int],  # noqa: ARG001
        operators_diagonals: list[list[list[complex]]],  # noqa: ARG001
        operators_offsets: list[list[int]],  # noqa: ARG001
        config: SimulationConfig,  # noqa: ARG001
    ) -> list[complex]:
        msg = "sse_solver_py not installed, add sse_solver_py feature"
        raise NotImplementedError(msg)


if TYPE_CHECKING:
    from collections.abc import Callable

    from sse_solver_py import SSEMethod

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
        TunnellingAMatrix,
    )
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBandsBasis,
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0 = TypeVar("_B0", bound=TunnellingSimulationBasis[Any, Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _B3 = TypeVar("_B3", bound=BasisLike[Any, Any])
    _B4 = TypeVar("_B4", bound=BasisLike[Any, Any])
    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def get_collapse_operators_from_a_matrix(
    matrix: TunnellingAMatrix[_B0],
) -> list[SingleBasisOperator[_B0]]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]]]
    """
    data = matrix["data"].reshape(matrix["basis"].shape)
    np.fill_diagonal(data, 0)
    return [
        {
            "basis": matrix["basis"],
            "data": np.array(
                scipy.sparse.coo_array(
                    ([data[idx]], ([np.int32(idx[0])], [np.int32(idx[1])])),  # type: ignore always idx[1]
                    shape=data.shape,
                ).toarray(),
                dtype=np.float64,
            ),
        }
        for idx in zip(*np.nonzero(data), strict=True)
    ]


def get_simplified_collapse_operators_from_a_matrix(
    matrix: TunnellingAMatrix[_B0], *, factor: float = 1
) -> list[SingleBasisOperator[_B0]]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]]]
    """
    util = BasisUtil(matrix["basis"][0])
    (n_x1, n_x2, n_bands) = util.shape
    jump_array = matrix["data"].reshape(*util.shape, *util.shape)[0, 0]
    out: list[SingleBasisOperator[_B0]] = []
    for n_0 in range(n_bands):
        for n_1 in range(n_bands):
            if n_0 == n_1:
                continue
            for hop in range(9):
                hop_shift = get_hop_shift(hop, 2)
                hop_val = factor * jump_array[n_0, hop_shift[0], hop_shift[1], n_1]
                if hop_val < 1:
                    continue

                operator = np.sqrt(hop_val) * build_hop_operator(hop, (n_x1, n_x2))
                array = np.zeros((*util.shape, *util.shape), dtype=np.complex128)
                array[:, :, n_1, :, :, n_0] = operator
                out.append({"basis": matrix["basis"], "data": array.reshape(-1)})

    return out


def get_collapse_operators_from_function(
    shape: tuple[_L0Inv, _L1Inv],
    bands_basis: TunnellingSimulationBandsBasis[_L2Inv],
    a_function: Callable[
        [
            int,
            int,
            tuple[int, int],
            tuple[int, int],
        ],
        float,
    ],
) -> list[
    SingleBasisOperator[
        TupleBasisLike[
            FundamentalBasis[_L0Inv],
            FundamentalBasis[_L1Inv],
            TunnellingSimulationBandsBasis[_L2Inv],
        ]
    ]
]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]]]
    """
    operators: list[
        SingleBasisOperator[
            TupleBasisLike[
                FundamentalBasis[_L0Inv],
                FundamentalBasis[_L1Inv],
                TunnellingSimulationBandsBasis[_L2Inv],
            ]
        ]
    ] = []
    n_sites = np.prod(shape).item()
    n_bands = bands_basis.fundamental_n
    basis = get_basis_from_shape(shape, n_bands, bands_basis)

    array = np.zeros((n_sites * n_bands, n_sites * n_bands))
    for i in range(array.shape[0]):
        for n1 in range(n_bands):
            for d1 in range(9):
                (i0, j0, n0) = np.unravel_index(i, (*shape, n_bands))
                d1_stacked = np.unravel_index(d1, (3, 3)) - np.array([1, 1])
                (i1, j1) = (i0 + d1_stacked[0], j0 + d1_stacked[1])
                j = np.ravel_multi_index((i1, j1, n1), (*shape, n_bands), mode="wrap")

                data = np.zeros(shape, dtype=np.complex128)
                data[i, j] = a_function(
                    int(n0), n1, (0, 0), (d1_stacked[0], d1_stacked[1])
                )
                # TODO: use a single point basis, and a normal np.array...
                operators.append(
                    {"basis": TupleBasis(basis, basis), "data": data.reshape(-1)}
                )
    return operators


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1]:
    ...


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1]:
    ...


def solve_stochastic_schrodinger_equation(  # type: ignore bad overload
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
) -> (
    StateVectorList[TupleBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1]
    | StateVectorList[TupleBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1]
):
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
    if collapse_operators is None:
        collapse_operators = []
    hamiltonian_qobj = qutip.Qobj(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape) / hbar
    ).to("CSR")
    initial_state_qobj = qutip.Qobj(initial_state["data"])

    sc_ops = [
        qutip.Qobj(op["data"].reshape(op["basis"].shape) / hbar).to("CSR")
        for op in collapse_operators
    ]
    result = qutip.ssesolve(
        hamiltonian_qobj,
        initial_state_qobj,
        times.times,
        sc_ops=sc_ops,
        e_ops=[],
        options={
            # No other scheme scales well enough to such a large number of heatbath modes
            # "method": "euler",
            "progress_bar": "enhanced",
            "store_states": True,
            "keep_runs_results": True,
            "map": "parallel",
            "num_cpus": n_trajectories,
            "dt": times.fundamental_dt,
        },
        ntraj=n_trajectories,  # cspell:disable-line
    )
    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": np.array(
            [
                np.asarray([state.full().reshape(-1) for state in trajectory])  # type: ignore unknown
                for trajectory in result.states  # type: ignore unknown
            ],
            dtype=np.complex128,
        ).reshape(-1),
    }


@overload
def solve_stochastic_schrodinger_equation_rust(
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1]:
    ...


@overload
def solve_stochastic_schrodinger_equation_rust(
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1]:
    ...


def solve_stochastic_schrodinger_equation_rust(  # type: ignore bad overload
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Any], _AX0Inv], _B1]:
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
    collapse_operators = [] if collapse_operators is None else collapse_operators

    data = solve_sse(
        list(initial_state["data"]),
        [
            list(x / hbar)
            for x in hamiltonian["data"].reshape(hamiltonian["basis"].shape)
        ],
        [
            [list(x / np.sqrt(hbar)) for x in o["data"].reshape(o["basis"].shape)]
            for o in collapse_operators
        ],
        SimulationConfig(
            n=times.n,
            step=times.step,
            dt=times.dt,
            n_trajectories=n_trajectories,
            method="Euler",
        ),
    )

    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": np.array(data).ravel(),
    }


def _get_operator_diagonals(
    operator: list[list[complex]],
) -> list[list[complex]]:
    return [
        np.diag(np.roll(operator, shift=-i, axis=0)).tolist()
        for i in range(len(operator))
    ]


def _get_banded_operator(
    operator: list[list[complex]], threshold: float
) -> tuple[list[list[complex]], list[int]]:
    diagonals = _get_operator_diagonals(operator)
    above_threshold = np.linalg.norm(diagonals, axis=1) > threshold

    diagonals_filtered = np.array(diagonals)[above_threshold]

    zero_imag = np.abs(np.imag(diagonals_filtered)) < threshold
    diagonals_filtered[zero_imag] = np.real(diagonals_filtered[zero_imag])
    zero_real = np.abs(np.real(diagonals_filtered)) < threshold
    diagonals_filtered[zero_real] = 1j * np.imag(diagonals_filtered[zero_real])

    diagonals_filtered = diagonals_filtered.tolist()

    offsets = np.arange(len(operator))[above_threshold].tolist()
    return (diagonals_filtered, offsets)


def _get_banded_operators(
    operators: list[list[list[complex]]], threshold: float
) -> list[tuple[list[list[complex]], list[int]]]:
    return [_get_banded_operator(o, threshold) for o in operators]


@overload
def solve_stochastic_schrodinger_equation_rust_banded(
    initial_state: StateVector[_B2],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B3]] | None = None,
    *,
    n_trajectories: _L1Inv,
    r_threshold: float = 1e-8,
    method: SSEMethod = "Euler",
) -> StateVectorList[TupleBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1]:
    ...


@overload
def solve_stochastic_schrodinger_equation_rust_banded(
    initial_state: StateVector[_B2],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B3]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
    r_threshold: float = 1e-8,
    method: SSEMethod = "Euler",
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1]:
    ...


def solve_stochastic_schrodinger_equation_rust_banded(  # type: ignore bad overload
    initial_state: StateVector[_B2],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[Operator[_B3, _B4]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
    r_threshold: float = 1e-8,
    method: SSEMethod = "Euler",
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Any], _AX0Inv], _B1]:
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
    data = np.zeros(
        (n_trajectories, times.n, initial_state["data"].size), dtype=np.complex128
    )
    collapse_operators = [] if collapse_operators is None else collapse_operators

    operators_data = [o["data"].reshape(o["basis"].shape) for o in collapse_operators]
    operators_norm = [np.linalg.norm(o) for o in operators_data]

    # We get the best numerical performace if we set the norm of the largest collapse operators
    # to be one. This prevents us from accumulating large errors when multiplying state * dt * operator * conj_operator
    max_norm = np.max(operators_norm)
    dt = (times.fundamental_dt * max_norm**2 / hbar).item()

    banded_collapse = _get_banded_operators(
        [
            [
                list(x / max_norm)
                for x in convert_operator_to_basis(o, hamiltonian["basis"])[
                    "data"
                ].reshape(hamiltonian["basis"].shape)
            ]
            for o in collapse_operators
        ],
        r_threshold / dt,
    )

    banded_h = _get_banded_operator(
        [
            list(x / max_norm**2)
            for x in hamiltonian["data"].reshape(hamiltonian["basis"].shape)
        ],
        r_threshold / dt,
    )
    initial_state_converted = convert_state_vector_to_basis(
        initial_state, hamiltonian["basis"][0]
    )
    data = solve_sse_banded(
        list(initial_state_converted["data"]),
        banded_h[0],
        banded_h[1],
        [b[0] for b in banded_collapse],
        [b[1] for b in banded_collapse],
        SimulationConfig(
            n=times.n,
            step=times.step,
            dt=dt,
            n_trajectories=n_trajectories,
            method=method,
        ),
    )

    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": np.array(data).ravel(),
    }


rng = np.random.default_rng()


def _select_random_localized_state(
    states: StateVectorList[_B2, _B1],
) -> StateVector[_B1]:
    """
    Select a random state built from states.

    This finds the states which diagonalize the overall density matrix,
    and selects states according to their overall occupation.

    Parameters
    ----------
    states : StateVectorList[_B2Inv, _B1Inv]

    Returns
    -------
    StateVector[_B1Inv]
    """
    states["data"].reshape(states["basis"].shape)

    op = calculate_inner_products(states, states)
    op["data"] /= np.linalg.norm(op["data"])
    # Vectors representing the combination of states that diagonalizes rho
    eigenstates = calculate_eigenvectors_hermitian(op)

    probabilities = eigenstates["eigenvalue"]
    probabilities /= np.sum(probabilities)

    idx = rng.choice(probabilities.size, p=probabilities)
    transformation = get_state_vector(eigenstates, idx)
    transformation["data"] /= np.sqrt(2)
    return get_weighted_state_vector(states, transformation)


@overload
def solve_stochastic_schrodinger_equation_localized(
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1]:
    ...


@overload
def solve_stochastic_schrodinger_equation_localized(
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1]:
    ...


def solve_stochastic_schrodinger_equation_localized(  # type: ignore bad overload
    initial_state: StateVector[_B1],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1],
    collapse_operators: list[SingleBasisOperator[_B1]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
    n_realizations: int = 2,
) -> StateVectorList[TupleBasisLike[FundamentalBasis[Any], _AX0Inv], _B1]:
    """
    Find the quantum trajectores, using the localized stochastic schrodinger approach.

    Returns
    -------
    StateVectorList[TupleBasisLike[FundamentalBasis[int], _AX0Inv], _B1Inv]
    """
    data = np.zeros((n_trajectories, times.n, initial_state["basis"].n), dtype=complex)

    for trajectory in range(n_trajectories):
        state = initial_state
        for t in range(times.n):
            result = solve_stochastic_schrodinger_equation(
                state,
                EvenlySpacedTimeBasis(2, times.step, 0, times.dt),
                hamiltonian,
                collapse_operators,
                n_trajectories=n_realizations,
            )
            # Re-localize our state
            state = _select_random_localized_state(
                get_state_along_axis(result, axes=(1,), idx=(1,))
            )

            data[trajectory, t] = state["data"]

    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": data.reshape(-1),
    }
