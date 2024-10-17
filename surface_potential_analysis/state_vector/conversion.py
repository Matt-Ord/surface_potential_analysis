from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    BasisWithLengthLike,
    StackedBasisWithVolumeLike,
    TransformedPositionBasis,
    TupleBasis,
    TupleBasisWithLengthLike,
    convert_dual_vector,
    convert_vector,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateDualVector,
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike)
    _B1 = TypeVar("_B1", bound=BasisLike)
    _B2 = TypeVar("_B2", bound=BasisLike)


def convert_state_vector_to_basis(
    state_vector: StateVector[_B0], basis: _B1
) -> StateVector[_B1]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    StateVector[_B1Inv]
    """
    converted = convert_vector(state_vector["data"], state_vector["basis"], basis)
    return {"basis": basis, "data": converted}  # type: ignore[typeddict-item]


def convert_state_vector_list_to_basis(
    state_vector: StateVectorList[_B0, _B1], basis: _B2
) -> StateVectorList[_B0, _B2]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    StateVector[_B1Inv]
    """
    stacked = state_vector["data"].reshape(state_vector["basis"].shape)
    converted = convert_vector(stacked, state_vector["basis"][1], basis).reshape(-1)
    return {
        "basis": VariadicTupleBasis((state_vector["basis"][0], basis), None),
        "data": converted,
    }


def convert_state_dual_vector_to_basis(
    state_vector: StateDualVector[_B0], basis: _B1
) -> StateDualVector[_B1]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    StateVector[_B1Inv]
    """
    converted = convert_dual_vector(state_vector["data"], state_vector["basis"], basis)
    return {"basis": basis, "data": converted}  # type: ignore[typeddict-item]


def convert_state_vector_to_position_basis(
    state_vector: StateVector[StackedBasisWithVolumeLike],
) -> StateVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis, ...]]
    """
    return convert_state_vector_to_basis(
        state_vector,
        tuple_basis_as_fundamental(state_vector["basis"]),
    )


def convert_state_vector_to_momentum_basis(
    state_vector: StateVector[StackedBasisWithVolumeLike],
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]
]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalMomentumBasis[Any, Any], ...]]
    """
    return convert_state_vector_to_basis(
        state_vector,
        stacked_basis_as_fundamental_momentum_basis(state_vector["basis"]),
    )


def convert_state_dual_vector_to_position_basis(
    state_vector: StateDualVector[StackedBasisWithVolumeLike],
) -> StateDualVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    state_vector : StateDualVector[_B0Inv]

    Returns
    -------
    StateDualVector[tuple[FundamentalPositionBasis, ...]]
    """
    return convert_state_dual_vector_to_basis(
        state_vector,
        tuple_basis_as_fundamental(state_vector["basis"]),
    )


def convert_state_dual_vector_to_momentum_basis(
    state_vector: StateDualVector[StackedBasisWithVolumeLike],
) -> StateDualVector[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]
]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateDualVector[_B0Inv]

    Returns
    -------
    StateDualVector[tuple[FundamentalMomentumBasis[Any, Any], ...]]
    """
    return convert_state_dual_vector_to_basis(
        state_vector,
        stacked_basis_as_fundamental_momentum_basis(state_vector["basis"]),
    )


def interpolate_state_vector_momentum(
    state_vector: StateVector[StackedBasisWithVolumeLike],
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> StateVector[TupleBasisWithLengthLike[*tuple[BasisWithLengthLike, ...]]]:
    """
    Given a state vector, get the equivalent vector in as a truncated vector in a larger basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    shape : _S0Inv
        Final fundamental shape of the basis

    Returns
    -------
    StateVector[tuple[MomentumBasis[Any, Any, Any], ...]]
    """
    converted_basis = stacked_basis_as_fundamental_momentum_basis(state_vector["basis"])
    converted = convert_state_vector_to_basis(state_vector, converted_basis)

    final_basis = TupleBasis[*tuple[BasisWithLengthLike, ...]](
        *tuple(
            TransformedPositionBasis(ax.delta_x, ax.n, shape[idx])
            if (
                idx := next((i for i, jax in enumerate(axes) if jax == iax), None)
                is not None
            )
            else ax
            for iax, ax in enumerate(converted["basis"])
        )
    )
    scaled = converted["data"] * np.sqrt(np.prod(shape) / converted_basis.n)
    return {"basis": final_basis, "data": scaled}
