from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import MomentumAxis
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalMomentumAxis,
        FundamentalPositionAxis,
    )
    from surface_potential_analysis.basis.basis import (
        Basis,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateVector,
    )

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


@timed
def convert_state_vector_to_basis(
    state_vector: StateVector[_B0Inv], basis: _B1Inv
) -> StateVector[_B1Inv]:
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
    converted = convert_vector(state_vector["vector"], state_vector["basis"], basis)
    return {"basis": basis, "vector": converted}  # type: ignore[typeddict-item]


def convert_state_vector_to_position_basis(
    state_vector: StateVector[_B0Inv],
) -> StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]
    """
    return convert_state_vector_to_basis(
        state_vector,
        basis_as_fundamental_position_basis(state_vector["basis"]),
    )


def convert_state_vector_to_momentum_basis(
    state_vector: StateVector[_B0Inv],
) -> StateVector[tuple[FundamentalMomentumAxis[Any, Any], ...]]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalMomentumAxis[Any, Any], ...]]
    """
    return convert_state_vector_to_basis(
        state_vector,
        basis_as_fundamental_momentum_basis(state_vector["basis"]),
    )


def interpolate_state_vector_momentum(
    state_vector: StateVector[_B0Inv], shape: _S0Inv
) -> StateVector[tuple[MomentumAxis[Any, Any, Any], ...]]:
    """
    Given a state vector, get the equivalent vector in as a truncated vector in a larger basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    shape : _S0Inv
        Final fundamental shape of the basis

    Returns
    -------
    StateVector[tuple[MomentumAxis[Any, Any, Any], ...]]
    """
    converted = convert_state_vector_to_momentum_basis(state_vector)
    util = BasisUtil(converted["basis"])
    final_basis = tuple(
        MomentumAxis(parent.delta_x, parent.n, n)
        for (parent, n) in zip(converted["basis"], shape, strict=True)
    )
    scaled = converted["vector"] * np.sqrt(np.prod(shape) / util.size)
    return {"basis": final_basis, "vector": scaled}