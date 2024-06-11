from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.basis_like import (
    BasisLike,
    BasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVector,
        ProbabilityVectorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


def convert_probability_vector_to_basis(
    probability_vector: ProbabilityVector[_B0], basis: _B1
) -> ProbabilityVector[_B1]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    probability_vector : StateVector[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    StateVector[_B1Inv]
    """
    converted = convert_vector(
        probability_vector["data"], probability_vector["basis"], basis
    )
    return {"basis": basis, "data": converted}  # type: ignore[typeddict-item]


def convert_probability_vector_list_to_basis(
    probability_vector: ProbabilityVectorList[_B0, _B1], basis: _B2
) -> ProbabilityVectorList[_B0, _B2]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    probability_vector : StateVector[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    StateVector[_B1Inv]
    """
    stacked = probability_vector["data"].reshape(probability_vector["basis"].shape)
    converted = convert_vector(stacked, probability_vector["basis"][1], basis).reshape(
        -1
    )
    return {
        "basis": TupleBasis(probability_vector["basis"][0], basis),
        "data": converted,
    }


def convert_probability_vector_to_position_basis(
    probability_vector: ProbabilityVector[TupleBasisLike[*tuple[_BL0, ...]]],
) -> ProbabilityVector[TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    probability_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis[Any, Any], ...]]
    """
    return convert_probability_vector_to_basis(
        probability_vector,
        stacked_basis_as_fundamental_position_basis(probability_vector["basis"]),
    )


def convert_probability_vector_to_momentum_basis(
    probability_vector: ProbabilityVector[TupleBasisLike[*tuple[_BL0, ...]]],
) -> ProbabilityVector[
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    probability_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalMomentumBasis[Any, Any], ...]]
    """
    return convert_probability_vector_to_basis(
        probability_vector,
        stacked_basis_as_fundamental_momentum_basis(probability_vector["basis"]),
    )
