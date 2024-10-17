from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    StackedBasisWithVolumeLike,
    TupleBasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVector,
        ProbabilityVectorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike)
    _B1 = TypeVar("_B1", bound=BasisLike)
    _B2 = TypeVar("_B2", bound=BasisLike)

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)


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
        "basis": VariadicTupleBasis((probability_vector["basis"][0], basis), None),
        "data": converted,
    }


def convert_probability_vector_to_position_basis(
    probability_vector: ProbabilityVector[_SBV0],
) -> ProbabilityVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    probability_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis, ...]]
    """
    return convert_probability_vector_to_basis(
        probability_vector,
        tuple_basis_as_fundamental(probability_vector["basis"]),
    )


def convert_probability_vector_to_momentum_basis(
    probability_vector: ProbabilityVector[_SBV0],
) -> ProbabilityVector[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]
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
        stacked_basis_as_transformed_basis(probability_vector["basis"]),
    )
