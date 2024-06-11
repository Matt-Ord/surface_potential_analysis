from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.basis_like import (
    BasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.potential.potential import Potential

    _SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])
    _SB1 = TypeVar("_SB1", bound=TupleBasisLike[*tuple[Any, ...]])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


def convert_potential_to_basis(
    potential: Potential[_SB0], basis: _SB1
) -> Potential[_SB1]:
    """
    Given an potential, calculate the potential in the given basis.

    Parameters
    ----------
    potential : Potential[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    Potential[_B1Inv]
    """
    converted = convert_vector(potential["data"], potential["basis"], basis)
    return {"basis": basis, "data": converted}


def convert_potential_to_position_basis(
    potential: Potential[TupleBasisLike[*tuple[_BL0, ...]]],
) -> Potential[TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    """
    Given an potential, convert to the fundamental position basis.

    Parameters
    ----------
    potential : Potential[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    Potential[_B1Inv]
    """
    return convert_potential_to_basis(
        potential, stacked_basis_as_fundamental_position_basis(potential["basis"])
    )
