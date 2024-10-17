from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    convert_vector,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.potential.potential import Potential

    _SB0 = TypeVar("_SB0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
    _SB1 = TypeVar("_SB1", bound=StackedBasisWithVolumeLike[Any, Any, Any])


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
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
) -> Potential[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
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


_B0 = TypeVar("_B0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def get_continuous_potential(
    potential: Potential[_B0],
) -> (
    Callable[[tuple[float, ...]], float]
    | Callable[
        [tuple[np.ndarray[Any, np.dtype[np.float64]], ...]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]
):
    """Given a potential, convert it to a continuous function of x.

    Parameters
    ----------
    potential : Potential[_B0]

    Returns
    -------
    Callable[[float], float]
    """
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_momentum_basis(potential["basis"]),
    )
    k_points = BasisUtil(converted["basis"]).fundamental_stacked_k_points

    @overload
    def _fn(
        x: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        ...

    @overload
    def _fn(
        x: tuple[float, ...],
    ) -> float:
        ...

    def _fn(
        x: tuple[float, ...] | tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        phases = 1j * np.einsum("ij,i...->j...", k_points, x)  # type:ignore unknown
        return np.einsum(  # type:ignore unknown
            "j,j...->...",
            converted["data"],
            np.exp(phases) / np.sqrt(converted["basis"].n),
        )

    return _fn


def get_potential_derivative(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    *,
    axis: int = 0,
) -> Potential[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    """Get the derivative of a potential."""
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_momentum_basis(potential["basis"]),
    )
    k_points = BasisUtil(converted["basis"]).k_points[axis]
    return {"basis": converted["basis"], "data": 1j * k_points * converted["data"]}
