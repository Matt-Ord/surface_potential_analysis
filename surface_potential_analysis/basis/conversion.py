from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
)

if TYPE_CHECKING:
    import numpy as np

    from .basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )

    _NDInv = TypeVar("_NDInv", bound=int)

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)


def basis_as_fundamental_position_basis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv],
) -> FundamentalPositionBasis[_NF0Inv, _NDInv]:
    """
    Get the fundamental position axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    FundamentalPositionBasis[_NF0Inv]
    """
    return FundamentalPositionBasis(axis.delta_x, axis.fundamental_n)


def basis_as_fundamental_momentum_basis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv],
) -> FundamentalTransformedPositionBasis[_NF0Inv, _NDInv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumBasis[_NF0Inv, _NDInv]
    """
    return FundamentalTransformedPositionBasis(axis.delta_x, axis.fundamental_n)


def basis_as_fundamental_transformed_basis(
    axis: BasisLike[_NF0Inv, _N0Inv],
) -> FundamentalTransformedBasis[_NF0Inv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumBasis[_NF0Inv, _NDInv]
    """
    return FundamentalTransformedBasis(axis.fundamental_n)


def basis_as_fundamental_basis(
    axis: BasisLike[_NF0Inv, _N0Inv],
) -> FundamentalBasis[_NF0Inv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumBasis[_NF0Inv, _NDInv]
    """
    return FundamentalBasis(axis.fundamental_n)


def basis_as_n_point_basis(
    delta_x: np.ndarray[tuple[int], np.dtype[np.float64]], *, n: _N1Inv
) -> FundamentalPositionBasis[_N1Inv, int]:
    """
    Get the corresponding n point axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv, _NDInv]
    n : _N1Inv

    Returns
    -------
    FundamentalPositionBasis[_N1Inv, _NDInv]
    """
    return FundamentalPositionBasis[_N1Inv, int](delta_x, n)


def basis_as_single_point_basis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv],
) -> FundamentalPositionBasis[Literal[1], _NDInv]:
    """
    Get the corresponding single point axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalPositionBasis[Literal[1], _NDInv]
    """
    return basis_as_n_point_basis(axis, n=1)
