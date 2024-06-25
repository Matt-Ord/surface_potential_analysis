from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
)
from surface_potential_analysis.basis.conversion import (
    basis_as_n_point_basis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import (
        BasisWithLengthLike,
    )

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


def stacked_basis_as_fundamental_transformed_basis(
    basis: StackedBasisLike[Any, Any, Any],
) -> TupleBasisLike[*tuple[FundamentalTransformedBasis[Any], ...]]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumBasis[Any, Any], ...]
    """
    return TupleBasis(
        *tuple(FundamentalTransformedBasis(n) for n in basis.fundamental_shape)
    )


def stacked_basis_as_fundamental_basis(
    basis: StackedBasisLike[Any, Any, Any],
) -> TupleBasisLike[*tuple[FundamentalBasis[Any], ...]]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumBasis[Any, Any], ...]
    """
    return TupleBasis(*tuple(FundamentalBasis(n) for n in basis.fundamental_shape))


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisWithLengthLike[_BL0],
) -> TupleBasisWithLengthLike[FundamentalTransformedPositionBasis[Any, Literal[1]]]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisWithLengthLike[_BL0, _BL0],
) -> TupleBasisWithLengthLike[
    FundamentalTransformedPositionBasis[Any, Literal[2]],
    FundamentalTransformedPositionBasis[Any, Literal[2]],
]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisWithLengthLike[_BL0, _BL0, _BL0],
) -> TupleBasisWithLengthLike[
    FundamentalTransformedPositionBasis[Any, Literal[3]],
    FundamentalTransformedPositionBasis[Any, Literal[3]],
    FundamentalTransformedPositionBasis[Any, Literal[3]],
]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
) -> TupleBasisWithLengthLike[
    *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
]:
    ...


def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisWithVolumeLike[Any, Any, Any]
    | TupleBasisWithLengthLike[_BL0]
    | TupleBasisWithLengthLike[_BL0, _BL0]
    | TupleBasisWithLengthLike[_BL0, _BL0, _BL0],
) -> TupleBasisWithLengthLike[
    *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumBasis[Any, Any], ...]
    """
    return TupleBasis(
        *tuple(
            starmap(
                FundamentalTransformedPositionBasis[int, int],
                zip(basis.delta_x_stacked, basis.fundamental_shape),
            )
        )
    )


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisWithLengthLike[_BL0],
) -> TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisWithLengthLike[_BL0, _BL0],
) -> TupleBasisWithLengthLike[
    FundamentalPositionBasis[Any, Literal[2]],
    FundamentalPositionBasis[Any, Literal[2]],
]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisWithLengthLike[_BL0, _BL0, _BL0],
) -> TupleBasisWithLengthLike[
    FundamentalPositionBasis[Any, Literal[3]],
    FundamentalPositionBasis[Any, Literal[3]],
    FundamentalPositionBasis[Any, Literal[3]],
]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
) -> TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    ...


def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisWithVolumeLike[Any, Any, Any]
    | TupleBasisWithLengthLike[_BL0]
    | TupleBasisWithLengthLike[_BL0, _BL0]
    | TupleBasisWithLengthLike[_BL0, _BL0, _BL0],
) -> TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    StackedBasisWithVolumeLike[tuple[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return TupleBasis(
        *tuple(
            starmap(
                FundamentalPositionBasis[int, int],
                zip(basis.delta_x_stacked, basis.fundamental_shape),
            )
        )
    )


def stacked_basis_as_fundamental_with_shape(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    shape: tuple[int, ...],
) -> TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    """
    Given a basis get a fundamental position basis with the given shape.

    Parameters
    ----------
    basis : Basis[_NDInv]
    shape : tuple[int, ...]

    Returns
    -------
    Basis[_NDInv]
    """
    return TupleBasis(
        *tuple(
            basis_as_n_point_basis(delta_x, n=n)
            for (delta_x, n) in zip(basis.delta_x_stacked, shape, strict=True)
        )
    )
