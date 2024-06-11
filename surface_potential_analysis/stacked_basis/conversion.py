from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_basis,
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    basis_as_fundamental_transformed_basis,
    basis_as_n_point_basis,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
        FundamentalTransformedBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def stacked_basis_as_fundamental_transformed_basis(
    basis: TupleBasisLike[*tuple[_B0, ...]],
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
        *tuple(basis_as_fundamental_transformed_basis(axis) for axis in basis)
    )


def stacked_basis_as_fundamental_basis(
    basis: TupleBasisLike[*tuple[_B0, ...]],
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
    return TupleBasis(*tuple(basis_as_fundamental_basis(axis) for axis in basis))


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisLike[_BL0],
) -> TupleBasisLike[FundamentalTransformedPositionBasis[Any, Literal[1]]]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisLike[_BL0, _BL0],
) -> TupleBasisLike[
    FundamentalTransformedPositionBasis[Any, Literal[2]],
    FundamentalTransformedPositionBasis[Any, Literal[2]],
]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisLike[_BL0, _BL0, _BL0],
) -> TupleBasisLike[
    FundamentalTransformedPositionBasis[Any, Literal[3]],
    FundamentalTransformedPositionBasis[Any, Literal[3]],
    FundamentalTransformedPositionBasis[Any, Literal[3]],
]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisLike[*tuple[_BL0, ...]],
) -> TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]:
    ...


def stacked_basis_as_fundamental_momentum_basis(
    basis: TupleBasisLike[*tuple[_BL0, ...]]
    | TupleBasisLike[_BL0]
    | TupleBasisLike[_BL0, _BL0]
    | TupleBasisLike[_BL0, _BL0, _BL0],
) -> TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]:
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
        *tuple(basis_as_fundamental_momentum_basis(axis) for axis in basis)
    )


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisLike[_BL0],
) -> TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisLike[_BL0, _BL0],
) -> TupleBasisLike[
    FundamentalPositionBasis[Any, Literal[2]],
    FundamentalPositionBasis[Any, Literal[2]],
]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisLike[_BL0, _BL0, _BL0],
) -> TupleBasisLike[
    FundamentalPositionBasis[Any, Literal[3]],
    FundamentalPositionBasis[Any, Literal[3]],
    FundamentalPositionBasis[Any, Literal[3]],
]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisLike[*tuple[Any, ...]],
) -> TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    ...


def stacked_basis_as_fundamental_position_basis(
    basis: TupleBasisLike[*tuple[_BL0, ...]]
    | TupleBasisLike[_BL0]
    | TupleBasisLike[_BL0, _BL0]
    | TupleBasisLike[_BL0, _BL0, _BL0],
) -> TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    TupleBasisLike[tuple[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return TupleBasis(
        *tuple(basis_as_fundamental_position_basis(axis) for axis in basis)
    )


def stacked_basis_as_fundamental_with_shape(
    basis: TupleBasisLike[*tuple[_BL0, ...]],
    shape: tuple[int, ...],
) -> TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
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
            basis_as_n_point_basis(ax, n=n)
            for (ax, n) in zip(basis, shape, strict=True)
        )
    )
