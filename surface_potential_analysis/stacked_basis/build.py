from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis3d,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisWithLengthLike

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])
    _BL1 = TypeVar("_BL1", bound=BasisWithLengthLike[Any, Any, Any])
    _BL2 = TypeVar("_BL2", bound=BasisWithLengthLike[Any, Any, Any])
    _S1Inv = TypeVar("_S1Inv", bound=tuple[int, int])


_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)


def position_basis_3d_from_parent(
    parent: TupleBasisLike[_BL0, _BL1, _BL2],
    resolution: tuple[_L0, _L1, _L2],
) -> TupleBasisLike[
    FundamentalPositionBasis[_L0, Literal[3]],
    FundamentalPositionBasis[_L1, Literal[3]],
    FundamentalPositionBasis[_L2, Literal[3]],
]:
    """
    Given a parent basis construct another basis with the same lattice vectors.

    Parameters
    ----------
    parent : _B3d0Inv
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]

    Returns
    -------
    FundamentalPositionTupleBasisLike[tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        _description_
    """
    return TupleBasis(
        FundamentalPositionBasis(parent[0].delta_x, resolution[0]),
        FundamentalPositionBasis(parent[1].delta_x, resolution[1]),
        FundamentalPositionBasis(parent[2].delta_x, resolution[2]),
    )


@overload
def fundamental_stacked_basis_from_shape(
    shape: tuple[_L0],
) -> TupleBasisLike[FundamentalBasis[_L0]]:
    ...


@overload
def fundamental_stacked_basis_from_shape(
    shape: tuple[_L0, _L1],
) -> TupleBasisLike[FundamentalBasis[_L0], FundamentalBasis[_L1]]:
    ...


@overload
def fundamental_stacked_basis_from_shape(
    shape: tuple[_L0, _L1, _L2],
) -> TupleBasisLike[
    FundamentalBasis[_L0], FundamentalBasis[_L1], FundamentalBasis[_L2]
]:
    ...


@overload
def fundamental_stacked_basis_from_shape(
    shape: tuple[int, ...],
) -> TupleBasisLike[*tuple[FundamentalBasis[Any], ...]]:
    ...


def fundamental_stacked_basis_from_shape(
    shape: tuple[Any, ...] | tuple[Any, Any, Any] | tuple[Any, Any] | tuple[Any],
) -> TupleBasisLike[*tuple[FundamentalBasis[Any], ...]]:
    """
    Given a resolution and a set of directions construct a FundamentalPositionBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalPositionTupleBasisLike[tuple[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    return TupleBasis(*tuple(FundamentalBasis(n) for n in shape))


def position_basis_from_shape(
    shape: tuple[int, ...],
    delta_x: np.ndarray[_S1Inv, np.dtype[np.float64]] | None = None,
) -> TupleBasisLike[*tuple[FundamentalPositionBasis[int, int], ...]]:
    """
    Given a resolution and a set of directions construct a FundamentalPositionBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalPositionTupleBasisLike[tuple[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    delta_x = np.eye(len(shape)) if delta_x is None else delta_x
    return TupleBasis(
        *tuple(starmap(FundamentalPositionBasis, zip(delta_x, shape, strict=True)))
    )


def position_basis_3d_from_shape(
    shape: tuple[_L0, _L1, _L2],
    delta_x: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
    | None = None,
) -> TupleBasisLike[
    FundamentalPositionBasis[_L0, Literal[3]],
    FundamentalPositionBasis[_L1, Literal[3]],
    FundamentalPositionBasis[_L2, Literal[3]],
]:
    """
    Given a shape generate a 3d position basis.

    Parameters
    ----------
    shape : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
    delta_x : np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]] | None, optional
        delta_x as list of individual delta_x, by default None

    Returns
    -------
    tuple[FundamentalPositionBasis[_NF0Inv, Literal[3]], FundamentalPositionBasis[_NF1Inv, Literal[3]], FundamentalPositionBasis[_NF2Inv, Literal[3]]]
    """
    return position_basis_from_shape(shape, delta_x)  # type: ignore[arg-type,return-value]


def momentum_basis_3d_from_resolution(
    resolution: tuple[_L0, _L1, _L2],
    delta_x: tuple[
        np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ]
    | None = None,
) -> TupleBasis[
    FundamentalTransformedPositionBasis3d[_L0],
    FundamentalTransformedPositionBasis3d[_L1],
    FundamentalTransformedPositionBasis3d[_L2],
]:
    """
    Given a resolution and a set of directions construct a FundamentalMomentumBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalMomentumTupleBasisLike[tuple[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    delta_x = (
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        if delta_x is None
        else delta_x
    )
    return TupleBasis(
        FundamentalTransformedPositionBasis(delta_x[0], resolution[0]),
        FundamentalTransformedPositionBasis(delta_x[1], resolution[1]),
        FundamentalTransformedPositionBasis(delta_x[2], resolution[2]),
    )
