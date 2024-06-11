from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    overload,
)

import numpy as np

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.build import (
    position_basis_from_shape,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.types import (
        ArrayStackedIndexLike,
        ArrayStackedIndexLike3d,
        SingleStackedIndexLike,
        SingleStackedIndexLike3d,
        StackedIndexLike,
        StackedIndexLike3d,
    )

    _S1Inv = TypeVar("_S1Inv", bound=tuple[int, ...])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _TS = TypeVarTuple("_TS")
    _L = TypeVar("_L", bound=int)


@overload
def fold_point_in_bragg_plane(
    bragg_point: SingleStackedIndexLike3d, coordinate: ArrayStackedIndexLike3d[_S1Inv]
) -> ArrayStackedIndexLike3d[_S1Inv]:
    ...


@overload
def fold_point_in_bragg_plane(
    bragg_point: SingleStackedIndexLike3d, coordinate: SingleStackedIndexLike3d
) -> SingleStackedIndexLike3d:
    ...


def fold_point_in_bragg_plane(
    bragg_point: SingleStackedIndexLike3d, coordinate: StackedIndexLike3d
) -> StackedIndexLike3d:
    """
    Given a bragg point and a StackedIndexLike fold the coordinates about the bragg plane.

    If the bragg_point corresponds to the closest brag plane outside the point
    this increments the brillouin zone by 1.
    If the bragg_point corresponds to the closest brag plane inside the point
    this increments the brillouin zone by 1.

    Parameters
    ----------
    bragg_point : SingleStackedIndexLike
        bragg point (point of symmetry corresponding to a bragg plane)
    coordinate : StackedIndexLike
        coordinate(s) to fold

    Returns
    -------
    StackedIndexLike
        folded coordinate(s)
    """
    # To fold a point about a bragg plane we
    # reflect it through the plane and then reflect it about
    # a plane parallel to this at the origin.
    # If we denote the plane by the normal n and point p, where p.n n = p
    # we find p is given by bragg_point / 2.
    # The two reflections are then equivalent to a translation a - 2p.
    return (
        (coordinate[0] - bragg_point[0]),  # type: ignore cant infer
        (coordinate[1] - bragg_point[1]),
        (coordinate[2] - bragg_point[2]),
    )


def get_bragg_point_basis(
    basis: TupleBasisLike[*tuple[Any, ...]], *, n_bands: int = 1
) -> TupleBasisLike[*tuple[FundamentalPositionBasis[int, int], ...]]:
    """
    Get the basis for the bragg points, where the k_points are the bragg points of the given basis.

    Parameters
    ----------
    basis : _B0Inv
        parent basis
    n_bands : int, optional
        number of bands, by default 1

    Returns
    -------
    FundamentalPositionTupleBasisLike[tuple[int, int, int]
    """
    width = 2 * n_bands + 1
    util = BasisUtil(basis)
    shape: tuple[int, ...] = (width,) * len(util.shape)
    # we want a basis where dk_1 = delta_k_original
    # Since dk = 2 * pi / delta_x we need to decrease delta_x by a factor of util.ni
    # as this will increase dk to the intended value.
    # This is equivalent to delta_x = util.dx
    return position_basis_from_shape(shape, util.dx_stacked)  # type: ignore[call-overload,no-any-return]


def get_all_brag_point_index(
    basis: TupleBasisLike[*tuple[Any, ...]], n_bands: int = 1
) -> ArrayStackedIndexLike[tuple[int]]:
    """
    Given a basis in 3D, get the stacked index of the brag points in the first n_bands.

    Parameters
    ----------
    basis : _B3d0Inv
    n_bands : int, optional
        number of bands of the brag points, by default 1

    Returns
    -------
    ArrayStackedIndexLike[tuple[int]]
        Index of each of the brag points, using the nk_points convention for the ArrayStackedIndexLike
    """
    bragg_point_basis = get_bragg_point_basis(basis, n_bands=n_bands)
    nk_points = BasisUtil(bragg_point_basis).fundamental_stacked_nk_points
    shape = BasisUtil(basis).shape
    return tuple(
        ni * nki_points for (ni, nki_points) in zip(shape, nk_points, strict=True)
    )


def get_all_brag_point(
    basis: TupleBasisLike[*tuple[Any, ...]], *, n_bands: int = 1
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Given a basis in 3D, get the coordinates of the brag points in the first n_bands.

    Parameters
    ----------
    basis : _B3d0Inv
    n_bands : int, optional
        number of bands of the brag points, by default 1

    Returns
    -------
    np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]
        Array of each of the brag points, using the nk_points convention
    """
    bragg_point_basis = get_bragg_point_basis(basis, n_bands=n_bands)
    return BasisUtil(bragg_point_basis).fundamental_stacked_k_points


@overload
def get_bragg_plane_distance(
    bragg_point: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    point: np.ndarray[tuple[Literal[3], _L, Unpack[_TS]], np.dtype[np.float64]],
) -> np.ndarray[tuple[_L, Unpack[_TS]], np.dtype[np.float64]]:
    ...


@overload
def get_bragg_plane_distance(
    bragg_point: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    point: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
) -> np.float64:
    ...


def get_bragg_plane_distance(
    bragg_point: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    point: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
    | np.ndarray[tuple[Literal[3], Unpack[_TS]], np.dtype[np.float64]],
) -> np.float64 | np.ndarray[tuple[Unpack[_TS]], np.dtype[np.float64]]:
    """
    Get the distance from the bragg plane for the given bragg_point.

    Parameters
    ----------
    bragg_point : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        bragg point (point of symmetry corresponding to a bragg plane)
    point : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]] | np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]

    Returns
    -------
    np.float_ | np.ndarray[tuple[int, ...], np.dtype[np.float_]]
    """
    bragg_point_norm = np.linalg.norm(bragg_point, axis=0)
    normalized_bragg_point = np.array(bragg_point) / bragg_point_norm
    # d = n.(a-p) = n.a - 0.5 * n.d where p = d/2 is the point on the bragg plane
    return np.tensordot(normalized_bragg_point, point, axes=(0, 0)) - (  # type: ignore[no-any-return]
        bragg_point_norm / 2
    )


def _get_decrement_tolerance(basis: TupleBasisLike[*tuple[Any, ...]]) -> float:
    r_tol = 1e-5
    util = BasisUtil(basis)
    return np.min(np.linalg.norm(util.dk_stacked, axis=1)) * r_tol


@overload
def decrement_brillouin_zone_3d(
    basis: TupleBasisLike[Any, Any, Any], coordinate: ArrayStackedIndexLike3d[_S0Inv]
) -> ArrayStackedIndexLike3d[_S0Inv]:
    ...


@overload
def decrement_brillouin_zone_3d(
    basis: TupleBasisLike[Any, Any, Any], coordinate: SingleStackedIndexLike3d
) -> SingleStackedIndexLike3d:
    ...


def decrement_brillouin_zone_3d(
    basis: TupleBasisLike[Any, Any, Any], coordinate: StackedIndexLike3d
) -> StackedIndexLike3d:
    """
    Given a basis, and a set of coordinates, decrement the brillouin zone of each coordinate.

    Note if the coordinate is in the 1st (ie lowest) zone it wil remain in the same location

    Parameters
    ----------
    basis : _B3d0Inv
    coordinate : StackedIndexLike

    Returns
    -------
    StackedIndexLike
        coordinates in the min(1, n-1) th brillouin zone
    """
    tolerance = _get_decrement_tolerance(basis)
    # Transpose used - we want the new axis to appear as the last axis not the first
    out = cast(np.ndarray[Any, Any], np.atleast_2d(np.transpose(coordinate))).T  # type: ignore cant spot array like cSpell:disable-line
    coordinate_points = BasisUtil(basis).get_k_points_at_index(tuple(out))  # type: ignore[arg-type]
    bragg_points = get_all_brag_point(basis, n_bands=1)

    closest_points = np.zeros_like(out[0], dtype=np.int_)
    distances = np.linalg.norm(coordinate_points, axis=0)
    # To fold a point into the previous brillouin zone we reflect it
    # in the closest bragg plane inside the point and then reflect it about
    # a plane parallel to this at the origin.
    for i, bragg_point in enumerate(np.array(bragg_points)[:, 1:].T):  # type: ignore unknown
        new_distances = get_bragg_plane_distance(bragg_point, coordinate_points)  # type: ignore[arg-type,var-annotated]
        is_closer = np.logical_and(new_distances > tolerance, new_distances < distances)  # type: ignore unknown arg
        closest_points[is_closer] = i + 1
        distances[is_closer] = new_distances[is_closer]

    bragg_point_index = get_all_brag_point_index(basis, n_bands=1)
    for i, bragg_point in enumerate(np.array(bragg_point_index).T):
        should_fold = closest_points == i
        folded = fold_point_in_bragg_plane(bragg_point, tuple(out[:, should_fold]))  # type: ignore[arg-type,var-annotated]
        out[:, should_fold] = folded  # type: ignore unknown

    if isinstance(coordinate[0], np.ndarray):
        # For 0D arrays we need to drop the additional axis here
        old_shape = coordinate[0].shape
        return (  # type: ignore unknown
            out[0].reshape(old_shape),
            out[1].reshape(old_shape),
            out[2].reshape(old_shape),
        )
    return (out.item(0), out.item(1), out.item(2))  # type: ignore unknown


@overload
def decrement_brillouin_zone(
    basis: TupleBasisLike[*tuple[Any, ...]], coordinate: ArrayStackedIndexLike[_S0Inv]
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


@overload
def decrement_brillouin_zone(
    basis: TupleBasisLike[*tuple[Any, ...]], coordinate: SingleStackedIndexLike
) -> SingleStackedIndexLike:
    ...


def decrement_brillouin_zone(
    basis: TupleBasisLike[*tuple[Any, ...]], coordinate: StackedIndexLike
) -> StackedIndexLike:
    """
    Given a basis, and a set of coordinates, decrement the brillouin zone of each coordinate.

    Note if the coordinate is in the 1st (ie lowest) zone it wil remain in the same location

    Parameters
    ----------
    basis : _B3d0Inv
    coordinate : StackedIndexLike

    Returns
    -------
    StackedIndexLike
        coordinates in the min(1, n-1) th brillouin zone
    """
    tolerance = _get_decrement_tolerance(basis)
    # Transpose used - we want the new axis to appear as the last axis not the first
    out = np.atleast_2d(np.transpose(coordinate)).T  # type: ignore unknown arg cSpell:disable-line
    coordinate_points = BasisUtil(basis).get_k_points_at_index(tuple(out))  # type: ignore[arg-type]
    bragg_points = get_all_brag_point(basis, n_bands=1)

    closest_points = np.zeros_like(out[0], dtype=np.int_)
    distances = np.linalg.norm(coordinate_points, axis=0)
    # To fold a point into the previous brillouin zone we reflect it
    # in the closest bragg plane inside the point and then reflect it about
    # a plane parallel to this at the origin.
    for i, bragg_point in enumerate(np.array(bragg_points)[:, 1:].T):
        new_distances = get_bragg_plane_distance(bragg_point, coordinate_points)  # type: ignore[arg-type,var-annotated]
        is_closer = np.logical_and(new_distances > tolerance, new_distances < distances)  # type: ignore unknown arg
        closest_points[is_closer] = i + 1
        distances[is_closer] = new_distances[is_closer]

    bragg_point_index = get_all_brag_point_index(basis, n_bands=1)
    for i, bragg_point in enumerate(np.array(bragg_point_index).T):
        should_fold = closest_points == i
        folded = fold_point_in_bragg_plane(bragg_point, tuple(out[:, should_fold]))  # type: ignore[arg-type,var-annotated]
        out[:, should_fold] = folded

    if isinstance(coordinate[0], np.ndarray):
        # For 0D arrays we need to drop the additional axis here
        old_shape = coordinate[0].shape
        return tuple(o.reshape(old_shape) for o in out)  # type: ignore unknown arg
    return tuple(o for o in out.flat)  # type: ignore unknown arg
