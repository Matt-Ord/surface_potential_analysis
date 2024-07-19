from __future__ import annotations

from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    TypeVarTuple,
    Unpack,
    overload,
)

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.util.util import (
    get_position_in_sorted,
    slice_ignoring_axes,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisLike,
    )
    from surface_potential_analysis.types import (
        ArrayStackedIndexLike,
        FloatLike_co,
        IntLike_co,
        SingleIndexLike,
        SingleStackedIndexLike,
        StackedIndexLike,
    )

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _BL0Inv = TypeVar("_BL0Inv", bound=BasisWithLengthLike[Any, Any, Any])
    _NDInv = TypeVar("_NDInv", bound=int)

    _TS = TypeVarTuple("_TS")


def project_k_points_along_axes(
    points: np.ndarray[tuple[_NDInv, Unpack[_TS]], np.dtype[np.float64]],
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float64]]:
    """
    Get the list of k points projected onto the plane including both axes.

    Parameters
    ----------
    points : np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)

    projected_axes = np.zeros((len(axes), util.ndim))
    for i, ax in enumerate(axes):
        projected = util.delta_k_stacked[ax]
        for j in range(i):
            projected -= projected_axes[j] * np.dot(projected_axes[j], projected)

        projected_axes[i] = projected / np.linalg.norm(projected)

    return np.tensordot(projected_axes, points, axes=(1, 0))


def get_fundamental_stacked_k_points_projected_along_axes(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)
    points = util.fundamental_stacked_k_points
    return project_k_points_along_axes(points, basis, axes)


def get_k_coordinates_in_axes(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    axes: tuple[int, ...],
    idx: SingleStackedIndexLike | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    idx = tuple(0 for _ in range(basis.ndim - len(axes))) if idx is None else idx
    points = get_fundamental_stacked_k_points_projected_along_axes(basis, axes)
    slice_ = slice_ignoring_axes(idx, axes)
    return np.transpose(
        points.reshape(-1, *basis.shape)[:, *slice_],
        (0, *(1 + np.array(get_position_in_sorted(axes)))),
    )  # type: ignore[no-any-return]


def project_x_points_along_axes(
    points: np.ndarray[tuple[_NDInv, Unpack[_TS]], np.dtype[np.float64]],
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float64]]:
    """
    Get the list of x points projected onto the plane including all axes.

    Parameters
    ----------
    points : np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)

    projected_axes = np.zeros((len(axes), basis.ndim))
    for i, ax in enumerate(axes):
        projected = util.delta_x_stacked[ax]
        for j in range(i):
            projected -= projected_axes[j] * np.dot(projected_axes[j], projected)

        projected_axes[i] = projected / np.linalg.norm(projected)

    return np.tensordot(projected_axes, points, axes=(1, 0))


def get_fundamental_stacked_x_points_projected_along_axes(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the fundamental_x_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)
    points = util.fundamental_x_points_stacked
    return project_x_points_along_axes(points, basis, axes)


def get_x_coordinates_in_axes(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    axes: tuple[int, ...],
    idx: SingleStackedIndexLike | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the fundamental_x_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    idx = tuple(0 for _ in range(basis.ndim - len(axes))) if idx is None else idx
    points = get_fundamental_stacked_x_points_projected_along_axes(basis, axes)
    slice_ = slice_ignoring_axes(idx, axes)
    return np.transpose(
        points.reshape(-1, *basis.shape)[:, *slice_],
        (0, *(1 + np.array(get_position_in_sorted(axes)))),
    )  # type: ignore[no-any-return]


@overload
def _wrap_distance(
    distance: float | np.float64, length: FloatLike_co, origin: FloatLike_co = 0
) -> np.float64:
    ...


@overload
def _wrap_distance(
    distance: np.ndarray[_S0Inv, np.dtype[np.float64]],
    length: FloatLike_co,
    origin: FloatLike_co = 0,
) -> np.ndarray[_S0Inv, np.dtype[np.float64]]:
    ...


def _wrap_distance(distance: Any, length: Any, origin: Any = 0) -> Any:
    return np.mod(distance - origin + length / 2, length) + origin - length / 2


@overload
def _wrap_index(
    distance: IntLike_co, length: IntLike_co, origin: IntLike_co = 0
) -> np.int_:
    ...


@overload
def _wrap_index(
    distance: np.ndarray[_S0Inv, np.dtype[np.int_]],
    length: IntLike_co,
    origin: IntLike_co = 0,
) -> np.ndarray[_S0Inv, np.dtype[np.int_]]:
    ...


def _wrap_index(distance: Any, length: Any, origin: Any = 0) -> Any:
    return np.mod(distance - origin + length // 2, length) + origin - length // 2


@overload
def wrap_index_around_origin(
    basis: TupleBasisLike[*tuple[_B0, ...]],
    idx: SingleStackedIndexLike,
    origin: SingleIndexLike | None = None,
    axes: tuple[int, ...] | None = None,
) -> SingleStackedIndexLike:
    ...


@overload
def wrap_index_around_origin(
    basis: TupleBasisLike[*tuple[_B0, ...]],
    idx: ArrayStackedIndexLike[_S0Inv],
    origin: SingleIndexLike | None = None,
    axes: tuple[int, ...] | None = None,
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


def wrap_index_around_origin(
    basis: TupleBasisLike[*tuple[_B0, ...]],
    idx: StackedIndexLike,
    origin: SingleIndexLike | None = None,
    axes: tuple[int, ...] | None = None,
) -> StackedIndexLike:
    """
    Given an index or list of indexes in stacked form, find the equivalent index closest to the origin.

    Parameters
    ----------
    basis : _B3d0Inv
    idx : StackedIndexLike | FlatIndexLike
    origin_idx : StackedIndexLike | FlatIndexLike, optional
        origin to wrap around, by default (0, 0, 0)

    Returns
    -------
    StackedIndexLike
    """
    util = BasisUtil(basis)
    origin = tuple(0 for _ in basis) if origin is None else origin
    origin = origin if isinstance(origin, tuple) else util.get_stacked_index(origin)
    return tuple(  # type: ignore[return-value]
        _wrap_index(idx[ax], util.shape[ax], origin[ax])
        if axes is None or ax in axes
        else idx[ax]
        for ax in range(util.ndim)
    )


def wrap_x_point_around_origin(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    points: np.ndarray[tuple[_NDInv, Unpack[_TS]], np.dtype[np.float64]],
    origin: np.ndarray[tuple[_NDInv], np.dtype[np.float64]] | None = None,
) -> np.ndarray[tuple[_NDInv, Unpack[_TS]], np.dtype[np.float64]]:
    """
    Given an index or list of indexes in stacked form, find the equivalent index closest to the origin.

    Parameters
    ----------
    basis : _B3d0Inv
    idx : StackedIndexLike | FlatIndexLike
    origin_idx : StackedIndexLike | FlatIndexLike, optional
        origin to wrap around, by default (0, 0, 0)

    Returns
    -------
    StackedIndexLike
    """
    util = BasisUtil(basis)
    origin = np.zeros(points.shape[0]) if origin is None else origin

    distance_along_axes = np.tensordot(
        np.linalg.inv(util.delta_x_stacked), points, axes=(0, 0)
    )
    origin_along_axes = np.tensordot(
        np.linalg.inv(util.delta_x_stacked), origin, axes=(0, 0)
    )
    wrapped_distances = np.array(
        [
            _wrap_distance(distance, 1, origin)
            for (distance, origin) in zip(
                distance_along_axes, origin_along_axes, strict=True
            )
        ],
        dtype=np.float64,
    )
    return np.tensordot(util.delta_x_stacked, wrapped_distances, axes=(0, 0))


_S2d0Inv = TypeVar("_S2d0Inv", bound=tuple[int, int])


def calculate_distances_along_path(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    path: np.ndarray[_S2d0Inv, np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """
    calculate cumulative distances along the given path.

    Parameters
    ----------
    basis : _B0Inv
        basis which the path is through
    path : np.ndarray[tuple[int, int], np.dtype[np.int_]]
        path through the basis, _ND by int points
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.int_]]
    """
    out = path[:, :-1] - path[:, 1:]
    if wrap_distances:
        util = BasisUtil(basis)
        return np.array(
            list(starmap(_wrap_distance, zip(out, util.shape, strict=True))),
            dtype=np.float64,
        )

    return out  # type:ignore[no-any-return]


def calculate_cumulative_x_distances_along_path(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    path: np.ndarray[_S2d0Inv, np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of indexes in the basis, _ND by int points
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    distances = calculate_distances_along_path(
        basis, path, wrap_distances=wrap_distances
    )

    util = BasisUtil(basis)
    x_distances = np.linalg.norm(
        np.tensordot(util.fundamental_dx_stacked, distances, axes=(0, 0)), axis=0
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


def calculate_cumulative_k_distances_along_path(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
    path: np.ndarray[_S2d0Inv, np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of indexes in the basis
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    distances = calculate_distances_along_path(
        basis, path, wrap_distances=wrap_distances
    )
    util = BasisUtil(basis)
    x_distances = np.linalg.norm(  # TODO: test
        np.tensordot(distances, util.dk_stacked, axes=(0, 0)),
        axis=0,
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


@overload
def get_x01_mirrored_index(idx: SingleStackedIndexLike) -> SingleStackedIndexLike:
    ...


@overload
def get_x01_mirrored_index(
    idx: ArrayStackedIndexLike[_S0Inv],
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


def get_x01_mirrored_index(idx: StackedIndexLike) -> StackedIndexLike:
    """
    Mirror the coordinate idx about x0=x1.

    Parameters
    ----------
    basis : _B3d0Inv
        the basis to mirror in
    idx : tuple[int, int, int] | int
        The index to mirror

    Returns
    -------
    tuple[int, int, int] | int
        The mirrored index
    """
    arr = list(idx)
    arr[0] = idx[1]
    arr[1] = idx[0]
    return tuple(arr)  # type: ignore[return-value]


def get_single_point_basis(
    basis: TupleBasisLike[Unpack[tuple[_BL0Inv, ...]]],
) -> tuple[FundamentalPositionBasis[Literal[1], Any], ...]:
    """
    Get the basis with a single point in position space.

    Parameters
    ----------
    basis : _B3d0Inv
        initial basis
    _type : Literal[&quot;position&quot;, &quot;momentum&quot;]
        type of the final basis

    Returns
    -------
    _SPB|_SMB
        the single point basis in either position or momentum basis
    """
    return tuple(FundamentalPositionBasis(b.delta_x, 1) for b in basis)  # type: ignore[return-value]


def get_max_idx(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[Any], np.dtype[np.complex128]],
    axes: tuple[int, ...],
) -> SingleStackedIndexLike:
    """
    Get the max index of data in the given axes.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[int], np.dtype[np.complex_]]
    axes : tuple[int, ...]

    Returns
    -------
    SingleStackedIndexLike
    """
    util = BasisUtil(basis)
    max_idx = util.get_stacked_index(np.argmax(np.abs(data)))
    return tuple(x for (i, x) in enumerate(max_idx) if i not in axes)
