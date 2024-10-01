from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    from surface_potential_analysis.types import IntLike_co, SingleStackedIndexLike

    P = ParamSpec("P")
    R = TypeVar("R")
    _DTInv = TypeVar("_DTInv", bound=np.dtype[Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def slice_along_axis(
    slice_at_axis: slice | IntLike_co | None, axis: IntLike_co = -1
) -> (
    tuple[EllipsisType | slice | IntLike_co | None, ...]
    | tuple[slice | IntLike_co | None, ...]
):
    """Return a slice such that the 1d slice provided by slice_at_axis, slices along the dimension provided."""
    from_end = False
    if axis < 0:  # choosing axis at the end
        from_end = True
        axis = -1 - axis
    # Pad the slice with slice(None)
    slice_padding = axis * (slice(None),)
    if from_end:
        return (Ellipsis, slice_at_axis, *slice_padding)

    return (*slice_padding, slice_at_axis)


def slice_ignoring_axes(
    old_slice: Sequence[slice | IntLike_co | None], axes: tuple[IntLike_co, ...]
) -> tuple[slice | IntLike_co | None, ...]:
    """
    Given a slice, insert slice(None) everywhere given in axes.

    Parameters
    ----------
    slice : list[slice  |  _IntLike_co  |  None]
        _description_
    axes : tuple[_IntLike_co]
        _description_

    Returns
    -------
    list[slice | _IntLike_co | None]
        _description_
    """
    old_slice = list(old_slice)
    for axis in sorted(int(a) for a in axes):
        old_slice.insert(axis, slice(None))
    return tuple(old_slice)


def get_position_in_sorted(axes: tuple[IntLike_co, ...]) -> tuple[IntLike_co, ...]:
    """
    Given a list of axes get the index in the sorted list.

    ie 2,4,1,3 -> 1,3,0,2

    Parameters
    ----------
    axes : _AX0Inv

    Returns
    -------
    _AX0Inv
    """
    return tuple(np.argsort(np.argsort(axes)))  # type: ignore Tuple is an array-like


@overload
def get_data_in_axes(
    data: np.ndarray[_S0Inv, _DTInv],
    axes: tuple[IntLike_co],
    idx: SingleStackedIndexLike,
) -> np.ndarray[tuple[int], _DTInv]:
    ...


@overload
def get_data_in_axes(
    data: np.ndarray[_S0Inv, _DTInv],
    axes: tuple[IntLike_co, IntLike_co],
    idx: SingleStackedIndexLike,
) -> np.ndarray[tuple[int, int], _DTInv]:
    ...


@overload
def get_data_in_axes(
    data: np.ndarray[_S0Inv, _DTInv],
    axes: tuple[IntLike_co, ...],
    idx: SingleStackedIndexLike,
) -> np.ndarray[tuple[int, ...], _DTInv]:
    ...


def get_data_in_axes(
    data: np.ndarray[_S0Inv, _DTInv],
    axes: tuple[IntLike_co, ...] | tuple[IntLike_co],
    idx: SingleStackedIndexLike,
) -> np.ndarray[tuple[int, ...], _DTInv] | np.ndarray[tuple[int], _DTInv]:
    """
    Given a slice, insert slice(None) everywhere given in axes.

    Parameters
    ----------
    slice : list[slice  |  _IntLike_co  |  None]
        slice excluding axes
    axes : tuple[_IntLike_co]
        axes to insert slice(None)

    Returns
    -------
    list[slice | _IntLike_co | None]
    """
    return np.transpose(
        data[slice_ignoring_axes(idx, axes)], get_position_in_sorted(axes)
    )  # type: ignore[no-any-return]


_LInv = TypeVar("_LInv", bound=int)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def calculate_cumulative_distances_along_path(
    path: np.ndarray[tuple[Literal[3], _LInv], np.dtype[np.int_]],
    coordinates: np.ndarray[
        tuple[Literal[3], _L0Inv, _L1Inv, _L2Inv], np.dtype[np.float64]
    ],
) -> np.ndarray[tuple[_LInv], np.dtype[np.float64]]:
    """
    Get a list of cumulative distances along a path ([x_coord, y_coord, z_coord]) given a grid of coordinates.

    Parameters
    ----------
    path : np.ndarray[tuple[Literal[3], _L], np.dtype[np.float_]]
        Path for which to calculate the distance along (as a list for each coordinate)
    coordinates : np.ndarray[tuple[_L0, _L1, _L2], np.dtype[np.float_]]
        Coordinate grid, with the same dimension as the coordinates given in the path

    Returns
    -------
    np.ndarray[tuple[_L], np.dtype[np.float_]]
        The distance from the starting coordinate along the given path
    """
    path_coordinates = coordinates[:, *path]
    distances = np.linalg.norm(
        path_coordinates[:, :-1] - path_coordinates[:, 1:], axis=0
    )
    cum_distances = np.cumsum(distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


Measure = Literal["real", "imag", "abs", "angle"]


def get_measured_data(
    data: np.ndarray[_S0Inv, _DTInv],
    measure: Measure,
) -> np.ndarray[_S0Inv, np.dtype[np.float64]]:
    """
    Transform data with the given measure.

    Parameters
    ----------
    data : np.ndarray[_S0Inv, np.dtype[np.complex_]]
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;]


    Returns
    -------
    np.ndarray[_SInv, np.dtype[np.float_]]
    """
    match measure:
        case "real":
            return np.real(data)  # type: ignore[no-any-return]
        case "imag":
            return np.imag(data)  # type: ignore[no-any-return]
        case "abs":
            return np.abs(data)  # type: ignore[no-any-return]
        case "angle":
            return np.unwrap(np.angle(data))  # type: ignore[no-any-return]
