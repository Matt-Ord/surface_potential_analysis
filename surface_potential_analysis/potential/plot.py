from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_basis,
    convert_potential_to_position_basis,
)
from surface_potential_analysis.stacked_basis.util import (
    calculate_cumulative_x_distances_along_path,
)
from surface_potential_analysis.util.plot import (
    animate_data_through_surface_x,
    get_figure,
    plot_data_1d_x,
    plot_data_2d_x,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)

from ._comparison_points import (
    get_100_comparison_points_x2,
    get_111_comparison_points_x2,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import FundamentalPositionBasis3d
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.potential.potential import Potential
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

# ruff: noqa: PLR0913


def plot_potential_1d_x(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the given axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int]
    axis : Literal[0, 1, 2, -1, -2, -3]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax, line = plot_data_1d_x(
        potential["basis"],
        potential["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure="real",
    )
    ax.set_ylabel("Energy /J")
    return fig, ax, line


def plot_potential_1d_comparison(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    comparison_points: Mapping[
        str, tuple[tuple[int, int], Literal[0, 1, 2, -1, -2, -3]]
    ],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the potential in 1d at the given comparison points.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    comparison_points : Mapping[ str, tuple[tuple[int, int], Literal[0, 1, 2,
        map of axis label to ((idx), axis) to pass to plot_potential_1d
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)
    lines: list[Line2D] = []
    for label, (idx, axis) in comparison_points.items():
        _, _, line = plot_potential_1d_x(potential, (axis,), idx, ax=ax, scale=scale)
        line.set_label(label)
        lines.append(line)
    ax.legend()
    return fig, ax, lines


def plot_potential_1d_x2_comparison_111(
    potential: Potential[
        TupleBasisWithLengthLike[
            FundamentalPositionBasis3d[_L0Inv],
            FundamentalPositionBasis3d[_L1Inv],
            FundamentalPositionBasis3d[_L2Inv],
        ]
    ],
    offset: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the potential along the x2 at the relevant 111 sites.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    offset : tuple[int, int]
        index of the fcc site
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    (s0, s1, _) = BasisUtil(potential["basis"]).shape
    points = get_111_comparison_points_x2((s0, s1), offset)
    comparison_points = {k: (v, 2) for (k, v) in points.items()}
    return plot_potential_1d_comparison(
        potential,
        comparison_points,
        ax=ax,
        scale=scale,  # type: ignore[arg-type]
    )


def plot_potential_1d_x2_comparison_100(
    potential: Potential[
        TupleBasisWithLengthLike[
            FundamentalPositionBasis3d[_L0Inv],
            FundamentalPositionBasis3d[_L1Inv],
            FundamentalPositionBasis3d[_L2Inv],
        ]
    ],
    offset: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the potential along the x2 at the relevant 100 sites.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    offset : tuple[int, int]
        index of the fcc site
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    (s0, s1, _) = BasisUtil(potential["basis"]).shape
    points = get_100_comparison_points_x2((s0, s1), offset)
    comparison_points = {k: (v, 2) for (k, v) in points.items()}
    return plot_potential_1d_comparison(
        potential,
        comparison_points,
        ax=ax,
        scale=scale,  # type: ignore[arg-type]
    )


def plot_potential_2d_x(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to z_axis at idx along z_axis.

    Parameters
    ----------
    potential : Potential[_B0Inv]
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to direction of plot
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_potential_to_position_basis(potential)

    return plot_data_2d_x(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure="real",
    )


def plot_potential_difference_2d_x(
    potential0: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    potential1: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two potentials in 2d, perpendicular to z_axis.

    Parameters
    ----------
    potential0 : Potential[_L0Inv, _L1Inv, _L2Inv]
    potential1 : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2,
        axis perpendicular to which to plot the data
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted_1 = convert_potential_to_basis(potential1, potential0["basis"])
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]] = {
        "basis": potential0["basis"],
        "data": potential0["data"] - converted_1["data"],
    }
    return plot_potential_2d_x(potential, axes, idx, ax=ax, scale=scale)


def animate_potential_3d_x(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate potential in the direction perpendicular to z_axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis to animate through
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    converted = convert_potential_to_position_basis(potential)
    points = potential["data"].reshape(converted["basis"].shape)
    fig, ax, ani = animate_data_through_surface_x(  # type: ignore[misc]
        potential["basis"],
        points,
        axes,
        idx,
        ax=ax,
        scale=scale,
        clim=clim,
        measure=measure,
    )
    ax.set_title(f"Animation of the potential perpendicular to the x{axes[2]} axis")

    return fig, ax, ani


def animate_potential_difference_3d_x(
    potential0: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    potential1: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the absolute relative difference between two potentials in 2d, perpendicular to z_axis.

    Parameters
    ----------
    potential0 : Potential[_L0Inv, _L1Inv, _L2Inv]
    potential1 : Potential[_L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2,
        axis perpendicular to which to plot the data
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted_1 = convert_potential_to_basis(potential1, potential0["basis"])
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]] = {
        "basis": potential0["basis"],
        "data": np.abs((potential0["data"] - converted_1["data"]) / potential0["data"]),
    }
    return animate_potential_3d_x(potential, axes, idx, ax=ax, scale=scale, clim=clim)


def plot_potential_along_path(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the given path.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[_ND], int], np.dtype[np.int_]]
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    converted = convert_potential_to_position_basis(potential)
    util = BasisUtil(converted["basis"])

    data = get_measured_data(
        np.take(
            converted["data"].reshape(util.shape),
            np.ravel_multi_index(path, util.shape, mode="wrap"),
        ),
        measure,
    )
    distances = calculate_cumulative_x_distances_along_path(
        converted["basis"], path, wrap_distances=wrap_distances
    )

    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    return fig, ax, line


def get_minimum_path(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    axis: int = 0,
) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
    """
    Find the minimum path, taking the smallest value along the given axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
        Potential to take the minimum of
    path : np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]
        Path for the coordinates perpendicular to axis
    axis : Literal[0, 1, 2,-1,-2,-3] optional
        Axis over which to search for the minimum of potential, by default -1

    Returns
    -------
    np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        The complete path after finding the minimum at each coordinate
    """
    axis = axis % potential["basis"].ndim
    min_idx = np.argmin(potential["data"].reshape(potential["basis"].shape), axis=axis)
    wrapped = np.ravel_multi_index(path, min_idx.shape, mode="wrap")
    return np.insert(path, axis, min_idx.flat[wrapped], axis=0)


def plot_potential_minimum_along_path(
    potential: Potential[StackedBasisWithVolumeLike[Any, Any, Any]],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    axis: int = 0,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the path, taking the minimum in the axis direction.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]
        path perpendicular to axis
    axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis to take the minimum along, by default -1
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    full_path = get_minimum_path(potential, path, axis)
    return plot_potential_along_path(potential, full_path, ax=ax, scale=scale)
