from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.util.plot import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from .point_potential import PointPotential3d

_L0Inv = TypeVar("_L0Inv", bound=int)


def get_point_potential_xy_locations(
    potential: PointPotential3d[_L0Inv],
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]:
    """
    Get xy locations in a point potential.

    Parameters
    ----------
    potential : PointPotential[_L0Inv]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    return np.array(  # type: ignore[no-any-return]
        [
            (x, y)
            for x in np.unique(potential["x_points"])
            for y in np.unique(
                np.array(potential["y_points"])[potential["x_points"] == x]
            )
        ]
    ).T


def plot_point_potential_location_xy(
    potential: PointPotential3d[Any], *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the xy locations of a point potential.

    Parameters
    ----------
    potential : PointPotential[Any]
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    points = get_point_potential_xy_locations(potential)
    (line,) = ax.plot(*points)  # type: ignore lib
    line.set_marker("x")
    line.set_linestyle("")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")  # type: ignore lib
    ax.set_ylabel("y")  # type: ignore lib
    ax.set_title("Plot of x,y points in the potential")  # type: ignore lib
    return fig, ax, line


def plot_point_potential_all_z(
    potential: PointPotential3d[Any], *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """
    Plot the z dependance of all xy locations in the point potential.

    Parameters
    ----------
    potential : PointPotential[Any]
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    points = get_point_potential_xy_locations(potential)
    for x, y in points.T:
        idx = np.argwhere(
            np.logical_and(
                potential["x_points"] == x,
                potential["y_points"] == y,
            )
        )
        points = potential["points"][idx]
        z_points = potential["z_points"][idx]
        (line,) = ax.plot(z_points, points)  # type: ignore lib
        line.set_label(f"{x:.2}, {y:.2}")

    ax.set_title("Plot of Energy against z for each (x,y) point")  # type: ignore lib
    ax.set_xlabel("z")  # type: ignore lib
    ax.set_ylabel("Energy /J")  # type: ignore lib
    return fig, ax
