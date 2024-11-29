from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.plot import get_figure

from ._comparison_points import (
    get_100_comparison_points_x2,
    get_111_comparison_points_x2,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.potential.potential import UnevenPotential3d
    from surface_potential_analysis.util.plot import Scale

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_uneven_potential_z(
    potential: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an uneven potential in 1d, in the x2 direction.

    Parameters
    ----------
    potential : UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int]
        [x0,x2] index
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    coordinates = potential["basis"][2].z_points
    util = BasisUtil(potential["basis"])
    data = potential["data"].reshape(util.shape)[*idx, :]

    (line,) = ax.plot(coordinates, data)  # type: ignore lib
    ax.set_xlabel("Z axis")  # type: ignore lib
    ax.set_ylabel("Energy /J")  # type: ignore lib
    ax.set_yscale(scale)  # type: ignore lib
    return fig, ax, line


def plot_uneven_potential_z_comparison(
    potential: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv],
    comparison_points: dict[str, tuple[int, int]],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot a nuneven potential in 1d along the x2 direction.

    Parameters
    ----------
    potential : UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
    comparison_points : dict[str, tuple[int, int]]
        dictionary mapping plot label to [x0 idx, x1 idx]
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
    for label, idx in comparison_points.items():
        (_, _, line) = plot_uneven_potential_z(potential, idx, ax=ax, scale=scale)
        line.set_label(label)
        lines.append(line)
    ax.legend()  # type: ignore lib
    return fig, ax, lines


def plot_uneven_potential_z_comparison_111(
    potential: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv],
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
    shape = (potential["basis"][0].size, potential["basis"][1].size)
    points = get_111_comparison_points_x2(shape, offset)
    return plot_uneven_potential_z_comparison(potential, points, ax=ax, scale=scale)


def plot_uneven_potential_z_comparison_100(
    potential: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv],
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
    shape = (potential["basis"][0].size, potential["basis"][1].size)
    points = get_100_comparison_points_x2(shape, offset)
    return plot_uneven_potential_z_comparison(potential, points, ax=ax, scale=scale)
