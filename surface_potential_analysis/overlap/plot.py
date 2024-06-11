from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
    convert_overlap_to_position_basis,
)
from surface_potential_analysis.stacked_basis.util import (
    calculate_cumulative_k_distances_along_path,
)
from surface_potential_analysis.util.plot import (
    get_figure,
    plot_data_2d_k,
    plot_data_2d_x,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import (
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import BasisWithLengthLike
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale

    from .overlap import SingleOverlap

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

    _B0 = TypeVar("_B0", bound=BasisWithLengthLike[Any, Any, Literal[3]])
    FundamentalMomentumOverlap = SingleOverlap[
        TupleBasisLike[
            FundamentalTransformedPositionBasis[_L0Inv, Literal[3]],
            FundamentalTransformedPositionBasis[_L1Inv, Literal[3]],
            FundamentalTransformedPositionBasis[_L2Inv, Literal[3]],
        ]
    ]


# ruff: noqa: PLR0913


def plot_overlap_2d_x(
    overlap: SingleOverlap[TupleBasisLike[*tuple[_B0, ...]]],
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the overlap in momentum space.

    Parameters
    ----------
    overlap : OverlapPosition
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2,-1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_overlap_to_position_basis(overlap)
    return plot_data_2d_x(
        converted["basis"][0],
        converted["data"],
        axes,
        idx,
        ax=ax,
        measure=measure,
        scale=scale,
    )


def plot_overlap_2d_k(
    overlap: SingleOverlap[TupleBasisLike[*tuple[_B0, ...]]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the overlap in momentum space.

    Parameters
    ----------
    overlap : OverlapMomentum
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2,-1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_overlap_to_momentum_basis(overlap)
    return plot_data_2d_k(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        measure=measure,
        scale=scale,
    )


def plot_overlap_along_path_k(
    overlap: SingleOverlap[TupleBasisLike[*tuple[_B0, ...]]],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the overlap transform along the given path.

    Parameters
    ----------
    overlap : OverlapMomentum
    path : np.ndarray[tuple[3, int], np.dtype[np.int_]]
        path, as a list of index for each coordinate
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)
    converted = convert_overlap_to_momentum_basis(overlap)

    points = converted["data"].reshape(converted["basis"].shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_k_distances_along_path(
        converted["basis"][0], path, wrap_distances=wrap_distances
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    ax.set_xlabel("Distance along path")
    return fig, ax, line


def plot_overlap_along_k_diagonal(
    overlap: SingleOverlap[TupleBasisLike[*tuple[_B0, ...]]],
    k2_ind: int = 0,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the overlap transform in the x0, x1 diagonal.

    Parameters
    ----------
    overlap : OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]
    k2_ind : int, optional
        index in the k2 direction, by default 0
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    path = np.array(
        [[i, i, k2_ind] for i in range(overlap["basis"].fundamental_shape[0])]
    ).T

    return plot_overlap_along_path_k(overlap, path, measure=measure, scale=scale, ax=ax)


def plot_overlap_along_k0(
    overlap: SingleOverlap[TupleBasisLike[*tuple[_B0, ...]]],
    k1_ind: int = 0,
    k2_ind: int = 0,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot overlap transform in the k0 direction.

    Parameters
    ----------
    overlap : OverlapMomentum[_L0Inv,_L1Inv,_L2Inv]
    k1_ind : int, optional
        index along k1, by default 0
    k2_ind : int, optional
        index along k2, by default 0
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    path = np.array(
        [[i, k1_ind, k2_ind] for i in range(overlap["basis"].fundamental_shape[0])]
    ).T

    return plot_overlap_along_path_k(overlap, path, measure=measure, scale=scale, ax=ax)
