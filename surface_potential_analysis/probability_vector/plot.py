from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.probability_vector.conversion import (
    convert_probability_vector_to_momentum_basis,
    convert_probability_vector_to_position_basis,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    ProbabilityVector,
    sum_probabilities,
)
from surface_potential_analysis.util.plot import (
    get_figure,
    plot_data_1d_k,
    plot_data_1d_x,
    plot_data_2d_k,
    plot_data_2d_x,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.basis.time_basis_like import BasisWithTimeLike
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVectorList,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

if TYPE_CHECKING:
    _B0 = TypeVar("_B0", bound=BasisWithTimeLike[int, int])
    _B1 = TypeVar("_B1", bound=BasisLike[int, int])
    _SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])


def plot_probability_against_time(
    probability: ProbabilityVectorList[_B0, _B1],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot probability against time.

    Parameters
    ----------
    probability : ProbabilityVectorList[_B0Inv, _L0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, list[Line2D]]
    """
    fig, ax = get_figure(ax)
    times = probability["basis"][0].times
    data = probability["data"].reshape(probability["basis"].shape)

    lines: list[Line2D] = []
    for n in range(probability["basis"].shape[1]):
        (line,) = ax.plot(times, (data[:, n]))
        lines.append(line)

    ax.set_title("Plot of probability against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    ax.set_yscale(scale)
    return fig, ax, lines


def plot_total_probability_against_time(
    probability: ProbabilityVectorList[_B0, _SB0],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot total probability against time.

    Parameters
    ----------
    probability : ProbabilityVectorList[_B0Inv, _L0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, list[Line2D]]
    """
    summed = sum_probabilities(probability)
    return plot_probability_against_time(summed, ax=ax, scale=scale)


# ruff: noqa: PLR0913


def plot_probability_1d_k(
    state: ProbabilityVector[TupleBasisLike[*tuple[Any, ...]]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given axis.

    Parameters
    ----------
    state : StateVector[_B0Inv]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = convert_probability_vector_to_momentum_basis(state)

    fig, ax, line = plot_data_1d_k(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("Probability")
    return fig, ax, line


def plot_probability_1d_x(
    state: ProbabilityVector[TupleBasisLike[*tuple[Any, ...]]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given axis.

    Parameters
    ----------
    state : StateVector[_B0Inv]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = convert_probability_vector_to_position_basis(state)

    fig, ax, line = plot_data_1d_x(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("Probability")
    return fig, ax, line


def plot_probability_2d_k(
    state: ProbabilityVector[TupleBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the probability in 2D in the k basis.

    Parameters
    ----------
    state : ProbabilityVector[TupleBasisLike
    axes : tuple[int, int], optional
        axes to plot, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_probability_vector_to_momentum_basis(state)

    fig, ax, mesh = plot_data_2d_k(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("Probability")
    return fig, ax, mesh


def plot_probability_2d_x(
    state: ProbabilityVector[TupleBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
     Plot the probability in 2D in the x basis.

    Parameters
    ----------
    state : ProbabilityVector[TupleBasisLike
    axes : tuple[int, int], optional
        axes to plot, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_probability_vector_to_position_basis(state)

    fig, ax, mesh = plot_data_2d_x(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("Probability")
    return fig, ax, mesh
