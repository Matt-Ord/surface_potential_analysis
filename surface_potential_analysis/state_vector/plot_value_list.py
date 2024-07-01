from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.util.plot import Scale, _get_scale_with_lim, get_figure
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.state_vector.eigenstate_collection import ValueList

_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def plot_value_list_against_time(
    values: ValueList[_AX0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data along axes in the k basis.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int], optional
        axes to plot in, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    data = values["data"]
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(data, measure)

    (line,) = ax.plot(values["basis"].times, measured_data)
    if measure == "abs":
        ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yscale(_get_scale_with_lim(scale, ax.get_ylim()))
    return fig, ax, line
