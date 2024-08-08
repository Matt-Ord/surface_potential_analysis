from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)
from surface_potential_analysis.util.plot import get_figure

from .plot import plot_state_1d_k, plot_state_1d_x

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _SB0 = TypeVar("_SB0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def plot_states_1d_x(
    states: StateVectorList[_B0, _SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot all states in a StateVectorList.

    Parameters
    ----------
    states : StateVectorList[_B0Inv, _L0Inv]
    axis : int, optional
        axis to plot along, by default 0
    idx : SingleStackedIndexLike | None, optional
        index in axes perpendicular to axis, by default None
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)
    return fig, ax


def plot_states_1d_k(
    states: StateVectorList[_B0, _SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot all states in a StateVectorList.

    Parameters
    ----------
    states : StateVectorList[_B0Inv, _L0Inv]
    axis : int, optional
        axis to plot along, by default 0
    idx : SingleStackedIndexLike | None, optional
        index in axes perpendicular to axis, by default None
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    for state in state_vector_list_into_iter(states):
        plot_state_1d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)
    return fig, ax
