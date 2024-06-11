from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.probability_vector.plot import (
    plot_probability_against_time,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    ProbabilityVectorList,
    average_probabilities,
    sum_probabilities,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.basis.time_basis_like import (
        BasisWithTimeLike,
        FundamentalTimeBasis,
    )
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.util.plot import Scale

    _B1Inv = TypeVar("_B1Inv", bound=TunnellingSimulationBasis[Any, Any, Any])
    _B0Inv = TypeVar("_B0Inv", bound=BasisWithTimeLike[int, int])
    _B0StackedInv = TypeVar(
        "_B0StackedInv",
        bound=TupleBasisLike[Any, FundamentalTimeBasis[int]],
    )


def plot_probability_per_band(
    probability: ProbabilityVectorList[_B0Inv, _B1Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the occupation of each band in the simulation.

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
    probability_per_band = sum_probabilities(probability, (0, 1))
    fig, ax, lines = plot_probability_against_time(
        probability_per_band, ax=ax, scale=scale
    )

    for n, line in enumerate(lines):
        line.set_label(f"band {n}")

    ax.legend()
    ax.set_title("Plot of occupation of each band against time")
    return fig, ax, lines


def plot_average_probability_per_band(
    probability: ProbabilityVectorList[_B0StackedInv, _B1Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot average probability of each band.

    Parameters
    ----------
    probability : list[ProbabilityVectorList[_B0StackedInv, _L0Inv]]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, list[Line2D]]
    """
    averaged = average_probabilities(probability, axis=(0,))
    fig, ax, lines = plot_probability_per_band(
        {
            "basis": TupleBasis(averaged["basis"][0][0], averaged["basis"][1]),
            "data": averaged["data"],
        },
        ax=ax,
        scale=scale,
    )

    ax.set_title(
        "Plot of occupation of each band against time,\n"
        f"averaged over {len(probability)} repeats"
    )
    return fig, ax, lines


def plot_probability_per_site(
    probability: ProbabilityVectorList[_B0Inv, _B1Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
     Plot the occupation of each site in the simulation.

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
    probability_per_site = sum_probabilities(probability, (2,))
    fig, ax, lines = plot_probability_against_time(
        probability_per_site, ax=ax, scale=scale
    )

    for n, line in enumerate(lines):
        line.set_label(f"site {n}")

    ax.legend()
    ax.set_title("Plot of occupation of each site against time")
    return fig, ax, lines
