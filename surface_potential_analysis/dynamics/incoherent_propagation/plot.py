from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from surface_potential_analysis.util.plot import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from surface_potential_analysis.dynamics.incoherent_propagation.isf import (
        RateDecomposition,
    )

_N0Inv = TypeVar("_N0Inv", bound=int)
_L0Inv = TypeVar("_L0Inv", bound=int)


def plot_rate_decomposition_against_temperature(
    rates: list[RateDecomposition[_L0Inv]],
    temperatures: np.ndarray[tuple[_N0Inv], np.dtype[np.float64]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the individual rates in the simulation against temperature.

    Parameters
    ----------
    rates : list[RateDecomposition[_L0Inv]]
    temperatures : np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    rate_constants = -np.real([rate.eigenvalues for rate in rates])
    coefficients = np.abs([rate.coefficients for rate in rates])
    relevant_rates = np.argsort(coefficients, axis=-1)
    sorted_rates = np.take_along_axis(rate_constants, relevant_rates, axis=-1)[:, ::-1]

    coefficients = np.abs([rate.coefficients for rate in rates])
    for i in range(50):
        ax.plot(temperatures, sorted_rates[:, i])  # type: ignore lib
    return fig, ax
