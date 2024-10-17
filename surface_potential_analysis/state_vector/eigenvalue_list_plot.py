from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.constants import Boltzmann  # type:ignore lib

from surface_potential_analysis.util.plot import get_figure
from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.legacy import BasisLike, BasisWithTimeLike
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        StatisticalDiagonalOperator,
    )
    from surface_potential_analysis.util.plot import Scale

    from .eigenstate_calculation import EigenstateList

    _B0_co = TypeVar(
        "_B0_co",
        bound=BasisWithTimeLike,
        covariant=True,
    )


def plot_eigenvalue_against_time(
    eigenvalues: SingleBasisDiagonalOperator[_B0_co]
    | StatisticalDiagonalOperator[_B0_co, _B0_co],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the eigenvalues against time.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_N0Inv]
        list of eigenvalues to plot
    times : np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
        Times for which to plot the eigenvalues
    ax : Axes | None, optional
        Plot axis, by default None
    measure : Measure, optional
        plot measure, by default "abs"
    scale : Scale, optional
        plot y scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    data = get_measured_data(eigenvalues["data"], measure)
    times = eigenvalues["basis"][0].times
    standard_deviation = eigenvalues.get("standard_deviation", None)
    if isinstance(standard_deviation, np.ndarray):
        line = ax.errorbar(times, data, yerr=standard_deviation).lines[0]  # type:ignore lib
    else:
        (line,) = ax.plot(times, data)  # type:ignore lib
    ax.set_ylabel("Eigenvalue")  # type:ignore lib
    ax.set_yscale(scale)  # type:ignore lib
    ax.set_xlabel("time /s")  # type:ignore lib
    ax.set_xlim(times[0], times[-1])
    return fig, ax, line


def plot_eigenstate_occupations(
    eigenstates: EigenstateList[BasisLike, BasisLike],
    temperature: float,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike, BasisLike]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    energies = eigenstates["eigenvalue"]
    energies -= np.min(energies)
    occupation = np.exp(-np.abs(energies) / (temperature * Boltzmann))
    occupation /= np.sum(occupation)

    a_sort = np.argsort(energies)
    energies = energies[a_sort]
    occupation = occupation[a_sort]

    (line,) = ax.plot(energies, occupation)  # type:ignore lib

    ax.set_yscale(scale)  # type:ignore lib
    ax.set_ylabel("Occupation")  # type:ignore lib
    ax.set_xlabel("Energy /J")  # type:ignore lib
    ax.set_title("Plot of Occupation against Energy")  # type:ignore lib

    return fig, ax, line


def plot_eigenvalues(
    eigenstates: EigenstateList[BasisLike, BasisLike],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike, BasisLike]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)
    a_sort = np.argsort(np.abs(eigenstates["eigenvalue"]))
    energies = get_measured_data(eigenstates["eigenvalue"], measure)[a_sort]

    (line,) = ax.plot(energies)  # type:ignore lib

    ax.set_yscale(scale)  # type:ignore lib
    ax.set_ylabel(f"{measure} Eigenvalue")  # type:ignore lib
    ax.set_title("Plot of Eigenvalues")  # type:ignore lib

    return fig, ax, line
