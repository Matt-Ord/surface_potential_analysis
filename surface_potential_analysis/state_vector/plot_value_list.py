from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import scipy.stats

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.time_basis_like import BasisWithTimeLike
from surface_potential_analysis.util.plot import (
    Scale,
    get_figure,
    plot_data_1d,
)
from surface_potential_analysis.util.util import get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList
    from surface_potential_analysis.util.util import (
        Measure,
    )

_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])
_B0 = TypeVar("_B0", bound=BasisLike[int, int])


def plot_value_list_against_nx(
    values: ValueList[_B0],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the data against time.

    Parameters
    ----------
    values : ValueList[_AX0Inv]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax, line = plot_data_1d(
        values["data"],
        np.arange(values["basis"].n).astype(np.float64),
        scale=scale,
        measure=measure,
        ax=ax,
    )
    return fig, ax, line


def plot_value_list_against_time(
    values: ValueList[_BT0],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the data against time.

    Parameters
    ----------
    values : ValueList[_AX0Inv]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax, line = plot_data_1d(
        values["data"], values["basis"].times, scale=scale, measure=measure, ax=ax
    )

    ax.set_xlabel("Times /s")
    return fig, ax, line


def plot_split_value_list_against_time(
    values: ValueList[TupleBasisLike[_B0, _BT0]],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes]:
    """
    Plot the data against time, split by _B0.

    Parameters
    ----------
    values : ValueList[_AX0Inv]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    stacked = values["data"].reshape(values["basis"].shape)
    cumulative = np.cumsum(stacked, axis=0)
    for band_data in cumulative:
        fig, ax, _line = plot_data_1d(
            band_data,
            values["basis"][1].times,
            scale=scale,
            measure=measure,
            ax=ax,
        )

    ax.set_xlabel("Times /s")
    return fig, ax


def plot_all_value_list_against_time(
    values: ValueList[TupleBasisLike[Any, _BT0]],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes]:
    """
    Plot all value lists against time.

    Parameters
    ----------
    values : ValueList[TupleBasis[Any, _AX0Inv]]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    for data in values["data"].reshape(values["basis"].shape):
        plot_value_list_against_time(
            {"basis": values["basis"][1], "data": data},
            ax=ax,
            scale=scale,
            measure=measure,
        )
    return fig, ax


def plot_average_value_list_against_time(
    values: ValueList[TupleBasisLike[Any, _BT0]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot all value lists against time.

    Parameters
    ----------
    values : ValueList[TupleBasis[Any, _AX0Inv]]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(values["data"], measure).reshape(
        values["basis"].shape
    )
    average_data = np.average(measured_data, axis=0)
    fig, ax, line = plot_data_1d(
        average_data, values["basis"][1].times, scale=scale, measure=measure, ax=ax
    )
    std_data = np.std(measured_data, axis=0) / np.sqrt(values["basis"].shape[0])
    ax.fill_between(
        values["basis"][1].times,
        average_data - std_data,
        average_data + std_data,
        alpha=0.2,
    )

    ax.set_xlabel("Times /s")
    ax.set_ylabel("Distance /m")
    return fig, ax, line


Distribution = Literal["normal", "exponential normal", "skew normal"]


def plot_value_list_distribution(
    values: ValueList[Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    distribution: Distribution | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of values in a list.

    Parameters
    ----------
    values : ValueList[TupleBasis[Any, _AX0Inv]]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(values["data"], measure)
    std = np.std(measured_data).item()
    average = np.average(measured_data).item()
    x_range = (
        (average - 4 * std, average + 4 * std)
        if distribution is not None
        else (np.min(measured_data).item(), np.max(measured_data).item())
    )
    n_bins = np.max([11, values["data"].size // 500]).item()

    ax.hist(measured_data, n_bins, x_range, density=True)

    if distribution == "normal":
        points = np.linspace(*x_range, 1000)
        (line,) = ax.plot(points, scipy.stats.norm.pdf(points, loc=average, scale=std))
        line.set_label("normal fit")

    if distribution == "exponential normal":
        points = np.linspace(*x_range, 1000)
        fit = scipy.stats.exponnorm.fit(measured_data)
        (line,) = ax.plot(points, scipy.stats.exponnorm.pdf(points, *fit))
        line.set_label("exponential normal fit")

    if distribution == "skew normal":
        points = np.linspace(*x_range, 1000)
        fit = scipy.stats.skewnorm.fit(measured_data)
        (line,) = ax.plot(points, scipy.stats.skewnorm.pdf(points, *fit))
        line.set_label("skew normal fit")

    ax.set_ylabel("Occupation")
    return fig, ax
