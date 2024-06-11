from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.scale import LinearScale, ScaleBase, SymmetricalLogScale

from surface_potential_analysis.stacked_basis.util import (
    get_k_coordinates_in_axes,
    get_max_idx,
    get_x_coordinates_in_axes,
)

from .util import Measure, get_data_in_axes, get_measured_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.types import SingleStackedIndexLike


Scale = Literal["symlog", "linear"]


def get_figure(ax: Axes | None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created

    Parameters
    ----------
    ax : Axes | None

    Returns
    -------
    tuple[Figure, Axes]

    """
    if ax is None:
        return cast(tuple[Figure, Axes], plt.subplots())  # type: ignore plt.subplots Unknown type

    fig = ax.get_figure()
    if fig is None:
        fig = plt.figure()  # type: ignore plt.figure Unknown type
        ax.set_figure(fig)
    return fig, ax


def _get_default_lim(
    measure: Measure, data: np.ndarray[Any, np.dtype[np.float64]]
) -> tuple[float, float]:
    if measure == "abs":
        return (0, float(np.max(data)))
    return (float(np.min(data)), float(np.max(data)))


def _get_lim(
    lim: tuple[float | None, float | None],
    measure: Measure,
    data: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float, float]:
    (default_min, default_max) = _get_default_lim(measure, data)
    l_max = default_max if lim[1] is None else lim[1]
    l_min = default_min if lim[0] is None else lim[0]
    return (l_min, l_max)


def _get_norm_with_lim(
    scale: Scale,
    lim: tuple[float, float],
) -> Normalize:
    match scale:
        case "linear":
            return Normalize(vmin=lim[0], vmax=lim[1])
        case "symlog":
            max_abs = max([np.abs(lim[0]), np.abs(lim[1])])
            return SymLogNorm(
                vmin=lim[0],
                vmax=lim[1],
                linthresh=1 if max_abs <= 0 else 1e-3 * max_abs,  # type: ignore No parameter named "linthresh"
            )


def _get_scale_with_lim(
    scale: Scale,
    lim: tuple[float, float],
) -> ScaleBase:
    match scale:
        case "linear":
            return LinearScale(axis=None)
        case "symlog":
            max_abs = max([np.abs(lim[0]), np.abs(lim[1])])
            return SymmetricalLogScale(
                axis=None,
                linthresh=1 if max_abs <= 0 else 1e-3 * max_abs,
            )


# https://stackoverflow.com/questions/49382105/set-different-margins-for-left-and-right-side
def _set_ymargin(ax: Axes, bottom: float = 0.0, top: float = 0.3) -> None:
    ax.set_autoscale_on(b=True)
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = lim[1] - lim[0]
    bottom = lim[0] - delta * bottom
    top = lim[1] + delta * top
    ax.set_ylim(bottom, top)


def plot_data_1d_k(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    axes: tuple[int,] = (0,),
    idx: SingleStackedIndexLike | None = None,
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
    fig, ax = get_figure(ax)
    idx = get_max_idx(basis, data, axes) if idx is None else idx

    coordinates = get_k_coordinates_in_axes(basis, axes, idx)
    data_in_axis = get_data_in_axes(data.reshape(basis.shape), axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    shifted_data = np.fft.fftshift(measured_data)
    shifted_coordinates = np.fft.fftshift(coordinates)

    (line,) = ax.plot(shifted_coordinates[0], shifted_data)
    ax.set_xlabel(f"k{(axes[0] % 3)} axis")
    ax.set_xmargin(0)
    _set_ymargin(ax, 0, 0.05)
    if measure == "abs":
        ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yscale(_get_scale_with_lim(scale, ax.get_ylim()))
    return fig, ax, line


def plot_data_1d_x(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    axes: tuple[int,] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data along axes in the x basis.

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
    fig, ax = get_figure(ax)
    idx = get_max_idx(basis, data, axes) if idx is None else idx

    coordinates = get_x_coordinates_in_axes(basis, axes, idx)
    data_in_axis = get_data_in_axes(data.reshape(basis.shape), axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    (line,) = ax.plot(coordinates[0], measured_data)
    ax.set_xlabel(f"x{(axes[0] % 3)} axis")
    ax.set_xmargin(0)
    _set_ymargin(ax, 0, 0.05)
    if measure == "abs":
        ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yscale(_get_scale_with_lim(scale, ax.get_ylim()))
    return fig, ax, line


def plot_data_2d_k(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the data in a 2d slice in k along the given axis.

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
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = get_figure(ax)
    idx = get_max_idx(basis, data, axes) if idx is None else idx

    coordinates = get_k_coordinates_in_axes(basis, axes, idx)
    data_in_axis = get_data_in_axes(data.reshape(basis.shape), axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    shifted_data = np.fft.fftshift(measured_data)
    shifted_coordinates = np.fft.fftshift(coordinates, axes=(1, 2))

    mesh = ax.pcolormesh(*shifted_coordinates, shifted_data, shading="nearest")
    clim = _get_lim((None, None), measure, shifted_data)
    norm = _get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel(f"k{axes[1]} axis")
    ax.text(
        0.05,
        0.95,
        f"k = {idx}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    return fig, ax, mesh


def plot_data_2d_x(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the data in 2d along the x axis in the given basis.

    Parameters
    ----------
    basis : TupleBasisLike
        basis to interpret the data in
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
        plot data
    axes : tuple[int, int, int], optional
        axes to plot in, by default (0, 1, 2)
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
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = get_figure(ax)
    idx = get_max_idx(basis, data, axes) if idx is None else idx

    coordinates = get_x_coordinates_in_axes(basis, axes, idx)
    data_in_axis = get_data_in_axes(data.reshape(basis.shape), axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    mesh = ax.pcolormesh(*coordinates, measured_data, shading="nearest")
    clim = _get_lim((None, None), measure, measured_data)
    norm = _get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    ax.text(
        0.05,
        0.95,
        f"x = {idx}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    return fig, ax, mesh


def build_animation(
    build_frame: Callable[[int, Axes], QuadMesh | AxesImage],
    n: int,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Build an animation from the data, set the scale and clim to the correct values.

    Parameters
    ----------
    build_frame : Callable[[int, Axes], QuadMesh | AxesImage]
        function to generate each frame
    n : int
        number of frames to generate
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        plot clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)

    mesh0 = build_frame(0, ax)
    frames = [[build_frame(d, ax)] for d in range(n)]

    clim = _get_lim(clim, measure, np.array([i[0].get_clim() for i in frames]))
    norm = _get_norm_with_lim(scale, clim)
    for (mesh,) in frames:
        mesh.set_norm(norm)
        mesh.set_clim(*clim)
    mesh0.set_norm(norm)
    mesh0.set_clim(*clim)

    return (fig, ax, ArtistAnimation(fig, frames))


_L0Inv = TypeVar("_L0Inv", bound=int)


# ruff: noqa: PLR0913


def animate_data_through_surface_x(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data on a given coordinate grid in 3D, animate through the surface.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    idx = tuple(0 for _ in range(basis.ndim - 3)) if idx is None else idx
    clim = (0.0, clim[1]) if clim[0] is None and measure == "abs" else clim

    coordinates = get_x_coordinates_in_axes(basis, axes, idx)
    data_in_axis = get_data_in_axes(data.reshape(basis.shape), axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    fig, ax, ani = build_animation(
        lambda i, ax: ax.pcolormesh(
            *coordinates[:2, :, :, i],
            measured_data[:, :, i],
            shading="nearest",
        ),
        data.shape[2],
        ax=ax,
        scale=scale,
        clim=clim,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(ax.collections[0], ax=ax, format="%4.1e")  # type: ignore Type of "collections" is unknown

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    return fig, ax, ani


def animate_data_through_list_1d_x(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    axes: tuple[int,] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)

    frames: list[list[Line2D]] = []

    for data_i in data:
        _, _, line = plot_data_1d_x(
            basis, data_i, axes, idx, ax=ax, scale=scale, measure=measure
        )
        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)
    return fig, ax, ani


def animate_data_through_list_1d_k(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    axes: tuple[int,] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)

    frames: list[list[Line2D]] = []

    for data_i in data:
        _, _, line = plot_data_1d_k(
            basis, data_i, axes, idx, ax=ax, scale=scale, measure=measure
        )
        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)

    return fig, ax, ani
