from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.basis.util import (
    get_k_coordinates_in_axes,
    get_x_coordinates_in_axes,
)
from surface_potential_analysis.state_vector.plot import (
    animate_eigenstate_3d_x,
    plot_eigenstate_2d_x,
    plot_state_vector_1d_x,
    plot_state_vector_2d_k,
    plot_state_vector_along_path,
    plot_state_vector_difference_2d_x,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

from .wavepacket import (
    Wavepacket,
    Wavepacket3d,
    get_sample_basis,
    get_wavepacket_sample_frequencies,
)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import (
        SingleFlatIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.util.plot import Scale

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _S03dInv = TypeVar("_S03dInv", bound=tuple[int, int, int])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


def plot_wavepacket_sample_frequencies(
    wavepacket: Wavepacket3dWith2dSamples[_NS0Inv, _NS1Inv, _B3d0Inv],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the frequencies used to sample the wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    frequencies = get_wavepacket_sample_frequencies(
        wavepacket["basis"], np.array(wavepacket["vectors"].shape)[0:2]
    )[:2, :]
    (line,) = ax.plot(*frequencies.reshape(2, -1))
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_xlabel("kx /$m^{-1}$")
    ax.set_ylabel("ky /$m^{-1}$")
    ax.set_title("Plot of sample points in the wavepacket")

    return fig, ax, line


def plot_wavepacket_energies_2d_k(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    basis = get_sample_basis(wavepacket["basis"], wavepacket["shape"])

    coordinates = get_k_coordinates_in_axes(basis, axes, idx)
    points = np.fft.ifftshift(wavepacket["energies"])

    shifted_coordinates = np.fft.ifftshift(coordinates, axes=(1, 2))

    mesh = ax.pcolormesh(*shifted_coordinates, points, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel("kx axis")
    ax.set_ylabel("ky axis")
    ax.set_title("Plot of the band energies against momentum")

    return fig, ax, mesh


def plot_wavepacket_energies_2d_x(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the fourier transform of energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
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
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    basis = get_sample_basis(wavepacket["basis"], wavepacket["shape"])
    coordinates = get_x_coordinates_in_axes(basis, axes, idx)

    data = np.fft.ifft2(wavepacket["energies"])
    data[0, 0] = 0
    points = get_measured_data(data, measure)

    mesh = ax.pcolormesh(*coordinates, points, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title("Plot of the fourier transform of the band energies against position")

    return fig, ax, mesh


def plot_wavepacket_1d_x(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: tuple[int, ...] | None = None,
    axis: int = 0,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the given axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through axis perpendicular to axis, by default (0,0)
    axis : Literal[0, 1, 2,, optional
        axis along which to plot, by default 2
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
    state = unfurl_wavepacket(wavepacket)
    return plot_state_vector_1d_x(state, axis, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_x0(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the x0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through x1, x2 axis, by default (0,0)
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
    state = unfurl_wavepacket(wavepacket)
    return plot_state_vector_1d_x(state, 0, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_x1(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the x1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through x2, x0 axis, by default (0,0)
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
    state = unfurl_wavepacket(wavepacket)
    return plot_state_vector_1d_x(state, 1, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_x2(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the x2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through x0, x1 axis, by default (0,0)
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
    state = unfurl_wavepacket(wavepacket)
    return plot_state_vector_1d_x(state, 2, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_2d_k(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket in 2D at idx along the given axis in momentum.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : SingleFlatIndexLike
        index along z_axis
    kz_axis : Literal[0, 1, 2]
        kz_axis, perpendicular to plotted direction
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
    state = unfurl_wavepacket(wavepacket)
    return plot_state_vector_2d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_k0k1(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    k2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the k2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    k2_idx : SingleFlatIndexLike
        index along k2 axis
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
    return plot_wavepacket_2d_k(
        wavepacket, (0, 1), (k2_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_k1k2(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    k0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the k0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    k0_idx : SingleFlatIndexLike
        index along k0 axis
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
    return plot_wavepacket_2d_k(
        wavepacket, (1, 2), (k0_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_k2k0(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    k1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the k1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    k1_idx : SingleFlatIndexLike
        index along k1 axis
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
    return plot_wavepacket_2d_k(
        wavepacket, (2, 0), (k1_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_2d_x(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket in 2D at idx along the given axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2]
        z_axis, perpendicular to plotted direction
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
    state = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_2d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_wavepacket_x0x1(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    x2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x2_idx : SingleFlatIndexLike
        index along x2 axis
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
    return plot_wavepacket_2d_x(
        wavepacket, (0, 1), (x2_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x1x2(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    x0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x0_idx : SingleFlatIndexLike
        index along x0 axis
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
    return plot_wavepacket_2d_x(
        wavepacket, (1, 2), (x0_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x2x0(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    x1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x1_idx : SingleFlatIndexLike
        index along x1 axis
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
    return plot_wavepacket_2d_x(
        wavepacket, (2, 0), (x1_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_difference_2d_x(
    wavepacket_0: Wavepacket[_S0Inv, _B0Inv],
    wavepacket_1: Wavepacket[_S0Inv, _B0Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two wavepackets in 2D.

    Parameters
    ----------
    wavepacket_0 : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    wavepacket_1 : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
        index along z_axis to plot
    z_axis : Literal[0, 1, 2,
        direction perpendicular to which to plot
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
    eigenstate_0 = unfurl_wavepacket(wavepacket_0)
    eigenstate_1 = unfurl_wavepacket(wavepacket_1)

    return plot_state_vector_difference_2d_x(
        eigenstate_0, eigenstate_1, axes, idx, ax=ax, measure=measure, scale=scale  # type: ignore[arg-type]
    )


def animate_wavepacket_3d_x(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    axes: tuple[int, int],
    z_axis: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to z_axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2,
        direction along which to animate
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    state = unfurl_wavepacket(wavepacket)
    return animate_eigenstate_3d_x(
        state, axes, z_axis, ax=ax, measure=measure, scale=scale  # type: ignore[arg-type]
    )


def animate_wavepacket_x0x1(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to x2.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_wavepacket_3d_x(
        wavepacket, (0, 1), 2, ax=ax, measure=measure, scale=scale
    )


def animate_wavepacket_x1x2(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to x0.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_wavepacket_3d_x(
        wavepacket, (1, 2), 0, ax=ax, measure=measure, scale=scale
    )


def animate_wavepacket_x2x0(
    wavepacket: Wavepacket3d[_S03dInv, _B3d0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to x1.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_wavepacket_3d_x(
        wavepacket, (2, 0), 1, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_along_path(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the wavepacket along the given path.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[int, int], np.dtype[np.int_]]
        path to plot, as a list of x0,x1,x2 coordinate lists
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstate: StateVector[Any] = unfurl_wavepacket(wavepacket)
    return plot_state_vector_along_path(
        eigenstate, path, ax=ax, measure=measure, scale=scale
    )
