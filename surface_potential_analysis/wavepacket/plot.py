from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar

from surface_potential_analysis.basis.block_fraction_basis import (
    ExplicitBlockFractionBasis,
)
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_3d_x,
    plot_state_1d_k,
    plot_state_1d_x,
    plot_state_2d_k,
    plot_state_2d_x,
    plot_state_along_path,
    plot_state_difference_2d_x,
)
from surface_potential_analysis.util.plot import (
    get_figure,
    plot_data_1d,
    plot_data_2d_k,
    plot_data_2d_x,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_data_in_axes,
    slice_ignoring_axes,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_with_eigenvalues_to_basis,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_all_wavepacket_states,
)

from .wavepacket import (
    BlochWavefunctionList,
    BlochWavefunctionListWithEigenvalues,
    BlochWavefunctionListWithEigenvaluesList,
    get_sample_basis,
    get_wavepacket_sample_fractions,
    get_wavepacket_sample_frequencies,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.block_fraction_basis import (
        BasisWithBlockFractionLike,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
        StackedBasisWithVolumeLike,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import EigenstateList
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale

    _TB0Inv = TypeVar("_TB0Inv", bound=TupleBasisLike[*tuple[Any, ...]])
    _TB1Inv = TypeVar("_TB1Inv", bound=TupleBasisLike[*tuple[Any, ...]])
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[Any, Any, Any])
    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _BF0 = TypeVar("_BF0", bound=BasisWithBlockFractionLike[Any, Any])
# ruff: noqa: PLR0913


def plot_wavepacket_sample_frequencies(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]],
        TupleBasisLike[*tuple[Any, ...]],
    ],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
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
    fig, ax = get_figure(ax)
    util = BasisUtil(wavepacket["basis"])
    idx = tuple(0 for _ in range(util.ndim - len(axes))) if idx is None else idx

    frequencies = get_wavepacket_sample_frequencies(wavepacket["basis"]).reshape(
        -1, *wavepacket["basis"][0].shape
    )
    frequencies = frequencies[list(axes), slice_ignoring_axes(idx, axes)]
    (line,) = ax.plot(*frequencies.reshape(2, -1))
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_xlabel(f"k{axes[0]} /$m^{-1}$")
    ax.set_ylabel(f"k{axes[1]} /$m^{-1}$")
    ax.set_title("Plot of sample points in the wavepacket")

    return fig, ax, line


def plot_wavepacket_eigenvalues_2d_k(
    wavepacket: BlochWavefunctionListWithEigenvalues[
        TupleBasisLike[*tuple[Any, ...]],
        TupleBasisWithLengthLike[*tuple[Any, ...]],
    ],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    basis = get_sample_basis(wavepacket["basis"])
    data = np.fft.ifftshift(wavepacket["eigenvalue"])

    fig, ax, mesh = plot_data_2d_k(
        basis, data, axes, idx, ax=ax, scale=scale, measure=measure
    )
    ax.set_title("Plot of the band energies against momentum")
    return fig, ax, mesh


def _get_projected_bloch_phases(
    collection: EigenstateList[TupleBasisLike[_B0, _BF0], _SBV0],
    direction: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    util = BasisUtil(collection["basis"][1])
    bloch_phases = np.tensordot(
        collection["basis"][0][1].bloch_fractions,
        util.fundamental_dk_stacked,
        axes=(0, 0),
    )
    normalized_direction = direction / np.linalg.norm(direction)
    return np.dot(bloch_phases, normalized_direction)


def plot_uneven_wavepacket_eigenvalues_1d_k(
    wavepacket: EigenstateList[
        TupleBasisLike[_B0, _BF0],
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    bands: list[int] | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    direction = BasisUtil(wavepacket["basis"][1]).dk_stacked[axes[0]]
    bloch_fractions = _get_projected_bloch_phases(wavepacket, direction)
    sorted_fractions = np.arange(bloch_fractions.size)

    bands = list(range(wavepacket["basis"][0][0].n)) if bands is None else bands
    data = wavepacket["eigenvalue"].reshape(wavepacket["basis"][0].shape)[bands, :]

    fig, ax = get_figure(ax)
    for band_data in data:
        _, _, line = plot_data_1d(
            band_data[sorted_fractions],
            bloch_fractions[sorted_fractions],
            ax=ax,
            scale=scale,
            measure=measure,
        )
        line.set_linestyle("--")
        line.set_marker("x")
    ax.set_xlabel("Bloch Phase")
    ax.set_ylabel("Energy / J")

    return (fig, ax)


def plot_wavepacket_eigenvalues_1d_k(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    bands: list[int] | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    bloch_fractions = get_wavepacket_sample_fractions(wavepacket["basis"][0][1])

    return plot_uneven_wavepacket_eigenvalues_1d_k(
        {
            "basis": TupleBasis(
                TupleBasis(
                    wavepacket["basis"][0][0],
                    ExplicitBlockFractionBasis(bloch_fractions),
                ),
                wavepacket["basis"][1],
            ),
            "data": wavepacket["data"],
            "eigenvalue": wavepacket["eigenvalue"],
        },
        axes,
        bands,
        ax=ax,
        measure=measure,
        scale=scale,
    )


def plot_wavepacket_eigenvalues_1d_x(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    bands: list[int] | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_wavepacket_with_eigenvalues_to_basis(
        wavepacket,
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0][1]),
    )

    bands = list(range(converted["basis"][0][0].n)) if bands is None else bands
    data = converted["eigenvalue"].reshape(converted["basis"][0].shape)[bands, :]

    fig, ax = get_figure(ax)
    nx_points = BasisUtil(converted["basis"][0][1][axes[0]]).fundamental_nx_points
    for band_data in data:
        _, _, line = plot_data_1d(
            band_data,
            nx_points.astype(np.float64),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        line.set_linestyle("--")
        line.set_marker("x")
    ax.set_xlabel("Delta X")
    ax.set_ylabel("Energy / J")

    return (fig, ax)


def plot_wavepacket_transformed_energy_1d(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    free_mass: float | None = None,
    axes: tuple[int,] = (0,),
    bands: list[int] | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_wavepacket_with_eigenvalues_to_basis(
        wavepacket,
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0][1]),
    )

    bands = list(range(converted["basis"][0][0].n)) if bands is None else bands
    data = converted["eigenvalue"].reshape(converted["basis"][0].shape)[bands, :]

    list_basis = converted["basis"][0][1]

    nx_points = BasisUtil(wavepacket["basis"][0]).nx_points[bands]
    fig, ax, line = plot_data_1d(
        data[:, *tuple(1 if i == axes[0] else 0 for i in range(list_basis.ndim))],
        nx_points.astype(np.float64),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_linestyle("--")
    line.set_marker("x")
    line.set_label("lowest fourier componet")

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Energy / J")

    if free_mass is not None:
        delta_x = np.linalg.norm(wavepacket["basis"][1].delta_x_stacked[axes[0]])
        norm = delta_x * np.sqrt(wavepacket["basis"][0][1].n) / (2 * np.pi)
        # By integrating explicitly we find
        # |E(\Delta x)| = (\Delta x)^{-3}(8\pi N + 4 \pi)
        # we add an additional np.sqrt(wavepacket["basis"][0][1].n) * delta_x / (2 * np.pi)
        # to account for the difference in fourier transform definitions

        offset = norm * ((4 * np.pi * hbar**2) / (2 * free_mass * delta_x**3))
        points = (2 * nx_points + 1) * offset

        (line,) = ax.plot(nx_points, points)
        line.set_label("free particle")

    return (fig, ax, line)


def plot_wavepacket_transformed_energy_effective_mass_1d(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    bands: list[int] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_wavepacket_with_eigenvalues_to_basis(
        wavepacket,
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0][1]),
    )

    bands = list(range(converted["basis"][0][0].n)) if bands is None else bands
    nx_points = BasisUtil(wavepacket["basis"][0]).nx_points[bands]
    data = converted["eigenvalue"].reshape(converted["basis"][0].shape)[bands, :]
    list_basis = converted["basis"][0][1]
    sliced_data = data[
        :, *tuple(1 if i == axes[0] else 0 for i in range(list_basis.ndim))
    ]

    offset = np.abs(sliced_data) / (2 * nx_points + 1)
    # By integrating explicitly we find
    # |E(\Delta x)| = (\Delta x)^{-3}(8\pi N + 4 \pi)
    # we add an additional np.sqrt(wavepacket["basis"][0][1].n) * delta_x / (2 * np.pi)
    # to account for the difference in fourier transform definitions
    delta_x = np.linalg.norm(wavepacket["basis"][1].delta_x_stacked[axes[0]])
    norm = np.sqrt(wavepacket["basis"][0][1].n) * delta_x / (2 * np.pi)
    effective_mass = norm * ((4 * np.pi * hbar**2) / (2 * offset * delta_x**3))

    fig, ax, line = plot_data_1d(
        effective_mass,
        nx_points.astype(np.float64),
        ax=ax,
        scale=scale,
        measure="real",
    )
    line.set_linestyle("--")
    line.set_marker("x")
    line.set_label("Effective Mass")

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Mass / Kg")
    ax.set_ylim([0, ax.get_ylim()[1]])

    return (fig, ax, line)


def plot_wavepacket_eigenvalues_2d_x(
    wavepacket: BlochWavefunctionListWithEigenvalues[
        TupleBasisLike[*tuple[Any, ...]],
        TupleBasisWithLengthLike[*tuple[Any, ...]],
    ],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the fourier transform of energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : WavepacketWithEigenvalues[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
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
    basis = get_sample_basis(wavepacket["basis"])

    data = np.fft.ifft2(wavepacket["eigenvalue"])
    data[0, 0] = 0

    fig, ax, mesh = plot_data_2d_x(
        basis, data, axes, idx, ax=ax, scale=scale, measure=measure
    )
    ax.set_title("Plot of the fourier transform of the band energies against position")
    return fig, ax, mesh


def plot_eigenvalues_1d_x(
    wavepacket: BlochWavefunctionListWithEigenvalues[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
    axes: tuple[int,] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the eigenvalues in an eigenstate collection against their projected phases.

    Parameters
    ----------
    collection : EigenstateColllection[_B0Inv, _L0Inv]
    direction : np.ndarray[tuple[int], np.dtype[np.float_]]
    band : int, optional
        band to plot, by default 0
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)
    util = BasisUtil(wavepacket["basis"][0])
    idx = tuple(0 for _ in range(util.ndim - 1)) if idx is None else idx

    eigenvalues = get_data_in_axes(
        wavepacket["eigenvalue"].reshape(wavepacket["basis"][0].shape), axes, idx
    )
    (line,) = ax.plot(eigenvalues)
    ax.set_yscale(scale)
    ax.set_xlabel("Bloch Phase")
    ax.set_ylabel("Energy / J")
    return fig, ax, line


def plot_wavepacket_1d_x(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
    axes: tuple[int] = (0,),
    idx: tuple[int, ...] | None = None,
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
    return plot_state_1d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_1d_k(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
    axes: tuple[int] = (0,),
    idx: tuple[int, ...] | None = None,
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
    return plot_state_1d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_2d_k(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
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
    return plot_state_2d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_all_wavepacket_states_2d_k(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> Generator[tuple[Figure, Axes, QuadMesh], None, None]:
    """
    Plot all states in a wavepacket in k at idx.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : SingleFlatIndexLike
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    Generator[tuple[Figure, Axes, QuadMesh], None, None]
    """
    states = get_all_wavepacket_states(wavepacket)
    return (
        plot_state_2d_k(state, axes, idx, measure=measure, scale=scale)
        for state in states
    )


def plot_wavepacket_2d_x(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
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
    return plot_state_2d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_all_wavepacket_states_2d_x(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> Generator[tuple[Figure, Axes, QuadMesh], None, None]:
    """
    Plot all states in a wavepacket in x at idx.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : SingleFlatIndexLike
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    Generator[tuple[Figure, Axes, QuadMesh], None, None]
    """
    states = get_all_wavepacket_states(wavepacket)
    return (
        plot_state_2d_x(state, axes, idx, measure=measure, scale=scale)
        for state in states
    )


def plot_wavepacket_difference_2d_x(
    wavepacket_0: BlochWavefunctionList[_TB0Inv, _TB1Inv],
    wavepacket_1: BlochWavefunctionList[_TB0Inv, _TB1Inv],
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

    return plot_state_difference_2d_x(
        eigenstate_0,
        eigenstate_1,
        axes,
        idx,
        ax=ax,
        measure=measure,
        scale=scale,  # type: ignore[arg-type]
    )


def animate_wavepacket_3d_x(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to.

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
    state = unfurl_wavepacket(wavepacket)
    return animate_state_3d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)  # type: ignore[arg-type]


def plot_wavepacket_along_path(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
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
    return plot_state_along_path(eigenstate, path, ax=ax, measure=measure, scale=scale)


def plot_occupation_against_band(
    collection: EigenstateList[TupleBasisLike[_B0, _B1], _B2],
    temperature: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the eigenvalues in an eigenstate collection against their projected phases.

    Parameters
    ----------
    collection : EigenstateColllection[_B0Inv, _L0Inv]
    direction : np.ndarray[tuple[int], np.dtype[np.float_]]
    band : int, optional
        band to plot, by default 0
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    eigenvalues = collection["eigenvalue"].reshape(*collection["basis"][0].shape, -1)
    occupations = np.exp(-eigenvalues / (temperature * Boltzmann))
    occupation_for_band = np.sum(occupations, axis=0) / np.sum(occupations)
    (line,) = ax.plot(occupation_for_band)
    ax.set_xlabel("band idx")
    ax.set_ylabel("Occupation / Au")
    return fig, ax, line
