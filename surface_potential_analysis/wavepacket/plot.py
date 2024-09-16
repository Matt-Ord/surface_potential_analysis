from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib.lines import Line2D
from scipy.constants import Boltzmann, hbar  # type: ignore lib

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
    stacked_basis_as_fundamental_transformed_basis,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_3d_x,
    get_average_band_energy,
    plot_state_1d_k,
    plot_state_1d_x,
    plot_state_2d_k,
    plot_state_2d_x,
    plot_state_along_path,
    plot_state_difference_2d_x,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_nx,
)
from surface_potential_analysis.util.plot import (
    get_figure,
    plot_data_1d,
    plot_data_1d_k,
    plot_data_1d_x,
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
    get_full_bloch_hamiltonian,
    get_full_wannier_hamiltonian,
)
from surface_potential_analysis.wavepacket.localization._wannier90 import (
    get_localization_operator_wannier90_individual_bands,
)

from .wavepacket import (
    BlochWavefunctionList,
    BlochWavefunctionListWithEigenvalues,
    BlochWavefunctionListWithEigenvaluesList,
    get_fundamental_sample_basis,
    get_wavepacket_basis,
    get_wavepacket_fundamental_sample_frequencies,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure

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
    from surface_potential_analysis.state_vector.eigenstate_list import (
        EigenstateList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.wavepacket.localization_operator import (
        LocalizationOperator,
    )

    _SB1 = TypeVar("_SB1", bound=StackedBasisLike[Any, Any, Any])
    _SBV1 = TypeVar("_SBV1", bound=StackedBasisWithVolumeLike[Any, Any, Any])
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
        TupleBasisWithLengthLike[*tuple[Any, ...]],
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

    frequencies = get_wavepacket_fundamental_sample_frequencies(
        wavepacket["basis"]
    ).reshape(-1, *wavepacket["basis"][0].shape)
    frequencies = frequencies[list(axes), *slice_ignoring_axes(idx, axes)]
    (line,) = ax.plot(*frequencies.reshape(2, -1))  # type: ignore lib
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_xlabel(f"k{axes[0]} /$m^{-1}$")  # type: ignore lib
    ax.set_ylabel(f"k{axes[1]} /$m^{-1}$")  # type: ignore lib
    ax.set_title("Plot of sample points in the wavepacket")  # type: ignore lib

    return fig, ax, line


def plot_wavepacket_eigenvalues_2d_k(
    wavepacket: BlochWavefunctionListWithEigenvalues[
        TupleBasisLike[*tuple[Any, ...]], _SBV0
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
    basis = get_fundamental_sample_basis(wavepacket["basis"])

    fig, ax, mesh = plot_data_2d_k(
        basis, wavepacket["eigenvalue"], axes, idx, ax=ax, scale=scale, measure=measure
    )
    ax.set_title("Plot of band energies against momentum")  # type: ignore lib
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
    sorted_fractions = np.argsort(bloch_fractions)

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
    ax.set_xlabel("Bloch Phase")  # type: ignore lib
    ax.set_ylabel("Energy / J")  # type: ignore lib

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
) -> tuple[Figure, Axes, list[Line2D]]:
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
        list_basis=stacked_basis_as_fundamental_transformed_basis(
            wavepacket["basis"][0][1]
        ),
    )

    bands = list(range(converted["basis"][0][0].n)) if bands is None else bands
    data = converted["eigenvalue"].reshape(converted["basis"][0][0].n, -1)[bands, :]

    fig, ax = get_figure(ax)
    lines = list[Line2D]()
    basis = get_fundamental_sample_basis(get_wavepacket_basis(wavepacket["basis"]))
    for band_data in data:
        _, _, line = plot_data_1d_k(
            basis,
            band_data,
            axes=axes,
            ax=ax,
            scale=scale,
            measure=measure,
        )
        line.set_linestyle("--")
        line.set_marker("x")
        lines.append(line)
    ax.set_xlabel("K")  # type: ignore lib
    ax.set_ylabel("Energy / J")  # type: ignore lib

    return fig, ax, lines


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
        list_basis=stacked_basis_as_fundamental_transformed_basis(
            wavepacket["basis"][0][1]
        ),
    )

    bands = list(range(converted["basis"][0][0].n)) if bands is None else bands
    data = converted["eigenvalue"].reshape(converted["basis"][0][0].n, -1)[bands, :]

    fig, ax = get_figure(ax)

    basis = get_fundamental_sample_basis(get_wavepacket_basis(wavepacket["basis"]))
    for band_data in data:
        _, _, line = plot_data_1d_x(
            basis,
            band_data,
            axes=axes,
            ax=ax,
            scale=scale,
            measure=measure,
        )
        line.set_linestyle("--")
        line.set_marker("x")
    ax.set_xlabel("Delta X")  # type: ignore lib
    ax.set_ylabel("Energy / J")  # type: ignore lib

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
    data = converted["eigenvalue"].reshape(
        converted["basis"][0][0].n, *converted["basis"][0][1].shape
    )[bands, :]

    list_basis = converted["basis"][0][1]

    nx_points = BasisUtil(wavepacket["basis"][0]).nx_points[bands]
    fig, ax, line = plot_data_1d(
        data[
            :,
            *tuple(1 if i == axes[0] else 0 for i in range(list_basis.ndim)),
        ],
        nx_points.astype(np.float64),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_linestyle("--")
    line.set_marker("x")
    line.set_label("lowest fourier componet")

    ax.set_xlabel("Band Index")  # type: ignore lib
    ax.set_ylabel("Energy / J")  # type: ignore lib

    if free_mass is not None:
        delta_x = np.linalg.norm(wavepacket["basis"][1].delta_x_stacked[axes[0]])
        norm = delta_x * np.sqrt(wavepacket["basis"][0][1].n) / (2 * np.pi)
        # By integrating explicitly we find
        # |E(\Delta x)| = (\Delta x)^{-3}(8\pi N + 4 \pi)
        # we add an additional np.sqrt(wavepacket["basis"][0][1].n) * delta_x / (2 * np.pi)
        # to account for the difference in fourier transform definitions

        offset = norm * ((4 * np.pi * hbar**2) / (2 * free_mass * delta_x**3))
        points = (2 * nx_points + 1) * offset

        (line,) = ax.plot(nx_points, points)  # type: ignore lib
        line.set_label("free particle")

    return fig, ax, line


def _get_free_energy(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    bands: np.ndarray[Any, np.dtype[np.int_]],
    axis: int = 0,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    # By integrating explicitly we find
    # |E(\Delta x)| = (\Delta x)^{-3}(8\pi N + 4 \pi)
    delta_x = np.linalg.norm(basis.delta_x_stacked[axis])
    return (4 * np.pi) * (2 * bands + 1) / delta_x**3


def get_wavepacket_transformed_energy_effective_mass(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axis: int = 0,
) -> ValueList[_B0]:
    """
    Get the effective mass of a wavepacket band along the axis direction.

    Parameters
    ----------
    wavepacket : BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SBV0]
    axis : int, optional
        axis index, by default 0

    Returns
    -------
    ValueList[_B0]
    """
    # E(\Delta x)
    converted = convert_wavepacket_with_eigenvalues_to_basis(
        wavefunctions,
        list_basis=stacked_basis_as_fundamental_basis(wavefunctions["basis"][0][1]),
    )

    nx_points = BasisUtil(wavefunctions["basis"][0][0]).nx_points
    data = converted["eigenvalue"].reshape(-1, *converted["basis"][0][1].shape)
    list_basis = converted["basis"][0][1]
    sliced_data = data[:, *tuple(1 if i == axis else 0 for i in range(list_basis.ndim))]

    actual_energy = np.abs(sliced_data)
    free_energy = _get_free_energy(wavefunctions["basis"][1], nx_points, axis)
    # By integrating explicitly we find
    # |E(\Delta x)| = (\Delta x)^{-3}(8\pi N + 4 \pi)
    # we add an additional np.sqrt(wavepacket["basis"][0][1].n) * delta_x / (2 * np.pi)
    # to account for the difference in fourier transform definitions
    delta_x = np.linalg.norm(wavefunctions["basis"][1].delta_x_stacked[axis])
    norm = np.sqrt(wavefunctions["basis"][0][1].n) * delta_x / (2 * np.pi)
    norm *= hbar**2 / 2
    effective_mass = norm * free_energy / actual_energy
    return {"basis": wavefunctions["basis"][0][0], "data": effective_mass}


def plot_wavepacket_transformed_energy_effective_mass_against_energy(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    *,
    true_mass: float | None = None,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]

    """
    energies = get_average_band_energy(get_full_bloch_hamiltonian(wavepacket))["data"]
    masses = get_wavepacket_transformed_energy_effective_mass(wavepacket, axes[0])
    data = (
        masses["data"]
        if true_mass is None
        else (masses["data"] - true_mass) / true_mass
    )
    fig, ax, line = plot_data_1d(
        data,
        np.real(energies),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_label("Effective Mass")

    ax.set_xlabel("Average Band Energy /J")  # type: ignore library type
    if true_mass:
        ax.set_ylabel("Mass - True Mass / True Mass")  # type: ignore library type
    else:
        ax.set_ylabel("Mass / kg")  # type: ignore library type
    ax.set_ylim((0.0, ax.get_ylim()[1]))

    return fig, ax, line


def plot_wavepacket_transformed_energy_effective_mass_against_band(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
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
    fig, ax, line = plot_value_list_against_nx(
        get_wavepacket_transformed_energy_effective_mass(wavepacket, axes[0]),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_marker("x")
    line.set_label("Effective Mass")

    ax.set_xlabel("Band Index")  # type: ignore lib
    ax.set_ylabel("Mass / Kg")  # type: ignore lib
    ax.set_ylim([0, ax.get_ylim()[1]])  # type: ignore lib

    return fig, ax, line


def get_wavepacket_localized_effective_mass(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    operator: LocalizationOperator[_SB0, _B0, _B0],
    axis: int = 0,
) -> ValueList[_B0]:
    """
    Get the effective mass of a wavepacket band along the axis direction.

    Parameters
    ----------
    wavepacket : BlochWavefunctionListWithEigenvaluesList[ _B0, _SB0, _SBV0, ]
    axis : int, optional
        axis index, by default 0

    Returns
    -------
    ValueList[_B0]
    """
    # E(\Delta x)
    localized = get_full_wannier_hamiltonian(wavefunctions, operator)
    diagonal = np.einsum(  # type: ignore lib
        "ijik->ijk",
        localized["data"].reshape(
            *localized["basis"][0].vectors["basis"][0].shape,
            *localized["basis"][1].vectors["basis"][0].shape,
        ),
    )[:, 0, :]

    list_basis = localized["basis"][0].vectors["basis"][0][1]
    actual_energy = diagonal.reshape(-1, *list_basis.shape)[
        :, *tuple(1 if i == axis else 0 for i in range(list_basis.ndim))
    ]
    actual_energy = np.abs(actual_energy)
    nx_points = BasisUtil(wavefunctions["basis"][0][0]).nx_points
    free_energy = _get_free_energy(wavefunctions["basis"][1], nx_points, axis)
    # By integrating explicitly we find
    # |E(\Delta x)| = (\Delta x)^{-3}(8\pi N + 4 \pi)
    # we add an additional np.sqrt(wavepacket["basis"][0][1].n) * delta_x / (2 * np.pi)
    # to account for the difference in fourier transform definitions
    delta_x = np.linalg.norm(wavefunctions["basis"][1].delta_x_stacked[axis])
    norm = np.sqrt(wavefunctions["basis"][0][1].n) * delta_x / (2 * np.pi)
    norm *= hbar**2 / 2
    effective_mass = norm * free_energy / actual_energy
    return {"basis": wavefunctions["basis"][0][0], "data": effective_mass}


def plot_wavepacket_localized_effective_mass_against_energy(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    axes: tuple[int,] = (0,),
    *,
    true_mass: float | None = None,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]

    """
    energies = get_average_band_energy(get_full_bloch_hamiltonian(wavepacket))["data"]
    operator = get_localization_operator_wannier90_individual_bands(wavepacket)
    masses = get_wavepacket_localized_effective_mass(wavepacket, operator, axes[0])
    data = (
        masses["data"]
        if true_mass is None
        else (masses["data"] - true_mass) / true_mass
    )
    fig, ax, line = plot_data_1d(
        data,
        np.real(energies),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_label("Effective Mass")

    ax.set_xlabel("Average Band Energy /J")  # type: ignore library type
    if true_mass:
        ax.set_ylabel("Mass - True Mass / True Mass")  # type: ignore library type
    else:
        ax.set_ylabel("Mass / kg")  # type: ignore library type
    ax.set_ylim((0.0, ax.get_ylim()[1]))

    return fig, ax, line


def plot_wavepacket_eigenvalues_2d_x(
    wavepacket: BlochWavefunctionListWithEigenvalues[
        TupleBasisLike[*tuple[Any, ...]],
        _SBV0,
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
    basis = get_fundamental_sample_basis(wavepacket["basis"])

    fig, ax, mesh = plot_data_2d_x(
        basis, wavepacket["eigenvalue"], axes, idx, ax=ax, scale=scale, measure=measure
    )
    ax.set_title("Plot of the fourier transform of the band energies against position")  # type: ignore lib
    return fig, ax, mesh


def plot_eigenvalues_1d_x(
    wavepacket: BlochWavefunctionListWithEigenvalues[
        TupleBasisLike[*tuple[Any, ...]], _SBV0
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
    (line,) = ax.plot(eigenvalues)  # type: ignore lib
    ax.set_yscale(scale)  # type: ignore lib
    ax.set_xlabel("Bloch Phase")  # type: ignore lib
    ax.set_ylabel("Energy / J")  # type: ignore lib
    return fig, ax, line


def plot_wavepacket_1d_x(
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    wavepacket_0: BlochWavefunctionList[_SB0, _SBV0],
    wavepacket_1: BlochWavefunctionList[_SB1, _SBV1],
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
    wavepacket: BlochWavefunctionList[TupleBasisLike[*tuple[Any, ...]], _SBV0],
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
    return animate_state_3d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)


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


def get_wavepacket_band_occupation(
    wavepacket: EigenstateList[TupleBasisLike[_B0, _B1], _B2],
    temperature: float,
) -> ValueList[_B0]:
    """
    Get the occupation of each band of a wavepacket.

    Parameters
    ----------
    wavepacket : EigenstateList[TupleBasisLike[_B0, _B1], _B2]
    temperature : float

    Returns
    -------
    ValueList[_B0]
    """
    eigenvalues = wavepacket["eigenvalue"].reshape(*wavepacket["basis"][0].shape)
    occupations = np.exp(-eigenvalues / (temperature * Boltzmann))
    occupation_for_band = np.sum(occupations, axis=1) / np.sum(occupations)
    return {"basis": wavepacket["basis"][0][0], "data": occupation_for_band.ravel()}


def plot_occupation_against_band(
    wavepacket: EigenstateList[TupleBasisLike[_B0, _B1], _B2],
    temperature: float,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
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
    fig, ax, line = plot_value_list_against_nx(
        get_wavepacket_band_occupation(wavepacket, temperature),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_label("Occupation")
    ax.set_xlabel("Band Idx")  # type:ignore lib
    ax.set_ylabel("Occupation / Au")  # type:ignore lib
    return fig, ax, line


def plot_occupation_against_band_average_energy(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SBV0],
    temperature: float,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
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
    fig, ax, line = plot_data_1d(
        get_wavepacket_band_occupation(wavepacket, temperature)["data"],
        get_average_band_energy(get_full_bloch_hamiltonian(wavepacket))["data"].astype(
            np.float64
        ),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_label("Occupation")
    ax.set_xlabel("Band Idx")  # type:ignore lib
    ax.set_ylabel("Occupation / Au")  # type:ignore lib
    return fig, ax, line
