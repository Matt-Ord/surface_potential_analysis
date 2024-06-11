from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.kernel.kernel import (
    SingleBasisDiagonalNoiseKernel,
    get_single_factorized_noise_operators,
    get_single_factorized_noise_operators_diagonal,
)
from surface_potential_analysis.operator.operator_list import (
    select_operator_diagonal,
)
from surface_potential_analysis.util.plot import (
    Scale,
    _get_lim,
    _get_norm_with_lim,
    _get_scale_with_lim,
    get_figure,
    plot_data_1d_x,
)
from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        NoiseKernel,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike


def plot_diagonal_kernel(
    kernel: DiagonalNoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the elements of a diagonal kernel.

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[Any, Any, Any, Any]
    ax : Axes | None, optional
        axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]

    """
    fig, ax = get_figure(ax)
    n_states = kernel["basis"][0].shape[0]
    data = kernel["data"].reshape(n_states, n_states)
    measured_data = get_measured_data(data, measure)

    mesh = ax.pcolormesh(measured_data)
    clim = _get_lim((None, None), measure, measured_data)
    norm = _get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def plot_kernel(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = get_figure(ax)
    data = kernel["data"].reshape(kernel["basis"].shape)
    measured_data = get_measured_data(data, measure)

    mesh = ax.pcolormesh(measured_data)
    clim = _get_lim((None, None), measure, measured_data)
    norm = _get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def plot_kernel_sparsity(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_single_factorized_noise_operators(kernel)

    data = get_measured_data(operators["eigenvalue"], measure)
    bins = np.logspace(
        np.log10(np.min(data)), np.log10(np.max(data)), 10000, dtype=np.float64
    )
    values, bins = np.histogram(data, bins=bins)

    cumulative = np.cumsum(values)
    ax.plot(bins[:-1], cumulative)

    ax.set_yscale(scale)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_xscale("log")

    return fig, ax


def plot_kernel_truncation_error(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    truncations: list[int] | None = None,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_single_factorized_noise_operators(kernel)
    sorted_eigenvalues = np.sort(np.abs(operators["eigenvalue"]))
    cumulative = np.empty(sorted_eigenvalues.size + 1)
    cumulative[0] = 0
    np.cumsum(sorted_eigenvalues, out=cumulative[1:])

    truncations = (
        list(range(0, cumulative.size, 1 + cumulative.size // 100))
        if truncations is None
        else truncations
    )

    ax.plot(truncations, cumulative[truncations])
    ax.set_yscale(scale)

    return fig, ax


def plot_diagonal_kernel_truncation_error(
    kernel: DiagonalNoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_single_factorized_noise_operators_diagonal(kernel)
    eigenvalues = np.sort(np.abs(operators["eigenvalue"]))
    cumulative = np.empty(eigenvalues.size + 1)
    cumulative[0] = 0
    np.cumsum(eigenvalues, out=cumulative[1:])

    truncations = np.arange(cumulative.size)

    ax.plot(truncations, cumulative)
    ax.set_yscale(_get_scale_with_lim(scale, ax.get_ylim()))

    return fig, ax


def plot_effective_potential_single_sample(
    kernel: SingleBasisDiagonalNoiseKernel[
        TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]],
    ],
    truncation: int | None = None,
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    truncation = kernel["basis"][0].n if truncation is None else truncation
    operators = get_single_factorized_noise_operators_diagonal(kernel)

    rng = np.random.default_rng()
    factors = (1 / np.sqrt(2)) * (
        rng.normal(size=truncation) + 1j * rng.normal(size=truncation)
    )

    measured_potential = np.zeros(operators["basis"][1][0].n, dtype=np.complex128)
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]
    for i, factor in zip(args[: truncation - 1], factors):
        operator = select_operator_diagonal(
            operators,
            idx=i,
        )
        measured_potential += (
            factor
            * np.lib.scimath.sqrt(operators["eigenvalue"][i] * hbar)
            * operator["data"]
        )

    ax.set_ylabel("Energy /J")

    return plot_data_1d_x(
        operators["basis"][1][0],
        measured_potential,
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
