from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, TypeVarTuple

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.kernel.conversion import (
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.kernel import (
    DiagonalNoiseOperatorList,
    NoiseOperatorList,
    SingleBasisDiagonalNoiseKernel,
    SingleBasisDiagonalNoiseOperatorList,
    as_diagonal_noise_operators,
    get_noise_operators,
    get_noise_operators_diagonal,
    truncate_diagonal_noise_operators,
)
from surface_potential_analysis.operator.operator_list import (
    select_operator_diagonal,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.plot import (
    Scale,
    get_figure,
    plot_data_1d,
    plot_data_1d_x,
    plot_data_2d,
)
from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisLike,
    )
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        NoiseKernel,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike

    _B0s = TypeVarTuple("_B0s")


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
    n_states = kernel["basis"][0].shape[0]
    data = kernel["data"].reshape(n_states, n_states)

    return plot_data_2d(data, ax=ax, scale=scale, measure=measure)


def plot_kernel(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    data = kernel["data"].reshape(kernel["basis"].shape)
    return plot_data_2d(data, ax=ax, scale=scale, measure=measure)


def plot_kernel_sparsity(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_noise_operators(kernel)

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

    operators = get_noise_operators(kernel)
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
) -> tuple[Figure, Axes, Line2D]:
    operators = get_noise_operators_diagonal(kernel)
    eigenvalues = np.sort(np.abs(operators["eigenvalue"]))
    cumulative = np.empty(eigenvalues.size + 1)
    cumulative[0] = 0
    np.cumsum(eigenvalues, out=cumulative[1:])

    truncations = np.arange(cumulative.size)

    return plot_data_1d(
        cumulative.astype(np.complex128),
        truncations.astype(np.complex128),
        ax=ax,
        scale=scale,
    )


def plot_diagonal_noise_operators_single_sample(
    operators: DiagonalNoiseOperatorList[
        FundamentalBasis[int],
        TupleBasisWithLengthLike[*_B0s],
        TupleBasisWithLengthLike[*_B0s],
    ],
    truncation: Iterable[int] | None = None,
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a single sample of the noise operators.

    Parameters
    ----------
    operators : SingleBasisDiagonalNoiseOperatorList[ FundamentalBasis[int], TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]], ]
    axes : tuple[int], optional
        axis to plot, by default (0,)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "real"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    truncation = (
        range(operators["eigenvalue"].size) if truncation is None else truncation
    )
    truncated = truncate_diagonal_noise_operators(operators, truncation)

    rng = np.random.default_rng()
    factors = (1 / np.sqrt(2)) * (
        rng.normal(size=truncated["eigenvalue"].size)
        + 1j * rng.normal(size=truncated["eigenvalue"].size)
    )

    measured_potential = np.zeros(truncated["basis"][1][0].n, dtype=np.complex128)
    for i, factor in enumerate(factors):
        operator = select_operator_diagonal(
            operators,
            idx=i,
        )
        measured_potential += (
            factor
            * np.lib.scimath.sqrt(operators["eigenvalue"][i] * hbar)
            * operator["data"]
        )

    measured_potential -= measured_potential[0]

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


def plot_noise_operators_single_sample_x(
    operators: NoiseOperatorList[
        FundamentalBasis[int],
        StackedBasisWithVolumeLike[Any, Any, Any],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    truncation: Iterable[int] | None = None,
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a single sample of the noise operators.

    Parameters
    ----------
    operators : SingleBasisDiagonalNoiseOperatorList[ FundamentalBasis[int], TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]], ]
    axes : tuple[int], optional
        axis to plot, by default (0,)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "real"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis_x = stacked_basis_as_fundamental_position_basis(operators["basis"][1][0])
    converted = convert_noise_operator_list_to_basis(
        operators, TupleBasis(basis_x, basis_x)
    )
    return plot_diagonal_noise_operators_single_sample(
        as_diagonal_noise_operators(converted),
        truncation=truncation,
        axes=axes,
        idx=idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_noise_kernel_single_sample(
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
    """
    Plot a singel sample form a noise kernel.

    Parameters
    ----------
    kernel : SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]], ]
    truncation : int | None, optional
        truncation, by default None
    axes : tuple[int], optional
        axes, by default (0,)
    idx : SingleStackedIndexLike | None, optional
        idx, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "real"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    operators = get_noise_operators_diagonal(kernel)

    return plot_diagonal_noise_operators_single_sample(
        operators,
        truncation,
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_diagonal_noise_operators_eigenvalues(
    operators: SingleBasisDiagonalNoiseOperatorList[
        FundamentalBasis[int],
        Any,
    ],
    truncation: int | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a single sample of the noise operators.

    Parameters
    ----------
    operators : SingleBasisDiagonalNoiseOperatorList[ FundamentalBasis[int], TupleBasisLike[FundamentalPositionBasis[Any, Literal[1]]], ]
    axes : tuple[int], optional
        axis to plot, by default (0,)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "real"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    eigenvalues = get_measured_data(operators["eigenvalue"], measure)
    args = np.argsort(eigenvalues)[:truncation:-1]

    (line,) = ax.plot(eigenvalues[args])
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale(scale)

    return fig, ax, line
