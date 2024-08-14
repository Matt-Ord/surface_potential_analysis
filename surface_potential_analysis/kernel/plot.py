from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, TypeVarTuple

import numpy as np
from scipy.constants import hbar  # type: ignore no stub

from surface_potential_analysis.basis.basis import FundamentalPositionBasis
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.kernel.build import (
    truncate_diagonal_noise_operator_list,
)
from surface_potential_analysis.kernel.conversion import (
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.kernel import (
    as_diagonal_noise_operators_from_full,
)
from surface_potential_analysis.kernel.solve import (
    get_noise_operators_diagonal_eigenvalue,
    get_noise_operators_eigenvalue,
)
from surface_potential_analysis.operator.operator_list import (
    select_diagonal_operator,
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
    plot_data_2d_x,
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
    )
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        DiagonalNoiseOperatorList,
        IsotropicNoiseKernel,
        NoiseKernel,
        NoiseOperatorList,
        SingleBasisDiagonalNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike

    _B0s = TypeVarTuple("_B0s")


def plot_diagonal_kernel_2d(
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
    data = np.fft.fftshift(data)

    return plot_data_2d(data, ax=ax, scale=scale, measure=measure)


def plot_kernel_2d(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot a kernel in 2d.

    Parameters
    ----------
    kernel : NoiseKernel[Any, Any, Any, Any]
    ax : Axes | None, optional
        ax, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    data = kernel["data"].reshape(kernel["basis"].shape)
    return plot_data_2d(data, ax=ax, scale=scale, measure=measure)


def plot_kernel_sparsity(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot sparsity of a kernel.

    Parameters
    ----------
    kernel : NoiseKernel[Any, Any, Any, Any]
    ax : Axes | None, optional
        ax, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    operators = get_noise_operators_eigenvalue(kernel)

    data = get_measured_data(operators["eigenvalue"], measure)
    bins = np.logspace(
        np.log10(np.min(data)), np.log10(np.max(data)), 10000, dtype=np.float64
    )
    values, bins = np.histogram(data, bins=bins)

    cumulative = np.cumsum(values)
    ax.plot(bins[:-1], cumulative)  # type: ignore library type

    ax.set_yscale(scale)  # type: ignore library type
    ax.set_xlabel("Value")  # type: ignore library type
    ax.set_ylabel("Density")  # type: ignore library type
    ax.set_xscale("log")  # type: ignore library type

    return fig, ax


def plot_kernel_truncation_error(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    truncations: list[int] | None = None,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the error from truncating a kernel.

    Parameters
    ----------
    kernel : NoiseKernel[Any, Any, Any, Any]
    truncations : list[int] | None, optional
        truncations, by default None
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    operators = get_noise_operators_eigenvalue(kernel)
    sorted_eigenvalues = np.sort(np.abs(operators["eigenvalue"]))
    cumulative = np.empty(sorted_eigenvalues.size + 1)
    cumulative[0] = 0
    np.cumsum(sorted_eigenvalues, out=cumulative[1:])

    truncations = (
        list(range(0, cumulative.size, 1 + cumulative.size // 100))
        if truncations is None
        else truncations
    )

    ax.plot(truncations, cumulative[truncations])  # type: ignore library type
    ax.set_yscale(scale)  # type: ignore library type

    return fig, ax


def plot_diagonal_kernel_truncation_error(
    kernel: DiagonalNoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """Plot the error from truncating a diagonal kernel.

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[Any, Any, Any, Any]
        kernel
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    operators = get_noise_operators_diagonal_eigenvalue(kernel)
    eigenvalues = np.sort(np.abs(operators["eigenvalue"]))
    cumulative = np.empty(eigenvalues.size + 1)
    cumulative[0] = 0
    np.cumsum(eigenvalues, out=cumulative[1:])

    truncations = np.arange(cumulative.size)

    return plot_data_1d(
        cumulative.astype(np.complex128),
        truncations.astype(np.float64),
        ax=ax,
        scale=scale,
    )


def plot_diagonal_noise_operators_single_sample(  # noqa: PLR0913
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
    truncated = truncate_diagonal_noise_operator_list(operators, truncation)

    rng = np.random.default_rng()
    factors = (1 / np.sqrt(2)) * (
        rng.normal(size=truncated["eigenvalue"].size)
        + 1j * rng.normal(size=truncated["eigenvalue"].size)
    )

    measured_potential = np.zeros(truncated["basis"][1][0].n, dtype=np.complex128)
    for i, factor in enumerate(factors):
        operator = select_diagonal_operator(
            operators,
            idx=i,
        )
        measured_potential += (
            factor
            * np.lib.scimath.sqrt(operators["eigenvalue"][i] * hbar)
            * operator["data"]
        )

    measured_potential -= measured_potential[0]

    ax.set_ylabel("Energy /J")  # type: ignore library type

    return plot_data_1d_x(
        operators["basis"][1][0],
        measured_potential,
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_noise_operators_single_sample_x(  # noqa: PLR0913
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
        as_diagonal_noise_operators_from_full(converted),
        truncation=truncation,
        axes=axes,
        idx=idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_noise_kernel_single_sample(  # noqa: PLR0913
    kernel: SingleBasisDiagonalNoiseKernel[
        TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]],
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
    operators = get_noise_operators_diagonal_eigenvalue(kernel)

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

    (line,) = ax.plot(eigenvalues[args])  # type: ignore library type
    ax.set_ylabel("Eigenvalue")  # type: ignore library type
    ax.set_yscale(scale)  # type: ignore library type

    return fig, ax, line


def plot_isotropic_noise_kernel_1d_x(  # noqa: PLR0913
    kernel: IsotropicNoiseKernel[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an isotropic kernal in 1d.

    Parameters
    ----------
    kernel : IsotropicNoiseKernel[StackedBasisWithVolumeLike[Any, Any, Any]]
    axes : tuple[int], optional
        axes, by default (0,)
    idx : SingleStackedIndexLike | None, optional
        idx, by default None
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "real"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax, line = plot_data_1d_x(
        kernel["basis"], kernel["data"], axes, idx, ax=ax, scale=scale, measure=measure
    )
    line.set_label(f"{measure} kernel")
    return fig, ax, line


def plot_isotropic_noise_kernel_2d_x(  # noqa: PLR0913
    kernel: IsotropicNoiseKernel[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an isotropic kernel in 1d.

    Parameters
    ----------
    kernel : IsotropicNoiseKernel[StackedBasisWithVolumeLike[Any, Any, Any]]
    axes : tuple[int], optional
        axes, by default (0,)
    idx : SingleStackedIndexLike | None, optional
        idx, by default None
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "real"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax, mesh = plot_data_2d_x(
        kernel["basis"], kernel["data"], axes, idx, ax=ax, scale=scale, measure=measure
    )
    return fig, ax, mesh
